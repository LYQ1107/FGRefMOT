#!/usr/bin/env python3
"""Train the causal FlexHook temporal head from frozen feature archives.

The command consumes only the registered ``train_core`` and ``internal_val``
archives.  It never accepts historical-test archives, never reads test GT, and
keeps the upstream FlexHook logits frozen.  Each physical track/expression is
processed chronologically so the evidence buffer cannot see a future frame.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable

import torch
from torch import Tensor
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.flexhook_temporal.adapter import TemporalEvidenceAdapter, run_causal_sequence


@dataclass(frozen=True)
class Sequence:
    """One chronological physical-track/expression sequence."""

    key: tuple[str, str, str]
    frames: Tensor
    features: Tensor
    language: Tensor
    logits: Tensor
    labels: Tensor
    bboxes: Tensor
    track_id: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_archive(archive_root: Path, expected_split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = archive_root / "FEATURE_ARCHIVE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCEEDED":
        raise ValueError(f"feature archive is not successful: {manifest_path}")
    if manifest.get("split") != expected_split:
        raise ValueError(f"expected {expected_split!r} archive, got {manifest.get('split')!r}")
    if "HISTORICAL_TEST" in str(manifest.get("split", "")):
        raise ValueError("historical-test feature archives are forbidden")
    if "NO_GT_AS_MODEL_OR_TRACKER_INPUT" not in str(manifest.get("policy", "")):
        raise ValueError("feature archive policy is missing the GT isolation declaration")
    shards: list[dict[str, Any]] = []
    for record in manifest.get("shards", []):
        path = Path(str(record["path"])).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        if record.get("sha256") and sha256_file(path) != record["sha256"]:
            raise ValueError(f"feature shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "features", "language", "logits", "labels", "bboxes", "clip_frame_ids",
            "video_ids", "track_ids", "expressions",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"feature shard {path} is missing {sorted(missing)}")
        shards.append(payload)
    if not shards:
        raise ValueError(f"feature archive has no shards: {archive_root}")
    return manifest, shards


def _as_int_track(track: str) -> int:
    try:
        return int(track)
    except ValueError:
        # The evidence key is still stable for non-numeric tracker IDs.  Keep
        # the integer namespace deterministic without using Python's salted hash.
        return int.from_bytes(hashlib.sha256(track.encode("utf-8")).digest()[:4], "big")


def build_sequences(
    manifest: dict[str, Any],
    shards: Iterable[dict[str, Any]],
    *,
    max_sequences: int | None = None,
) -> list[Sequence]:
    """Flatten archive samples and group them causally by track/expression."""

    grouped: dict[tuple[str, str, str], list[tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor]]] = defaultdict(list)
    feature_dim: int | None = None
    language_dim: int | None = None
    for payload in shards:
        features = payload["features"].float()
        language = payload["language"].float()
        logits = payload["logits"].float()
        labels = payload["labels"].long()
        bboxes = payload["bboxes"].float()
        count, candidates, feature_dim_now = features.shape
        if language.shape[:2] != (count, candidates):
            raise ValueError("language and visual feature sample shapes differ")
        if logits.shape[:2] != (count, candidates) or logits.shape[-1] != 2:
            raise ValueError("archive logits must have shape [samples, candidates, 2]")
        if labels.shape != (count, candidates):
            raise ValueError("archive labels must have shape [samples, candidates]")
        if bboxes.shape != (count, 4):
            raise ValueError("archive bboxes must have shape [samples, 4]")
        feature_dim = feature_dim_now if feature_dim is None else feature_dim
        language_dim_now = int(language.shape[-1])
        language_dim = language_dim_now if language_dim is None else language_dim
        if feature_dim != feature_dim_now or language_dim != language_dim_now:
            raise ValueError("archive shards disagree on feature dimensions")
        metadata_count = len(payload["video_ids"])
        if metadata_count != count:
            raise ValueError("archive metadata count does not match tensors")
        for sample in range(count):
            frame_values = payload["clip_frame_ids"][sample]
            if not frame_values:
                raise ValueError("archive sample has no clip frame IDs")
            frame = int(frame_values[-1])
            video = str(payload["video_ids"][sample])
            track = str(payload["track_ids"][sample])
            expressions = payload["expressions"][sample]
            if len(expressions) != candidates:
                raise ValueError("archive expression count does not match candidate count")
            for candidate in range(candidates):
                expression = str(expressions[candidate])
                if not expression:
                    raise ValueError("archive contains an empty expression")
                key = (video, track, expression)
                grouped[key].append(
                    (
                        frame,
                        features[sample, candidate],
                        language[sample, candidate],
                        logits[sample, candidate],
                        labels[sample, candidate],
                        bboxes[sample],
                    )
                )

    sequences: list[Sequence] = []
    for key, values in sorted(grouped.items()):
        # Overlapping clips can end on the same frame.  Keep the last archived
        # observation deterministically; feeding duplicate frame IDs would
        # violate the causal EvidenceBuffer contract.
        by_frame: dict[int, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = {}
        for value in values:
            by_frame[value[0]] = value[1:]
        ordered = sorted(by_frame.items())
        frames = torch.tensor([item[0] for item in ordered], dtype=torch.long)
        features = torch.stack([item[1][0] for item in ordered]).float()
        language = torch.stack([item[1][1] for item in ordered]).float()
        logits = torch.stack([item[1][2] for item in ordered]).float()
        labels = torch.stack([item[1][3] for item in ordered]).long()
        bboxes = torch.stack([item[1][4] for item in ordered]).float()
        sequences.append(
            Sequence(
                key=key,
                frames=frames,
                features=features,
                language=language,
                logits=logits,
                labels=labels,
                bboxes=bboxes,
                track_id=_as_int_track(key[1]),
            )
        )
        if max_sequences is not None and len(sequences) >= max_sequences:
            break
    if not sequences:
        raise ValueError("archive produced no temporal sequences")
    return sequences


def _sequence_loss(adapter: TemporalEvidenceAdapter, sequence: Sequence, *, dynamic_window: int) -> Tensor:
    output = run_causal_sequence(
        adapter,
        current=sequence.features.unsqueeze(1),
        bboxes=sequence.bboxes.unsqueeze(1),
        track_ids=torch.tensor([sequence.track_id], dtype=torch.long, device=sequence.features.device),
        language=sequence.language.unsqueeze(1),
        namespace=sequence.key,
        frame_indices=sequence.frames,
        base_logits=sequence.logits.unsqueeze(1),
        dynamic_window=dynamic_window,
    )
    positive_logit = output[:, 0, 1] - output[:, 0, 0]
    return F.binary_cross_entropy_with_logits(positive_logit, sequence.labels.float())


def _run_epoch(
    adapter: TemporalEvidenceAdapter,
    sequences: list[Sequence],
    *,
    optimizer: torch.optim.Optimizer | None,
    dynamic_window: int,
) -> float:
    training = optimizer is not None
    adapter.train(training)
    total = 0.0
    weight = 0
    order = list(range(len(sequences)))
    if training:
        random.shuffle(order)
    for index in order:
        sequence = sequences[index]
        if training:
            optimizer.zero_grad(set_to_none=True)
        loss = _sequence_loss(adapter, sequence, dynamic_window=dynamic_window)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
            optimizer.step()
        count = int(sequence.labels.numel())
        total += float(loss.detach()) * count
        weight += count
    return total / max(weight, 1)


def train_temporal_head(
    *,
    train_archive: Path,
    val_archive: Path,
    output_root: Path,
    variant: str,
    epochs: int,
    learning_rate: float,
    hidden_dim: int | None,
    dynamic_window: int,
    seed: int,
    device: str,
    max_sequences: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    if epochs <= 0 or learning_rate <= 0 or dynamic_window <= 0:
        raise ValueError("epochs, learning_rate and dynamic_window must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root is non-empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    selected_device = torch.device(device)
    train_manifest, train_shards = _load_archive(train_archive.expanduser().resolve(), "train_core")
    val_manifest, val_shards = _load_archive(val_archive.expanduser().resolve(), "internal_val")
    train_sequences = build_sequences(train_manifest, train_shards, max_sequences=max_sequences)
    val_sequences = build_sequences(val_manifest, val_shards, max_sequences=max_sequences)
    feature_dim = int(train_sequences[0].features.shape[-1])
    language_dim = int(train_sequences[0].language.shape[-1])
    if any(sequence.features.shape[-1] != feature_dim for sequence in val_sequences):
        raise ValueError("train and validation feature dimensions differ")
    if any(sequence.language.shape[-1] != language_dim for sequence in val_sequences):
        raise ValueError("train and validation language dimensions differ")
    train_sequences = [
        Sequence(sequence.key, sequence.frames.to(selected_device), sequence.features.to(selected_device),
                 sequence.language.to(selected_device), sequence.logits.to(selected_device),
                 sequence.labels.to(selected_device), sequence.bboxes.to(selected_device), sequence.track_id)
        for sequence in train_sequences
    ]
    val_sequences = [
        Sequence(sequence.key, sequence.frames.to(selected_device), sequence.features.to(selected_device),
                 sequence.language.to(selected_device), sequence.logits.to(selected_device),
                 sequence.labels.to(selected_device), sequence.bboxes.to(selected_device), sequence.track_id)
        for sequence in val_sequences
    ]
    adapter = TemporalEvidenceAdapter(
        feature_dim=feature_dim,
        language_dim=language_dim,
        hidden_dim=hidden_dim,
        variant=variant,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    best_val = math.inf
    best_epoch = -1
    for epoch in range(epochs):
        train_loss = _run_epoch(adapter, train_sequences, optimizer=optimizer, dynamic_window=dynamic_window)
        with torch.no_grad():
            val_loss = _run_epoch(adapter, val_sequences, optimizer=None, dynamic_window=dynamic_window)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        torch.save(
            {"state_dict": adapter.state_dict(), "config": {
                "feature_dim": feature_dim, "language_dim": language_dim,
                "hidden_dim": hidden_dim, "variant": variant,
                "dynamic_window": dynamic_window, "seed": seed,
            }},
            output_root / "temporal_head_last.pt",
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {"state_dict": adapter.state_dict(), "config": {
                    "feature_dim": feature_dim, "language_dim": language_dim,
                    "hidden_dim": hidden_dim, "variant": variant,
                    "dynamic_window": dynamic_window, "seed": seed,
                }},
                output_root / "temporal_head_best.pt",
            )
    result = {
        "schema_version": "flexhook-temporal-training-0.1",
        "status": "SUCCEEDED",
        "variant": variant,
        "policy": "TRAIN_CORE_TARGETS_ONLY; INTERNAL_VAL_SELECTION_ONLY; HISTORICAL_TEST_OFFLINE_ONLY",
        "train_archive": str(train_archive.expanduser().resolve()),
        "train_archive_sha256": sha256_file(train_archive.expanduser().resolve() / "FEATURE_ARCHIVE_MANIFEST.json"),
        "val_archive": str(val_archive.expanduser().resolve()),
        "val_archive_sha256": sha256_file(val_archive.expanduser().resolve() / "FEATURE_ARCHIVE_MANIFEST.json"),
        "train_sequence_count": len(train_sequences),
        "val_sequence_count": len(val_sequences),
        "feature_dim": feature_dim,
        "language_dim": language_dim,
        "hidden_dim": hidden_dim,
        "dynamic_window": dynamic_window,
        "seed": seed,
        "device": str(selected_device),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "checkpoint": str(output_root / "temporal_head_best.pt"),
    }
    _atomic_json(output_root / "TEMPORAL_TRAINING_MANIFEST.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", required=True)
    parser.add_argument("--val-archive", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", default="H8", choices=[f"H{i}" for i in range(9)])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dynamic-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = train_temporal_head(
        train_archive=Path(args.train_archive),
        val_archive=Path(args.val_archive),
        output_root=Path(args.output_root),
        variant=args.variant,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dynamic_window=args.dynamic_window,
        seed=args.seed,
        device=args.device,
        max_sequences=args.max_sequences,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else f"temporal_training: {result['status']} best_val_loss={result['best_val_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
