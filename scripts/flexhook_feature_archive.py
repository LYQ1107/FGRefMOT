#!/usr/bin/env python3
"""Capture deterministic FlexHook temporal features without changing upstream.

The archive is a bridge between the frozen FlexHook checkpoint and the
project-level causal temporal head.  It records the final cross-attention query
representation, the frozen logits, clip/frame metadata and supervised labels.
Labels are stored as training targets only; they are never passed to the
FlexHook model or the physical tracker.  The command is intentionally limited
to the registered train/core-validation layouts and does not accept test GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def normalize_expression_batch(value: Any, batch_size: int) -> list[list[str]]:
    """Normalize PyTorch's transposed list collation for expression strings."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(value, (list, tuple)):
        raise ValueError("expression batch must be a list or tuple")
    # ``default_collate`` transposes the list of eight expression strings and
    # returns a list of tuples, one tuple per expression.  Check that shape
    # first because it is ambiguous when the batch size also equals eight.
    if all(isinstance(item, tuple) and len(item) == batch_size for item in value):
        return [
            [str(value[expression_index][sample_index]) for expression_index in range(len(value))]
            for sample_index in range(batch_size)
        ]
    if len(value) == batch_size and all(isinstance(item, (list, tuple)) for item in value):
        rows = [[str(item) for item in row] for row in value]
        if rows and len({len(row) for row in rows}) == 1:
            return rows
    if len(value) == batch_size and all(isinstance(item, str) for item in value):
        return [[str(item)] for item in value]
    raise ValueError("could not normalize expression batch")


def split_data_key(data_key: str) -> tuple[str, str]:
    """Return the video and physical track ID from a loader data key."""

    parts = str(data_key).split("#")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid FlexHook data key: {data_key!r}")
    return parts[0], parts[1]


def _dataset_trajectory(dataset: Any, data_key: str) -> dict[str, Any]:
    video, track_id = split_data_key(data_key)
    base_key = f"{video}#{track_id}"
    data = getattr(dataset, "data", None)
    if not isinstance(data, dict):
        raise ValueError("FlexHook dataset has no structured trajectory data")
    if "SportsMOT" in data and isinstance(data["SportsMOT"], dict):
        trajectory = data["SportsMOT"].get(base_key)
    else:
        trajectory = data.get(base_key)
    if not isinstance(trajectory, dict):
        raise KeyError(f"trajectory not found for {base_key}")
    return trajectory


def frame_ids_for_key(dataset: Any, data_key: str, sample_indices: Iterable[int]) -> list[int]:
    trajectory = _dataset_trajectory(dataset, data_key)
    boxes = trajectory.get("bbox")
    if not isinstance(boxes, list):
        raise ValueError(f"trajectory has no bbox list: {data_key}")
    frame_ids: list[int] = []
    for raw_index in sample_indices:
        index = int(raw_index)
        if index < 0 or index >= len(boxes):
            raise IndexError(f"sample index {index} outside trajectory for {data_key}")
        frame_ids.append(int(boxes[index][0]))
    return frame_ids


def _sample_indices_for_key(dataset: Any, data_key: str) -> Any:
    """Look up a validation sample in flat or mixed-dataset storage.

    ``RMOT_Dataset_mix`` keeps ``sample_dict`` and ``data`` nested by dataset
    name, while the public ``data_keys`` list contains the flattened keys.
    Single-dataset loaders expose the same mapping flat.  The archive only
    needs the sampled indices, so support both layouts without changing the
    upstream dataset or loader ordering.
    """

    sample_dict = getattr(dataset, "sample_dict", None)
    if not isinstance(sample_dict, dict):
        raise ValueError("FlexHook dataset has no sample_dict mapping")
    if data_key in sample_dict:
        return sample_dict[data_key]
    for nested in sample_dict.values():
        if isinstance(nested, dict) and data_key in nested:
            return nested[data_key]
    raise KeyError(data_key)


def _val_batch_metadata(dataset: Any, start: int, batch_size: int) -> tuple[list[str], list[list[int]], list[list[str]]]:
    keys = list(dataset.data_keys[start : start + batch_size])
    frame_ids: list[list[int]] = []
    expressions: list[list[str]] = []
    for key in keys:
        indices = _sample_indices_for_key(dataset, key)
        frame_ids.append(frame_ids_for_key(dataset, "#".join(str(key).split("#")[:2]), indices))
        expressions.append(str(key).split("#")[-1].split("+"))
    return [str(key) for key in keys], frame_ids, expressions


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _write_shard(output_root: Path, shard_index: int, batches: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not batches:
        raise ValueError("cannot write an empty feature shard")
    payload: dict[str, Any] = {
        "schema_version": "flexhook-feature-shard-0.2",
        "features": torch.cat([item["features"] for item in batches], dim=0).float(),
        "language": torch.cat([item["language"] for item in batches], dim=0).float(),
        "logits": torch.cat([item["logits"] for item in batches], dim=0).float(),
        "labels": torch.cat([item["labels"] for item in batches], dim=0).long(),
        "bboxes": torch.cat([item["bboxes"] for item in batches], dim=0).float(),
        "clip_frame_ids": [frame for item in batches for frame in item["clip_frame_ids"]],
        "video_ids": [value for item in batches for value in item["video_ids"]],
        "track_ids": [value for item in batches for value in item["track_ids"]],
        "data_keys": [value for item in batches for value in item["data_keys"]],
        "expressions": [value for item in batches for value in item["expressions"]],
    }
    final_path = output_root / "shards" / f"features_{shard_index:05d}.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=final_path.parent, prefix=f".{final_path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(final_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(final_path),
        "size_bytes": final_path.stat().st_size,
        "sha256": sha256_file(final_path),
        "sample_count": int(payload["features"].shape[0]),
        "feature_shape": list(payload["features"].shape),
        "language_shape": list(payload["language"].shape),
    }


def archive_features(
    *,
    config_path: Path,
    layout_root: Path,
    tracker_root: Path,
    checkpoint: Path,
    runtime_root: Path,
    output_root: Path,
    split: str,
    gpu: int,
    batch_size: int,
    workers: int,
    shard_batches: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    if split not in {"train_core", "internal_val"}:
        raise ValueError("split must be train_core or internal_val")
    for path in (config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not layout_root.is_dir() or not tracker_root.is_dir():
        raise FileNotFoundError("layout or tracker root is missing")
    if batch_size <= 0 or workers <= 0 or shard_batches <= 0:
        raise ValueError("batch_size, workers and shard_batches must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"feature archive is non-empty: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    random.seed(seed)

    upstream_root = REPO_ROOT / "third_party" / "FlexHook"
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(upstream_root / "TrackEval"))
    from scripts.transformers_compat import install_legacy_tokenizer_alias

    install_legacy_tokenizer_alias()
    import torch
    import torch.distributed as dist
    from config import _C
    from data import build_loader
    from logger import create_logger
    from models import build_model
    from utils import load_ckpt_my

    torch.manual_seed(seed)
    dist.init_process_group(backend="gloo", init_method="env://", rank=0, world_size=1)
    try:
        runtime_root = runtime_root.expanduser().resolve()
        cfg = _C.clone()
        cfg.defrost()
        cfg.merge_from_file(str(upstream_root / "configs/train/train-mix.yaml"))
        cfg.EVAL_MODE = False
        cfg.LOCAL_RANK = 0
        cfg.DATA.DATASET = "mix"
        cfg.DATA.TRAIN = ["SportsMOT"]
        cfg.DATA.TEST = ["SportsMOT"]
        cfg.DATA.BATCH_SIZE = int(batch_size)
        cfg.DATA.VAL_BATCH_SIZE = int(batch_size)
        cfg.DATA.NUM_WORKERS = int(workers)
        cfg.data_root = str(layout_root)
        cfg.val_data_root = str(layout_root)
        cfg.track_root = str(tracker_root)
        cfg.sample_frame_stride = [30]
        cfg.val_sample_frame_stride = [24]
        cfg.val_set_scale = [5]
        cfg.ROBERTA_PATH = str((runtime_root / "pretrained/roberta-base").resolve())
        cfg.MODEL.PRETRAINED = str(checkpoint)
        cfg.MODEL.RESUME = str(checkpoint)
        cfg.OUTPUT = str(output_root)
        cfg.freeze()

        os.chdir(runtime_root)
        dataset_train, dataset_val, loader_train, loader_val, _ = build_loader(cfg)
        dataset = dataset_train if split == "train_core" else dataset_val
        loader = loader_train if split == "train_core" else loader_val
        logger = create_logger(str(output_root), dist_rank=0, name="flexhook_feature_archive")
        model = build_model(cfg).cuda()
        load_ckpt_my(cfg, model, logger)
        model.eval()

        captured: list[torch.Tensor] = []
        language_captured: list[torch.Tensor] = []

        def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not hasattr(output, "detach"):
                raise TypeError("output_ca_qkv hook returned a non-tensor value")
            captured.append(output.detach().cpu())

        def capture_language(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            """Capture the frozen text encoder sequence for the adapter input."""

            hidden = getattr(output, "last_hidden_state", output)
            if not hasattr(hidden, "detach"):
                raise TypeError("text encoder hook returned no tensor hidden state")
            language_captured.append(hidden.detach().cpu())

        handle = model.output_ca_qkv[-1].register_forward_hook(capture)
        language_handle = model.text_encoder.register_forward_hook(capture_language)
        manifest: dict[str, Any] = {
            "schema_version": "flexhook-feature-archive-0.2",
            "status": "RUNNING",
            "split": split,
            "policy": "LABELS_ARE_TRAINING_TARGETS_ONLY; NO_GT_AS_MODEL_OR_TRACKER_INPUT",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "layout_root": str(layout_root),
            "tracker_root": str(tracker_root),
            "seed": seed,
            "dataset_count": len(dataset),
            "batch_size": batch_size,
            "workers": workers,
            "shard_batches": shard_batches,
            "shards": [],
            "samples": 0,
            "feature_shape": None,
            "language_shape": None,
            "logit_shape": None,
        }
        _atomic_json(output_root / "FEATURE_ARCHIVE_MANIFEST.json", manifest)

        shard_batches_data: list[dict[str, Any]] = []
        shard_index = 0
        cursor = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    images, pes, bbox_gt, labels, expid, expma = batch[:6]
                    current_batch = int(images.shape[0])
                    if split == "train_core":
                        data_keys = [str(value) for value in batch[6]]
                        sample_indices = batch[7]
                        frame_ids = [
                            frame_ids_for_key(dataset, data_keys[index], sample_indices[index].tolist())
                            for index in range(current_batch)
                        ]
                        expressions = normalize_expression_batch(batch[8], current_batch)
                    else:
                        data_keys, frame_ids, expressions = _val_batch_metadata(
                            dataset, cursor, current_batch
                        )
                    cursor += current_batch
                    captured.clear()
                    language_captured.clear()
                    outputs, _regular = model(
                        images.cuda(non_blocking=True),
                        pes.cuda(non_blocking=True),
                        bbox_gt.cuda(non_blocking=True),
                        expid.cuda(non_blocking=True),
                        expma.cuda(non_blocking=True),
                    )
                    if len(captured) != 1:
                        raise RuntimeError(f"expected one hook output, got {len(captured)}")
                    if len(language_captured) != 1:
                        raise RuntimeError(f"expected one text hook output, got {len(language_captured)}")
                    features = captured[0]
                    language_tokens = language_captured[0]
                    if language_tokens.ndim != 3:
                        raise RuntimeError(
                            "expected text encoder output [B*N,L,C], "
                            f"got {tuple(language_tokens.shape)}"
                        )
                    sample_expression_count = int(labels.shape[1])
                    if language_tokens.shape[0] != current_batch * sample_expression_count:
                        raise RuntimeError(
                            "text encoder batch does not match sampled expression count: "
                            f"{tuple(language_tokens.shape)} vs {tuple(labels.shape)}"
                        )
                    token_mask = expma.detach().cpu().float()
                    if token_mask.ndim != 3 or tuple(token_mask.shape[:2]) != tuple(labels.shape):
                        raise RuntimeError(
                            f"expression mask shape {tuple(token_mask.shape)} does not match labels "
                            f"{tuple(labels.shape)}"
                        )
                    token_mask = token_mask.reshape(
                        language_tokens.shape[0], token_mask.shape[-1], 1
                    )
                    token_count = min(token_mask.shape[1], language_tokens.shape[1])
                    token_mask = token_mask[:, :token_count]
                    language_tokens = language_tokens[:, :token_count]
                    language = (language_tokens * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
                    language = language.reshape(current_batch, sample_expression_count, language.shape[-1])
                    logits = outputs[:, -1].detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    bboxes = bbox_gt[:, -1].detach().cpu()
                    video_ids: list[str] = []
                    track_ids: list[str] = []
                    for key in data_keys:
                        video, track = split_data_key(key)
                        video_ids.append(video)
                        track_ids.append(track)
                    shard_batches_data.append(
                        {
                            "features": features,
                            "language": language,
                            "logits": logits,
                            "labels": labels_cpu,
                            "bboxes": bboxes,
                            "clip_frame_ids": frame_ids,
                            "video_ids": video_ids,
                            "track_ids": track_ids,
                            "data_keys": data_keys,
                            "expressions": expressions,
                        }
                    )
                    manifest["samples"] += current_batch
                    manifest["feature_shape"] = list(features.shape[1:])
                    manifest["language_shape"] = list(language.shape[1:])
                    manifest["logit_shape"] = list(logits.shape[1:])
                    if len(shard_batches_data) >= shard_batches:
                        record = _write_shard(output_root, shard_index, shard_batches_data)
                        manifest["shards"].append(record)
                        _atomic_json(output_root / "FEATURE_ARCHIVE_MANIFEST.json", manifest)
                        shard_index += 1
                        shard_batches_data = []
        finally:
            handle.remove()
            language_handle.remove()
            for candidate_loader in (loader_train, loader_val):
                if hasattr(candidate_loader, "_iterator") and candidate_loader._iterator is not None:
                    candidate_loader._iterator._shutdown_workers()

        if shard_batches_data:
            record = _write_shard(output_root, shard_index, shard_batches_data)
            manifest["shards"].append(record)
        manifest["status"] = "SUCCEEDED"
        _atomic_json(output_root / "FEATURE_ARCHIVE_MANIFEST.json", manifest)
        return manifest
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("train_core", "internal_val"), required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = archive_features(
        config_path=Path(args.config).expanduser().resolve(),
        layout_root=Path(args.layout_root).expanduser().resolve(),
        tracker_root=Path(args.tracker_root).expanduser().resolve(),
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        runtime_root=Path(args.runtime_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        split=args.split,
        gpu=args.gpu,
        batch_size=args.batch_size,
        workers=args.workers,
        shard_batches=args.shard_batches,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"flexhook_feature_archive: {result['status']} samples={result['samples']} shards={len(result['shards'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
