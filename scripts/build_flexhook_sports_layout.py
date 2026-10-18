#!/usr/bin/env python3
"""Materialize a symlink-only SportsMOT layout for FlexHook's mix loader.

FlexHook's checked-in ``data/mydataloader_mix.py`` expects normalized
``SportsMOT_labels.json`` records, ``SportsMOT_RESOLUTION.json``,
``annotations_v1/<split>/SportsMOT`` language files, and MOT-style image paths.
This adapter creates those small metadata files and symlinks the existing raw
frame directories. It never copies frames, writes tracker results, or uses GT
as detector input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_videos(train_list: Path, test_list: Path) -> tuple[set[str], set[str]]:
    train: set[str] = set()
    for raw in train_list.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split("/")
        if len(parts) >= 4 and parts[3]:
            train.add(parts[3])
    test: set[str] = set()
    for raw in test_list.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value:
            test.add(value.split("+", 1)[0])
    return train, test


def sequence_dir(dataset_root: Path, video_id: str) -> Path:
    for split in ("train", "val", "test"):
        candidate = dataset_root / split / video_id
        if (candidate / "img1").is_dir() and (candidate / "seqinfo.ini").is_file():
            return candidate
    raise FileNotFoundError(f"SportsMOT sequence not found: {video_id}")


def sequence_size(sequence: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for raw in (sequence / "seqinfo.ini").read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key.strip() in {"imWidth", "imHeight"}:
            values[key.strip()] = int(value.strip())
    return values["imWidth"], values["imHeight"]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "expression"


def read_expressions(archive: zipfile.ZipFile, videos: set[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info in archive.infolist():
        if info.is_dir() or not info.filename.endswith(".json"):
            continue
        parts = info.filename.split("/")
        if len(parts) < 3 or parts[1] not in videos:
            continue
        payload = json.loads(archive.read(info.filename))
        label = payload.get("label") if isinstance(payload.get("label"), dict) else {}
        target_by_frame: dict[str, set[int]] = {}
        for frame_raw, ids in label.items():
            try:
                frame = str(int(frame_raw))
            except (TypeError, ValueError):
                continue
            if isinstance(ids, list):
                target_by_frame[frame] = {int(track_id) for track_id in ids}
        output[parts[1]].append(
            {
                "member": info.filename,
                "sentence": str(payload.get("sentence") or ""),
                "target_by_frame": target_by_frame,
            }
        )
    for values in output.values():
        values.sort(key=lambda row: row["member"])
    return output


def read_labels(labels_root: Path, video_id: str, expressions: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], int, int, int]:
    sequence = labels_root / video_id
    if not sequence.is_dir():
        raise FileNotFoundError(f"labels_with_ids sequence not found: {sequence}")
    labels: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    expression_targets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for expression in expressions:
        for frame, ids in expression["target_by_frame"].items():
            for track_id in ids:
                expression_targets[frame][str(track_id)].append(expression["sentence"])

    frame_count = 0
    box_count = 0
    missing_targets = 0
    for path in sorted(sequence.glob("*.txt")):
        frame_match = re.search(r"(\d+)$", path.stem)
        if not frame_match:
            continue
        frame = str(int(frame_match.group(1)))
        frame_count += 1
        for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            fields = raw.split()
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number}: expected class, ID, cx, cy, w, h")
            try:
                track_id = str(int(float(fields[1])))
                cx, cy, width, height = (float(value) for value in fields[2:6])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid normalized box") from exc
            x = cx - width / 2.0
            y = cy - height / 2.0
            labels[track_id][frame] = {
                "bbox": [x, y, width, height],
                "expression_raw": sorted(set(expression_targets[frame].get(track_id, []))),
            }
            box_count += 1

    # ``labels`` is keyed by the global SportsMOT track ID, then by frame.
    # Checking the outer keys preserves the ID namespace; iterating inner
    # rows would incorrectly compare expression IDs with frame numbers.
    present = set(labels)
    for expression in expressions:
        for ids in expression["target_by_frame"].values():
            missing_targets += sum(str(track_id) not in present for track_id in ids)
    return dict(labels), frame_count, box_count, missing_targets


def write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression-zip", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    train_videos, test_videos = split_videos(args.train_list, args.test_list)
    videos = train_videos | test_videos
    with zipfile.ZipFile(args.expression_zip) as archive:
        expressions = read_expressions(archive, videos)

    counts: Counter[str] = Counter()
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source_split, videos_for_split in (("train", train_videos), ("val", test_videos)):
        sequence_root = args.output / "sequences" / source_split / "SportsMOT-sequences"
        sequence_root.mkdir(parents=True, exist_ok=True)
        labels_payload: dict[str, Any] = {}
        resolution_payload: dict[str, list[int]] = {}
        expression_root = args.output / "annotations_v1" / source_split / "SportsMOT"
        expression_root.mkdir(parents=True, exist_ok=True)
        for video_id in sorted(videos_for_split):
            sequence = sequence_dir(args.dataset_root, video_id)
            labels, frame_count, box_count, missing_targets = read_labels(args.labels_root, video_id, expressions.get(video_id, []))
            labels_payload[video_id] = labels
            resolution_payload[video_id] = list(sequence_size(sequence))
            link = sequence_root / video_id
            link.symlink_to(sequence, target_is_directory=True)
            for ordinal, expression in enumerate(expressions.get(video_id, [])):
                expression_file = expression_root / f"{video_id}_{ordinal:06d}.json"
                write_json(expression_file, {"language": expression["sentence"], "source_member": expression["member"]})
            if missing_targets:
                warnings.append(f"{video_id}: {missing_targets} expression target IDs absent from labels_with_ids")
            counts[source_split] += len(expressions.get(video_id, []))
            metadata.append(
                {
                    "video_id": video_id,
                    "flexhook_split": source_split,
                    "source_sequence": str(sequence.resolve()),
                    "symlink": str(link),
                    "frame_count": frame_count,
                    "box_count": box_count,
                    "expression_count": len(expressions.get(video_id, [])),
                    "missing_expression_target_ids": missing_targets,
                }
            )
        split_root = args.output / "sequences" / source_split
        # The upstream loader asserts that frame IDs are strictly increasing.
        # Preserve the numeric file traversal order here; JSON ``sort_keys``
        # would place frame 10 before frame 2 lexicographically.
        write_json(split_root / "SportsMOT_labels.json", labels_payload, sort_keys=False)
        write_json(split_root / "SportsMOT_RESOLUTION.json", resolution_payload)
        write_json(split_root / "TAO_frame2file.json", {})

    report = {
        "schema_version": "flexhook-sports-layout-0.1",
        "source_expression_archive": str(args.expression_zip.resolve()),
        "source_expression_archive_sha256": sha256(args.expression_zip),
        "source_train_list_sha256": sha256(args.train_list),
        "source_test_list_sha256": sha256(args.test_list),
        "source_dataset_root": str(args.dataset_root.resolve()),
        "source_labels_root": str(args.labels_root.resolve()),
        "symlink_only": True,
        "counts_by_flexhook_split": dict(counts),
        "video_count": len(metadata),
        "metadata": metadata,
        "warnings": warnings,
        "inference_data_policy": "LAYOUT_METADATA_ONLY; NO_GT_AS_TRACKER_INPUT",
    }
    write_json(args.output / "manifest.json", report)
    print(json.dumps({"output": str(args.output), "video_count": len(metadata), "counts_by_flexhook_split": dict(counts), "warnings": warnings}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
