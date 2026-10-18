#!/usr/bin/env python3
"""Build a symlink-aware Refer-KITTI layout for YesTrack's SportsMOT adapter.

The pinned YesTrack loader assumes zero-based ``000000.png`` images and
``labels_with_ids`` boxes whose x/y values are normalized top-left coordinates.
SportsMOT stores one-based ``.jpg`` frames and center-format normalized boxes.
This adapter creates per-frame image symlinks, rewrites the small annotation
files into the loader's coordinate/frame convention, and shifts expression
keys from one-based to zero-based.  It never creates tracker outputs.
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "expression"


def read_expressions(
    archive: zipfile.ZipFile, videos: set[str]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info in archive.infolist():
        if info.is_dir() or not info.filename.endswith(".json"):
            continue
        parts = info.filename.split("/")
        if len(parts) < 3 or parts[1] not in videos:
            continue
        payload = json.loads(archive.read(info.filename))
        shifted: dict[str, list[int]] = {}
        for frame_raw, ids in (payload.get("label") or {}).items():
            source_frame = int(frame_raw)
            if source_frame < 1:
                raise ValueError(f"{info.filename}: expected one-based frame, got {source_frame}")
            shifted[str(source_frame - 1)] = [int(track_id) for track_id in (ids or [])]
        output[parts[1]].append(
            {
                "member": info.filename,
                "sentence": str(payload.get("sentence") or ""),
                "label": shifted,
            }
        )
    for values in output.values():
        values.sort(key=lambda row: row["member"])
    return output


def convert_label_file(source: Path, target: Path) -> int:
    lines: list[str] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = raw.split()
        if len(fields) < 6:
            raise ValueError(f"{source}:{line_number}: expected class, ID, cx, cy, w, h")
        try:
            cx, cy, width, height = (float(value) for value in fields[2:6])
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: invalid normalized box") from exc
        x = cx - width / 2.0
        y = cy - height / 2.0
        converted = [fields[0], fields[1], f"{x:.8f}", f"{y:.8f}", f"{width:.8f}", f"{height:.8f}"]
        converted.extend(fields[6:])
        lines.append(" ".join(converted))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    with zipfile.ZipFile(args.expression_zip) as archive:
        expressions = read_expressions(archive, train_videos | test_videos)

    counts: Counter[str] = Counter()
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    for output_split, videos in (("train", train_videos), ("val", test_videos)):
        root = args.output / output_split
        image_root = root / "KITTI" / "training" / "image_02"
        labels_root = root / "KITTI" / "labels_with_ids" / "image_02"
        expression_root = root / "expression"
        sequence_count = 0
        frame_count = 0
        box_count = 0
        expression_count = 0
        for video_id in sorted(videos):
            sequence = sequence_dir(args.dataset_root, video_id)
            source_images = sorted((sequence / "img1").glob("*.jpg"), key=lambda p: int(p.stem))
            source_labels = args.labels_root / video_id
            if len(source_images) == 0:
                raise ValueError(f"no SportsMOT images: {sequence}")
            image_dir = image_root / video_id
            label_dir = labels_root / video_id
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            seq_boxes = 0
            for frame_index, source_image in enumerate(source_images):
                image_link = image_dir / f"{frame_index:06d}.png"
                image_link.symlink_to(source_image.resolve())
                source_label = source_labels / f"{int(source_image.stem):06d}.txt"
                if not source_label.is_file():
                    raise FileNotFoundError(f"missing labels_with_ids frame: {source_label}")
                seq_boxes += convert_label_file(source_label, label_dir / f"{frame_index:06d}.txt")
            for ordinal, expression in enumerate(expressions.get(video_id, [])):
                write_json(
                    expression_root / video_id / f"{safe_name(video_id)}_{ordinal:06d}.json",
                    {
                        "video_name": video_id,
                        "sentence": expression["sentence"],
                        "label": expression["label"],
                        "source_member": expression["member"],
                    },
                )
            sequence_count += 1
            frame_count += len(source_images)
            box_count += seq_boxes
            expression_count += len(expressions.get(video_id, []))
            metadata.append(
                {
                    "video_id": video_id,
                    "yestrack_split": output_split,
                    "source_sequence": str(sequence.resolve()),
                    "frame_count": len(source_images),
                    "box_count": seq_boxes,
                    "expression_count": len(expressions.get(video_id, [])),
                    "image_symlink_root": str(image_dir),
                }
            )
        write_json(
            root / "manifest.json",
            {
                "layout_split": output_split,
                "video_count": sequence_count,
                "frame_count": frame_count,
                "box_count": box_count,
                "expression_count": expression_count,
            },
        )
        counts[f"{output_split}_videos"] = sequence_count
        counts[f"{output_split}_frames"] = frame_count
        counts[f"{output_split}_boxes"] = box_count
        counts[f"{output_split}_expressions"] = expression_count

    report = {
        "schema_version": "yestrack-sports-layout-0.1",
        "source_expression_archive": str(args.expression_zip.resolve()),
        "source_expression_archive_sha256": sha256(args.expression_zip),
        "source_train_list_sha256": sha256(args.train_list),
        "source_test_list_sha256": sha256(args.test_list),
        "source_dataset_root": str(args.dataset_root.resolve()),
        "source_labels_root": str(args.labels_root.resolve()),
        "frame_index_policy": "SPORTSMOT_1_BASED_TO_YESTRACK_0_BASED",
        "box_policy": "SPORTSMOT_CENTER_NORMALIZED_TO_YESTRACK_TOP_LEFT_NORMALIZED",
        "symlink_only_for_frames": True,
        "counts": dict(counts),
        "video_count": len(metadata),
        "metadata": metadata,
        "warnings": warnings,
        "inference_data_policy": "ANNOTATIONS_FOR_YESTRACK_DATASET_ONLY; NO_GT_AS_TRACKER_INPUT",
    }
    write_json(args.output / "manifest.json", report)
    print(json.dumps({"output": str(args.output), "video_count": len(metadata), "counts": dict(counts), "warnings": warnings}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
