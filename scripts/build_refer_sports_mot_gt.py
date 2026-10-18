#!/usr/bin/env python3
"""Build MOTChallenge-style referring GT from the audited SportsMOT labels.

This converter is deliberately separate from detector/tracker inference.  It
filters the published SportsMOT ``labels_with_ids`` rows by the positive track
IDs and frames in each Refer-Sports expression.  Those rows use the global IDs
stored by the expression archive, unlike the per-video IDs in ``gt/gt.txt``.
The converter never invents a box, propagates an ID across an unlabeled frame,
or writes GT into a tracker-output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
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


def locate_sequence(images_root: Path, video_id: str) -> Path:
    candidates = [images_root / split / video_id for split in ("train", "val", "test")]
    for candidate in candidates:
        if (candidate / "gt" / "gt.txt").is_file():
            return candidate
    raise FileNotFoundError(f"no SportsMOT GT found for {video_id}")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value or "expression"


def sequence_size(sequence: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    for raw in (sequence / "seqinfo.ini").read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key.strip() in {"imWidth", "imHeight"}:
            values[key.strip()] = int(value.strip())
    try:
        return values["imWidth"], values["imHeight"]
    except KeyError as exc:
        raise ValueError(f"missing image size in {sequence / 'seqinfo.ini'}") from exc


def read_label_rows(path: Path, image_width: int, image_height: int) -> dict[int, dict[int, str]]:
    """Read one normalized YOLO-ID file and return MOT rows by global ID."""

    by_frame: dict[int, dict[int, str]] = {}
    if not path.is_file():
        return by_frame
    frame_match = re.search(r"(\d+)$", path.stem)
    if not frame_match:
        raise ValueError(f"cannot infer frame number from {path}")
    frame = int(frame_match.group(1))
    rows: dict[int, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = raw.split()
        if len(fields) < 6:
            raise ValueError(f"{path}:{line_number}: expected class, ID, cx, cy, w, h")
        try:
            track_id = int(float(fields[1]))
            cx, cy, width, height = (float(value) for value in fields[2:6])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid normalized box") from exc
        x = (cx - width / 2.0) * image_width
        y = (cy - height / 2.0) * image_height
        pixel_width = width * image_width
        pixel_height = height * image_height
        rows[track_id] = f"{frame},{track_id},{x:.6f},{y:.6f},{pixel_width:.6f},{pixel_height:.6f},1,1,1"
    by_frame[frame] = rows
    return by_frame


def read_label_sequence(labels_dir: Path, image_width: int, image_height: int) -> dict[int, dict[int, str]]:
    by_frame: dict[int, dict[int, str]] = {}
    for path in sorted(labels_dir.glob("*.txt")):
        by_frame.update(read_label_rows(path, image_width, image_height))
    return by_frame


def read_gt(path: Path) -> dict[int, dict[int, str]]:
    by_frame: dict[int, dict[int, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            raise ValueError(f"{path}:{line_number}: expected at least 6 comma-separated fields")
        try:
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid frame or track ID") from exc
        by_frame.setdefault(frame, {})[track_id] = ",".join(fields)
    return by_frame


def expression_split(video_id: str, train_videos: set[str], test_videos: set[str]) -> str:
    if video_id in train_videos:
        return "train"
    if video_id in test_videos:
        return "test"
    return "UNKNOWN"


def choose_records(split: str, train_videos: set[str], test_videos: set[str], archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    records: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        if info.is_dir() or not info.filename.endswith(".json"):
            continue
        parts = info.filename.split("/")
        if len(parts) < 3:
            continue
        video_id = parts[1]
        if split == "all" or expression_split(video_id, train_videos, test_videos) == split:
            records.append(info)
    return sorted(records, key=lambda item: item.filename)


def write_seqmap(names: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name"])
        writer.writerows([[name] for name in names])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression-zip", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True, help="SportsMOT labels_with_ids directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seqmap_dir = args.output / "seqmaps"
    gt_root = args.output / "gt"
    seqmap_dir.mkdir()
    gt_root.mkdir()

    train_videos, test_videos = split_videos(args.train_list, args.test_list)
    sequence_cache: dict[str, tuple[Path, dict[int, dict[int, str]]]] = {}
    names: list[str] = []
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_names: set[str] = set()
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(args.expression_zip) as archive:
        infos = choose_records(args.split, train_videos, test_videos, archive)
        for info in infos:
            payload = json.loads(archive.read(info.filename))
            parts = info.filename.split("/")
            video_id = str(payload.get("video_name") or parts[1])
            expression_stem = Path(parts[-1]).stem
            name = f"{safe_name(video_id)}__{safe_name(expression_stem)}"
            if name in seen_names:
                raise ValueError(f"expression name collision: {name}")
            seen_names.add(name)
            split = expression_split(video_id, train_videos, test_videos)
            sequence = locate_sequence(args.images_root, video_id)
            if video_id not in sequence_cache:
                image_width, image_height = sequence_size(sequence)
                labels_dir = args.labels_root / video_id
                if not labels_dir.is_dir():
                    raise FileNotFoundError(f"no labels_with_ids directory for {video_id}: {labels_dir}")
                sequence_cache[video_id] = (
                    sequence,
                    read_label_sequence(labels_dir, image_width, image_height),
                )
            _sequence, gt_by_frame = sequence_cache[video_id]
            label = payload.get("label") if isinstance(payload.get("label"), dict) else {}
            target_by_frame: dict[int, set[int]] = {}
            for frame_raw, ids in label.items():
                try:
                    frame = int(frame_raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(ids, list):
                    target_by_frame[frame] = {int(track_id) for track_id in ids}

            output_dir = gt_root / name
            output_dir.mkdir()
            output_file = output_dir / "gt.txt"
            rows = 0
            missing_rows: list[tuple[int, int]] = []
            with output_file.open("w", encoding="utf-8") as handle:
                for frame in sorted(target_by_frame):
                    source_rows = gt_by_frame.get(frame, {})
                    for track_id in sorted(target_by_frame[frame]):
                        row = source_rows.get(track_id)
                        if row is None:
                            missing_rows.append((frame, track_id))
                            continue
                        handle.write(row + "\n")
                        rows += 1
            if missing_rows:
                warnings.append(f"{name}: {len(missing_rows)} labeled frame/track pairs missing from source GT")
            names.append(name)
            counts[split] += 1
            metadata.append(
                {
                    "name": name,
                    "video_id": video_id,
                    "expression_member": info.filename,
                    "sentence": str(payload.get("sentence") or ""),
                    "split": split,
                    "target_track_ids": sorted({track_id for ids in target_by_frame.values() for track_id in ids}),
                    "labeled_frame_count": len(target_by_frame),
                    "gt_rows_written": rows,
                    "source_labels_with_ids": str((args.labels_root / video_id).resolve()),
                    "output_gt": str(output_file),
                }
            )

    write_seqmap(names, seqmap_dir / f"{args.split}.txt")
    report = {
        "schema_version": "refer-sports-mot-gt-0.1",
        "split": args.split,
        "expression_archive": str(args.expression_zip.resolve()),
        "expression_archive_sha256": sha256(args.expression_zip),
        "train_list_sha256": sha256(args.train_list),
        "test_list_sha256": sha256(args.test_list),
        "images_root": str(args.images_root.resolve()),
        "counts_by_split": dict(counts),
        "expression_count": len(metadata),
        "seqmap": str((seqmap_dir / f"{args.split}.txt").resolve()),
        "metadata": metadata,
        "warnings": warnings,
        "inference_data_policy": "GT_ONLY_EVALUATION_TARGET; NEVER_TRACKER_INPUT",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "expression_count": len(metadata), "counts_by_split": dict(counts), "warnings": warnings}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
