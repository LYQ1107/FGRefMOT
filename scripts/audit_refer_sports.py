#!/usr/bin/env python3
"""Audit the archived Refer-Sports expressions without copying raw frames.

The expression archive is small and is read from its existing NAS path.  Frame
counts and image dimensions are checked against the local SportsMOT frame tree.
Every generated record keeps source paths, sizes, timestamps and content hashes
so later adapters can be reproduced without treating historical claims as
ground truth.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(path: Path) -> str:
    return sha256(path)


def parse_train_videos(path: Path) -> tuple[set[str], dict[str, int]]:
    videos: set[str] = set()
    counts: Counter[str] = Counter()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("/")
        if len(parts) < 4:
            continue
        video = parts[3]
        videos.add(video)
        counts[video] += 1
    return videos, dict(counts)


def parse_test_videos(path: Path) -> set[str]:
    videos: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            videos.add(line.split("+", 1)[0])
    return videos


def parse_test_expression_names(path: Path) -> set[tuple[str, str]]:
    entries: set[tuple[str, str]] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "+" in line:
            video, expression = line.split("+", 1)
            entries.add((video, expression))
    return entries


def jpeg_size(path: Path) -> tuple[int | None, int | None]:
    """Read JPEG dimensions without requiring Pillow or OpenCV."""

    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None, None
        while True:
            byte = handle.read(1)
            if not byte:
                return None, None
            if byte != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker or marker in (b"\xd8", b"\xd9"):
                continue
            length_raw = handle.read(2)
            if len(length_raw) != 2:
                return None, None
            length = int.from_bytes(length_raw, "big")
            if length < 2:
                return None, None
            if marker[0] in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                payload = handle.read(length - 2)
                if len(payload) >= 5:
                    height = int.from_bytes(payload[1:3], "big")
                    width = int.from_bytes(payload[3:5], "big")
                    return width, height
            else:
                handle.seek(length - 2, os.SEEK_CUR)


def png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        if handle.read(24)[:8] != b"\x89PNG\r\n\x1a\n":
            return None, None
        handle.seek(16)
        width = int.from_bytes(handle.read(4), "big")
        height = int.from_bytes(handle.read(4), "big")
        return width, height


def image_size(path: Path) -> tuple[int | None, int | None]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return jpeg_size(path)
    if suffix == ".png":
        return png_size(path)
    return None, None


def frame_files(images_root: Path, video_id: str) -> list[Path]:
    candidates: list[Path] = []
    for split in ("train", "val", "test"):
        for image_dir in (images_root / split / video_id / "img1", images_root / split / video_id):
            if image_dir.is_dir():
                candidates.extend(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                if candidates:
                    break
        if candidates:
            break
    return sorted(candidates)


def frame_number(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def contiguous_ranges(frames: Iterable[int]) -> list[list[int]]:
    values = sorted(set(frames))
    if not values:
        return []
    result: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append([start, previous])
            start = value
        previous = value
    result.append([start, previous])
    return result


def labels_with_ids_ids(labels_root: Path, video_id: str) -> set[int]:
    """Return global SportsMOT IDs used by the expression annotations."""

    video_dir = labels_root / video_id
    ids: set[int] = set()
    if not video_dir.is_dir():
        return ids
    for path in video_dir.glob("*.txt"):
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = raw.split()
            if len(fields) < 2:
                continue
            try:
                ids.add(int(float(fields[1])))
            except ValueError:
                continue
    return ids


def make_record(
    zip_path: Path,
    info: zipfile.ZipInfo,
    data: dict[str, Any],
    images_root: Path,
    train_videos: set[str],
    test_videos: set[str],
    frame_cache: dict[str, list[Path]],
    labels_cache: dict[str, set[int]],
) -> dict[str, Any]:
    relative = info.filename
    video_id = str(data.get("video_name") or relative.split("/")[1])
    sentence = str(data.get("sentence") or "")
    label = data.get("label") if isinstance(data.get("label"), dict) else {}
    ignore = data.get("ignore") if isinstance(data.get("ignore"), dict) else {}
    frames = sorted(int(frame) for frame in label if str(frame).isdigit())
    by_track: dict[int, list[int]] = {}
    for frame_raw, target_ids in label.items():
        try:
            frame = int(frame_raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(target_ids, list):
            continue
        for target_raw in target_ids:
            try:
                by_track.setdefault(int(target_raw), []).append(frame)
            except (TypeError, ValueError):
                continue
    interval_rows = []
    for target_id in sorted(by_track):
        interval_rows.append(
            {
                "track_id": target_id,
                "intervals": contiguous_ranges(by_track[target_id]),
            }
        )

    images = frame_cache.get(video_id, [])
    image_numbers = [value for value in (frame_number(path) for path in images) if value is not None]
    width = height = None
    if images:
        width, height = image_size(images[0])
    # Expression IDs are the global IDs in SportsMOT labels_with_ids.  The
    # ordinary gt/gt.txt files reset IDs per video, so they are not the correct
    # namespace for this overlap check.
    gt = labels_cache.get(video_id, set())
    target_ids = sorted(by_track)
    split = "train" if video_id in train_videos else "test" if video_id in test_videos else "UNKNOWN"
    raw_bytes = zipfile.ZipFile(zip_path).read(info.filename)
    source_stat = zip_path.stat()
    sport = sentence.split()[0] if sentence else None
    return {
        "video_id": video_id,
        "original_sportsmot_id": video_id,
        "source_video_id": None,
        "match_id": None,
        "sport_type": sport,
        "original_split": split,
        "images_root": str(images_root),
        "video_image_dir": str(images[0].parent) if images else None,
        "num_frames": len(images),
        "image_width": width,
        "image_height": height,
        "fps": None,
        "image_frame_min": min(image_numbers) if image_numbers else None,
        "image_frame_max": max(image_numbers) if image_numbers else None,
        "image_frame_gaps": max((max(image_numbers) - min(image_numbers) + 1) - len(set(image_numbers)), 0) if image_numbers else None,
        "expression_id": f"{video_id}:{Path(info.filename).stem}",
        "parent_expression_id": None,
        "sentence": sentence,
        "target_track_ids": target_ids,
        "temporal_intervals": interval_rows,
        "annotation_status": "PARTIAL_POSITIVE_TRACK_LABELS" if label else "UNKNOWN",
        "label_state_semantics": {"positive": "TRUE", "negative": "UNKNOWN", "outside_interval": "UNKNOWN"},
        "frame_index_base": 1,
        "num_labeled_frames": len(frames),
        "labeled_frame_min": min(frames) if frames else None,
        "labeled_frame_max": max(frames) if frames else None,
        "ignore_frame_count": len(ignore),
        "gt_id_overlap": sorted(set(target_ids) & gt) if gt else None,
        "target_ids_missing_from_gt": sorted(set(target_ids) - gt) if gt else None,
        "gt_id_namespace": "SPORTSMOT_LABELS_WITH_IDS_GLOBAL" if gt else "UNKNOWN",
        "id_mapping_status": "MATCHED_BY_LABELS_WITH_IDS_GLOBAL_ID" if images and gt else "UNVERIFIED",
        "source_archive": str(zip_path),
        "source_member": relative,
        "source_file_size": len(raw_bytes),
        "source_file_mtime": datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat(),
        "source_member_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression-zip", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True, help="SportsMOT labels_with_ids directory")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    train_videos, train_image_line_counts = parse_train_videos(args.train_list)
    test_videos = parse_test_videos(args.test_list)
    test_expression_names = parse_test_expression_names(args.test_list)
    overlap = sorted(train_videos & test_videos)
    records: list[dict[str, Any]] = []
    expression_videos: set[str] = set()
    with zipfile.ZipFile(args.expression_zip) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".json"):
                continue
            expression_videos.add(info.filename.split("/")[1])
    frame_cache = {video: frame_files(args.images_root, video) for video in sorted(expression_videos)}
    labels_cache = {video: labels_with_ids_ids(args.labels_root, video) for video in sorted(expression_videos)}
    with zipfile.ZipFile(args.expression_zip) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".json"):
                continue
            data = json.loads(archive.read(info.filename))
            records.append(
                make_record(
                    args.expression_zip,
                    info,
                    data,
                    args.images_root,
                    train_videos,
                    test_videos,
                    frame_cache,
                    labels_cache,
                )
            )
    records.sort(key=lambda item: item["expression_id"])
    observed_videos = sorted({record["video_id"] for record in records})
    archive_test_expression_names = {
        (record["video_id"], Path(record["source_member"]).stem)
        for record in records
        if record["original_split"] == "test"
    }
    test_expression_missing_from_list = sorted(archive_test_expression_names - test_expression_names)
    test_expression_unknown_in_list = sorted(test_expression_names - archive_test_expression_names)
    split_counts = Counter(record["original_split"] for record in records)
    video_split_counts = Counter()
    for video in observed_videos:
        if video in train_videos:
            video_split_counts["train"] += 1
        elif video in test_videos:
            video_split_counts["test"] += 1
        else:
            video_split_counts["UNKNOWN"] += 1
    warnings: list[str] = []
    if len(observed_videos) != 41 or len(records) != 492:
        warnings.append(
            "Historical reference counts (41 videos and approximately 492 expressions) "
            f"do not match this archive: observed {len(observed_videos)} videos and {len(records)} expressions."
        )
    if overlap:
        warnings.append(f"Train/test video overlap: {overlap}")
    missing_split = sorted(set(observed_videos) - train_videos - test_videos)
    if missing_split:
        warnings.append(f"Expression videos absent from both split lists: {missing_split}")
    if test_expression_missing_from_list or test_expression_unknown_in_list:
        warnings.append(
            "Test expression list does not exactly cover the archive: "
            f"missing={test_expression_missing_from_list}, unknown={test_expression_unknown_in_list}"
        )
    payload = {
        "schema_version": "refer-sports-canonical-0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "expression_archive": str(args.expression_zip.resolve()),
            "expression_archive_size_bytes": args.expression_zip.stat().st_size,
            "expression_archive_sha256": sha256(args.expression_zip),
            "train_list": str(args.train_list.resolve()),
            "train_list_sha256": text_hash(args.train_list),
            "test_list": str(args.test_list.resolve()),
            "test_list_sha256": text_hash(args.test_list),
            "images_root": str(args.images_root.resolve()),
            "labels_root": str(args.labels_root.resolve()),
        },
        "split_audit": {
            "train_video_count": len(train_videos),
            "test_video_count": len(test_videos),
            "train_test_video_overlap": overlap,
            "train_image_list_line_counts": train_image_line_counts,
            "test_list_line_count": sum(1 for line in args.test_list.read_text(encoding="utf-8").splitlines() if line.strip()),
            "archive_test_expression_count": len(archive_test_expression_names),
            "test_expression_list_count": len(test_expression_names),
            "test_expressions_missing_from_list": test_expression_missing_from_list,
            "test_expression_list_entries_not_in_archive": test_expression_unknown_in_list,
        },
        "summary": {
            "video_count": len(observed_videos),
            "expression_count": len(records),
            "video_count_by_split": dict(video_split_counts),
            "expression_count_by_split": dict(split_counts),
            "sport_type_counts": dict(Counter(record["sport_type"] for record in records)),
            "total_unique_image_frames": sum(len(frame_cache.get(video, [])) for video in observed_videos),
            "fps_status": "UNKNOWN_NO_VIDEO_METADATA_IN_ARCHIVE",
            "id_namespace": "SPORTSMOT_LABELS_WITH_IDS_GLOBAL",
        },
        "warnings": warnings,
        "records": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "video_id", "original_sportsmot_id", "source_video_id", "match_id", "sport_type", "original_split",
        "num_frames", "image_width", "image_height", "fps", "expression_id", "parent_expression_id",
        "sentence", "target_track_ids", "temporal_intervals", "annotation_status", "frame_index_base",
        "num_labeled_frames", "labeled_frame_min", "labeled_frame_max", "source_archive", "source_member",
        "source_file_size", "source_file_mtime", "source_member_sha256",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            for field in ("target_track_ids", "temporal_intervals"):
                row[field] = json.dumps(row[field], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    print(json.dumps({"json": str(args.out_json), "csv": str(args.out_csv), "summary": payload["summary"], "warnings": warnings}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
