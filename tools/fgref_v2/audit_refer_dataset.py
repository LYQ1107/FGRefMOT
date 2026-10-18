#!/usr/bin/env python3
"""Create a bounded, read-only audit of the Refer-KITTI/RMOT assets.

The NAS tree is treated as an input only.  The report is written to a local
output path and records link targets, split coverage, expression statistics,
frame/label consistency, and conservative language categories.  It does not
infer a train/test protocol when no test manifest is present.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data1/liuyeqiang/fgref_v2_runs/dataset_audit/refer_kitti.json"),
    )
    return parser


def _link_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
        "link_target": str(path.readlink()) if path.is_symlink() else None,
        "resolved": str(path.resolve(strict=False)),
    }


def _split_records(root: Path) -> dict[str, Any]:
    training = root / "refer-kitti" / "KITTI" / "training"
    records: dict[str, Any] = {}
    for path in sorted(training.glob("refer-kitti*.train")):
        rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        videos = [Path(row).parts[-2] for row in rows if len(Path(row).parts) >= 2]
        records[path.name] = {
            "path": str(path),
            "rows": len(rows),
            "unique_rows": len(set(rows)),
            "duplicate_rows": len(rows) - len(set(rows)),
            "videos": len(set(videos)),
            "video_ids": sorted(set(videos)),
            "rows_by_video": dict(sorted(Counter(videos).items())),
        }
    return records


def _source_group(video_id: str) -> str:
    return video_id.rsplit("_c", 1)[0] if "_c" in video_id else video_id


def _categories(sentence: str) -> list[str]:
    text = sentence.lower()
    categories: list[str] = []
    if re.search(r"\b(number|no\.?\s*\d+|#\s*\d+|jersey|uniform|sportswear)\b|\b\d+\b", text):
        categories.append("identity_or_number")
    if re.search(r"\b(blue|yellow|red|green|black|white|orange|purple|pink)\b", text):
        categories.append("color_or_appearance")
    if re.search(r"\b(left|right|front|back|court|net|side|near|far|middle|center)\b", text):
        categories.append("spatial_relation")
    if re.search(r"\b(holding|serving|serve|running|jumping|walking|shooting|kicking|playing|moving|standing)\b", text):
        categories.append("action_or_state")
    if re.search(r"\b(players|people|men|women|goalkeepers|group|team|ball)\b", text):
        categories.append("group_or_entity")
    return categories or ["uncategorized"]


def _expression_audit(root: Path) -> dict[str, Any]:
    dataset = root / "refer-kitti"
    expression_root = dataset / "expression"
    image_root = dataset / "KITTI" / "training" / "image_02"
    label_root = dataset / "KITTI" / "labels_with_ids" / "image_02"

    expressions: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    sentence_videos: defaultdict[str, set[str]] = defaultdict(set)
    video_summary: dict[str, dict[str, Any]] = {}
    missing_image_dirs: list[str] = []
    missing_label_dirs: list[str] = []
    frame_mismatches: list[dict[str, Any]] = []

    video_dirs = sorted(p for p in expression_root.iterdir() if p.is_dir())
    for video_dir in video_dirs:
        video = video_dir.name
        image_dir = image_root / video
        label_dir = label_root / video
        if not image_dir.is_dir():
            missing_image_dirs.append(video)
        if not label_dir.is_dir():
            missing_label_dirs.append(video)
        image_frames = {
            int(p.stem) for p in image_dir.glob("*.jpg") if p.stem.isdigit()
        } if image_dir.is_dir() else set()
        label_frames = {
            int(p.stem) for p in label_dir.glob("*.txt") if p.stem.isdigit()
        } if label_dir.is_dir() else set()
        video_summary[video] = {
            "source_group": _source_group(video),
            "expression_files": 0,
            "image_frames": len(image_frames),
            "label_frames": len(label_frames),
        }
        for expression_path in sorted(video_dir.glob("*.json")):
            payload = json.loads(expression_path.read_text())
            label = payload.get("label", {})
            frame_ids = sorted(int(frame) for frame in label if str(frame).isdigit())
            target_counts = [len(label[str(frame)]) for frame in frame_ids]
            sentence = str(payload.get("sentence", ""))
            cats = _categories(sentence)
            category_counts.update(cats)
            sentence_videos[sentence].add(video)
            video_summary[video]["expression_files"] += 1
            expression_record = {
                "video": video,
                "source_group": _source_group(video),
                "expression_file": str(expression_path),
                "expression": expression_path.stem,
                "sentence": sentence,
                "categories": cats,
                "frames_with_labels": len(frame_ids),
                "first_label_frame": frame_ids[0] if frame_ids else None,
                "last_label_frame": frame_ids[-1] if frame_ids else None,
                "target_count_min": min(target_counts) if target_counts else 0,
                "target_count_max": max(target_counts) if target_counts else 0,
                "target_count_mean": (
                    sum(target_counts) / len(target_counts) if target_counts else 0.0
                ),
                "unique_target_ids": sorted(
                    {int(target) for ids in label.values() for target in ids}
                ),
                "ignore_frames": len(payload.get("ignore", {})),
                "label_frames_missing_image": sorted(set(frame_ids) - image_frames),
                "label_frames_missing_box_file": sorted(set(frame_ids) - label_frames),
            }
            if expression_record["label_frames_missing_image"] or expression_record[
                "label_frames_missing_box_file"
            ]:
                frame_mismatches.append(expression_record)
            expressions.append(expression_record)

    duplicated_sentences = {
        sentence: sorted(videos)
        for sentence, videos in sentence_videos.items()
        if len(videos) > 1
    }
    source_groups: defaultdict[str, list[str]] = defaultdict(list)
    for video in video_summary:
        source_groups[_source_group(video)].append(video)

    return {
        "video_count": len(video_dirs),
        "expression_count": len(expressions),
        "category_counts": dict(sorted(category_counts.items())),
        "videos": video_summary,
        "expressions": expressions,
        "duplicate_sentence_across_videos": duplicated_sentences,
        "source_groups": {key: sorted(value) for key, value in sorted(source_groups.items())},
        "missing_image_dirs": missing_image_dirs,
        "missing_label_dirs": missing_label_dirs,
        "frame_mismatches": frame_mismatches,
    }


def main() -> None:
    args = _parser().parse_args()
    root = args.dataset_root
    refer_kitti = root / "refer-kitti"
    payload = {
        "audit": "bounded_read_only_refer_kitti",
        "dataset_root": str(root),
        "assets": {
            "refer_kitti": _link_info(refer_kitti),
            "refer_kitti_v1": _link_info(root / "refer-kitti-v1"),
            "refer_kitti_v2": _link_info(root / "refer-kitti-v2"),
            "refer_kitti_v2_kitti": _link_info(root / "refer-kitti-v2" / "KITTI"),
            "refer_dance": _link_info(root / "Refer-DanceTrack"),
        },
        "splits": _split_records(root),
        "refer_kitti": _expression_audit(root),
        "notes": [
            "No test manifest was found or assumed by this audit.",
            "Source-group overlap is a structural diagnostic, not proof of leakage without the official split provenance.",
            "Language categories are conservative regex diagnostics and are not human annotation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    report = payload["refer_kitti"]
    print("DATASET_AUDIT_OK")
    print("videos", report["video_count"])
    print("expressions", report["expression_count"])
    print("category_counts", json.dumps(report["category_counts"], sort_keys=True))
    print("frame_mismatches", len(report["frame_mismatches"]))
    print("split_files", json.dumps(payload["splits"], sort_keys=True))
    print("output", args.output)


if __name__ == "__main__":
    main()
