#!/usr/bin/env python3
"""Build a detector-training annotation split with evaluation videos removed.

The Refer-Sports evaluation videos are a subset of the public SportsMOT train
split.  This utility makes a COCO-style annotation file containing only the
remaining train videos, so a detector checkpoint can be trained without
seeing evaluation frames or their labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-train-json", type=Path, required=True)
    ap.add_argument("--eval-sequence-list", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    source = json.loads(args.source_train_json.read_text())
    eval_names = set(read_names(args.eval_sequence_list))
    source_videos = {int(v["id"]): v["file_name"] for v in source["videos"]}
    missing = sorted(eval_names - set(source_videos.values()))
    if missing:
        raise SystemExit(f"evaluation sequences absent from source train JSON: {missing}")

    held_out_ids = {vid for vid, name in source_videos.items() if name in eval_names}
    keep_videos = [v for v in source["videos"] if int(v["id"]) not in held_out_ids]
    keep_images = [i for i in source["images"] if int(i["video_id"]) not in held_out_ids]
    keep_image_ids = {int(i["id"]) for i in keep_images}
    keep_annotations = [
        a for a in source["annotations"] if int(a["image_id"]) in keep_image_ids
    ]

    out = {
        "images": keep_images,
        "annotations": keep_annotations,
        "videos": keep_videos,
        "categories": source["categories"],
    }

    ann_dir = args.output_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_json = ann_dir / "clean_train.json"
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    manifest = {
        "schema_version": "rmot-sports-leakage-safe-detector-split-0.1",
        "source_train_json": str(args.source_train_json),
        "source_train_json_sha256": sha256(args.source_train_json),
        "evaluation_sequence_list": str(args.eval_sequence_list),
        "evaluation_sequence_list_sha256": sha256(args.eval_sequence_list),
        "evaluation_sequence_count": len(eval_names),
        "evaluation_sequences": sorted(eval_names),
        "source_train_video_count": len(source["videos"]),
        "kept_train_video_count": len(keep_videos),
        "source_train_image_count": len(source["images"]),
        "kept_train_image_count": len(keep_images),
        "source_train_annotation_count": len(source["annotations"]),
        "kept_train_annotation_count": len(keep_annotations),
        "output_json": str(out_json),
        "output_json_sha256": sha256(out_json),
        "used_evaluation_frames": False,
        "used_evaluation_ground_truth": False,
    }
    (args.output_root / "SPLIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
