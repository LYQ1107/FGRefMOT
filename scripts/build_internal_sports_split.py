#!/usr/bin/env python3
"""Create a deterministic, video-disjoint Refer-Sports train/validation split.

The historical 22/19 archive split is retained for final comparison.  This
script creates a fixed validation holdout from the 22 training videos so model
selection does not inspect the historical 19-video test set.  The emitted list
format is accepted by the existing symlink-only FlexHook and YesTrack layout
builders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_HOLDOUT = (
    "v_-6Os86HzwCs_c001",
    "v_1LwtoLPw2TU_c006",
    "v_1yHWGw8DH4A_c029",
    "v_2j7kLB-vEEk_c005",
    "v_ApPxnw_Jffg_c001",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"canonical manifest has no records: {path}")
    return [record for record in records if isinstance(record, dict)]


def build_split(
    *, manifest_path: Path, output_root: Path, holdout_videos: tuple[str, ...]
) -> dict[str, Any]:
    records = load_records(manifest_path)
    train_records = [record for record in records if record.get("original_split") == "train"]
    by_video: dict[str, dict[str, Any]] = {}
    for record in train_records:
        video = str(record.get("video_id") or "")
        if video and video not in by_video:
            by_video[video] = record
    holdout = set(holdout_videos)
    missing = sorted(holdout - set(by_video))
    if missing:
        raise ValueError(f"holdout videos are not in the canonical training split: {missing}")
    core = sorted(set(by_video) - holdout)
    if not core or not holdout:
        raise ValueError("both SPORTS_TRAIN_CORE and SPORTS_VAL must be non-empty")

    output_root.mkdir(parents=True, exist_ok=True)
    core_list = output_root / "sports_train_core_images.txt"
    val_list = output_root / "sports_val_expressions.txt"
    holdout_list = output_root / "sports_val_videos.txt"

    # Layout builders only use the video component (index 3) of each training
    # image path.  Keep all frame names so the list is an auditable manifest,
    # while avoiding any copy of image data.
    core_lines: list[str] = []
    for video in core:
        frames = int(by_video[video].get("num_frames") or 0)
        if frames <= 0:
            raise ValueError(f"invalid frame count for {video}: {frames}")
        core_lines.extend(
            f"SportsMOT/train/image_02/{video}/{frame:06d}.jpg" for frame in range(1, frames + 1)
        )
    val_records = [record for record in train_records if str(record.get("video_id")) in holdout]
    val_lines = [
        f"{record['video_id']}+{record['expression_id'].split(':', 1)[-1]}"
        for record in sorted(val_records, key=lambda item: str(item.get("expression_id")))
    ]
    core_list.write_text("\n".join(core_lines) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    holdout_list.write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")

    sport_by_video = {video: str(by_video[video].get("sport_type") or "UNKNOWN") for video in by_video}
    return {
        "schema_version": "refer-sports-internal-split-0.1",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "selection_policy": "fixed_video_disjoint_holdout; stratified_by_sport; no_test_video_used",
        "sports_train_core_videos": core,
        "sports_val_videos": sorted(holdout),
        "video_count": {"train_core": len(core), "val": len(holdout)},
        "sport_count": {
            "train_core": dict(Counter(sport_by_video[video] for video in core)),
            "val": dict(Counter(sport_by_video[video] for video in holdout)),
        },
        "expression_count": {"train_core": sum(str(r.get("video_id")) in set(core) for r in train_records), "val": len(val_records)},
        "frame_count": {"train_core": len(core_lines), "val": sum(int(by_video[v].get("num_frames") or 0) for v in holdout)},
        "lists": {
            "sports_train_core_images": str(core_list.resolve()),
            "sports_train_core_images_sha256": sha256(core_list),
            "sports_val_expressions": str(val_list.resolve()),
            "sports_val_expressions_sha256": sha256(val_list),
            "sports_val_videos": str(holdout_list.resolve()),
            "sports_val_videos_sha256": sha256(holdout_list),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/refer_sports_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-video", action="append", dest="holdout_videos")
    args = parser.parse_args()
    holdout = tuple(args.holdout_videos or DEFAULT_HOLDOUT)
    payload = build_split(
        manifest_path=args.manifest.expanduser().resolve(),
        output_root=args.output.expanduser().resolve(),
        holdout_videos=holdout,
    )
    report_path = args.output.expanduser().resolve() / "manifest.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
