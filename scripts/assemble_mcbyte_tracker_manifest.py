#!/usr/bin/env python3
"""Assemble one race-free manifest after parallel McByte workers finish."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence-file", required=True)
    p.add_argument("--test-size", default="512x896")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--nms", type=float, default=0.70)
    p.add_argument("--track-thresh", type=float, default=0.50)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--match-thresh", type=float, default=0.80)
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    sequences = [
        line.strip()
        for line in Path(args.sequence_file).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    results = []
    missing = []
    for seq in sequences:
        seq_dir = data_root / "train" / seq
        frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted((seq_dir / "img1").glob("*.png"))
        out_path = output_root / f"{seq}.txt"
        if not out_path.is_file():
            missing.append(seq)
            continue
        rows = sum(1 for _ in out_path.open())
        results.append(
            {
                "sequence": seq,
                "frames": len(frame_paths),
                "rows": rows,
                "output": str(out_path),
            }
        )
    if missing:
        raise SystemExit(f"missing tracker outputs: {', '.join(missing)}")

    h, w = (int(x) for x in args.test_size.lower().split("x", 1))
    manifest = {
        "tracker": "McByte-SportsMOT",
        "variant": "shared detector/tracker outputs for FlexHook and YesTrack+McByte",
        "detector_checkpoint": str(checkpoint),
        "detector_checkpoint_sha256": sha256_file(checkpoint),
        "data_root": str(data_root),
        "sequence_file": str(Path(args.sequence_file).resolve()),
        "sequences": sequences,
        "sequence_count": len(sequences),
        "used_ground_truth": False,
        "used_evaluation_frames_for_training": False,
        "used_evaluation_ground_truth": False,
        "test_size": [h, w],
        "conf": args.conf,
        "nms": args.nms,
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "results": results,
    }
    (output_root / "TRACKER_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output_root / "TRACKER_MANIFEST.json"), "sequence_count": len(results)}))


if __name__ == "__main__":
    main()
