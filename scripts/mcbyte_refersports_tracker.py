#!/usr/bin/env python3
"""Run the shared, leakage-safe McByte tracker on Refer-Sports frames.

The script reads only image files from the 19 evaluation sequences.  It never
opens ``gt/`` or any annotation file, so the tracker outputs cannot consume
evaluation ground truth.  A worker may process one or more complete sequences;
sequence boundaries reset ByteTrack state and track IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.byte_tracker.basetrack import BaseTrack
from yolox.byte_tracker.byte_tracker import BYTETracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-file", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--sequences", nargs="+", required=True)
    p.add_argument("--test-size", default="512x896", help="HxW")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--nms", type=float, default=0.70)
    p.add_argument("--track-thresh", type=float, default=0.50)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--match-thresh", type=float, default=0.80)
    p.add_argument("--min-box-area", type=float, default=100.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=None, help="debug limit; omit for full sequence")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(args: argparse.Namespace, test_size: tuple[int, int]):
    exp = get_exp(args.exp_file, None)
    exp.test_size = test_size
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    model = exp.get_model()
    device = torch.device(args.device)
    model.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    if args.fp16:
        model.half()
    return exp, model, device


def image_paths(seq_dir: Path) -> list[Path]:
    paths = sorted((seq_dir / "img1").glob("*.jpg"))
    if not paths:
        paths = sorted((seq_dir / "img1").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no frames under {seq_dir / 'img1'}")
    return paths


def tracker_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=False,
    )


def write_results(path: Path, rows: list[tuple[int, int, np.ndarray, float]]) -> None:
    with path.open("w") as f:
        for frame_id, track_id, tlwh, score in rows:
            x, y, w, h = [float(v) for v in tlwh]
            f.write(
                f"{frame_id},{track_id},{x:.1f},{y:.1f},{w:.1f},{h:.1f},"
                f"{float(score):.2f},-1,-1,-1\n"
            )


@torch.inference_mode()
def process_sequence(
    seq: str,
    data_root: Path,
    output_root: Path,
    exp,
    model,
    device: torch.device,
    args: argparse.Namespace,
    test_size: tuple[int, int],
) -> dict:
    seq_dir = data_root / "train" / seq
    frames = image_paths(seq_dir)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    out_path = output_root / f"{seq}.txt"
    targs = tracker_args(args)
    BaseTrack._count = 0
    tracker = BYTETracker(targs, frame_rate=30)
    rows: list[tuple[int, int, np.ndarray, float]] = []
    processed = 0

    for start in range(0, len(frames), args.batch_size):
        batch_paths = frames[start : start + args.batch_size]
        originals = []
        tensors = []
        for path in batch_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"failed to read {path}")
            originals.append(image)
            tensor, _ratio = preproc(
                image,
                test_size,
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            )
            tensors.append(tensor)
        inputs = torch.from_numpy(np.stack(tensors, axis=0)).to(device)
        if args.fp16:
            inputs = inputs.half()
        else:
            inputs = inputs.float()
        outputs = postprocess(model(inputs), exp.num_classes, args.conf, args.nms)

        for offset, (image, output) in enumerate(zip(originals, outputs)):
            frame_id = start + offset + 1
            if output is None:
                processed += 1
                continue
            info = [image.shape[0], image.shape[1]]
            online_targets = tracker.update(output, info, test_size)
            for track in online_targets:
                tlwh = track.tlwh
                if tlwh[2] <= 0 or tlwh[3] <= 0:
                    continue
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    rows.append((frame_id, track.track_id, tlwh, track.score))
            processed += 1
        if processed and processed % 200 == 0:
            logger.info("{}: {}/{} frames", seq, processed, len(frames))

    write_results(out_path, rows)
    logger.info("{}: wrote {} rows to {}", seq, len(rows), out_path)
    return {"sequence": seq, "frames": len(frames), "rows": len(rows), "output": str(out_path)}


def main() -> None:
    args = parse_args()
    h, w = (int(x) for x in args.test_size.lower().split("x", 1))
    test_size = (h, w)
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    exp, model, device = load_model(args, test_size)
    logger.info("loaded checkpoint {} sha256={} device={}", checkpoint, sha256_file(checkpoint), device)

    results = []
    for seq in args.sequences:
        results.append(process_sequence(seq, data_root, output_root, exp, model, device, args, test_size))

    manifest = {
        "tracker": "McByte-SportsMOT",
        "detector_checkpoint": str(checkpoint),
        "detector_checkpoint_sha256": sha256_file(checkpoint),
        "data_root": str(data_root),
        "sequences": list(args.sequences),
        "used_ground_truth": False,
        "test_size": list(test_size),
        "conf": args.conf,
        "nms": args.nms,
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "results": results,
    }
    (output_root / "TRACKER_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
