#!/usr/bin/env python3
"""Measure complete-model FPS on the same expression-conditioned clip.

This is an engineering speed measurement, not an accuracy evaluator.  It
times the complete ``inference_single_image`` path (backbone, decoder, TAM,
STReasoner, QIM, and tracker) and separately records wall time including image
decode and transforms.  No GT is read.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.refer_sports import make_transforms_for_sports
from main import get_args_parser
from models import build_model


ORIGINAL = ("standard", "lstm", "temporal_attention", "mamba")
FAST6 = (
    "fast_mamba_spatial", "fast_mamba_temporal", "fast_mamba_streaming",
    "fast_mamba_motion", "fast_mamba_language", "fast_mamba_reliability",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "expression"


def find_video(root: Path, video: str) -> Path:
    for split in ("test", "val", "train"):
        candidate = root / "dataset" / split / video
        if (candidate / "img1").is_dir():
            return candidate
    raise FileNotFoundError(video)


def load_model(checkpoint_path: Path, variant: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = copy.deepcopy(checkpoint["args"])
    args.qim_variant = variant
    args.device = str(device)
    args.pretrained = None
    args.resume = ""
    args.use_checkpoint = False
    model, _, _ = build_model(args)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [
        key for key in unexpected
        if not key.endswith("total_params") and not key.endswith("total_ops")
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"{variant} checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    return model


@torch.no_grad()
def benchmark(model, image_paths, sentence, transform, device, warmup_frames):
    model.clear()
    track_instances = None
    model_ms = []
    wall_ms = []
    active_counts = []
    for index, image_path in enumerate(image_paths):
        wall_start = time.perf_counter()
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        raw_width, raw_height = image.size
        images, _ = transform([image], [None])
        tensor = images[0].to(device)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = model.inference_single_image(
                [tensor], [sentence], (raw_height, raw_width), track_instances
            )
            end.record()
            end.synchronize()
            elapsed_model = float(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            result = model.inference_single_image(
                [tensor], [sentence], (raw_height, raw_width), track_instances
            )
            elapsed_model = (time.perf_counter() - start_time) * 1000.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        track_instances = result["track_instances"]
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        active_counts.append(int((track_instances.obj_idxes >= 0).sum().item()))
        if index >= warmup_frames:
            model_ms.append(elapsed_model)

    model_values = np.asarray(model_ms, dtype=np.float64)
    wall_values = np.asarray(wall_ms[warmup_frames:], dtype=np.float64)
    result = {
        "frames_total": len(image_paths),
        "warmup_frames": warmup_frames,
        "timed_frames": int(len(model_values)),
        "model_mean_ms": float(model_values.mean()),
        "model_median_ms": float(np.median(model_values)),
        "model_p95_ms": float(np.percentile(model_values, 95)),
        "full_model_fps": float(1000.0 / model_values.mean()),
        "wall_mean_ms": float(wall_values.mean()),
        "wall_median_ms": float(np.median(wall_values)),
        "wall_p95_ms": float(np.percentile(wall_values, 95)),
        "end_to_end_wall_fps": float(1000.0 / wall_values.mean()),
        "active_tracks_mean": float(np.mean(active_counts[warmup_frames:])),
        "active_tracks_max": int(max(active_counts[warmup_frames:])),
        "active_tracks_per_frame": active_counts,
    }
    if device.type == "cuda":
        result["peak_allocated_gb"] = float(torch.cuda.max_memory_allocated(device) / 2**30)
        result["peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(device) / 2**30)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-checkpoint-root", type=Path, required=True)
    parser.add_argument("--fast-checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sportsmot-root", type=Path, required=True)
    parser.add_argument("--video", default="")
    parser.add_argument("--expression-id", default="")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(ORIGINAL + FAST6))
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the requested benchmark")
        torch.cuda.set_device(0 if device.index is None else device.index)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row.get("original_split") == "test"]
    if args.expression_id:
        records = [row for row in records if row["expression_id"] == args.expression_id]
    elif args.video:
        records = [row for row in records if row["original_sportsmot_id"] == args.video]
    if not records:
        raise SystemExit("no matching test expression in manifest")
    record = sorted(records, key=lambda row: row["expression_id"])[0]
    video_dir = find_video(args.sportsmot_root, record["original_sportsmot_id"])
    image_paths = sorted((video_dir / "img1").glob("*"), key=lambda path: int(path.stem))
    image_paths = image_paths[:args.frames]
    if len(image_paths) <= args.warmup_frames:
        raise ValueError("need more frames than warmup-frames")
    transform = make_transforms_for_sports("val")
    report = {
        "protocol": {
            "type": "complete-model expression-conditioned speed benchmark",
            "video": record["original_sportsmot_id"],
            "expression_id": record["expression_id"],
            "sentence": record["sentence"],
            "frames": len(image_paths),
            "warmup_frames": args.warmup_frames,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "precision": "fp32",
            "uses_gt": False,
            "timed_path": "backbone+TAM+decoder+STReasoner+QIM+RuntimeTracker",
        },
        "variants": {},
    }
    checkpoint_roots = {variant: args.original_checkpoint_root for variant in ORIGINAL}
    checkpoint_roots.update({variant: args.fast_checkpoint_root for variant in FAST6})
    checkpoint_names = {variant: "checkpoint_step00005000.pth" for variant in ORIGINAL}
    checkpoint_names.update({variant: "checkpoint_step00020000.pth" for variant in FAST6})
    for variant in args.variants:
        if variant not in checkpoint_roots:
            raise SystemExit(f"unknown variant: {variant}")
        checkpoint = checkpoint_roots[variant] / variant / checkpoint_names[variant]
        print(f"building {variant}", flush=True)
        model = load_model(checkpoint, variant, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = benchmark(model, image_paths, record["sentence"], transform, device, args.warmup_frames)
        result.update({
            "checkpoint": str(checkpoint),
            "qim_params": int(sum(p.numel() for p in model.track_embed.parameters())),
            "full_model_params": int(sum(p.numel() for p in model.parameters())),
        })
        report["variants"][variant] = result
        print(json.dumps({variant: result}, indent=2), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
