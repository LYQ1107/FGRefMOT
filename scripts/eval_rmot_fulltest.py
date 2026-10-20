#!/usr/bin/env python3
"""Run expression-conditioned FGRefMOT inference on the 19-video Test set."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make ``python scripts/eval_rmot_fulltest.py`` resolve repository modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import get_args_parser
from models import build_model
from datasets.refer_sports import make_transforms_for_sports


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "expression"


def find_video(root: Path, video: str) -> Path:
    for split in ("test", "val", "train"):
        candidate = root / "dataset" / split / video
        if (candidate / "img1").is_dir():
            return candidate
    raise FileNotFoundError(video)


def write_prediction(path, frame, track_instances, score_threshold, refer_threshold):
    if track_instances is None:
        return 0
    keep = (
        (track_instances.obj_idxes >= 0)
        & (track_instances.scores >= score_threshold)
        & (track_instances.refers >= refer_threshold)
    )
    ids = track_instances.obj_idxes[keep].detach().cpu().tolist()
    boxes = track_instances.boxes[keep].detach().cpu().numpy()
    scores = track_instances.scores[keep].detach().cpu().tolist()
    refers = track_instances.refers[keep].detach().cpu().tolist()
    with path.open("a", encoding="utf-8") as handle:
        for object_id, box, score, refer in zip(ids, boxes, scores, refers):
            x0, y0, x1, y1 = (float(value) for value in box)
            handle.write(
                f"{frame},{int(object_id)},{x0:.6f},{y0:.6f},{x1 - x0:.6f},{y1 - y0:.6f},"
                f"{float(score):.8f},{float(refer):.8f},1\n"
            )
    return len(ids)


@torch.no_grad()
def run_record(model, transform, record, sports_root, prediction_root, device, score_threshold, refer_threshold, max_frames):
    video = record["original_sportsmot_id"]
    sequence = find_video(sports_root, video)
    image_paths = sorted((sequence / "img1").glob("*"), key=lambda item: int(item.stem))
    if max_frames > 0:
        image_paths = image_paths[:max_frames]
    expression_dir = prediction_root / safe_name(record["expression_id"])
    expression_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = expression_dir / "predict.txt"
    if prediction_file.exists():
        prediction_file.unlink()
    model.clear()
    track_instances = None
    start = time.perf_counter()
    kept = 0
    transform_start = time.perf_counter()
    for frame_index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        raw_width, raw_height = image.size
        images, _ = transform([image], [None])
        result = model.inference_single_image(
            [images[0].to(device)], [record["sentence"]], (raw_height, raw_width), track_instances
        )
        track_instances = result["track_instances"]
        kept += write_prediction(prediction_file, frame_index, track_instances, score_threshold, refer_threshold)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    timing = {
        "video": video,
        "expression_id": record["expression_id"],
        "frames": len(image_paths),
        "expression_frame_pairs": len(image_paths),
        "wall_seconds": elapsed,
        "end_to_end_fps": len(image_paths) / elapsed if elapsed else 0.0,
        "mean_kept_referring_tracks": kept / max(1, len(image_paths)),
        "score_threshold": score_threshold,
        "refer_threshold": refer_threshold,
        "inference_uses_gt": False,
        "timing_note": "FPS is video frames for one expression; expression-frame pairs are reported separately",
        "preprocess_seconds_inclusive": time.perf_counter() - transform_start,
    }
    (expression_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    return timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qim-variant", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sportsmot-root", type=Path, default=Path("/data/DATASETS/TRACKING/JDE/SportsMOT/sportsmot_publish"))
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--historical-141", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--refer-threshold", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-expressions", type=int, default=0,
                        help="development smoke limit; zero means all selected expressions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row.get("original_split") == "test"]
    if args.historical_141:
        records = [row for row in records if row.get("expression_id") != "v_4LXTUim5anY_c010:players"]
    records.sort(key=lambda row: row["expression_id"])
    if args.max_expressions > 0:
        records = records[:args.max_expressions]
    model_args = get_args_parser().parse_args([])
    model_args.meta_arch = "temp_rmot"
    model_args.dataset_file = "e2e_rmot_sports"
    model_args.device = str(device)
    model_args.with_box_refine = True
    model_args.dropout = 0.0
    model_args.random_drop = 0.1
    model_args.fp_ratio = 0.3
    model_args.merger_dropout = 0.0
    model_args.update_query_pos = True
    model_args.use_checkpoint = False
    model_args.qim_variant = args.qim_variant
    model_args.sampler_lengths = [4, 4, 4, 4]
    model_args.sports_split = "test"
    model_args.sportsmot_root = str(args.sportsmot_root)
    model, _, _ = build_model(model_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [key for key in unexpected if not key.endswith("total_params") and not key.endswith("total_ops")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    transform = make_transforms_for_sports("val")
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    timings = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record['expression_id']}", flush=True)
        timings.append(run_record(
            model, transform, record, args.sportsmot_root, args.prediction_root,
            device, args.score_threshold, args.refer_threshold, args.max_frames,
        ))
    summary = {
        "checkpoint": str(args.checkpoint),
        "qim_variant": args.qim_variant,
        "manifest": str(args.manifest),
        "historical_141": bool(args.historical_141),
        "num_expressions": len(records),
        "num_videos": len({row["original_sportsmot_id"] for row in records}),
        "inference_uses_gt": False,
        "filter_policy": "obj_idxes>=0, scores>=score_threshold, refers>=refer_threshold",
        "timings": timings,
        "mean_end_to_end_fps_per_expression": float(np.mean([row["end_to_end_fps"] for row in timings])) if timings else 0.0,
        "total_frames": int(sum(row["frames"] for row in timings)),
        "total_expression_frame_pairs": int(sum(row["expression_frame_pairs"] for row in timings)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("num_expressions", "num_videos", "total_frames", "mean_end_to_end_fps_per_expression")}, indent=2))


if __name__ == "__main__":
    main()
