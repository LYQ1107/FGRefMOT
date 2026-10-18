#!/usr/bin/env python3
"""Evaluate many historical expression sequences with one restored model.

Each worker owns one GPU and writes only to a local output root.  Jobs are
restricted to expression directories that also exist in an optional archived
results root, which makes the comparison set explicit and resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_ROOT))

from historical_sequence_eval import (  # noqa: E402
    _filter_predictions,
    _load_expression,
    _load_gt_boxes,
    _load_tracker_rows,
    _preprocess_image,
    _prepare_model,
    _run_metrics,
)


DEFAULT_DATA_ROOT = Path(
    "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--archive-results-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--det-threshold", type=float, default=0.6)
    parser.add_argument("--refer-threshold", type=float, default=0.4)
    parser.add_argument("--area-threshold", type=float, default=100.0)
    parser.add_argument("--keep-all-referring", action="store_true")
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    expression_root = args.data_root / "expression"
    all_jobs = []
    archive_root = args.archive_results_root
    for video_dir in sorted(p for p in expression_root.iterdir() if p.is_dir()):
        for expression_path in sorted(video_dir.glob("*.json")):
            if archive_root is not None:
                archived = archive_root / video_dir.name / expression_path.stem
                if not archived.is_dir():
                    continue
            all_jobs.append((video_dir.name, expression_path.name))
    if args.limit >= 0:
        all_jobs = all_jobs[: args.limit]
    return all_jobs[args.worker_id :: args.workers]


def _evaluate_one(model, device, args: argparse.Namespace, video: str, expression: str) -> dict:
    expression_info, expression_path = _load_expression(args.data_root, video, expression)
    image_root = args.data_root / "KITTI" / "training" / "image_02" / video
    image_paths = sorted(image_root.glob("*.jpg")) + sorted(image_root.glob("*.png"))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"no frames found under {image_root}")
    checkpoint_name = Path(args.checkpoint).stem
    output_dir = args.output_root / checkpoint_name / video / expression_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predict.txt"
    gt_path = output_dir / "gt.txt"
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text())
            if previous.get("complete_sequence") and previous.get("frames_processed") == len(image_paths):
                return previous
        except json.JSONDecodeError:
            pass
    for path in (prediction_path, gt_path):
        path.unlink(missing_ok=True)

    model.clear()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    track_instances = None
    gt_frames = {}
    total_inference_seconds = 0.0
    total_predictions = 0
    peak_memory = 0
    max_detection_score = 0.0
    max_referring_score = 0.0
    max_active_tracks = 0
    active_tracks_above_runtime_threshold = 0
    active_track_snapshot = []
    sentence = [expression_info["sentence"]]
    with prediction_path.open("w") as prediction_file, gt_path.open("w") as gt_file:
        for frame_index, image_path in enumerate(image_paths, start=1):
            image_tensor, original = _preprocess_image(image_path)
            height, width = original.shape[:2]
            image_tensor = image_tensor.to(device)
            start = time.perf_counter()
            with torch.no_grad():
                result = model.inference_single_image(
                    image_tensor, sentence, (height, width), track_instances
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            total_inference_seconds += time.perf_counter() - start
            track_instances = result["track_instances"]
            cpu_instances = track_instances.to(torch.device("cpu"))
            max_detection_score = max(max_detection_score, float(cpu_instances.scores.max().item()))
            max_referring_score = max(max_referring_score, float(cpu_instances.refers.max().item()))
            active_tracks = cpu_instances.obj_idxes >= 0
            max_active_tracks = max(max_active_tracks, int(active_tracks.sum().item()))
            active_tracks_above_runtime_threshold += int(
                ((cpu_instances.obj_idxes >= 0) & (cpu_instances.scores >= args.det_threshold)).sum().item()
            )
            if not active_track_snapshot:
                active_track_snapshot = [
                    {
                        "track_id": int(track_id),
                        "detection_score": float(score),
                        "referring_score": float(referring),
                    }
                    for track_id, score, referring in zip(
                        cpu_instances.obj_idxes.tolist(),
                        cpu_instances.scores.tolist(),
                        cpu_instances.refers.tolist(),
                    )
                    if track_id >= 0
                ]
            boxes, scores, track_ids = _filter_predictions(
                track_instances,
                args.det_threshold,
                args.refer_threshold,
                args.area_threshold,
                args.keep_all_referring,
            )
            total_predictions += len(boxes)
            for box, score, track_id in zip(boxes, scores, track_ids):
                x1, y1, x2, y2 = box
                prediction_file.write(
                    f"{frame_index},{int(track_id)},{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},"
                    f"{float(score):.6f},1,1\n"
                )
            target_ids = {
                int(value) for value in expression_info.get("label", {}).get(str(frame_index), [])
            }
            gt_boxes, gt_ids = _load_gt_boxes(
                args.data_root, video, frame_index, target_ids, width, height
            )
            gt_frames[frame_index] = (gt_boxes, gt_ids)
            for box, object_id in zip(gt_boxes, gt_ids):
                x1, y1, x2, y2 = box
                gt_file.write(
                    f"{frame_index},{object_id},{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},1,1,1\n"
                )
    prediction_rows = _load_tracker_rows(prediction_path)
    metrics = _run_metrics(gt_frames, prediction_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=True)
    metadata = {
        "category": "CHECKPOINT_REPRODUCED",
        "source_root": args.source_root,
        "checkpoint": args.checkpoint,
        "checkpoint_size_bytes": Path(args.checkpoint).stat().st_size,
        "deepfuse_removed_for_historical_namespace": bool(
            getattr(model, "_fgref_v2_deepfuse_removed", False)
        ),
        "data_root": str(args.data_root),
        "video": video,
        "expression": str(expression_path),
        "sentence": expression_info["sentence"],
        "frames_processed": len(image_paths),
        "complete_sequence": True,
        "det_threshold": args.det_threshold,
        "refer_threshold": args.refer_threshold,
        "area_threshold": args.area_threshold,
        "total_predictions": total_predictions,
        "max_detection_score": max_detection_score,
        "max_referring_score": max_referring_score,
        "max_active_tracks": max_active_tracks,
        "active_tracks_above_runtime_threshold": active_tracks_above_runtime_threshold,
        "active_track_snapshot": active_track_snapshot,
        "inference_seconds": total_inference_seconds,
        "fps": len(image_paths) / max(total_inference_seconds, 1e-12),
        "peak_memory_mib": peak_memory / 2**20,
        "metrics": {key: float(metrics.iloc[0][key]) for key in metrics.columns},
        "metrics_file": str(metrics_path),
        "prediction_file": str(prediction_path),
        "gt_file": str(gt_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> int:
    args = _args()
    jobs = _jobs(args)
    if args.worker_id >= args.workers:
        raise ValueError("worker-id must be smaller than workers")
    model, device = _prepare_model(args)
    worker_output = args.output_root / f"worker_{args.worker_id:02d}.jsonl"
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    with worker_output.open("a") as handle:
        for index, (video, expression) in enumerate(jobs, start=1):
            started = time.perf_counter()
            try:
                metadata = _evaluate_one(model, device, args, video, expression)
                record = {"status": "ok", "elapsed_seconds": time.perf_counter() - started, **metadata}
            except Exception as error:  # retain a resumable failure record
                record = {
                    "status": "error",
                    "video": video,
                    "expression": expression,
                    "error": repr(error),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {"worker": args.worker_id, "index": index, "total": len(jobs), **record},
                    sort_keys=True,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
