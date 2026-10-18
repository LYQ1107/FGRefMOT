"""Run an auditable online evaluation for the historical FGRefMOT checkpoint.

The script deliberately imports the historical TempRMOT source tree instead of
the research working tree.  This keeps the checkpoint reproduction path
separate from the optional FGRefMOT-v2 modules.  NAS inputs are read only;
predictions and metrics are written to a local output directory.

By default this is a bounded sequence diagnostic.  ``--max-frames -1`` runs
the complete selected video/expression pair and writes files that can be
consumed by the repository's TrackEval fork.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import torch
import torchvision.transforms.functional as F
from transformers import RobertaModel as HFModel
from transformers import RobertaTokenizerFast as HFTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPooler


DEFAULT_SOURCE_ROOT = "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT"
DEFAULT_CHECKPOINT = (
    "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/"
    "exps/official_checkpoint/checkpoint_0060.pth"
)
DEFAULT_DATA_ROOT = "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti"
DEFAULT_VIDEO = "v_-6Os86HzwCs_c001"
DEFAULT_EXPRESSION = "basketball-players-with-blue-uniforms.json"
DEFAULT_OUTPUT_ROOT = "/data1/liuyeqiang/fgref_v2_runs/historical_sequence_eval"
DEFAULT_HF_ROOT = (
    "/home/liuyeqiang/.cache/huggingface/hub/models--roberta-base/"
    "snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)


class _OfflineTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFTokenizer.from_pretrained(DEFAULT_HF_ROOT, local_files_only=True)


class _OfflineModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFModel.from_pretrained(DEFAULT_HF_ROOT, local_files_only=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--expression", default=DEFAULT_EXPRESSION)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-frames", type=int, default=8,
                        help="number of frames; -1 means the complete sequence")
    parser.add_argument("--det-threshold", type=float, default=0.6)
    parser.add_argument("--refer-threshold", type=float, default=0.4)
    parser.add_argument("--area-threshold", type=float, default=100.0)
    parser.add_argument("--keep-all-referring", action="store_true",
                        help="write every referring-positive prediction, including low detection scores")
    return parser.parse_args()


def _prepare_model(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, args.source_root)

    import models.backbone as backbone  # noqa: WPS433
    import models.transrmot_pro as transrmot  # noqa: WPS433

    backbone.is_main_process = lambda: False
    transrmot.RobertaTokenizerFast = _OfflineTokenizer
    transrmot.RobertaModel = _OfflineModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["args"]
    model_args.device = args.device
    model_args.meta_arch = "temp_rmot"
    model_args.use_checkpoint = False
    model_args.sampler_lengths = [1, 1, 1, 1]
    model_args.sampler_steps = [60, 80, 90]
    model_args.rmot_path = args.data_root
    model_args.distributed = False
    model_args.rank = 0
    model_args.world_size = 1
    model_args.gpu = 0

    model, _, _ = transrmot.build(model_args)
    checkpoint_state = checkpoint["model"]
    # TempMam was edited after this checkpoint was produced: the current tree
    # contains an additional Triple_Alignment module, while the archived Mamba
    # checkpoints do not contain any ``deepfuse.*`` tensors.  Remove that
    # post-checkpoint module before strict loading and activate the source's
    # explicit ablation switch so the forward path matches the archived model.
    deepfuse_removed = False
    if hasattr(model, "deepfuse") and not any(
        key.startswith("deepfuse.") for key in checkpoint_state
    ):
        del model.deepfuse
        model._ablate_deepfuse = True
        deepfuse_removed = True
    # The historical model removed the unused pooler and Transformers 4.35
    # makes position_ids non-persistent.  Restore both historical details so
    # that this is a true strict checkpoint restore, not a partial load.
    if "text_encoder.pooler.dense.weight" in checkpoint_state:
        model.text_encoder.pooler = RobertaPooler(model.text_encoder.config)
    if "text_encoder.embeddings.position_ids" in checkpoint_state:
        model.text_encoder.embeddings._non_persistent_buffers_set.discard("position_ids")
    model.load_state_dict(checkpoint_state, strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    model._fgref_v2_deepfuse_removed = deepfuse_removed
    return model, device


def _preprocess_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()
    original_height, original_width = image.shape[:2]
    target_height = 800
    target_width = 1536
    scale = target_height / min(original_height, original_width)
    if max(original_height, original_width) * scale > target_width:
        scale = target_width / max(original_height, original_width)
    resized = cv2.resize(
        image,
        (int(original_width * scale), int(original_height * scale)),
    )
    tensor = F.normalize(
        F.to_tensor(resized),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    # ``inference_single_image`` follows the historical inference.py caller,
    # which passes a one-item image batch rather than a bare CHW tensor.
    return tensor.unsqueeze(0), original


def _load_expression(data_root: Path, video: str, expression: str) -> tuple[dict, Path]:
    path = data_root / "expression" / video / expression
    if not path.is_file():
        candidates = sorted((data_root / "expression" / video).glob("*.json"))
        if not candidates:
            raise FileNotFoundError(path)
        raise FileNotFoundError(f"expression not found: {path}; available={candidates}")
    with path.open() as handle:
        return json.load(handle), path


def _load_gt_boxes(data_root: Path, video: str, frame_id: int, target_ids: set[int], width: int, height: int):
    label_path = (
        data_root / "KITTI" / "labels_with_ids" / "image_02" / video
        / f"{frame_id:06d}.txt"
    )
    if not label_path.is_file():
        return np.empty((0, 4), dtype=np.float32), []
    rows = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
    boxes = []
    ids = []
    for row in rows:
        object_id = int(row[1])
        if object_id not in target_ids:
            continue
        x1 = row[2] * width
        y1 = row[3] * height
        x2 = (row[2] + row[4]) * width
        y2 = (row[3] + row[5]) * height
        boxes.append([x1, y1, x2, y2])
        ids.append(object_id)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), ids


def _xyxy_to_tlwh(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    output = boxes.copy()
    output[:, 2] = boxes[:, 2] - boxes[:, 0]
    output[:, 3] = boxes[:, 3] - boxes[:, 1]
    return output


def _filter_predictions(track_instances, det_threshold: float, refer_threshold: float,
                        area_threshold: float, keep_all_referring: bool):
    instances = track_instances.to(torch.device("cpu"))
    keep = instances.obj_idxes >= 0
    if not keep_all_referring:
        keep &= instances.scores > det_threshold
    keep &= instances.refers > refer_threshold
    boxes = instances.boxes[keep].detach().numpy().astype(np.float32)
    scores = instances.scores[keep].detach().numpy().astype(np.float32)
    ids = instances.obj_idxes[keep].detach().numpy().astype(np.int64) + 1
    if len(boxes):
        wh = boxes[:, 2:4] - boxes[:, :2]
        keep_area = (wh[:, 0] * wh[:, 1]) > area_threshold
        boxes, scores, ids = boxes[keep_area], scores[keep_area], ids[keep_area]
    return boxes, scores, ids


def _load_tracker_rows(path: Path):
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        values = line.split(",")
        if len(values) < 7:
            continue
        frame = int(values[0])
        rows.setdefault(frame, []).append((
            np.asarray([float(values[2]), float(values[3]), float(values[2]) + float(values[4]),
                        float(values[3]) + float(values[5])], dtype=np.float32),
            int(float(values[1])),
        ))
    return rows


def _run_metrics(gt_frames, prediction_rows):
    accumulator = mm.MOTAccumulator(auto_id=True)
    true_positive = false_positive = false_negative = 0
    for frame_id in sorted(set(gt_frames) | set(prediction_rows)):
        gt_boxes, gt_ids = gt_frames.get(frame_id, (np.empty((0, 4)), []))
        predictions = prediction_rows.get(frame_id, [])
        pred_boxes = np.asarray([row[0] for row in predictions], dtype=np.float32).reshape(-1, 4)
        pred_ids = [row[1] for row in predictions]
        distance = mm.distances.iou_matrix(
            _xyxy_to_tlwh(gt_boxes), _xyxy_to_tlwh(pred_boxes), max_iou=0.5,
        )
        accumulator.update(gt_ids, pred_ids, distance)
        matched_rows, matched_cols = mm.lap.linear_sum_assignment(distance)
        if len(matched_rows):
            matched = ~np.isnan(distance[matched_rows, matched_cols])
            true_positive += int(matched.sum())
        false_negative += len(gt_boxes) - int((~np.isnan(distance[matched_rows, matched_cols])).sum()) if len(matched_rows) else len(gt_boxes)
        false_positive += len(pred_boxes) - int((~np.isnan(distance[matched_rows, matched_cols])).sum()) if len(matched_rows) else len(pred_boxes)

    summary = mm.metrics.create().compute(
        accumulator,
        metrics=["idf1", "num_switches", "mota", "precision", "recall"],
        name="historical_checkpoint",
    )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    summary["referring_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
    summary["referring_tp"] = true_positive
    summary["referring_fp"] = false_positive
    summary["referring_fn"] = false_negative
    return summary


def main() -> int:
    args = _parse_args()
    model, device = _prepare_model(args)
    data_root = Path(args.data_root)
    expression_info, expression_path = _load_expression(data_root, args.video, args.expression)
    image_root = data_root / "KITTI" / "training" / "image_02" / args.video
    image_paths = sorted(image_root.glob("*.jpg")) + sorted(image_root.glob("*.png"))
    image_paths = sorted(image_paths)
    if args.max_frames >= 0:
        image_paths = image_paths[:args.max_frames]
    if not image_paths:
        raise FileNotFoundError(f"no frames found under {image_root}")

    checkpoint_name = Path(args.checkpoint).stem
    run_name = f"{checkpoint_name}/{args.video}/{expression_path.stem}"
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predict.txt"
    gt_path = output_dir / "gt.txt"
    metadata_path = output_dir / "metadata.json"
    for path in (prediction_path, gt_path):
        path.unlink(missing_ok=True)

    sentence = [expression_info["sentence"]]
    track_instances = None
    gt_frames = {}
    total_inference_seconds = 0.0
    peak_memory = 0
    total_predictions = 0
    max_detection_score = 0.0
    max_referring_score = 0.0
    max_active_tracks = 0
    active_tracks_above_runtime_threshold = 0
    active_track_snapshot = []

    with prediction_path.open("w") as prediction_file, gt_path.open("w") as gt_file:
        for frame_index, image_path in enumerate(image_paths, start=1):
            image_tensor, original = _preprocess_image(image_path)
            height, width = original.shape[:2]
            image_tensor = image_tensor.to(device)
            start = time.perf_counter()
            with torch.no_grad():
                result = model.inference_single_image(
                    image_tensor,
                    sentence,
                    (height, width),
                    track_instances,
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
                ((cpu_instances.obj_idxes >= 0) & (cpu_instances.scores >= 0.6)).sum().item()
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
                    f"{frame_index},{int(track_id)},{x1:.4f},{y1:.4f},{x2 - x1:.4f},{y2 - y1:.4f},"
                    f"{float(score):.6f},1,1\n"
                )

            target_ids = {int(value) for value in expression_info.get("label", {}).get(str(frame_index), [])}
            gt_boxes, gt_ids = _load_gt_boxes(data_root, args.video, frame_index, target_ids, width, height)
            gt_frames[frame_index] = (gt_boxes, gt_ids)
            for box, object_id in zip(gt_boxes, gt_ids):
                x1, y1, x2, y2 = box
                gt_file.write(
                    f"{frame_index},{object_id},{x1:.4f},{y1:.4f},{x2 - x1:.4f},{y2 - y1:.4f},1,1,1\n"
                )

    prediction_rows = _load_tracker_rows(prediction_path)
    metrics = _run_metrics(gt_frames, prediction_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=True)
    metadata = {
        "category": "CHECKPOINT_REPRODUCED" if args.max_frames < 0 else "CHECKPOINT_DIAGNOSTIC",
        "source_root": args.source_root,
        "checkpoint": args.checkpoint,
        "checkpoint_size_bytes": Path(args.checkpoint).stat().st_size,
        "deepfuse_removed_for_historical_namespace": bool(
            getattr(model, "_fgref_v2_deepfuse_removed", False)
        ),
        "data_root": args.data_root,
        "video": args.video,
        "expression": str(expression_path),
        "sentence": expression_info["sentence"],
        "frames_processed": len(image_paths),
        "complete_sequence": args.max_frames < 0,
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
        "fps": len(image_paths) / total_inference_seconds,
        "peak_memory_mib": peak_memory / 2**20,
        "metrics_file": str(metrics_path),
        "prediction_file": str(prediction_path),
        "gt_file": str(gt_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("HISTORICAL_SEQUENCE_EVAL_OK")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(metrics.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
