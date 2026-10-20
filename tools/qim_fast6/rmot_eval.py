"""Protocol-safe Refer-Sports RMOT metric adapters.

The functions here evaluate expression-conditioned prediction rows.  They do
not create predictions and they never accept GT as an inference input.  GT is
read only by the separate evaluator after prediction files have been written.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKEVAL_ROOT = REPO_ROOT / "TrackEval"
if str(TRACKEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKEVAL_ROOT))

from trackeval.datasets._base_dataset import _BaseDataset
from trackeval.metrics import CLEAR, HOTA, Identity


HISTORICAL_141_EXCLUSION = "v_4LXTUim5anY_c010:players"


def select_test_records(records, historical_141=False):
    selected = [row for row in records if row.get("original_split") == "test"]
    if historical_141:
        selected = [row for row in selected if row.get("expression_id") != HISTORICAL_141_EXCLUSION]
    return sorted(selected, key=lambda row: row.get("expression_id", ""))


def _compact_ids(frame_rows, index=0):
    values = sorted({int(row[index]) for rows in frame_rows for row in rows})
    return {value: position for position, value in enumerate(values)}, values


def metric_data_from_rows(gt_frames, pred_frames):
    """Build the pinned TrackEval sequence schema from ``(id, xywh)`` rows."""

    if len(gt_frames) != len(pred_frames):
        raise ValueError("GT and prediction frame counts differ")
    gt_map, gt_ids = _compact_ids(gt_frames)
    pred_map, pred_ids = _compact_ids(pred_frames)
    gt_id_frames = []
    pred_id_frames = []
    similarities = []
    gt_dets = pred_dets = 0
    for gt_rows, pred_rows in zip(gt_frames, pred_frames):
        gt_id_frames.append(np.asarray([gt_map[int(row[0])] for row in gt_rows], dtype=np.int64))
        pred_id_frames.append(np.asarray([pred_map[int(row[0])] for row in pred_rows], dtype=np.int64))
        gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=np.float32).reshape(-1, 4)
        similarities.append(
            _BaseDataset._calculate_box_ious(gt_boxes, pred_boxes, box_format="xywh")
        )
        gt_dets += len(gt_rows)
        pred_dets += len(pred_rows)
    return {
        "gt_ids": gt_id_frames,
        "tracker_ids": pred_id_frames,
        "similarity_scores": similarities,
        "num_gt_ids": len(gt_ids),
        "num_tracker_ids": len(pred_ids),
        "num_gt_dets": gt_dets,
        "num_tracker_dets": pred_dets,
        "num_timesteps": len(gt_frames),
    }


def evaluate_sequence_rows(gt_frames, pred_frames):
    data = metric_data_from_rows(gt_frames, pred_frames)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    identity = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": float(np.mean(hota["HOTA"])),
        "HOTA_05": float(hota["HOTA"][9]),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "IDF1": float(identity["IDF1"]),
        "IDSW": int(clear["IDSW"]),
        "DetRe": float(hota.get("DetRe", [0.0])[9]),
        "DetPr": float(hota.get("DetPr", [0.0])[9]),
        "LocA": float(hota.get("LocA", [0.0])[9]),
        "gt_dets": int(data["num_gt_dets"]),
        "pred_dets": int(data["num_tracker_dets"]),
        "frames": int(data["num_timesteps"]),
    }


def summarize_sequence_metrics(results):
    if not results:
        raise ValueError("cannot summarize an empty RMOT result set")
    keys = ("HOTA", "HOTA_05", "DetA", "AssA", "IDF1", "DetRe", "DetPr", "LocA")
    summary = {key: float(np.mean([float(row[key]) for row in results])) for key in keys}
    summary["IDSW"] = int(sum(int(row["IDSW"]) for row in results))
    summary["num_sequences"] = len(results)
    summary["num_videos"] = len({row["video"] for row in results})
    return summary


def read_mot_rows(path: Path):
    """Read a MOT text file into one-indexed frame rows of ``(id, xywh)``."""

    rows = {}
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) < 6:
            raise ValueError(f"{path}:{line_number}: expected at least six MOT fields")
        frame = int(float(fields[0]))
        track_id = int(float(fields[1]))
        box = np.asarray([float(value) for value in fields[2:6]], dtype=np.float32)
        rows.setdefault(frame, []).append((track_id, box))
    return rows


def rows_for_length(frame_map, length):
    return [frame_map.get(frame, []) for frame in range(1, length + 1)]


def write_seqmap(names, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name"])
        writer.writerows([[name] for name in names])

