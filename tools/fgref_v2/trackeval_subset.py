"""Run the repository's TrackEval fork on one locally generated RMOT pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKEVAL_ROOT = REPO_ROOT / "TrackEval"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True,
                        help="directory containing <video>/<expression>/{gt,predict}.txt")
    parser.add_argument("--video", required=True)
    parser.add_argument("--expression", required=True,
                        help="expression directory name without .json")
    parser.add_argument("--num-frames", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    sequence = f"{args.video}+{args.expression}"
    gt_path = run_root / args.video / args.expression / "gt.txt"
    prediction_path = run_root / args.video / args.expression / "predict.txt"
    if not gt_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing generated files: {gt_path} / {prediction_path}")

    sys.path.insert(0, str(TRACKEVAL_ROOT))
    import trackeval  # noqa: E402

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(run_root),
        "TRACKERS_FOLDER": str(run_root),
        "TRACKERS_TO_EVAL": [str(run_root)],
        "TRACKER_DISPLAY_NAMES": ["historical_checkpoint"],
        "SEQ_INFO": {sequence: args.num_frames},
        "SKIP_SPLIT_FOL": True,
        "GT_LOC_FORMAT": "{gt_folder}/{video_id}/{expression_id}/gt.txt",
        "PRINT_CONFIG": False,
        "OUTPUT_FOLDER": str(run_root / "trackeval"),
        "PLOT_CURVES": False,
    })
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "DISPLAY_LESS_PROGRESS": True,
    })
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    result, messages = evaluator.evaluate([dataset], metrics)
    dataset_name = dataset.get_name()
    tracker_result = result[dataset_name][str(run_root)]
    combined = tracker_result["COMBINED_SEQ"]["pedestrian"]
    summary = {
        "run_root": str(run_root),
        "sequence": sequence,
        "num_frames": args.num_frames,
        "status": messages[dataset_name][str(run_root)],
        # TrackEval reports HOTA/DetA/AssA over its 19 alpha thresholds;
        # the official summary uses the mean, matching pedestrian_summary.txt.
        "HOTA": float(100.0 * combined["HOTA"]["HOTA"].mean()),
        "DetA": float(100.0 * combined["HOTA"]["DetA"].mean()),
        "AssA": float(100.0 * combined["HOTA"]["AssA"].mean()),
        "IDF1": float(100.0 * combined["Identity"]["IDF1"]),
        "IDSW": float(combined["CLEAR"]["IDSW"]),
        "trackeval_output": str(run_root / "trackeval"),
    }
    print("TRACKEVAL_SUBSET_OK")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
