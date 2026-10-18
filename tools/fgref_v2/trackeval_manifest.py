"""Run the repository TrackEval fork over every completed manifest pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKEVAL_ROOT = REPO_ROOT / "TrackEval"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True,
                        help="checkpoint-specific root containing video/expression pairs")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    jobs = {}
    for metadata_path in sorted(run_root.glob("*/*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            continue
        if not metadata.get("complete_sequence"):
            continue
        video = metadata_path.parent.parent.name
        expression = metadata_path.parent.name
        gt_path = metadata_path.parent / "gt.txt"
        prediction_path = metadata_path.parent / "predict.txt"
        if not gt_path.is_file() or not prediction_path.is_file():
            continue
        sequence = f"{video}+{expression}"
        jobs[sequence] = int(metadata["frames_processed"])
    if not jobs:
        raise RuntimeError(f"no complete generated pairs under {run_root}")

    sys.path.insert(0, str(TRACKEVAL_ROOT))
    import trackeval  # noqa: E402

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(run_root),
        "TRACKERS_FOLDER": str(run_root),
        "TRACKERS_TO_EVAL": [str(run_root)],
        "TRACKER_DISPLAY_NAMES": ["historical_checkpoint"],
        "SEQ_INFO": jobs,
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
        "sequences": len(jobs),
        "frames": int(sum(jobs.values())),
        "status": messages[dataset_name][str(run_root)],
        "HOTA": float(100.0 * combined["HOTA"]["HOTA"].mean()),
        "DetA": float(100.0 * combined["HOTA"]["DetA"].mean()),
        "AssA": float(100.0 * combined["HOTA"]["AssA"].mean()),
        "IDF1": float(100.0 * combined["Identity"]["IDF1"]),
        "IDSW": float(combined["CLEAR"]["IDSW"]),
        "trackeval_output": str(run_root / "trackeval"),
    }
    (run_root / "trackeval_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("TRACKEVAL_MANIFEST_OK")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
