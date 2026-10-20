#!/usr/bin/env python3
"""Score expression-specific RMOT prediction files with pinned TrackEval."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make ``python scripts/score_rmot_fulltest.py`` resolve repository modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.qim_fast6.rmot_eval import (
    read_mot_rows,
    rows_for_length,
    select_test_records,
    summarize_sequence_metrics,
    evaluate_sequence_rows,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "expression"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-141", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = select_test_records(manifest["records"], historical_141=args.historical_141)
    results = []
    missing = []
    for record in records:
        name = safe_name(record["expression_id"])
        gt_file = args.gt_root / name / "gt.txt"
        prediction_file = args.prediction_root / name / "predict.txt"
        if not gt_file.is_file() or not prediction_file.is_file():
            missing.append({"expression_id": record["expression_id"], "gt": str(gt_file), "prediction": str(prediction_file)})
            continue
        length = int(record["num_frames"])
        gt = rows_for_length(read_mot_rows(gt_file), length)
        pred = rows_for_length(read_mot_rows(prediction_file), length)
        metrics = evaluate_sequence_rows(gt, pred)
        metrics.update({"expression_id": record["expression_id"], "video": record["original_sportsmot_id"]})
        results.append(metrics)
    if missing:
        raise SystemExit("missing prediction/GT files for {} expression(s); first={}".format(len(missing), missing[0]))
    summary = summarize_sequence_metrics(results)
    output = {
        "manifest": str(args.manifest),
        "historical_141": bool(args.historical_141),
        "protocol": "expression-conditioned RMOT; referring-filtered predictions; GT read only during scoring",
        "inference_uses_gt": False,
        "summary": summary,
        "sequences": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
