#!/usr/bin/env python3
"""Run the four original-QIM 5k full historical RMOT evaluations in order."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = ("standard", "lstm", "temporal_attention", "mamba")


def run(command, environment, log):
    print("RUN", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("RUN " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"command failed with {result.returncode}: {' '.join(command)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sportsmot-root", type=Path, required=True)
    parser.add_argument("--gt-full", type=Path, required=True)
    parser.add_argument("--gt-141", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--refer-threshold", type=float, default=0.4)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log = args.output_root / "original_5k_eval.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["OMP_NUM_THREADS"] = "1"
    manifest = {
        "gpu": args.gpu,
        "variants": [],
        "protocol": "19 historical videos, 142 recovered expressions; predictions generated once, 142/141 scored separately",
    }
    for variant in VARIANTS:
        checkpoint = args.checkpoint_root / variant / "checkpoint_step00005000.pth"
        if not checkpoint.is_file():
            raise SystemExit(f"missing 5k checkpoint: {checkpoint}")
        variant_root = args.output_root / variant
        prediction_root = variant_root / "predictions"
        inference_summary = variant_root / "inference_142.json"
        metrics_142 = variant_root / "metrics_142.json"
        metrics_141 = variant_root / "metrics_141.json"
        eval_command = [
            args.python, "scripts/eval_rmot_fulltest.py",
            "--checkpoint", str(checkpoint), "--qim-variant", variant,
            "--manifest", str(args.manifest), "--sportsmot-root", str(args.sportsmot_root),
            "--prediction-root", str(prediction_root), "--output", str(inference_summary),
            "--device", "cuda", "--score-threshold", str(args.score_threshold),
            "--refer-threshold", str(args.refer_threshold),
        ]
        run(eval_command, environment, log)
        for historical_141, metrics_path, gt_root in (
            (False, metrics_142, args.gt_full), (True, metrics_141, args.gt_141)
        ):
            score_command = [
                args.python, "scripts/score_rmot_fulltest.py",
                "--manifest", str(args.manifest), "--gt-root", str(gt_root),
                "--prediction-root", str(prediction_root), "--output", str(metrics_path),
            ]
            if historical_141:
                score_command.append("--historical-141")
            run(score_command, environment, log)
        manifest["variants"].append({
            "variant": variant,
            "checkpoint": str(checkpoint),
            "inference_summary": str(inference_summary),
            "metrics_142": str(metrics_142),
            "metrics_141": str(metrics_141),
        })
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("ORIGINAL_5K_FULL_RMOT_DONE", flush=True)


if __name__ == "__main__":
    main()
