#!/usr/bin/env python3
"""Launch one complete historical RMOT evaluation per finished fast-QIM run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = (
    "fast_mamba_spatial",
    "fast_mamba_temporal",
    "fast_mamba_streaming",
    "fast_mamba_motion",
    "fast_mamba_language",
    "fast_mamba_reliability",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sportsmot-root", type=Path, required=True)
    parser.add_argument("--gt-full", type=Path, required=True)
    parser.add_argument("--gt-141", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=6, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint-name", default="checkpoint_step00020000.pth")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--refer-threshold", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("evaluation GPUs must be distinct")
    for path in (args.manifest, args.gt_full, args.gt_141):
        if not path.exists():
            raise SystemExit(f"missing evaluation input: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for variant, gpu in zip(VARIANTS, args.gpus):
        checkpoint = args.checkpoint_root / variant / args.checkpoint_name
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint for {variant}: {checkpoint}")
        output = args.output_root / variant
        output.mkdir(parents=True, exist_ok=True)
        log = output / "launcher.log"
        command = [
            args.python, "tools/qim_fast6/run_one_full_eval.py",
            "--repo", str(args.repo), "--variant", variant,
            "--checkpoint", str(checkpoint), "--manifest", str(args.manifest),
            "--sportsmot-root", str(args.sportsmot_root), "--gt-full", str(args.gt_full),
            "--gt-141", str(args.gt_141), "--output-root", str(output),
            "--gpu", str(gpu), "--score-threshold", str(args.score_threshold),
            "--refer-threshold", str(args.refer_threshold),
        ]
        entry = {
            "variant": variant,
            "physical_gpu": gpu,
            "checkpoint": str(checkpoint),
            "output_root": str(output),
            "log": str(log),
            "command": command,
        }
        if args.dry_run:
            entries.append(entry)
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        environment["OMP_NUM_THREADS"] = "1"
        with log.open("ab") as handle:
            process = subprocess.Popen(
                command, cwd=args.repo, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        entry["pid"] = process.pid
        entry["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        entries.append(entry)
        print(f"started {variant} gpu={gpu} pid={process.pid} log={log}", flush=True)

    manifest = {
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_name": args.checkpoint_name,
        "output_root": str(args.output_root),
        "variants": entries,
        "protocol": "19 historical videos, recovered 142 and compatible 141 expressions",
        "dry_run": args.dry_run,
    }
    path = args.output_root / "launch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"launch manifest: {path}")


if __name__ == "__main__":
    main()
