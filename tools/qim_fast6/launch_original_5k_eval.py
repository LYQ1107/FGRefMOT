#!/usr/bin/env python3
"""Detach the sequential original-QIM 5k evaluator on one explicit GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    command = [
        args.python, "tools/qim_fast6/run_original_5k_eval.py",
        "--repo", str(args.repo), "--checkpoint-root", str(args.checkpoint_root),
        "--manifest", str(args.manifest), "--sportsmot-root", str(args.sportsmot_root),
        "--gt-full", str(args.gt_full), "--gt-141", str(args.gt_141),
        "--output-root", str(args.output_root), "--gpu", str(args.gpu),
    ]
    log = args.output_root / "launcher.log"
    entry = {"gpu": args.gpu, "command": command, "log": str(log), "dry_run": args.dry_run}
    if args.dry_run:
        print(json.dumps(entry, indent=2))
    else:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        environment["OMP_NUM_THREADS"] = "1"
        with log.open("ab") as handle:
            process = subprocess.Popen(
                command, cwd=args.repo, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        entry["pid"] = process.pid
        print(f"started original 5k evaluator gpu={args.gpu} pid={process.pid} log={log}")
    (args.output_root / "launcher_manifest.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
