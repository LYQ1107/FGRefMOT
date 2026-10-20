#!/usr/bin/env python3
"""Collect final tables once all ten RMOT protocol outputs exist."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = (
    "standard", "lstm", "temporal_attention", "mamba",
    "fast_mamba_spatial", "fast_mamba_temporal", "fast_mamba_streaming",
    "fast_mamba_motion", "fast_mamba_language", "fast_mamba_reliability",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--fast-root", type=Path, required=True)
    parser.add_argument("--original-checkpoint-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fps-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    log = args.docs_root / "wait_and_collect_reports.log"
    args.docs_root.mkdir(parents=True, exist_ok=True)

    def report(message):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    while True:
        missing = []
        for variant in VARIANTS:
            root = args.original_root if variant in VARIANTS[:4] else args.fast_root
            for name in ("inference_142.json", "metrics_142.json", "metrics_141.json"):
                if not (root / variant / name).is_file():
                    missing.append(f"{variant}/{name}")
        if not args.fps_json.is_file():
            missing.append(str(args.fps_json))
        if missing:
            report("waiting for " + ",".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
            time.sleep(args.poll_seconds)
            continue
        command = [
            sys.executable, "tools/qim_fast6/collect_final_reports.py",
            "--original-checkpoint-root", str(args.original_checkpoint_root),
            "--original-run-root", str(args.original_root),
            "--fast-run-root", str(args.fast_root),
            "--profile", str(args.profile), "--fps-json", str(args.fps_json),
            "--manifest", str(args.manifest), "--docs-root", str(args.docs_root),
        ]
        report("all ten 142/141 metric sets and FPS evidence exist; collecting reports")
        result = subprocess.run(command, cwd=args.repo, env=os.environ.copy())
        if result.returncode:
            report(f"report collector failed with return code {result.returncode}")
            raise SystemExit(result.returncode)
        report("final reports collected")
        return


if __name__ == "__main__":
    main()
