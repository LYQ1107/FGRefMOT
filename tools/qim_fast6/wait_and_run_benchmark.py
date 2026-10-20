#!/usr/bin/env python3
"""Run the isolated ten-model full-model FPS benchmark after fast6 training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = (
    "fast_mamba_spatial", "fast_mamba_temporal", "fast_mamba_streaming",
    "fast_mamba_motion", "fast_mamba_language", "fast_mamba_reliability",
)


def command_for(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sportsmot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=9)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    log = args.output.parent / "wait_and_run_benchmark.log"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def report(message):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    launch = args.run_root / "launch_manifest.json"
    while True:
        if not launch.is_file():
            report("waiting for six-GPU launch manifest")
            time.sleep(args.poll_seconds)
            continue
        payload = json.loads(launch.read_text(encoding="utf-8"))
        missing = [
            variant for variant in VARIANTS
            if not (args.run_root / variant / "checkpoint_step00020000.pth").is_file()
        ]
        live = []
        for entry in payload.get("variants", []):
            variant = entry.get("variant")
            pid = entry.get("pid")
            if variant in VARIANTS and pid:
                cmd = command_for(int(pid))
                if "main.py" in cmd and variant in cmd:
                    live.append(f"{variant}:{pid}")
        if missing or live:
            report("waiting: missing=" + ",".join(missing) + " live=" + ",".join(live))
            time.sleep(args.poll_seconds)
            continue
        command = [
            args.python if hasattr(args, "python") else sys.executable,
            "tools/qim_fast6/benchmark_full_fps.py",
            "--original-checkpoint-root", str(args.original_root),
            "--fast-checkpoint-root", str(args.run_root),
            "--manifest", str(args.manifest),
            "--sportsmot-root", str(args.sportsmot_root),
            "--frames", "60", "--warmup-frames", "5", "--device", "cuda",
            "--output", str(args.output),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        environment["OMP_NUM_THREADS"] = "1"
        report("training finished; starting isolated full-model FPS benchmark")
        result = subprocess.run(command, cwd=args.repo, env=environment)
        if result.returncode:
            report(f"benchmark failed with return code {result.returncode}")
            raise SystemExit(result.returncode)
        report("benchmark completed")
        return


if __name__ == "__main__":
    main()
