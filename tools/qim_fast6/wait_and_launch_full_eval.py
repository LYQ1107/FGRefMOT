#!/usr/bin/env python3
"""Wait for all six 20k checkpoints, then launch their full RMOT jobs once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = (
    "fast_mamba_spatial",
    "fast_mamba_temporal",
    "fast_mamba_streaming",
    "fast_mamba_motion",
    "fast_mamba_language",
    "fast_mamba_reliability",
)


def pid_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sportsmot-root", type=Path, required=True)
    parser.add_argument("--gt-full", type=Path, required=True)
    parser.add_argument("--gt-141", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=6, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--checkpoint-name", default="checkpoint_step00020000.pth")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log = args.output_root / "wait_and_launch.log"

    def report(message):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    launch_manifest = args.run_root / "launch_manifest.json"
    while True:
        if not launch_manifest.is_file():
            report(f"waiting for launch manifest: {launch_manifest}")
            time.sleep(args.poll_seconds)
            continue
        manifest = json.loads(launch_manifest.read_text(encoding="utf-8"))
        entries = {row["variant"]: row for row in manifest.get("variants", [])}
        missing = []
        live = []
        for variant in VARIANTS:
            checkpoint = args.run_root / variant / args.checkpoint_name
            if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000_000:
                missing.append(variant)
            entry = entries.get(variant, {})
            pid = entry.get("pid")
            if pid and pid_command(int(pid)):
                command = pid_command(int(pid))
                if "main.py" in command and variant in command:
                    live.append(f"{variant}:{pid}")
        if missing or live:
            report("waiting: missing=" + ",".join(missing) + " live=" + ",".join(live))
            time.sleep(args.poll_seconds)
            continue
        report("all six checkpoints present and training PIDs exited; launching full RMOT")
        command = [
            args.python, "tools/qim_fast6/launch_fast6_full_eval.py",
            "--repo", str(args.repo), "--checkpoint-root", str(args.run_root),
            "--manifest", str(args.manifest), "--sportsmot-root", str(args.sportsmot_root),
            "--gt-full", str(args.gt_full), "--gt-141", str(args.gt_141),
            "--output-root", str(args.output_root), "--gpus",
            *[str(gpu) for gpu in args.gpus], "--checkpoint-name", args.checkpoint_name,
        ]
        environment = os.environ.copy()
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        environment["OMP_NUM_THREADS"] = "1"
        result = subprocess.run(command, cwd=args.repo, env=environment)
        if result.returncode:
            report(f"full RMOT launcher failed with return code {result.returncode}")
            raise SystemExit(result.returncode)
        report("full RMOT launcher started; watcher exiting")
        return


if __name__ == "__main__":
    main()
