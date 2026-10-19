#!/usr/bin/env python3
"""Wait for YesTrack Y1, then run the fixed Refer-Sports tracker on a free GPU.

This watcher is intentionally independent of GPU assignment.  It never kills
or reconfigures another process: after the training process exits successfully,
it selects a physical GPU with no compute application and less than 1 GiB in
use, then invokes the existing SambaMOTR launcher.  Tracker outputs stay on
the NAS path supplied below.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_matches(fragment: str) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    )
    current = os.getpid()
    matches: list[int] = []
    for line in result.stdout.splitlines():
        raw_pid, _, command = line.strip().partition(" ")
        if not raw_pid.isdigit() or int(raw_pid) == current:
            continue
        if "post_training_tracker_watch.py" in command:
            continue
        if fragment in command:
            matches.append(int(raw_pid))
    return matches


def gpu_snapshot() -> list[tuple[int, list[int], int]]:
    indexes = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
    )
    snapshot = []
    for raw in indexes.splitlines():
        raw = raw.strip()
        if not raw.isdigit():
            continue
        gpu = int(raw)
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        pids = [int(value.strip()) for value in apps.splitlines() if value.strip().isdigit()]
        memory = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        snapshot.append((gpu, pids, memory))
    return snapshot


def choose_free_gpu(max_memory_mib: int) -> tuple[int | None, list[tuple[int, list[int], int]]]:
    snapshot = gpu_snapshot()
    for gpu, apps, memory in snapshot:
        if not apps and memory < max_memory_mib:
            return gpu, snapshot
    return None, snapshot


def tracker_manifest_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return None


def run(args: argparse.Namespace) -> int:
    log_path = Path(args.watch_log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{now()} {message}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    tracker_manifest = Path(args.output_root).expanduser().resolve() / "TRACKER_RUN_MANIFEST.json"
    log("watch_started")
    while process_matches(args.training_fragment):
        time.sleep(args.poll_seconds)

    train_log = Path(args.training_log).expanduser().resolve()
    if not train_log.is_file() or "__YESTRACK_EXIT__=0" not in train_log.read_text(
        encoding="utf-8", errors="replace"
    ):
        log("training_did_not_exit_successfully")
        return 2
    if tracker_manifest_status(tracker_manifest) == "SUCCEEDED":
        log("tracker_already_succeeded")
        return 0

    while True:
        gpu, snapshot = choose_free_gpu(args.max_memory_mib)
        if gpu is not None:
            log(f"selected_gpu={gpu} snapshot={snapshot}")
            break
        log(f"waiting_for_free_gpu snapshot={snapshot}")
        time.sleep(args.poll_seconds)

    command = [
        args.python,
        str(Path(args.project) / "scripts" / "run_sambamotr_refersports_tracker.py"),
        "--source-root",
        args.source_root,
        "--model-root",
        args.model_root,
        "--dataset-root",
        args.dataset_root,
        "--staging-root",
        args.staging_root,
        "--output-root",
        args.output_root,
        "--gpu",
        str(gpu),
        "--run",
        "--json",
    ]
    log("starting_tracker=" + " ".join(command))
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=args.project, stdout=handle, stderr=subprocess.STDOUT)
    log(f"tracker_exit={completed.returncode}")
    return completed.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--training-fragment", default="scripts/yestrack_sportsmot_adapter.py")
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--watch-log", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-memory-mib", type=int, default=1000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
