#!/usr/bin/env python3
"""Run the two formal Refer-Sports evaluations after tracker generation.

The watcher waits for a successful fixed-tracker manifest, discovers the Y1
LoRA checkpoint on NAS, selects a genuinely free GPU, and runs the YesTrack and
FlexHook evaluation adapters sequentially.  It never feeds GT to inference and
never modifies another GPU process.
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


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def free_gpu(max_memory_mib: int) -> tuple[int | None, list[dict]]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    snapshot = []
    for line in raw.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        gpu, memory = int(fields[0]), int(fields[1])
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
        pids = [value.strip() for value in apps.splitlines() if value.strip()]
        item = {"gpu": gpu, "memory_mib": memory, "compute_pids": pids}
        snapshot.append(item)
        if not pids and memory < max_memory_mib:
            return gpu, snapshot
    return None, snapshot


def run_step(name: str, command: list[str], project: Path, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp()} start_{name}={' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"{timestamp()} exit_{name}={completed.returncode}\n")
    return completed.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--tracker-manifest", required=True)
    parser.add_argument("--lora-root", required=True)
    parser.add_argument(
        "--training-log",
        help="Optional YesTrack wrapper log used to distinguish an active run from a failed run.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--gt-manifest-root", required=True)
    parser.add_argument("--yestrack-output-root", required=True)
    parser.add_argument(
        "--sequence-list",
        help="Optional exact sequence list forwarded to the YesTrack evaluator.",
    )
    parser.add_argument(
        "--skip-flexhook",
        action="store_true",
        help="Run only the YesTrack evaluation; useful for an internal-selection split.",
    )
    parser.add_argument("--flexhook-config", required=True)
    parser.add_argument("--flexhook-layout-root", required=True)
    parser.add_argument("--flexhook-checkpoint", required=True)
    parser.add_argument("--flexhook-output-root", required=True)
    parser.add_argument("--watch-log", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-memory-mib", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project = Path(args.project).expanduser().resolve()
    tracker_manifest = Path(args.tracker_manifest).expanduser().resolve()
    watch_log = Path(args.watch_log).expanduser().resolve()
    watch_log.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with watch_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp()} {message}\n")

    log("eval_watch_started")
    while True:
        if tracker_manifest.is_file():
            status = load_json(tracker_manifest).get("status")
            if status == "SUCCEEDED":
                break
            if status == "FAILED":
                log("tracker_failed")
                return 2
        time.sleep(args.poll_seconds)

    lora_root = Path(args.lora_root).expanduser().resolve()
    training_log = (
        Path(args.training_log).expanduser().resolve() if args.training_log else None
    )
    while True:
        lora_candidates = sorted(
            (path for path in lora_root.glob("**/lora_epoch_0") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        if lora_candidates:
            break
        if training_log and training_log.is_file():
            text = training_log.read_text(encoding="utf-8", errors="replace")
            exit_markers = [line for line in text.splitlines() if "__YESTRACK_EXIT__=" in line]
            if exit_markers and not exit_markers[-1].endswith("__YESTRACK_EXIT__=0"):
                log(f"training_failed_before_lora={exit_markers[-1]}")
                return 3
        log("waiting_for_lora_epoch_0")
        time.sleep(args.poll_seconds)
    lora_path = lora_candidates[-1]
    log(f"lora_path={lora_path}")

    while True:
        gpu, snapshot = free_gpu(args.max_memory_mib)
        if gpu is not None:
            log(f"selected_gpu={gpu} snapshot={snapshot}")
            break
        log(f"waiting_for_free_gpu snapshot={snapshot}")
        time.sleep(args.poll_seconds)

    yestrack_manifest = Path(args.yestrack_output_root).expanduser().resolve() / "YESTRACK_EVAL_MANIFEST.json"
    if not (yestrack_manifest.is_file() and load_json(yestrack_manifest).get("status") == "SUCCEEDED"):
        yestrack_results = Path(args.yestrack_output_root).expanduser().resolve() / "results"
        command = [
            args.python,
            str(project / "scripts" / "eval_yestrack_refersports.py"),
            "--model-name",
            args.model_name,
            "--lora-path",
            str(lora_path),
            "--data-root",
            args.data_root,
            "--tracker-root",
            args.tracker_root,
            "--results-root",
            str(yestrack_results),
            "--gt-manifest-root",
            args.gt_manifest_root,
            "--output-root",
            args.yestrack_output_root,
            "--gpu",
            str(gpu),
            "--infer-every-n-frames",
            "1",
            "--run-inference",
            "--json",
        ]
        if args.sequence_list:
            command.extend(["--sequence-list", args.sequence_list])
        if run_step("yestrack_eval", command, project, watch_log) != 0:
            return 4
    else:
        log("yestrack_eval_already_succeeded")

    if args.skip_flexhook:
        log("flexhook_eval_skipped")
        log("eval_watch_succeeded")
        return 0

    flexhook_manifest = Path(args.flexhook_output_root).expanduser().resolve() / "FLEXHOOK_EVAL_MANIFEST.json"
    if not (flexhook_manifest.is_file() and load_json(flexhook_manifest).get("status") == "SUCCEEDED"):
        while True:
            next_gpu, snapshot = free_gpu(args.max_memory_mib)
            if next_gpu is not None:
                gpu = next_gpu
                log(f"selected_gpu_for_flexhook={gpu} snapshot={snapshot}")
                break
            log(f"waiting_for_free_gpu_before_flexhook snapshot={snapshot}")
            time.sleep(args.poll_seconds)
        command = [
            args.python,
            str(project / "scripts" / "eval_flexhook_refersports.py"),
            "--config",
            args.flexhook_config,
            "--layout-root",
            args.flexhook_layout_root,
            "--tracker-root",
            args.tracker_root,
            "--checkpoint",
            args.flexhook_checkpoint,
            "--output-root",
            args.flexhook_output_root,
            "--gpu",
            str(gpu),
            "--batch-size",
            "32",
            "--workers",
            "4",
            "--json",
        ]
        if run_step("flexhook_eval", command, project, watch_log) != 0:
            return 5
    else:
        log("flexhook_eval_already_succeeded")

    log("eval_watch_succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
