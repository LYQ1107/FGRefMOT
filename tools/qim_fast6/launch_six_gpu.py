#!/usr/bin/env python3
"""Launch the six fast-QIM training jobs with one explicit GPU each.

The launcher never kills or restarts a process.  It records the exact command,
PID, GPU assignment, and output path before returning so a later audit can
distinguish a completed run from a manually interrupted one.
"""

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


def build_command(repo, variant, output, common_checkpoint, args):
    return [
        args.python,
        "main.py",
        "--meta_arch", "temp_rmot",
        "--dataset_file", "e2e_rmot_sports",
        "--sports_split", "train",
        "--sports_ref_protocol", "unknown_mask",
        "--with_box_refine",
        "--dropout", "0.0",
        "--random_drop", "0.1",
        "--fp_ratio", "0.3",
        "--merger_dropout", "0.0",
        "--update_query_pos",
        "--qim_variant", variant,
        "--qim_fast_dim", str(args.qim_fast_dim),
        "--qim_fast_layers", str(args.qim_fast_layers),
        "--qim_fast_state", str(args.qim_fast_state),
        "--qim_fast_history_len", str(args.qim_fast_history_len),
        "--init_common_checkpoint", str(common_checkpoint),
        "--reset_qim",
        "--train_qim_only",
        "--batch_size", "1",
        "--num_workers", "0",
        "--sampler_steps",
        "--sampler_lengths", "4",
        "--max_train_steps", str(args.max_train_steps),
        "--epochs", "9",
        "--lr_drop", "999999",
        "--benchmark_train",
        "--save_dir", str(output),
        "--save_steps", "200", "5000", "10000", "20000",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--common-checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=6, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-train-steps", type=int, default=20000)
    parser.add_argument("--qim-fast-dim", type=int, default=64)
    parser.add_argument("--qim-fast-layers", type=int, default=1)
    parser.add_argument("--qim-fast-state", type=int, default=8)
    parser.add_argument("--qim-fast-history-len", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != 6:
        raise SystemExit("six training GPUs must be distinct")
    if not args.common_checkpoint.is_file():
        raise SystemExit(f"common checkpoint does not exist: {args.common_checkpoint}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for variant, gpu in zip(VARIANTS, args.gpus):
        output = args.output_root / variant
        output.mkdir(parents=True, exist_ok=True)
        log_path = args.output_root / f"{variant}.log"
        command = build_command(args.repo, variant, output, args.common_checkpoint, args)
        entry = {
            "variant": variant,
            "physical_gpu": gpu,
            "cuda_visible_devices": str(gpu),
            "output_dir": str(output),
            "log": str(log_path),
            "command": command,
        }
        if args.dry_run:
            entries.append(entry)
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        # This host loads two OpenMP runtimes through the vision/text stack;
        # keep the one-process training contract explicit and bounded.
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        environment["OMP_NUM_THREADS"] = "1"
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        entry["pid"] = process.pid
        entry["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        entries.append(entry)
        print(f"started {variant} gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    manifest = {
        "repo": str(args.repo),
        "output_root": str(args.output_root),
        "common_checkpoint": str(args.common_checkpoint),
        "max_train_steps": args.max_train_steps,
        "variants": entries,
        "dry_run": args.dry_run,
    }
    manifest_path = args.output_root / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"launch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
