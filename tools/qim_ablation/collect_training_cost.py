#!/usr/bin/env python3
"""Collect measured training-cost evidence from QIM ablation runs.

This utility is intentionally read-only.  Checkpoint timestamps provide a
wall-clock observation for each milestone while the training log's final
``Training time`` line (when present) is the authoritative end-to-end time.
It does not extrapolate missing milestones or turn an ETA into a measured
cost.
"""

import argparse
import ast
import datetime as dt
import json
import re
from pathlib import Path


DEFAULT_VARIANTS = ("standard", "lstm", "temporal_attention", "mamba")
MILESTONES = (200, 5000, 20000, 60000, 120000)
_BENCHMARK_RE = re.compile(r"TRAIN_BENCHMARK\s+(\{.*\})")
_TRAINING_TIME_RE = re.compile(r"Training time\s+(\d+):([0-5]\d):([0-5]\d)\s*$")
_SOURCE_RE = re.compile(r"^\s*sha:\s*([0-9a-f]{40}),\s*status:\s*(.+)$", re.MULTILINE)


def utc_iso(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def parse_log(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}

    text = path.read_text(errors="replace")
    benchmarks = []
    for match in _BENCHMARK_RE.finditer(text):
        try:
            value = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            benchmarks.append(value)

    training_time = None
    for line in reversed(text.splitlines()):
        match = _TRAINING_TIME_RE.search(line)
        if match:
            hours, minutes, seconds = (int(part) for part in match.groups())
            training_time = hours * 3600 + minutes * 60 + seconds
            break

    source = _SOURCE_RE.search(text)
    stat = path.stat()
    return {
        "status": "present",
        "path": str(path),
        "bytes": stat.st_size,
        "last_modified_utc": utc_iso(stat.st_mtime),
        "source_sha": source.group(1) if source else None,
        "source_worktree_status": source.group(2).strip() if source else None,
        "training_time_s": training_time,
        "benchmark_epochs": len(benchmarks),
        "latest_benchmark": benchmarks[-1] if benchmarks else None,
        "explicit_no_grad_log_events": text.count("NO_GRAD_STEP"),
    }


def checkpoint_info(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    stat = path.stat()
    return {
        "status": "present",
        "path": str(path),
        "bytes": stat.st_size,
        "modified_utc": utc_iso(stat.st_mtime),
    }


def collect_variant(run_root: Path, variant: str) -> dict:
    variant_root = run_root / variant
    log = parse_log(run_root / f"{variant}_long.log")
    checkpoints = {}
    for step in MILESTONES:
        checkpoints[str(step)] = checkpoint_info(
            variant_root / f"checkpoint_step{step:08d}.pth"
        )

    baseline = checkpoints["200"]
    baseline_time = baseline.get("modified_utc")
    baseline_epoch = None
    if baseline_time is not None:
        baseline_epoch = dt.datetime.fromisoformat(baseline_time).timestamp()

    for step in MILESTONES:
        item = checkpoints[str(step)]
        if baseline_epoch is None or item.get("status") != "present":
            item["checkpoint_wallclock_from_step200_s"] = None
            item["observed_gpu_hours_from_step200"] = None
            continue
        item_epoch = dt.datetime.fromisoformat(item["modified_utc"]).timestamp()
        elapsed = max(0.0, item_epoch - baseline_epoch)
        item["checkpoint_wallclock_from_step200_s"] = elapsed
        item["observed_gpu_hours_from_step200"] = elapsed / 3600.0

    return {
        "variant": variant,
        "log": log,
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/data1/liuyeqiang/fg_v2_runs/qim_ablation_sports_20261019"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qim_ablation/training_cost_20261019.json"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        metavar="VARIANT",
        help="variant directory names under --run-root (defaults to the original four QIM variants)",
    )
    args = parser.parse_args()

    result = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(args.run_root),
        "milestones": list(MILESTONES),
        "measurement_notes": [
            "checkpoint_wallclock_from_step200_s is measured from checkpoint file modification times",
            "observed_gpu_hours_from_step200 is one GPU-hour per active variant GPU",
            "missing milestones remain missing; no ETA or extrapolation is recorded",
            "training_time_s is taken only from the log's final Training time line",
        ],
        "variant_names": args.variants,
        "variants": {
            variant: collect_variant(args.run_root, variant) for variant in args.variants
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
