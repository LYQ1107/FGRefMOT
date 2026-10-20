#!/usr/bin/env python3
"""Collect completed RMOT, efficiency, and training artifacts into reports.

The collector is intentionally evidence-only: absent inference, scoring,
benchmark, or checkpoint files remain ``PENDING`` instead of being inferred
from a log or a training loss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ORIGINAL = ("standard", "lstm", "temporal_attention", "mamba")
FAST6 = (
    "fast_mamba_spatial", "fast_mamba_temporal", "fast_mamba_streaming",
    "fast_mamba_motion", "fast_mamba_language", "fast_mamba_reliability",
)


def read_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def value(mapping, key):
    if mapping is None or key not in mapping:
        return "PENDING"
    return mapping[key]


def sha256(path):
    if not path.is_file():
        return "PENDING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics_row(path, suffix=""):
    payload = read_json(path)
    summary = payload.get("summary") if payload else None
    return {
        f"HOTA{suffix}": value(summary, "HOTA"),
        f"DetA{suffix}": value(summary, "DetA"),
        f"AssA{suffix}": value(summary, "AssA"),
        f"IDF1{suffix}": value(summary, "IDF1"),
        f"IDSW{suffix}": value(summary, "IDSW"),
        f"MOTA{suffix}": "PENDING",
        f"DetRe{suffix}": value(summary, "DetRe"),
        f"DetPr{suffix}": value(summary, "DetPr"),
        f"LocA{suffix}": value(summary, "LocA"),
        f"expressions{suffix}": value(summary, "num_sequences"),
        f"videos{suffix}": value(summary, "num_videos"),
    }


def timing_data(path):
    payload = read_json(path)
    if not payload:
        return {"full_fps": "PENDING", "expression_frame_fps": "PENDING"}
    timings = payload.get("timings", [])
    total_frames = sum(float(row.get("frames", 0)) for row in timings)
    total_seconds = sum(float(row.get("wall_seconds", 0)) for row in timings)
    return {
        "full_fps": value(payload, "mean_end_to_end_fps_per_expression"),
        "expression_frame_fps": total_frames / total_seconds if total_seconds else "PENDING",
    }


def checkpoint_cost(run_root, variant, step, launch_manifest):
    checkpoint = run_root / variant / f"checkpoint_step{step:08d}.pth"
    if not checkpoint.is_file():
        return {"completed_steps": "PENDING", "wall_hours": "PENDING", "sha256": "PENDING"}
    started = None
    for entry in (launch_manifest or {}).get("variants", []):
        if entry.get("variant") == variant:
            started = entry.get("started_at_utc")
            break
    wall_hours = "PENDING"
    if started:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end_dt = datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=timezone.utc)
        wall_hours = max(0.0, (end_dt - start_dt).total_seconds() / 3600.0)
    return {
        "completed_steps": step,
        "wall_hours": wall_hours,
        "sha256": sha256(checkpoint),
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(args):
    profile = read_json(args.profile)
    benchmark = read_json(args.fps_json)
    launch = read_json(args.fast_run_root / "launch_manifest.json")
    groups = []
    for variant in ORIGINAL:
        root = args.original_run_root / variant
        groups.append((variant, root, 5000, "ORIGINAL_5K"))
    for variant in FAST6:
        root = args.fast_run_root / variant
        groups.append((variant, root, 20000, "FAST6_20K"))

    rows = []
    training_rows = []
    parameter_rows = []
    fps_rows = []
    error_inputs = []
    for variant, root, step, protocol in groups:
        metrics_142 = metrics_row(root / "metrics_142.json", "_142")
        metrics_141 = metrics_row(root / "metrics_141.json", "_141")
        inference = read_json(root / "inference_142.json")
        timing = timing_data(root / "inference_142.json")
        benchmark_entry = (benchmark or {}).get("variants", {}).get(variant, {})
        profile_entry = (profile or {}).get("variants", {}).get(variant, {})
        profile_20 = (profile_entry.get("active_tracks", {}) if profile_entry else {}).get("20", {})
        complete = metrics_142.get("HOTA_142") != "PENDING" and metrics_141.get("HOTA_141") != "PENDING"
        rows.append({
            "variant": variant, "training_steps": step, "protocol": protocol,
            **metrics_142, **metrics_141,
            "full_fps": value(benchmark_entry, "full_model_fps"),
            "end_to_end_wall_fps": value(benchmark_entry, "end_to_end_wall_fps"),
            "expression_frame_fps": timing["expression_frame_fps"],
            "status": "COMPLETE" if complete else "PENDING",
        })
        cost = checkpoint_cost(args.fast_run_root, variant, step, launch) if variant in FAST6 else {
            "completed_steps": step if (args.original_checkpoint_root / variant / "checkpoint_step00005000.pth").is_file() else "PENDING",
            "wall_hours": "PENDING", "sha256": sha256(args.original_checkpoint_root / variant / "checkpoint_step00005000.pth"),
        }
        training_rows.append({
            "variant": variant, "physical_gpu": value(next((x for x in (launch or {}).get("variants", []) if x.get("variant") == variant), {}), "physical_gpu"),
            "planned_updates": step, "completed_updates": cost["completed_steps"],
            "wall_hours": cost["wall_hours"], "checkpoint_sha256": cost["sha256"],
            "status": "COMPLETE" if cost["completed_steps"] == step else "PENDING",
        })
        parameter_rows.append({
            "variant": variant,
            "trainable_qim_parameters": value(profile_entry, "qim_params"),
            "total_model_parameters": value(benchmark_entry, "full_model_params"),
            "protocol": protocol,
            "status": "COMPLETE" if profile_entry else "PENDING",
        })
        fps_rows.append({
            "variant": variant,
            "active_tracks": 20,
            "qim_mean_ms": value(profile_20, "mean_ms"),
            "qim_median_ms": value(profile_20, "median_ms"),
            "qim_p95_ms": value(profile_20, "p95_ms"),
            "qim_peak_memory_mb": (float(profile_20["peak_allocated_gb"]) * 1024
                                    if profile_20.get("peak_allocated_gb") != "PENDING" else "PENDING"),
            "full_model_fps": value(benchmark_entry, "full_model_fps"),
            "end_to_end_wall_fps": value(benchmark_entry, "end_to_end_wall_fps"),
            "expression_frame_fps": timing["expression_frame_fps"],
            "status": "COMPLETE" if benchmark_entry else "PENDING",
        })
        if metrics_142["HOTA_142"] != "PENDING":
            error_inputs.append((variant, read_json(root / "metrics_142.json")))
    return rows, training_rows, parameter_rows, fps_rows, error_inputs


def write_error_analysis(path, error_inputs, manifest):
    records = {row["expression_id"]: row for row in manifest.get("records", [])}
    lines = ["# Error analysis", "", "Generated only from completed expression-level RMOT metrics.", ""]
    if not error_inputs:
        lines.append("Status: `NOT_YET_COMPLETE`; no scored 142-expression result is available.")
    for variant, payload in error_inputs:
        sequences = payload.get("sequences", [])
        lines.extend([f"## {variant}", ""])
        if not sequences:
            lines.append("No sequence rows.")
            continue
        worst_hota = sorted(sequences, key=lambda row: row.get("HOTA", 0.0))[:5]
        worst_idsw = sorted(sequences, key=lambda row: row.get("IDSW", 0), reverse=True)[:5]
        lines.append("Worst HOTA expressions:")
        for row in worst_hota:
            lines.append(f"- `{row['expression_id']}` ({row['video']}): HOTA={row['HOTA']:.6f}, IDSW={row['IDSW']}")
        lines.append("")
        lines.append("Highest-IDSW expressions:")
        for row in worst_idsw:
            lines.append(f"- `{row['expression_id']}` ({row['video']}): IDSW={row['IDSW']}, AssA={row['AssA']:.6f}")
        by_sport = {}
        for row in sequences:
            sport = records.get(row["expression_id"], {}).get("sport_type", "UNKNOWN")
            by_sport.setdefault(sport, []).append(row)
        lines.append("")
        lines.append("Sport breakdown:")
        for sport, values in sorted(by_sport.items()):
            lines.append(
                f"- {sport}: n={len(values)}, HOTA={sum(x['HOTA'] for x in values)/len(values):.6f}, "
                f"AssA={sum(x['AssA'] for x in values)/len(values):.6f}, IDSW={sum(x['IDSW'] for x in values)}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-checkpoint-root", type=Path, required=True)
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--fast-run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fps-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows, training_rows, parameter_rows, fps_rows, error_inputs = collect_rows(args)
    report_root = args.docs_root
    full_fields = list(rows[0].keys())
    write_csv(report_root / "ORIGINAL_QIM_5K_FULL_TEST.csv", full_fields, rows[:4])
    write_csv(report_root / "FAST6_20K_FULL_TEST.csv", full_fields, rows[4:])
    write_csv(report_root / "TRAINING_COST.csv", list(training_rows[0].keys()), training_rows[4:])
    write_csv(report_root / "QIM_PARAMETER_REPORT.csv", list(parameter_rows[0].keys()), parameter_rows)
    write_csv(report_root / "FPS_REPORT.csv", list(fps_rows[0].keys()), fps_rows)
    write_error_analysis(report_root / "ERROR_ANALYSIS.md", error_inputs, manifest)

    completed = [row for row in rows if row["status"] == "COMPLETE"]
    lines = ["# FGRefMOT fast6 final report", "", f"Completed scored models: {len(completed)}/{len(rows)}.", ""]
    if not completed:
        lines.append("Status: `IN_PROGRESS`; no complete paired 142/141 result is available yet.")
    else:
        lines.extend(["The following rows are evidence-backed 142-expression summaries:", "", "| Variant | HOTA | DetA | AssA | IDF1 | IDSW | Full FPS |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in completed:
            lines.append(f"| {row['variant']} | {row['HOTA_142']} | {row['DetA_142']} | {row['AssA_142']} | {row['IDF1_142']} | {row['IDSW_142']} | {row['full_fps']} |")
    lines.extend(["", "No metric is inferred from training loss or parameter count."])
    (report_root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
