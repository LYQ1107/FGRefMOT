#!/usr/bin/env python3
"""Derive reproducible throughput and memory records from run manifests.

The research protocol reports three different rates: the physical tracker
rate, the referring-module rate, and the complete pipeline rate.  A tracker
process is normally run once and its outputs are then reused for every
expression, so the pipeline rate is computed against the unique video-frame
count and includes the recorded tracker wall time plus the referring-module
inference time.  The referring-module rate additionally records
expression-frame units, which is the workload seen by a referring model.

This module only reads manifests and timing files.  It never reads labels or
ground truth and never changes an existing run.
"""

from __future__ import annotations

import datetime as _datetime
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = _datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
    return parsed.timestamp()


def tracker_wall_seconds(manifest: dict[str, Any]) -> float | None:
    """Return measured tracker wall time, including legacy manifests."""

    for key in ("wall_seconds", "elapsed_seconds", "inference_seconds"):
        value = _float(manifest.get(key))
        if value is not None:
            return value
    started = _timestamp_seconds(manifest.get("started_at"))
    finished = _timestamp_seconds(manifest.get("finished_at"))
    if started is None or finished is None or finished < started:
        return None
    return finished - started


def tracker_frame_count(manifest: dict[str, Any]) -> int | None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        return None
    total = 0
    seen = False
    for item in outputs:
        if not isinstance(item, dict):
            continue
        value = item.get("max_frame", item.get("frame_count"))
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        total += count
        seen = True
    return total if seen else None


def tracker_efficiency(manifest: dict[str, Any]) -> dict[str, Any]:
    seconds = tracker_wall_seconds(manifest)
    frames = tracker_frame_count(manifest)
    fps = frames / seconds if frames is not None and seconds and seconds > 0 else None
    return {
        "schema_version": "rmot-efficiency-0.1",
        "status": manifest.get("status"),
        "physical_gpu": manifest.get("physical_gpu"),
        "frame_count": frames,
        "wall_seconds": seconds,
        "tracker_fps": fps,
        "timing_source": (
            "manifest.wall_seconds_or_elapsed_seconds"
            if any(key in manifest for key in ("wall_seconds", "elapsed_seconds", "inference_seconds"))
            else "manifest.started_at_finished_at"
        ),
    }


def _referring_timing(manifest: dict[str, Any], timing: dict[str, Any] | None) -> dict[str, Any]:
    run = manifest.get("run")
    if not isinstance(run, dict):
        run = {}
    source = timing or {}
    recorded = manifest.get("efficiency")
    if not isinstance(recorded, dict):
        recorded = {}
    inference_seconds = _float(run.get("inference_seconds"))
    if inference_seconds is None:
        inference_seconds = _float(source.get("elapsed_seconds_max"))
    if inference_seconds is None:
        inference_seconds = _float(recorded.get("inference_seconds"))
    model_loading_seconds = _float(run.get("model_loading_seconds"))
    if model_loading_seconds is None:
        model_loading_seconds = _float(source.get("model_loading_seconds_max"))
    if model_loading_seconds is None:
        model_loading_seconds = _float(recorded.get("model_loading_seconds"))
    peak_memory = _float(run.get("peak_gpu_memory_mib"))
    if peak_memory is None:
        peak_memory = _float(source.get("peak_gpu_memory_mib_max"))
    if peak_memory is None:
        peak_memory = _float(recorded.get("peak_gpu_memory_mib"))
    model_load_peak = _float(run.get("model_load_peak_gpu_memory_mib"))
    if model_load_peak is None:
        model_load_peak = _float(source.get("model_load_peak_gpu_memory_mib_max"))
    if model_load_peak is None:
        model_load_peak = _float(recorded.get("model_load_peak_gpu_memory_mib"))
    return {
        "inference_seconds": inference_seconds,
        "model_loading_seconds": model_loading_seconds,
        "peak_gpu_memory_mib": peak_memory,
        "model_load_peak_gpu_memory_mib": model_load_peak,
        "timing_source": "eval_manifest.run_or_timing_file",
    }


def referring_efficiency(
    manifest: dict[str, Any],
    *,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = manifest.get("records")
    seq_info = manifest.get("seq_info")
    video_lengths: dict[str, int] = {}
    expression_frames = 0
    if isinstance(records, list) and isinstance(seq_info, dict):
        for record in records:
            if not isinstance(record, dict):
                continue
            sequence = record.get("sequence_name")
            video = record.get("video")
            try:
                frames = int(seq_info[sequence])
            except (KeyError, TypeError, ValueError):
                continue
            if frames < 0:
                continue
            expression_frames += frames
            if isinstance(video, str):
                video_lengths[video] = max(video_lengths.get(video, 0), frames)
    recorded = manifest.get("efficiency")
    if not isinstance(recorded, dict):
        recorded = {}
    if not video_lengths:
        frame_count = recorded.get("video_frame_count", manifest.get("video_frame_count"))
        if isinstance(frame_count, (int, float)):
            video_lengths = {"_recorded": int(frame_count)}
    if not expression_frames:
        expression_frames = int(recorded.get("expression_frame_count") or 0)
    video_frames = sum(video_lengths.values()) if video_lengths else None
    rates = _referring_timing(manifest, timing)
    seconds = rates["inference_seconds"]
    rates.update(
        {
            "video_frame_count": video_frames,
            "expression_frame_count": expression_frames or None,
            "video_frame_fps": video_frames / seconds if video_frames and seconds else None,
            "referring_module_fps": expression_frames / seconds if expression_frames and seconds else None,
        }
    )
    return rates


def combine_efficiency(
    tracker: dict[str, Any],
    referring: dict[str, Any],
) -> dict[str, Any]:
    video_frames = referring.get("video_frame_count")
    tracker_seconds = _float(tracker.get("wall_seconds"))
    referring_seconds = _float(referring.get("inference_seconds"))
    pipeline_seconds = None
    if tracker_seconds is not None and referring_seconds is not None:
        pipeline_seconds = tracker_seconds + referring_seconds
    return {
        "schema_version": "rmot-efficiency-0.1",
        "tracker": tracker,
        "referring_module": referring,
        "complete_pipeline": {
            "video_frame_count": video_frames,
            "wall_seconds": pipeline_seconds,
            "complete_pipeline_fps": (
                video_frames / pipeline_seconds
                if video_frames and pipeline_seconds and pipeline_seconds > 0
                else None
            ),
            "formula": "unique_video_frames / (tracker_wall_seconds + referring_inference_seconds)",
        },
    }
