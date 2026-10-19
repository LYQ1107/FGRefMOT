#!/usr/bin/env python3
"""Build a protocol-controlled Refer-Sports baseline table.

The table is deliberately conservative: validation metrics are emitted only
from a completed internal-validation manifest, and formal metrics only from a
completed historical-test manifest. Missing or in-progress experiments remain
``NOT_RUN``/``RUNNING`` instead of being inferred from training losses or
partial files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


VALIDATION_METRIC_FIELDS = (
    "validation_hota",
    "validation_deta",
    "validation_assa",
    "validation_idf1",
    "validation_idsw",
    "validation_ref_f1",
)
FORMAL_METRIC_FIELDS = (
    "formal_hota",
    "formal_deta",
    "formal_assa",
    "formal_idf1",
    "formal_idsw",
    "formal_ref_f1",
)
METRIC_FIELDS = (*VALIDATION_METRIC_FIELDS, *FORMAL_METRIC_FIELDS, "fps")
DEFAULT_YESTRACK_THRESHOLD = 0.5
TABLE_FIELDS = (
    "method",
    "variant",
    "pretraining",
    "sports_training",
    "physical_tracker",
    "referring_model",
    "validation_split",
    "evaluation_split",
    *METRIC_FIELDS,
    "trainable_parameters",
    "checkpoint_path",
    "validation_manifest",
    "source_manifest",
    "status",
)


def _validation_fallback_manifest(nas_root: Path, method: str) -> Path:
    if method == "flexhook":
        return nas_root / "runs/gpu/flexhook_f1_internal_eval_20261019_v1/FLEXHOOK_EVAL_MANIFEST.json"
    if method == "yestrack":
        return nas_root / "runs/gpu/yestrack_y1_internal_eval_20261019_v1/YESTRACK_EVAL_MANIFEST.json"
    raise ValueError(f"unknown validation method: {method}")


def resolve_flexhook_validation_manifest(nas_root: Path) -> Path:
    pattern = "runs/gpu/flexhook_f1_internal_eval_20261019_v*/FLEXHOOK_EVAL_MANIFEST.json"
    candidates = sorted(nas_root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return _validation_fallback_manifest(nas_root, "flexhook")
    successful = [path for path in candidates if manifest_status(path) == "SUCCEEDED"]
    return (successful or candidates)[-1]


def resolve_yestrack_validation_manifest(nas_root: Path) -> Path:
    pattern = "runs/gpu/yestrack_y1_internal_eval_20261019_v*/YESTRACK_EVAL_MANIFEST.json"
    candidates = sorted(nas_root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return _validation_fallback_manifest(nas_root, "yestrack")
    successful = [path for path in candidates if manifest_status(path) == "SUCCEEDED"]
    return (successful or candidates)[-1]


def manifest_status(path: Path) -> str:
    if not path.is_file():
        return "NOT_RUN"
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return "MANIFEST_INVALID"
    return str(value or "STATUS_UNKNOWN")


def manifest_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def resolve_flexhook_eval_manifest(nas_root: Path) -> Path:
    """Select the newest FlexHook formal-evaluation manifest.

    Compatibility retries use versioned output roots (``v2``, ``v3``, ...)
    so a failed run is never deleted on NAS.  Prefer the newest successful
    manifest; while no run has succeeded yet, expose the newest run's status
    instead of silently falling back to an older directory.
    """

    pattern = "runs/gpu/flexhook_f1_common_eval_20261019_v*/FLEXHOOK_EVAL_MANIFEST.json"
    candidates = sorted(nas_root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return nas_root / "runs/gpu/flexhook_f1_common_eval_20261019_v2/FLEXHOOK_EVAL_MANIFEST.json"
    successful = [path for path in candidates if manifest_status(path) == "SUCCEEDED"]
    return (successful or candidates)[-1]


def resolve_yestrack_eval_manifest(nas_root: Path) -> Path:
    """Select the newest versioned YesTrack formal-evaluation manifest.

    A completed training run may need more than one evaluator retry (for
    example after a transient GPU or output-tree failure).  Keep every retry
    on NAS and choose the newest successful manifest, or the newest pending
    manifest while no retry has succeeded yet.
    """

    pattern = "runs/gpu/yestrack_y1_common_eval_20261019_v*/YESTRACK_EVAL_MANIFEST.json"
    candidates = sorted(nas_root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return nas_root / "runs/gpu/yestrack_y1_common_eval_20261019_v2/YESTRACK_EVAL_MANIFEST.json"
    successful = [path for path in candidates if manifest_status(path) == "SUCCEEDED"]
    return (successful or candidates)[-1]


def _mean_percent(value: Any) -> str:
    """Format TrackEval's unit-scale scalar/array as a percentage."""

    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
        value = sum(values) / len(values) if values else None
    if value is None:
        return "NOT_RUN"
    return f"{100.0 * float(value):.6f}"


def _find_combined_results(value: Any) -> dict[str, Any] | None:
    """Find the TrackEval combined sequence/class result in nested JSON."""

    if isinstance(value, dict):
        if isinstance(value.get("COMBINED_SEQ"), dict):
            combined = value["COMBINED_SEQ"]
            if isinstance(combined.get("all"), dict):
                return combined["all"]
            for candidate in combined.values():
                if isinstance(candidate, dict):
                    return candidate
        for candidate in value.values():
            found = _find_combined_results(candidate)
            if found is not None:
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = _find_combined_results(candidate)
            if found is not None:
                return found
    return None


def _manifest_threshold(payload: dict[str, Any], fallback: float = DEFAULT_YESTRACK_THRESHOLD) -> float:
    """Return a manifest's preselected threshold, with a compatibility fallback."""

    value = payload.get("selected_threshold")
    try:
        return float(value) if value is not None else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def _selected_trackeval_results(
    payload: dict[str, Any], selected_threshold: float | None = None
) -> Any:
    """Return the registered TrackEval result from a manifest payload.

    A direct ``trackeval_results`` payload is already selected. For manifests
    containing a threshold grid, use the threshold selected on internal
    validation. Older manifests lack that metadata and retain the registered
    0.50 compatibility fallback.
    """

    results: Any = payload.get("trackeval_results")
    if results is None:
        evaluations = payload.get("evaluations")
        if isinstance(evaluations, list):
            threshold = (
                _manifest_threshold(payload)
                if selected_threshold is None
                else float(selected_threshold)
            )
            selected = next(
                (
                    item.get("results")
                    for item in evaluations
                    if isinstance(item, dict)
                    and abs(float(item.get("threshold", -1.0)) - threshold) < 1e-9
                ),
                None,
            )
            results = selected
    return results


def _metrics(
    payload: dict[str, Any], prefix: str, selected_threshold: float | None = None
) -> dict[str, str]:
    combined = _find_combined_results(
        _selected_trackeval_results(payload, selected_threshold=selected_threshold)
    )
    fields = VALIDATION_METRIC_FIELDS if prefix == "validation" else FORMAL_METRIC_FIELDS
    if not isinstance(combined, dict):
        return {field: "NOT_RUN" for field in fields}
    hota = combined.get("HOTA", {})
    clear = combined.get("CLEAR", {})
    identity = combined.get("Identity", {})
    return {
        f"{prefix}_hota": _mean_percent(hota.get("HOTA")) if isinstance(hota, dict) else "NOT_RUN",
        f"{prefix}_deta": _mean_percent(hota.get("DetA")) if isinstance(hota, dict) else "NOT_RUN",
        f"{prefix}_assa": _mean_percent(hota.get("AssA")) if isinstance(hota, dict) else "NOT_RUN",
        f"{prefix}_idf1": _mean_percent(identity.get("IDF1")) if isinstance(identity, dict) else "NOT_RUN",
        f"{prefix}_idsw": str(int(clear["IDSW"])) if isinstance(clear, dict) and "IDSW" in clear else "NOT_RUN",
        f"{prefix}_ref_f1": str(payload.get("ref_f1", "NOT_RUN")),
    }


def formal_metrics(
    payload: dict[str, Any], selected_threshold: float | None = None
) -> dict[str, str]:
    """Extract registered metrics from a completed formal test manifest."""

    values = _metrics(payload, "formal", selected_threshold=selected_threshold)
    values["fps"] = str(payload.get("fps", "NOT_RUN"))
    return values


def validation_metrics(payload: dict[str, Any]) -> dict[str, str]:
    """Extract registered metrics from a completed internal validation manifest."""

    return _metrics(payload, "validation")


def _parameter_count_from_log(path: Path) -> str:
    """Read an explicitly logged model parameter count without loading weights."""

    if not path.is_file():
        return "NOT_RUN"
    pattern = re.compile(r"number of params:\s*(\d+)", re.IGNORECASE)
    try:
        for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            match = pattern.search(line)
            if match:
                return match.group(1)
    except OSError:
        pass
    return "NOT_RUN"


def _safetensors_parameter_count(path: Path) -> str:
    """Count LoRA tensors from a safetensors header without loading them to GPU."""

    if not path.is_file():
        return "NOT_RUN"
    try:
        from safetensors import safe_open

        total = 0
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                count = 1
                for dimension in handle.get_slice(key).get_shape():
                    count *= int(dimension)
                total += count
        return str(total)
    except (ImportError, OSError, RuntimeError, ValueError):
        return "NOT_RUN"


def row(
    *,
    method: str,
    variant: str,
    pretraining: str,
    sports_training: str,
    physical_tracker: str,
    referring_model: str,
    status: str,
    validation_split: str = "NOT_RUN",
    evaluation_split: str = "NOT_RUN",
    trainable_parameters: str = "NOT_RUN",
    checkpoint_path: str = "NOT_RUN",
    validation_manifest: str = "NOT_RUN",
    source_manifest: str = "NOT_RUN",
    metrics: dict[str, str] | None = None,
) -> dict[str, str]:
    values = {
        "method": method,
        "variant": variant,
        "pretraining": pretraining,
        "sports_training": sports_training,
        "physical_tracker": physical_tracker,
        "referring_model": referring_model,
        "validation_split": validation_split,
        "evaluation_split": evaluation_split,
        "trainable_parameters": trainable_parameters,
        "checkpoint_path": checkpoint_path,
        "validation_manifest": validation_manifest,
        "status": status,
        "source_manifest": source_manifest,
    }
    values.update({field: "NOT_RUN" for field in METRIC_FIELDS})
    if metrics:
        values.update({field: str(metrics.get(field, "NOT_RUN")) for field in METRIC_FIELDS})
    return values


def build_rows(project: Path, nas_root: Path) -> list[dict[str, str]]:
    tracker_manifest = nas_root / "assets/pretrained/common_tracker/sambamotr_dab_refersports_train_20261019/TRACKER_RUN_MANIFEST.json"
    # The first Y1 attempt was invalidated by the UNKNOWN-frame audit.  Only
    # the corrected v2 run is eligible for the baseline table.
    y1_manifest = project / "runs/gpu/yestrack_y1_sports_lora_seed42_v2/RUN_MANIFEST.json"
    y1_data = manifest_payload(y1_manifest)
    y1_checkpoint_root = (
        Path(y1_data["outputs"]["run_root_mount"])
        if y1_data.get("outputs", {}).get("run_root_mount")
        else None
    )
    y1_eval_manifest = resolve_yestrack_eval_manifest(nas_root)
    y1_validation_manifest = resolve_yestrack_validation_manifest(nas_root)
    f1_checkpoint = nas_root / "runs/gpu/flexhook_f1_sports_trained_seed42/ckpt_epoch_best_0.pth"
    f1_parameter_count = _parameter_count_from_log(
        nas_root / "runs/gpu/flexhook_f1_sports_trained_seed42/log_rank0.txt"
    )
    f1_eval_manifest = resolve_flexhook_eval_manifest(nas_root)
    f1_validation_manifest = resolve_flexhook_validation_manifest(nas_root)

    y1_eval_status = manifest_status(y1_eval_manifest)
    y1_eval_data = manifest_payload(y1_eval_manifest)
    y1_validation_status = manifest_status(y1_validation_manifest)
    y1_validation_data = manifest_payload(y1_validation_manifest)
    y1_selected_threshold = (
        _manifest_threshold(y1_validation_data)
        if y1_validation_status == "SUCCEEDED"
        else DEFAULT_YESTRACK_THRESHOLD
    )
    y1_status = manifest_status(y1_manifest)
    y1_parameter_count = _safetensors_parameter_count(
        y1_checkpoint_root / "adapter_model.safetensors"
        if y1_checkpoint_root is not None
        else Path()
    )
    if y1_status.startswith("RUNNING"):
        y1_status = "TRAINING_RUNNING"
    if y1_eval_status == "SUCCEEDED":
        y1_status = "FORMAL_EVAL_SUCCEEDED"
    elif y1_status == "SUCCEEDED" and y1_data.get("protocol_valid", True):
        y1_status = "TRAINING_SUCCEEDED_FORMAL_EVAL_PENDING"
    elif y1_status == "SUCCEEDED":
        y1_status = "PROTOCOL_INVALID"
    f1_eval_status = manifest_status(f1_eval_manifest)
    f1_eval_data = manifest_payload(f1_eval_manifest)
    f1_validation_status = manifest_status(f1_validation_manifest)
    f1_validation_data = manifest_payload(f1_validation_manifest)
    f1_status = "FORMAL_EVAL_SUCCEEDED" if f1_eval_status == "SUCCEEDED" else (
        "CHECKPOINT_READY_FORMAL_EVAL_PENDING" if f1_checkpoint.is_file() else "NOT_RUN"
    )
    tracker_status = manifest_status(tracker_manifest)
    if tracker_status == "PREPARED_NO_GPU_RUN":
        tracker_status = "TRACKER_GPU_RUN_PENDING"
    f1_metrics: dict[str, str] = {}
    if f1_validation_status == "SUCCEEDED":
        f1_metrics.update(validation_metrics(f1_validation_data))
    if f1_eval_status == "SUCCEEDED":
        f1_metrics.update(formal_metrics(f1_eval_data))
    y1_metrics: dict[str, str] = {}
    if y1_validation_status == "SUCCEEDED":
        y1_metrics.update(validation_metrics(y1_validation_data))
    if y1_eval_status == "SUCCEEDED":
        y1_metrics.update(formal_metrics(y1_eval_data, selected_threshold=y1_selected_threshold))
    return [
        row(
            method="FGRefMOT",
            variant="historical-paper",
            pretraining="historical",
            sports_training="historical",
            physical_tracker="historical-protocol",
            referring_model="historical-protocol",
            status="HISTORICAL_SEPARATE_PROTOCOL",
            evaluation_split="PAPER_REPORTED",
            source_manifest="NOT_RUN",
        ),
        row(
            method="FGRefMOT",
            variant="current-checkpoint",
            pretraining="existing",
            sports_training="NOT_RUN",
            physical_tracker="NOT_RUN",
            referring_model="NOT_RUN",
            status="NOT_RUN",
        ),
        row(
            method="FlexHook",
            variant="F0",
            pretraining="public-pretrained",
            sports_training="zero-shot",
            physical_tracker="fixed-common-SambaMOTR",
            referring_model="FlexHook",
            status="NOT_RUN",
        ),
        row(
            method="FlexHook",
            variant="F1",
            pretraining="public-pretrained",
            sports_training="Sports-trained",
            physical_tracker="fixed-common-SambaMOTR",
            referring_model="FlexHook",
            status=f1_status,
            validation_split="INTERNAL_VAL_5" if f1_validation_status == "SUCCEEDED" else "NOT_RUN",
            evaluation_split="HISTORICAL_TEST_19",
            trainable_parameters=f1_parameter_count,
            checkpoint_path=str(f1_checkpoint) if f1_checkpoint.is_file() else "NOT_RUN",
            validation_manifest=str(f1_validation_manifest),
            source_manifest=str(f1_eval_manifest),
            metrics=f1_metrics or None,
        ),
        row(
            method="YesTrack",
            variant="Y0",
            pretraining="public-pretrained",
            sports_training="zero-shot",
            physical_tracker="fixed-common-SambaMOTR",
            referring_model="YesTrack",
            status="NOT_RUN",
        ),
        row(
            method="YesTrack",
            variant="Y1",
            pretraining="Qwen3-VL-2B",
            sports_training="Sports-LoRA",
            physical_tracker="fixed-common-SambaMOTR",
            referring_model="YesTrack",
            status=y1_status,
            validation_split="INTERNAL_VAL_5" if y1_validation_status == "SUCCEEDED" else "NOT_RUN",
            evaluation_split="HISTORICAL_TEST_19",
            trainable_parameters=y1_parameter_count,
            checkpoint_path=str(y1_checkpoint_root) if y1_checkpoint_root and y1_checkpoint_root.is_dir() else "NOT_RUN",
            validation_manifest=str(y1_validation_manifest),
            source_manifest=str(y1_eval_manifest),
            metrics=y1_metrics or None,
        ),
        row(
            method="YesTrack",
            variant="Y2",
            pretraining="Qwen3-VL-2B",
            sports_training="Sports-LoRA + temporal variant",
            physical_tracker="fixed-common-SambaMOTR",
            referring_model="YesTrack",
            status="NOT_RUN",
        ),
        row(
            method="YesTrack",
            variant="Y3",
            pretraining="Qwen3-VL-2B",
            sports_training="Sports-LoRA + proposed mechanism",
            physical_tracker="fixed-common-SambaMOTR",
            referring_model="YesTrack",
            status="NOT_RUN",
        ),
        row(
            method="SambaMOTR",
            variant="common-tracker-control",
            pretraining="SportsMOT-public-checkpoint",
            sports_training="inference-only",
            physical_tracker="SambaMOTR-DAB-D-DETR",
            referring_model="none",
            status=tracker_status,
            evaluation_split="HISTORICAL_TEST_19",
            source_manifest=str(tracker_manifest),
        ),
    ]


def write_outputs(rows: list[dict[str, str]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "BASELINE_TABLE.json"
    csv_path = output_root / "BASELINE_TABLE.csv"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "rmot-sports-baseline-table-0.1",
                "metric_policy": "VALIDATION_AND_FORMAL_METRICS_REQUIRE_COMPLETED_MANIFESTS; OTHERWISE_NOT_RUN",
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--nas-root", default="/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_rows(Path(args.project).expanduser().resolve(), Path(args.nas_root).expanduser().resolve())
    write_outputs(rows, Path(args.output_root).expanduser().resolve())
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(f"baseline_table_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
