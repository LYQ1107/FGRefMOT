#!/usr/bin/env python3
"""Build a protocol-controlled Refer-Sports baseline table.

The table is deliberately conservative: a metric is emitted only when a
completed formal-evaluation manifest is available.  Missing or in-progress
experiments remain ``NOT_RUN``/``RUNNING`` instead of being inferred from
training losses or partial files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "validation_hota",
    "validation_deta",
    "validation_assa",
    "validation_idf1",
    "validation_idsw",
    "validation_ref_f1",
    "fps",
)
TABLE_FIELDS = (
    "method",
    "variant",
    "pretraining",
    "sports_training",
    "physical_tracker",
    "referring_model",
    *METRIC_FIELDS,
    "trainable_parameters",
    "checkpoint_path",
    "status",
    "source_manifest",
)


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


def row(
    *,
    method: str,
    variant: str,
    pretraining: str,
    sports_training: str,
    physical_tracker: str,
    referring_model: str,
    status: str,
    checkpoint_path: str = "NOT_RUN",
    source_manifest: str = "NOT_RUN",
) -> dict[str, str]:
    values = {
        "method": method,
        "variant": variant,
        "pretraining": pretraining,
        "sports_training": sports_training,
        "physical_tracker": physical_tracker,
        "referring_model": referring_model,
        "trainable_parameters": "NOT_RUN",
        "checkpoint_path": checkpoint_path,
        "status": status,
        "source_manifest": source_manifest,
    }
    values.update({field: "NOT_RUN" for field in METRIC_FIELDS})
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
    y1_eval_manifest = nas_root / "runs/gpu/yestrack_y1_common_eval_20261019_v2/YESTRACK_EVAL_MANIFEST.json"
    f1_checkpoint = nas_root / "runs/gpu/flexhook_f1_sports_trained_seed42/ckpt_epoch_best_0.pth"
    f1_eval_manifest = nas_root / "runs/gpu/flexhook_f1_common_eval_20261019_v2/FLEXHOOK_EVAL_MANIFEST.json"

    y1_status = manifest_status(y1_manifest)
    if y1_status.startswith("RUNNING"):
        y1_status = "TRAINING_RUNNING"
    elif y1_status == "SUCCEEDED" and y1_data.get("protocol_valid", True):
        y1_status = "TRAINING_SUCCEEDED_FORMAL_EVAL_PENDING"
    elif y1_status == "SUCCEEDED":
        y1_status = "PROTOCOL_INVALID"
    f1_status = "FORMAL_EVAL_SUCCEEDED" if manifest_status(f1_eval_manifest) == "SUCCEEDED" else (
        "CHECKPOINT_READY_FORMAL_EVAL_PENDING" if f1_checkpoint.is_file() else "NOT_RUN"
    )
    tracker_status = manifest_status(tracker_manifest)
    if tracker_status == "PREPARED_NO_GPU_RUN":
        tracker_status = "TRACKER_GPU_RUN_PENDING"
    return [
        row(
            method="FGRefMOT",
            variant="historical-paper",
            pretraining="historical",
            sports_training="historical",
            physical_tracker="historical-protocol",
            referring_model="historical-protocol",
            status="HISTORICAL_SEPARATE_PROTOCOL",
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
            checkpoint_path=str(f1_checkpoint) if f1_checkpoint.is_file() else "NOT_RUN",
            source_manifest=str(f1_eval_manifest),
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
            checkpoint_path=str(y1_checkpoint_root) if y1_checkpoint_root and y1_checkpoint_root.is_dir() else "NOT_RUN",
            source_manifest=str(y1_eval_manifest),
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
                "metric_policy": "ONLY_COMPLETED_FORMAL_MANIFESTS; OTHERWISE_NOT_RUN",
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
