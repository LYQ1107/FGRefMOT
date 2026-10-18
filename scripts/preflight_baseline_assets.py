#!/usr/bin/env python3
"""Check that baseline assets are complete before any GPU job is started.

This is deliberately a metadata and bounded-file check.  It never walks the
dataset, creates a tracker output, or treats labels/GT as detector input.  A
missing external asset is reported as ``MISSING`` and makes the overall result
``PENDING``; it is not silently replaced by a similarly named local file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


READY = "READY"
MISSING = "MISSING"
INVALID = "INVALID"
UNVERIFIED = "UNVERIFIED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PARTS = {"gt", "gts", "label", "labels", "groundtruth", "annotations"}


def _display_path(path: str | Path | None) -> str:
    if path is None or str(path).strip() == "":
        return "<not provided>"
    return str(Path(path).expanduser())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path, hash_files: bool) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    if hash_files:
        item["sha256"] = _file_sha256(path)
    return item


def check_directory(
    name: str,
    path_value: str | Path | None,
    required_files: Iterable[str],
    *,
    hash_files: bool = False,
) -> dict[str, Any]:
    """Check a directory and a finite list of files beneath it."""

    path_text = _display_path(path_value)
    result: dict[str, Any] = {
        "name": name,
        "kind": "directory",
        "path": path_text,
        "required_for_training": True,
        "status": MISSING,
        "files": [],
        "reasons": [],
    }
    if path_text == "<not provided>":
        result["reasons"].append("path_not_provided")
        return result

    path = Path(path_text)
    if not path.exists():
        result["reasons"].append("path_does_not_exist")
        return result
    if not path.is_dir():
        result["status"] = INVALID
        result["reasons"].append("path_is_not_a_directory")
        return result

    missing: list[str] = []
    for relative in required_files:
        candidate = path / relative
        if not candidate.is_file():
            missing.append(relative)
            continue
        result["files"].append(_file_info(candidate, hash_files))
    if missing:
        result["reasons"].append("missing_files:" + ",".join(missing))
        return result
    if not hash_files:
        result["status"] = UNVERIFIED
        result["reasons"].append("run_with_hash_files_to_record_external_asset_hashes")
        return result
    result["status"] = READY
    return result


def check_checkpoint(
    name: str,
    path_value: str | Path | None,
    *,
    hash_files: bool = False,
) -> dict[str, Any]:
    """Check one task checkpoint without accepting annotation files."""

    path_text = _display_path(path_value)
    result: dict[str, Any] = {
        "name": name,
        "kind": "checkpoint",
        "path": path_text,
        "required_for_training": True,
        "status": MISSING,
        "reasons": [],
    }
    if path_text == "<not provided>":
        result["reasons"].append("path_not_provided")
        return result
    path = Path(path_text)
    if not path.exists():
        result["reasons"].append("path_does_not_exist")
        return result
    if not path.is_file():
        result["status"] = INVALID
        result["reasons"].append("path_is_not_a_file")
        return result
    if path.suffix.lower() not in {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}:
        result["status"] = INVALID
        result["reasons"].append("unsupported_checkpoint_suffix")
        return result
    if path.stat().st_size == 0:
        result["status"] = INVALID
        result["reasons"].append("empty_checkpoint")
        return result
    if not hash_files:
        result["status"] = UNVERIFIED
        result["reasons"].append("run_with_hash_files_to_record_checkpoint_hash")
        return result
    result["file"] = _file_info(path, hash_files)
    result["status"] = READY
    return result


def _safe_relative_path(value: str) -> bool:
    candidate = Path(value)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _contains_forbidden_part(value: str) -> bool:
    for part in Path(value).parts:
        tokens = set(re.split(r"[^a-z0-9]+", part.lower()))
        if tokens & _FORBIDDEN_PARTS:
            return True
    return False


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def check_tracker_root(
    name: str,
    path_value: str | Path | None,
    *,
    hash_files: bool = False,
) -> dict[str, Any]:
    """Validate a provenance-complete physical-tracker output manifest.

    The manifest is intentionally strict: a folder of MOT text files without
    detector provenance is not accepted, and ``used_ground_truth`` must be the
    literal boolean ``false``.
    """

    path_text = _display_path(path_value)
    result: dict[str, Any] = {
        "name": name,
        "kind": "physical_tracker_outputs",
        "path": path_text,
        "required_for_training": True,
        "status": MISSING,
        "reasons": [],
    }
    if path_text == "<not provided>":
        result["reasons"].append("path_not_provided")
        return result
    if _contains_forbidden_part(path_text):
        result["status"] = INVALID
        result["reasons"].append("path_looks_like_annotation_or_ground_truth")
        return result

    root = Path(path_text)
    if not root.exists():
        result["reasons"].append("path_does_not_exist")
        return result
    if not root.is_dir():
        result["status"] = INVALID
        result["reasons"].append("path_is_not_a_directory")
        return result

    manifest_path = root / "tracker_manifest.json"
    if not manifest_path.is_file():
        result["reasons"].append("tracker_manifest.json_missing")
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = INVALID
        result["reasons"].append(f"invalid_tracker_manifest:{exc.__class__.__name__}")
        return result
    if not isinstance(manifest, dict):
        result["status"] = INVALID
        result["reasons"].append("tracker_manifest_not_an_object")
        return result

    expected = {
        "asset_role": "physical_tracker_outputs",
        "dataset": "SportsMOT",
        "input_type": "detector_outputs",
        "used_ground_truth": False,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            result["reasons"].append(f"manifest_{key}_must_equal_{expected_value!r}")
    for key in ("split", "detector_checkpoint_sha256", "tracker_config_sha256", "source_manifest_sha256"):
        value = manifest.get(key)
        if not value:
            result["reasons"].append(f"manifest_{key}_missing")
        elif key.endswith("_sha256") and not _valid_sha256(value):
            result["reasons"].append(f"manifest_{key}_not_sha256")

    for key in ("detector_checkpoint_path", "tracker_config_path"):
        value = manifest.get(key)
        if not isinstance(value, str) or not _safe_relative_path(value):
            result["reasons"].append(f"manifest_{key}_missing_or_unsafe")
            continue
        if _contains_forbidden_part(value):
            result["reasons"].append(f"manifest_{key}_looks_like_annotation:{value}")
            continue
        candidate = root / value
        if not candidate.is_file() or candidate.stat().st_size == 0:
            result["reasons"].append(f"manifest_{key}_file_missing_or_empty:{value}")
            continue
        if hash_files:
            declared_key = "detector_checkpoint_sha256" if key == "detector_checkpoint_path" else "tracker_config_sha256"
            if _file_sha256(candidate) != manifest.get(declared_key):
                result["reasons"].append(f"manifest_{key}_hash_mismatch:{value}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        result["reasons"].append("manifest_outputs_empty")
        outputs = []
    checked_outputs: list[dict[str, Any]] = []
    for entry in outputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            result["reasons"].append("manifest_output_entry_invalid")
            continue
        relative = entry["path"]
        if not _safe_relative_path(relative) or _contains_forbidden_part(relative):
            result["reasons"].append(f"output_path_rejected:{relative}")
            continue
        output_path = root / relative
        if not output_path.is_file() or output_path.stat().st_size == 0:
            result["reasons"].append(f"output_file_missing_or_empty:{relative}")
            continue
        declared_hash = entry.get("sha256")
        if not _valid_sha256(declared_hash):
            result["reasons"].append(f"output_hash_missing_or_invalid:{relative}")
            continue
        output_info = _file_info(output_path, hash_files)
        if hash_files and output_info["sha256"] != declared_hash:
            result["reasons"].append(f"output_hash_mismatch:{relative}")
            continue
        output_info["relative_path"] = relative
        checked_outputs.append(output_info)

    result["manifest"] = {
        "path": str(manifest_path),
        "output_count": len(checked_outputs),
        "declared_output_count": len(outputs),
        "split": manifest.get("split"),
    }
    result["outputs"] = checked_outputs
    if result["reasons"]:
        result["status"] = INVALID
        return result
    if not hash_files:
        result["status"] = UNVERIFIED
        result["reasons"].append("run_with_hash_files_to_verify_tracker_hashes")
        return result
    result["status"] = READY
    return result


def check_layout(name: str, path_value: str | Path | None) -> dict[str, Any]:
    """Check one generated layout manifest and its no-GT policy marker."""

    path_text = _display_path(path_value)
    result: dict[str, Any] = {
        "name": name,
        "kind": "dataset_layout",
        "path": path_text,
        "required_for_training": True,
        "status": MISSING,
        "reasons": [],
    }
    if path_text == "<not provided>":
        result["reasons"].append("path_not_provided")
        return result
    root = Path(path_text)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        result["reasons"].append("manifest.json_missing")
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = INVALID
        result["reasons"].append(f"invalid_layout_manifest:{exc.__class__.__name__}")
        return result
    policy = str(manifest.get("inference_data_policy", ""))
    if "NO_GT_AS_TRACKER_INPUT" not in policy:
        result["status"] = INVALID
        result["reasons"].append("layout_manifest_missing_no_gt_policy")
        return result
    result["status"] = READY
    result["manifest"] = {
        "path": str(manifest_path),
        "inference_data_policy": policy,
    }
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks = [
        check_layout("flexhook_layout", args.flexhook_layout),
        check_layout("yestrack_layout", args.yestrack_layout),
        check_directory(
            "roberta_base",
            args.roberta_dir,
            ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.json"),
            hash_files=args.hash_files,
        ),
        check_directory(
            "rope_swin_tiny",
            args.swin_dir,
            ("config.json", "pytorch_model.bin"),
            hash_files=args.hash_files,
        ),
        check_checkpoint("flexhook_task_checkpoint", args.flexhook_checkpoint, hash_files=args.hash_files),
        check_checkpoint("yestrack_lora_checkpoint", args.yestrack_lora, hash_files=args.hash_files),
        check_tracker_root("common_sports_tracker", args.tracker_root, hash_files=args.hash_files),
    ]
    pending = [item["name"] for item in checks if item["status"] != READY]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(Path(__file__).resolve().parents[1]),
        "policy": {
            "gt_or_labels_as_tracker_input": "REJECTED",
            "unverified_external_asset_substitution": "REJECTED",
            "gpu_jobs_started": False,
        },
        "checks": checks,
        "overall_status": "READY" if not pending else "PENDING",
        "pending_checks": pending,
        "next_action": (
            "Acquire and hash every missing official asset, then rerun this preflight."
            if pending
            else "All gates are ready; obtain an explicit GPU-run approval before starting training."
        ),
    }


def _default_path(relative: str) -> str:
    return str(Path(__file__).resolve().parents[1] / relative)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flexhook-layout", default=os.environ.get("FLEXHOOK_LAYOUT_ROOT", _default_path("runs/flexhook_sports_layout_20261019_v2")))
    parser.add_argument("--yestrack-layout", default=os.environ.get("YESTRACK_LAYOUT_ROOT", _default_path("runs/yestrack_sports_layout_20261019_v1")))
    parser.add_argument("--roberta-dir", default=os.environ.get("FLEXHOOK_ROBERTA_DIR", ""))
    parser.add_argument("--swin-dir", default=os.environ.get("FLEXHOOK_SWIN_DIR", ""))
    parser.add_argument("--flexhook-checkpoint", default=os.environ.get("FLEXHOOK_TASK_CHECKPOINT", ""))
    parser.add_argument("--yestrack-lora", default=os.environ.get("YESTRACK_LORA_CHECKPOINT", ""))
    parser.add_argument("--tracker-root", default=os.environ.get("COMMON_TRACKER_ROOT", ""))
    parser.add_argument("--output", default=_default_path("runs/baseline_asset_preflight_20261019/report.json"))
    parser.add_argument("--hash-files", action="store_true", help="hash finite listed files; required for output hash verification")
    parser.add_argument("--json", action="store_true", help="also print the complete JSON report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for item in report["checks"]:
            print(f"{item['name']}: {item['status']} — {item['path']}")
        print(f"baseline_asset_preflight: {report['overall_status']}")
        print(f"report: {output}")
    return 0 if report["overall_status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
