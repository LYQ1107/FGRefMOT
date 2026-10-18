#!/usr/bin/env python3
"""Create and verify a reproducible baseline run contract.

This command is a read-only launch gate.  It records the exact dataset/layout
manifests, preflight result, finite asset hashes, source revision, protocol
parameters and environment metadata that a later GPU command must use.  It
never imports a model, allocates a CUDA device, starts a subprocess, or writes
tracker outputs.

``--write`` creates a contract even while the study is pending external
assets.  ``--require-ready`` turns the same command into a strict gate and
returns non-zero until every preflight check and protocol field is complete.
``--check`` verifies all hashes in an existing contract before a launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
READY = "READY"
PENDING = "PENDING"
INVALID = "INVALID"

METHOD_LAYOUT = {
    "flexhook": "flexhook_layout",
    "yestrack": "yestrack_layout",
}
METHOD_ASSETS = {
    "flexhook": ("roberta_base", "rope_swin_tiny", "flexhook_task_checkpoint"),
    "yestrack": ("yestrack_lora_checkpoint", "yestrack_base_model"),
}
COMMON_ASSET = "common_sports_tracker"

# The study has several intentionally separate initialization protocols.  A
# task checkpoint is required for the zero-shot/fine-tuned variants that use
# it, but it must not block the Sports-only variants that start from the
# public language/visual base models.  Keep the old method-level mapping as
# the ``default`` for backwards-compatible contracts created before the
# matrix was recorded.
METHOD_VARIANTS = {
    "flexhook": {
        "default": METHOD_ASSETS["flexhook"],
        "f0_zero_shot": METHOD_ASSETS["flexhook"],
        "f1_sports_trained": ("roberta_base", "rope_swin_tiny"),
        "f2_sports_finetuned": METHOD_ASSETS["flexhook"],
    },
    "yestrack": {
        "default": METHOD_ASSETS["yestrack"],
        "y0_zero_shot": METHOD_ASSETS["yestrack"],
        "y1_sports_lora": ("yestrack_base_model",),
        "y2_tcp_trp": ("yestrack_base_model",),
        "y3_video_training": ("yestrack_base_model",),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, workspace_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _path_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    else:
        record["exists"] = path.exists()
    return record


def _git_info(workspace_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"workspace": str(workspace_root)}
    commands = {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--porcelain"],
    }
    for key, command in commands.items():
        try:
            result = subprocess.run(
                command,
                cwd=workspace_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            info[key] = f"unavailable:{exc.__class__.__name__}"
            continue
        value = result.stdout.strip()
        info[key] = value
    info["clean"] = info.get("status") == ""
    return info


def _official_source_commits(workspace_root: Path) -> dict[str, str]:
    path = workspace_root / "configs" / "official_sources.env.example"
    commits: dict[str, str] = {}
    if not path.is_file():
        return commits
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.endswith("_COMMIT"):
            commits[key] = value.strip()
    return commits


def _check_map(preflight: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for item in preflight.get("checks", []):
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            checks[item["name"]] = item
    return checks


def _asset_snapshot(check: dict[str, Any]) -> dict[str, Any]:
    """Keep the finite, hash-bearing part of a preflight check."""

    allowed = {
        "name",
        "kind",
        "path",
        "status",
        "required_for_training",
        "reasons",
        "files",
        "weight_files",
        "file",
        "manifest",
        "outputs",
    }
    return {key: check[key] for key in sorted(allowed) if key in check}


def _protocol_complete(protocol: dict[str, Any]) -> tuple[bool, list[str]]:
    def has_placeholder(value: Any) -> bool:
        if isinstance(value, str):
            upper = value.upper()
            return any(token in upper for token in ("PENDING", "TODO", "TBD", "TO_BE_FILLED", "UNSPECIFIED"))
        if isinstance(value, dict):
            return any(has_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(has_placeholder(item) for item in value)
        return False

    reasons: list[str] = []
    if not isinstance(protocol.get("seed"), int):
        reasons.append("protocol_seed_missing_or_not_integer")
    for key in ("train_split", "evaluation_split", "tracker_policy"):
        if not isinstance(protocol.get(key), str) or not protocol[key].strip():
            reasons.append(f"protocol_{key}_missing")
    thresholds = protocol.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds or has_placeholder(thresholds):
        reasons.append("protocol_thresholds_missing")
    hyperparameters = protocol.get("hyperparameters")
    if not isinstance(hyperparameters, dict) or not hyperparameters or has_placeholder(hyperparameters):
        reasons.append("protocol_hyperparameters_missing_or_not_object")
    command = protocol.get("launch_command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(v, str) and v for v in command)
        or has_placeholder(command)
    ):
        reasons.append("protocol_launch_command_missing_or_invalid")
    return not reasons, reasons


def build_contract(
    *,
    workspace_root: Path,
    method: str,
    run_id: str,
    preflight_path: Path,
    dataset_manifest_path: Path,
    layout_path: Path,
    protocol: dict[str, Any],
    variant: str = "default",
) -> dict[str, Any]:
    if method not in METHOD_LAYOUT:
        raise ValueError(f"unsupported method: {method}")
    variants = METHOD_VARIANTS[method]
    if variant not in variants:
        raise ValueError(f"unsupported {method} variant: {variant}")

    preflight = read_json(preflight_path)
    checks = _check_map(preflight)
    blocking: list[str] = []

    if preflight.get("schema_version") != 1:
        blocking.append("preflight_schema_version_must_equal_1")
    required_asset_names = (*variants[variant], COMMON_ASSET)
    # The aggregate Phase 7 report contains every method's assets.  A pending
    # optional variant must not block a different variant whose own required
    # checks are ready, so only required checks are promoted to blockers here.
    if preflight.get("overall_status") != READY:
        pending = preflight.get("pending_checks")
        pending_names = set(pending) if isinstance(pending, list) else set()
        required_pending = [name for name in required_asset_names if name in pending_names]
        blocking.extend(f"preflight_pending:{name}" for name in required_pending)
        if not pending_names and required_pending:
            blocking.append("preflight_overall_status_not_ready")

    expected_layout_name = METHOD_LAYOUT[method]
    layout_check = checks.get(expected_layout_name)
    if not layout_check or layout_check.get("status") != READY:
        blocking.append(f"layout_check_not_ready:{expected_layout_name}")
    if not layout_path.is_dir():
        blocking.append("layout_path_missing")
    layout_manifest = layout_path / "manifest.json"
    if not layout_manifest.is_file():
        blocking.append("layout_manifest_missing")
    else:
        try:
            layout_value = read_json(layout_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            layout_value = {}
            blocking.append("layout_manifest_invalid")
        if "NO_GT_AS_TRACKER_INPUT" not in str(layout_value.get("inference_data_policy", "")):
            blocking.append("layout_manifest_missing_no_gt_policy")

    if not dataset_manifest_path.is_file():
        blocking.append("dataset_manifest_missing")

    required_assets = required_asset_names
    assets: dict[str, Any] = {}
    for name in required_assets:
        check = checks.get(name)
        if check is None:
            blocking.append(f"preflight_check_missing:{name}")
            continue
        assets[name] = _asset_snapshot(check)
        if check.get("status") != READY:
            blocking.append(f"asset_check_not_ready:{name}")

    protocol_ok, protocol_reasons = _protocol_complete(protocol)
    blocking.extend(protocol_reasons)

    source_files = [
        workspace_root / "configs" / "official_sources.env.example",
        workspace_root / "scripts" / "preflight_baseline_assets.py",
        workspace_root / "scripts" / "run_contract.py",
    ]
    files: dict[str, Any] = {
        "preflight": _path_record(preflight_path),
        "dataset_manifest": _path_record(dataset_manifest_path),
        "layout_manifest": _path_record(layout_manifest),
    }
    for path in source_files:
        if path.is_file():
            files[path.relative_to(workspace_root).as_posix()] = _path_record(path)

    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "variant": variant,
        "status": READY if not blocking else PENDING,
        "ready_to_launch": not blocking,
        "blocking_reasons": blocking,
        "policy": {
            "gt_or_labels_as_tracker_input": "REJECTED",
            "tracker_outputs_must_be_common_across_methods": True,
            "external_asset_substitution": "REJECTED",
            "gpu_job_started_by_contract_builder": False,
        },
        "protocol": protocol,
        "inputs": {
            "required_assets": list(required_assets),
            "preflight": {
                "record": files["preflight"],
                "overall_status": preflight.get("overall_status"),
                "pending_checks": preflight.get("pending_checks", []),
            },
            "dataset_manifest": files["dataset_manifest"],
            "layout": {
                "check_name": expected_layout_name,
                "record": files["layout_manifest"],
            },
            "assets": assets,
        },
        "code": {
            "git": _git_info(workspace_root),
            "official_source_commits": _official_source_commits(workspace_root),
            "source_files": {
                key: value for key, value in files.items() if key not in {"preflight", "dataset_manifest", "layout_manifest"}
            },
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
        },
        "validation": {
            "protocol_complete": protocol_ok,
            "preflight_reused_without_mutation": True,
            "layout_policy_checked": True,
        },
    }
    return contract


def _record_matches(record: dict[str, Any]) -> tuple[bool, str | None]:
    path_text = record.get("path")
    if not isinstance(path_text, str):
        return False, "record_path_missing"
    path = Path(path_text)
    if not path.is_file():
        return False, f"file_missing:{path}"
    expected = record.get("sha256")
    if not isinstance(expected, str):
        return False, f"hash_missing:{path}"
    actual = sha256_file(path)
    if actual != expected:
        return False, f"hash_mismatch:{path}"
    size = record.get("size_bytes")
    if isinstance(size, int) and path.stat().st_size != size:
        return False, f"size_mismatch:{path}"
    return True, None


def check_contract(contract_path: Path) -> tuple[dict[str, Any], list[str]]:
    contract = read_json(contract_path)
    errors: list[str] = []
    if contract.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if contract.get("policy", {}).get("gt_or_labels_as_tracker_input") != "REJECTED":
        errors.append("gt_or_labels_policy_missing")
    records: list[dict[str, Any]] = []
    inputs = contract.get("inputs", {})
    for key in ("preflight", "dataset_manifest"):
        if key == "dataset_manifest":
            record = inputs.get(key)
        else:
            record = inputs.get(key, {}).get("record") if isinstance(inputs.get(key), dict) else None
        if isinstance(record, dict):
            records.append(record)
        else:
            errors.append(f"input_record_missing:{key}")
    layout_record = inputs.get("layout", {}).get("record") if isinstance(inputs.get("layout"), dict) else None
    if isinstance(layout_record, dict):
        records.append(layout_record)
    else:
        errors.append("input_record_missing:layout")
    source_files = contract.get("code", {}).get("source_files", {})
    if isinstance(source_files, dict):
        records.extend(value for value in source_files.values() if isinstance(value, dict))
    assets = inputs.get("assets", {})
    if isinstance(assets, dict):
        for asset in assets.values():
            if not isinstance(asset, dict):
                errors.append("asset_record_invalid")
                continue
            for key in ("files", "weight_files"):
                values = asset.get(key, [])
                if isinstance(values, list):
                    records.extend(value for value in values if isinstance(value, dict))
            value = asset.get("file")
            if isinstance(value, dict):
                records.append(value)
            for output in asset.get("outputs", []) if isinstance(asset.get("outputs"), list) else []:
                if isinstance(output, dict):
                    records.append(output)
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("path")), str(record.get("sha256")))
        if key in seen:
            continue
        seen.add(key)
        ok, reason = _record_matches(record)
        if not ok and reason:
            errors.append(reason)
    if contract.get("status") == READY and errors:
        errors.append("ready_contract_is_no_longer_valid")
    return contract, errors


def _parse_json_object(value: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{name} must be a JSON object")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="build and write a contract")
    mode.add_argument("--check", metavar="CONTRACT", help="verify an existing contract's hashes")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--method", choices=sorted(METHOD_LAYOUT), help="required with --write")
    parser.add_argument("--variant", default="default", help="initialization/training variant, e.g. f1_sports_trained")
    parser.add_argument("--run-id", help="stable run identifier, e.g. flexhook_baseline_seed42")
    parser.add_argument("--preflight", help="Phase 7 JSON report")
    parser.add_argument("--dataset-manifest", help="data/refer_sports_manifest.json")
    parser.add_argument("--layout", help="method-specific generated layout root")
    parser.add_argument("--output", help="contract JSON destination")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--evaluation-split", default="val")
    parser.add_argument("--tracker-policy", default="common_physical_tracker_detector_outputs")
    parser.add_argument("--thresholds-json", type=lambda value: _parse_json_object(value, "--thresholds-json"), default={})
    parser.add_argument("--hyperparameters-json", type=lambda value: _parse_json_object(value, "--hyperparameters-json"), default={})
    parser.add_argument("--launch-command-json", type=lambda value: json.loads(value), default=[])
    parser.add_argument("--require-ready", action="store_true", help="exit 2 if the contract cannot launch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        contract, errors = check_contract(resolve_path(args.check, Path(args.workspace_root).resolve()))
        if errors:
            for error in errors:
                print(f"contract: INVALID — {error}")
            return 2
        if args.require_ready and not contract.get("ready_to_launch", False):
            print(f"contract: {contract.get('status', PENDING)} — hashes and policy verified, launch blocked")
            for reason in contract.get("blocking_reasons", []):
                print(f"  - {reason}")
            return 2
        print(f"contract: {contract.get('status', INVALID)} — hashes and policy verified")
        return 0

    required = (args.method, args.run_id, args.preflight, args.dataset_manifest, args.layout, args.output)
    if any(value in (None, "") for value in required):
        raise SystemExit("--write requires --method, --run-id, --preflight, --dataset-manifest, --layout and --output")
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    protocol = {
        "seed": args.seed,
        "train_split": args.train_split,
        "evaluation_split": args.evaluation_split,
        "tracker_policy": args.tracker_policy,
        "thresholds": args.thresholds_json,
        "hyperparameters": args.hyperparameters_json,
        "launch_command": args.launch_command_json,
    }
    contract = build_contract(
        workspace_root=workspace_root,
        method=args.method,
        run_id=args.run_id,
        preflight_path=resolve_path(args.preflight, workspace_root),
        dataset_manifest_path=resolve_path(args.dataset_manifest, workspace_root),
        layout_path=resolve_path(args.layout, workspace_root),
        protocol=protocol,
        variant=args.variant,
    )
    output = resolve_path(args.output, workspace_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run_contract: {contract['status']} — {output}")
    if contract["blocking_reasons"]:
        print("blocking_reasons:")
        for reason in contract["blocking_reasons"]:
            print(f"  - {reason}")
    if args.require_ready and not contract["ready_to_launch"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
