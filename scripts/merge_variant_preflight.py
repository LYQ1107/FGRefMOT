#!/usr/bin/env python3
"""Merge independently hash-audited asset preflights for variant contracts.

The NAS login shell exposes the project at ``/home/Valadmin/A100`` while the
server uses a read-only mount at ``/data3/liuyeqiang/nas/A100``.  Phase 21
audited the pretrained directories and Phase 23 audited the common tracker;
this command joins those finite records without silently changing a status or
inventing a hash.  The resulting report remains ``PENDING`` for the two
official task-checkpoint gates that were not acquired.

This command reuses hash-bearing source reports.  ``run_contract.py --check``
is still required immediately before a launch and hashes the current files.
Use ``--verify-hashes`` only when a full current-file rehash is acceptable on
the NAS mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NAS_LOGIN_PREFIX = "/home/Valadmin/A100"
NAS_MOUNT_PREFIX = "/data3/liuyeqiang/nas/A100"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _translate(value: str) -> str:
    return value.replace(NAS_LOGIN_PREFIX, NAS_MOUNT_PREFIX)


def _translate_paths(item: Any) -> Any:
    if isinstance(item, dict):
        return {key: _translate_paths(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_translate_paths(value) for value in item]
    if isinstance(item, str):
        return _translate(item)
    return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_current_sizes(check: dict[str, Any]) -> list[str]:
    """Check existence and recorded sizes without rereading every byte."""

    errors: list[str] = []
    for key in ("files", "weight_files"):
        for record in check.get(key, []):
            path = Path(record["path"])
            if not path.is_file():
                errors.append(f"missing:{path}")
            elif isinstance(record.get("size_bytes"), int) and path.stat().st_size != record["size_bytes"]:
                errors.append(f"size_mismatch:{path}")
    for record in check.get("outputs", []):
        path = Path(record["path"])
        if not path.is_file():
            errors.append(f"missing:{path}")
        elif isinstance(record.get("size_bytes"), int) and path.stat().st_size != record["size_bytes"]:
            errors.append(f"size_mismatch:{path}")
    return errors


def merge_reports(
    *,
    base_report: Path,
    tracker_report: Path,
    output: Path,
    generated_at: str | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    base = _read(base_report)
    tracker = _read(tracker_report)
    base_checks = {item["name"]: item for item in base.get("checks", []) if isinstance(item, dict)}
    tracker_checks = {item["name"]: item for item in tracker.get("checks", []) if isinstance(item, dict)}

    names = ("flexhook_layout", "yestrack_layout", "roberta_base", "rope_swin_tiny", "yestrack_base_model")
    checks: list[dict[str, Any]] = []
    for name in names:
        if name not in base_checks:
            raise ValueError(f"base report missing check: {name}")
        item = _translate_paths(json.loads(json.dumps(base_checks[name])))
        if item.get("status") != "READY":
            raise ValueError(f"base report is not READY for {name}: {item.get('status')}")
        errors = _check_current_sizes(item)
        if errors:
            raise ValueError(f"current size check failed for {name}: {errors[:3]}")
        if verify_hashes:
            records = [*item.get("files", []), *item.get("weight_files", [])]
            for record in records:
                if _sha256(Path(record["path"])) != record.get("sha256"):
                    raise ValueError(f"hash mismatch for {record['path']}")
        checks.append(item)

    checks.extend(
        [
            {
                "kind": "checkpoint",
                "name": "flexhook_task_checkpoint",
                "path": "<not provided>",
                "required_for_training": True,
                "status": "MISSING",
                "reasons": ["official_checkpoint_not_acquired"],
            },
            {
                "kind": "checkpoint",
                "name": "yestrack_lora_checkpoint",
                "path": "<not provided>",
                "required_for_training": True,
                "status": "MISSING",
                "reasons": ["official_lora_not_acquired"],
            },
        ]
    )

    tracker_check = tracker_checks.get("common_sports_tracker")
    if not tracker_check or tracker_check.get("status") != "READY":
        raise ValueError("tracker report does not contain a READY common_sports_tracker check")
    tracker_check = _translate_paths(json.loads(json.dumps(tracker_check)))
    errors = _check_current_sizes(tracker_check)
    if errors:
        raise ValueError(f"current size check failed for common_sports_tracker: {errors[:3]}")
    if verify_hashes:
        manifest_path = Path(tracker_check["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        for relative, declared in ((entry["path"], entry["sha256"]) for entry in manifest["outputs"]):
            if _sha256(root / relative) != declared:
                raise ValueError(f"tracker output hash mismatch: {relative}")
    checks.append(tracker_check)

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "workspace": str(output.parents[2].resolve()),
        "overall_status": "PENDING",
        "pending_checks": ["flexhook_task_checkpoint", "yestrack_lora_checkpoint"],
        "next_action": "Acquire and provenance-audit the official FlexHook RMOT task checkpoint and YesTrack public V2 LoRA; F1/Y1/Y2/Y3 variant contracts may be evaluated separately under the matrix.",
        "policy": {
            "gpu_jobs_started": False,
            "gt_or_labels_as_tracker_input": "REJECTED",
            "unverified_external_asset_substitution": "REJECTED",
            "hash_provenance": "READY checks reuse hash-bearing Phase 21/23 reports; run_contract --check is the final current-file hash verification before launch.",
        },
        "evidence": {
            "base_preflight": str(base_report),
            "tracker_preflight": str(tracker_report),
            "path_translation": f"{NAS_LOGIN_PREFIX} on the NAS login shell maps to {NAS_MOUNT_PREFIX} on this read-only mount.",
            "ready_asset_names": ["flexhook_layout", "yestrack_layout", "roberta_base", "rope_swin_tiny", "yestrack_base_model", "common_sports_tracker"],
            "size_check_performed": True,
            "hash_recheck_performed": bool(verify_hashes),
        },
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-preflight", type=Path, required=True)
    parser.add_argument("--tracker-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    report = merge_reports(
        base_report=args.base_preflight,
        tracker_report=args.tracker_preflight,
        output=args.output,
        generated_at=args.generated_at,
        verify_hashes=args.verify_hashes,
    )
    print(f"merge_variant_preflight: {report['overall_status']} — {args.output}")
    for item in report["checks"]:
        print(f"{item['name']}: {item['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
