#!/usr/bin/env python3
"""Validate the pre-registered Refer-Sports experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_METRICS = {"HOTA", "DetA", "AssA", "IDF1", "IDSW", "Ref-F1", "FPS"}


def validate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "Refer-Sports":
        raise ValueError("matrix must target Refer-Sports")
    if payload.get("primary_split") != "SPORTS_VAL":
        raise ValueError("matrix primary split must be SPORTS_VAL")
    if set(payload.get("metrics", [])) != REQUIRED_METRICS:
        raise ValueError("matrix metrics do not match the registered protocol")
    policy = payload.get("tracker_input_policy", {})
    if policy.get("gt_or_labels_as_tracker_input") is not False:
        raise ValueError("GT/labels tracker input must be explicitly rejected")
    if policy.get("test_gt_for_tuning_or_checkpoint_selection") is not False:
        raise ValueError("test GT tuning/checkpoint selection must be rejected")
    variants = payload.get("variants", [])
    ids = [item.get("id") for item in variants]
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("variants must have unique non-empty ids")
    proposed = [item for item in variants if item.get("stage") == "proposed_ablation"]
    if not proposed or any(item.get("status") != "WAIT_FOR_BASELINE" for item in proposed):
        raise ValueError("proposed variants must wait for formal baselines")
    return {"path": str(path), "variant_count": len(variants), "status": "VALID"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="configs/experiment_matrix_20261019.json")
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.path).expanduser().resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
