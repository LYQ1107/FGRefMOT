#!/usr/bin/env python3
"""Merge completed parallel FlexHook evaluation shards and run TrackEval.

The shard jobs each contain a disjoint set of videos.  This command links the
video directories into one TrackEval root, combines their manifests, and runs
the existing FlexHook TrackEval configuration once over all expressions.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_LIST = REPO_ROOT / "configs" / "refer_sports_eval_sequences_20261019.txt"


def read_sequence_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing shard manifest: {path}")
    value = json.loads(path.read_text())
    if value.get("status") not in {"PREDICTIONS_WRITTEN", "SUCCEEDED"}:
        raise RuntimeError(f"shard is not complete: {path} status={value.get('status')!r}")
    return value


def merge_shards(*, base_root: Path, output_root: Path, sequence_list: Path, overwrite: bool) -> dict[str, Any]:
    expected = read_sequence_list(sequence_list)
    shard_paths = sorted(base_root.glob("shard_*/FLEXHOOK_EVAL_MANIFEST.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard manifests under {base_root}")
    manifests = [_manifest(path) for path in shard_paths]

    seen: set[str] = set()
    ordered_manifests: list[dict[str, Any]] = []
    for name in expected:
        match = next((item for item in manifests if name in item.get("sequence_names", [])), None)
        if match is None:
            raise RuntimeError(f"sequence {name} is missing from completed shard manifests")
        if name in seen:
            raise RuntimeError(f"duplicate sequence in shard manifests: {name}")
        seen.add(name)
        ordered_manifests.append(match)
    if seen != set(expected):
        raise RuntimeError(f"shard sequence set differs from requested list: missing={set(expected)-seen}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    seq_info: dict[str, int] = {}
    source_shards: dict[str, str] = {}
    total_prediction_rows = 0
    total_dataset_count = 0
    shard_runs: list[dict[str, Any]] = []
    for manifest in manifests:
        shard_root = Path(manifest["output_root"])
        for video in manifest["sequence_names"]:
            destination = output_root / video
            source = shard_root / video
            if destination.exists() or destination.is_symlink():
                raise RuntimeError(f"duplicate output directory: {destination}")
            if not source.is_dir():
                raise FileNotFoundError(f"missing completed video output: {source}")
            destination.symlink_to(source, target_is_directory=True)
            source_shards[video] = str(shard_root)
        for record in manifest.get("records", []):
            item = copy.deepcopy(record)
            video = item["video"]
            expression_dir = item["expression_dir"]
            item["gt_path"] = str(output_root / video / expression_dir / "gt.txt")
            item["predict_path"] = str(output_root / video / expression_dir / "predict.txt")
            records.append(item)
        seq_info.update({str(key): int(value) for key, value in manifest.get("seq_info", {}).items()})
        total_prediction_rows += int(manifest.get("prediction_rows", manifest.get("run", {}).get("prediction_rows", 0)))
        total_dataset_count += int(manifest.get("run", {}).get("dataset_count", 0))
        if manifest.get("run"):
            shard_runs.append({"shard_root": str(shard_root), "run": manifest["run"]})

    first = copy.deepcopy(manifests[0])
    first.update(
        {
            "status": "PREDICTIONS_WRITTEN",
            "output_root": str(output_root),
            "sequence_names": expected,
            "sequence_count": len(expected),
            "seq_info": seq_info,
            "records": records,
            "prediction_rows": total_prediction_rows,
            "dataset_count": total_dataset_count,
            "source_shards": source_shards,
            "shard_count": len(manifests),
            "shard_runs": shard_runs,
            "merge_policy": "video directories are disjoint symlinks; no model inference is repeated",
        }
    )
    manifest_path = output_root / "FLEXHOOK_EVAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from eval_flexhook_refersports import run_trackeval

    run_trackeval(manifest=first, output_root=output_root)
    return json.loads(manifest_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sequence-list", type=Path, default=DEFAULT_SEQUENCE_LIST)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = merge_shards(
        base_root=args.base_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        sequence_list=args.sequence_list.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    track = result.get("trackeval_results", {})
    print(json.dumps({"status": result.get("status"), "sequence_count": result.get("sequence_count"), "trackeval_results": track}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
