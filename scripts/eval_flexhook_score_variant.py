#!/usr/bin/env python3
"""Evaluate a causal FlexHook score variant on an existing score archive.

The archive is produced by ``eval_flexhook_refersports.py --save-scores``.
This adapter thresholds only the archived model probabilities, links the
already prepared offline GT files, and runs the same TrackEval configuration
as the formal FlexHook evaluator. It never reads GT to transform scores.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.temporal_score_control import FLEXHOOK_SCORE_VARIANTS
from scripts.apply_temporal_score_variant import apply_variant
from scripts.eval_flexhook_refersports import _write_rows, run_trackeval, sha256_file


def _load_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            raise ValueError(f"score row has fewer than seven fields: {path}")
        rows.append(fields)
    return rows


def prepare_score_eval_tree(
    *,
    score_root: Path,
    archive_root: Path,
    output_root: Path,
    variant: str,
    threshold: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    if variant not in FLEXHOOK_SCORE_VARIANTS:
        raise ValueError(f"variant must be one of {FLEXHOOK_SCORE_VARIANTS}, got {variant}")
    archive_manifest_path = archive_root / "FLEXHOOK_EVAL_MANIFEST.json"
    if not archive_manifest_path.is_file():
        raise FileNotFoundError(f"score archive manifest missing: {archive_manifest_path}")
    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root is non-empty: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = deepcopy(archive_manifest)
    manifest.update(
        {
            "schema_version": "flexhook-refersports-score-variant-0.1",
            "status": "PREDICTIONS_WRITTEN",
            "archive_root": str(archive_root),
            "score_root": str(score_root),
            "output_root": str(output_root),
            "variant": variant,
            "threshold": float(threshold),
            "policy": {
                "gt_or_labels_as_model_input": "REJECTED",
                "model_input": "frozen_archived_model_scores",
                "gt_usage": "offline_trackeval_tree_only",
                "future_frames": "REJECTED",
            },
        }
    )
    prediction_rows = 0
    score_rows = 0
    for record in manifest["records"]:
        video = record["video"]
        expression_dir = record["expression_dir"]
        gt_source = archive_root / video / expression_dir / "gt.txt"
        score_source = score_root / video / expression_dir / "predict_scores.txt"
        if not gt_source.is_file():
            raise FileNotFoundError(f"archive GT missing: {gt_source}")
        if not score_source.is_file():
            raise FileNotFoundError(f"variant score file missing: {score_source}")
        target_dir = output_root / video / expression_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "gt.txt").symlink_to(gt_source.resolve())
        kept: list[list[float | int]] = []
        score_file_rows = _load_rows(score_source)
        for fields in score_file_rows:
            score_rows += 1
            if float(fields[6]) < float(threshold):
                continue
            kept.append(
                [
                    int(float(fields[0])),
                    int(float(fields[1])),
                    *[float(value) for value in fields[2:6]],
                    1,
                    1,
                    1,
                    1,
                ]
            )
        kept.sort(key=lambda row: (int(row[0]), int(row[1])))
        _write_rows(target_dir / "predict.txt", kept)
        record["prediction_rows"] = len(kept)
        record["score_rows"] = len(score_file_rows)
        record["score_source"] = str(score_source)
        record["prediction_path"] = str(target_dir / "predict.txt")
        prediction_rows += len(kept)
    manifest["prediction_rows"] = prediction_rows
    manifest["score_rows"] = score_rows
    manifest_path = output_root / "FLEXHOOK_EVAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", required=True, help="Root containing variant predict_scores.txt files")
    parser.add_argument("--archive-root", required=True, help="Completed --save-scores FlexHook evaluation root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=FLEXHOOK_SCORE_VARIANTS, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_score_eval_tree(
        score_root=Path(args.score_root).expanduser().resolve(),
        archive_root=Path(args.archive_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        variant=args.variant,
        threshold=float(args.threshold),
        overwrite=args.overwrite,
    )
    run_trackeval(manifest=manifest, output_root=Path(args.output_root).expanduser().resolve())
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"flexhook_score_variant: {manifest['status']}")
        print(f"variant={args.variant} prediction_rows={manifest['prediction_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
