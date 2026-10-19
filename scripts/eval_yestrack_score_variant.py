#!/usr/bin/env python3
"""Evaluate a causal YesTrack score variant without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.temporal_score_control import TEMPORAL_VARIANTS
from scripts.apply_temporal_score_variant import apply_variant
from scripts.eval_yestrack_refersports import evaluate_threshold, read_sequence_list, _jsonable


def evaluate_score_variant(
    *,
    input_results_root: Path,
    score_root: Path,
    output_root: Path,
    variant: str,
    data_root: Path,
    gt_manifest_root: Path,
    sequence_list: Path,
    thresholds: list[float],
    overwrite: bool = False,
) -> dict[str, Any]:
    if variant not in TEMPORAL_VARIANTS:
        raise ValueError(f"variant must be one of {TEMPORAL_VARIANTS}, got {variant}")
    for path in (input_results_root, data_root, gt_manifest_root, sequence_list):
        if not path.exists():
            raise FileNotFoundError(path)
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root is non-empty: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    apply_variant(
        input_root=input_results_root,
        output_root=score_root,
        variant=variant,
        filename="predict_with_conf.txt",
        overwrite=overwrite,
    )
    sequences = read_sequence_list(sequence_list)
    evaluations = [
        evaluate_threshold(
            gt_manifest_root=gt_manifest_root,
            data_root=data_root,
            results_root=score_root,
            output_root=output_root,
            sequence_names=sequences,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]
    payload: dict[str, Any] = {
        "schema_version": "yestrack-refersports-score-variant-0.1",
        "status": "SUCCEEDED",
        "variant": variant,
        "input_results_root": str(input_results_root),
        "score_root": str(score_root),
        "output_root": str(output_root),
        "data_root": str(data_root),
        "gt_manifest_root": str(gt_manifest_root),
        "sequence_names": sequences,
        "thresholds": thresholds,
        "evaluations": evaluations,
        "policy": {
            "model_inference": "NOT_RUN; frozen archived YesTrack scores",
            "gt_or_labels_for_score_transform": False,
            "gt_usage": "offline TrackEval metric computation only",
            "future_frames": False,
        },
    }
    (output_root / "YESTRACK_SCORE_EVAL_MANIFEST.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-results-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=TEMPORAL_VARIANTS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--gt-manifest-root", required=True)
    parser.add_argument("--sequence-list", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = evaluate_score_variant(
        input_results_root=Path(args.input_results_root),
        score_root=Path(args.score_root),
        output_root=Path(args.output_root),
        variant=args.variant,
        data_root=Path(args.data_root),
        gt_manifest_root=Path(args.gt_manifest_root),
        sequence_list=Path(args.sequence_list),
        thresholds=[float(value) for value in args.thresholds],
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    else:
        print(f"yestrack_score_variant: {payload['status']}")
        print(f"variant={args.variant} thresholds={len(payload['evaluations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
