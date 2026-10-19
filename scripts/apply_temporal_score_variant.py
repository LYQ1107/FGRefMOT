#!/usr/bin/env python3
"""Apply a causal score-level temporal ablation to MOT prediction files.

This utility operates only on model prediction files.  It never reads GT or
labels, and it keeps each sequence/expression directory in its own namespace.
It is intended for the first YB2--YB6 post-inference controls while a learned
temporal head remains a separate experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapters.temporal_score_control import (
    ALL_VARIANTS,
    ScoreObservation,
    TemporalScoreController,
)


def _parse_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            raise ValueError(f"prediction row has fewer than seven fields: {path}")
        rows.append(fields)
    return rows


def _process_file(
    *,
    source: Path,
    target: Path,
    namespace: str,
    controller: TemporalScoreController,
    min_score: float | None,
) -> dict[str, Any]:
    rows = _parse_rows(source)
    by_frame: dict[int, list[tuple[int, list[str]]]] = defaultdict(list)
    for index, fields in enumerate(rows):
        frame = int(float(fields[0]))
        track_id = int(float(fields[1]))
        bbox = tuple(float(value) for value in fields[2:6])
        score = float(fields[6])
        by_frame[frame].append(
            (
                index,
                [
                    str(frame),
                    str(track_id),
                    *[str(value) for value in bbox],
                    str(score),
                ],
            )
        )

    output_fields: dict[int, list[str]] = {}
    reliability_sum = 0.0
    writes = 0
    for frame in sorted(by_frame):
        items = by_frame[frame]
        observations = [
            ScoreObservation(
                frame_index=frame,
                track_id=int(fields[1]),
                bbox=tuple(float(value) for value in fields[2:6]),
                score=float(fields[6]),
            )
            for _, fields in items
        ]
        updates = controller.process_frame(namespace, observations)
        for (index, fields), update in zip(items, updates):
            fields = list(rows[index])
            fields[6] = f"{update.score:.6f}"
            output_fields[index] = fields
            reliability_sum += update.reliability
            writes += int(update.wrote_identity)

    target.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with target.open("w", encoding="utf-8") as handle:
        for index in range(len(rows)):
            fields = output_fields[index]
            if min_score is not None and float(fields[6]) < min_score:
                continue
            handle.write(",".join(fields) + "\n")
            kept += 1
    return {
        "source": str(source),
        "target": str(target),
        "rows": len(rows),
        "kept_rows": kept,
        "mean_reliability": reliability_sum / len(rows) if rows else 0.0,
        "identity_writes": writes,
    }


def apply_variant(
    *,
    input_root: Path,
    output_root: Path,
    variant: str,
    filename: str = "predict_with_conf.txt",
    min_score: float | None = None,
    overwrite: bool = False,
    dynamic_window: int = 4,
    max_verification_interval: int = 8,
    max_track_gap: int = 60,
) -> dict[str, Any]:
    if variant not in ALL_VARIANTS:
        raise ValueError(f"variant must be one of {ALL_VARIANTS}, got {variant}")
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root is non-empty: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(path for path in input_root.rglob(filename) if path.is_file())
    if not sources:
        raise FileNotFoundError(f"no {filename} files under {input_root}")
    controller = TemporalScoreController(
        variant=variant, dynamic_window=dynamic_window,
        max_verification_interval=max_verification_interval,
        max_track_gap=max_track_gap,
    )
    files: list[dict[str, Any]] = []
    for source in sources:
        relative = source.relative_to(input_root)
        files.append(
            _process_file(
                source=source,
                target=output_root / relative,
                namespace=relative.parent.as_posix(),
                controller=controller,
                min_score=min_score,
            )
        )
    manifest = {
        "schema_version": "temporal-score-variant-0.1",
        "status": "SUCCEEDED",
        "variant": variant,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "filename": filename,
        "min_score": min_score,
        "policy": {
            "gt_or_labels_read": False,
            "causal_state": "current_and_past_frames_only",
            "expression_namespace": "relative_prediction_directory",
        },
        "files": files,
    }
    (output_root / "TEMPORAL_SCORE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    parser.add_argument("--filename", default="predict_with_conf.txt")
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--dynamic-window", type=int, default=4)
    parser.add_argument("--max-verification-interval", type=int, default=8)
    parser.add_argument("--max-track-gap", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = apply_variant(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        variant=args.variant,
        filename=args.filename,
        min_score=args.min_score,
        overwrite=args.overwrite,
        dynamic_window=args.dynamic_window,
        max_verification_interval=args.max_verification_interval,
        max_track_gap=args.max_track_gap,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"temporal_score_variant: {manifest['status']}")
        print(f"variant={args.variant} files={len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
