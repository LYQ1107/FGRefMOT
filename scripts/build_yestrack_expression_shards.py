#!/usr/bin/env python3
"""Split Refer-Sports expressions into symlink-only YesTrack shards.

The upstream ByteTrack mode enumerates every tracker file and every expression
under each sequence.  Splitting at expression level lets several workers share
one A100 while keeping each worker's data and tracker directory self-contained.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def read_sequences(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value and not value.startswith("#")]
    if not values or len(values) != len(set(values)):
        raise ValueError("sequence list must contain unique non-empty names")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--sequence-list", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    sequence_list = args.sequence_list.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sequences = read_sequences(sequence_list)
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    source_manifest_path = source_root.parent / "manifest.json"
    if not source_manifest_path.is_file():
        source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    metadata_by_video = {
        record["video_id"]: record
        for record in source_manifest.get("metadata", [])
        if record.get("video_id") in set(sequences)
    }
    missing = sorted(set(sequences) - set(metadata_by_video))
    if missing:
        raise ValueError(f"source validation layout does not cover: {missing}")

    units: list[tuple[str, Path]] = []
    for sequence in sequences:
        expression_dir = source_root / "expression" / sequence
        if not expression_dir.is_dir():
            raise FileNotFoundError(expression_dir)
        files = sorted(expression_dir.glob("*.json"))
        if not files:
            raise ValueError(f"no expressions for {sequence}")
        units.extend((sequence, path) for path in files)

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    shard_units: list[list[tuple[str, Path]]] = [[] for _ in range(args.shards)]
    for index, unit in enumerate(units):
        shard_units[index % args.shards].append(unit)

    for index, selected in enumerate(shard_units):
        shard_root = output_root / f"val_{index:02d}"
        image_root = shard_root / "KITTI" / "training" / "image_02"
        expression_root = shard_root / "expression"
        image_root.mkdir(parents=True, exist_ok=True)
        expression_root.mkdir(parents=True, exist_ok=True)
        by_video: dict[str, list[Path]] = defaultdict(list)
        for sequence, expression_path in selected:
            by_video[sequence].append(expression_path)
        for sequence, expression_paths in by_video.items():
            image_source = source_root / "KITTI" / "training" / "image_02" / sequence
            if not image_source.is_dir():
                raise FileNotFoundError(image_source)
            (image_root / sequence).symlink_to(image_source)
            expression_destination = expression_root / sequence
            expression_destination.mkdir(parents=True, exist_ok=True)
            for expression_path in expression_paths:
                (expression_destination / expression_path.name).symlink_to(expression_path)

        metadata = []
        for sequence, expression_paths in by_video.items():
            item = dict(metadata_by_video[sequence])
            item["expression_count"] = len(expression_paths)
            metadata.append(item)
        names = sorted(by_video)
        manifest = {
            "schema_version": "yestrack-sports-expression-shard-0.1",
            "source_root": str(source_root),
            "source_manifest": str(source_manifest_path),
            "sequence_count": len(names),
            "sequence_names": names,
            "expression_count": len(selected),
            "metadata": metadata,
            "symlink_only": True,
            "split_unit": "expression_json",
        }
        (shard_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (shard_root / "sequence_list.txt").write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")

    summary = {
        "output_root": str(output_root),
        "shards": args.shards,
        "sequence_count": len(sequences),
        "expression_count": len(units),
        "shard_sizes": [len(items) for items in shard_units],
        "symlink_only": True,
        "split_unit": "expression_json",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
