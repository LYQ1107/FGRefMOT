#!/usr/bin/env python3
"""Build lightweight, symlink-only YesTrack evaluation shards.

The upstream YesTrack evaluator discovers every sequence below ``data_root``.
For a wall-clock speedup we give independent processes disjoint sequence
subtrees.  Images and expression JSON files remain symlinks to the audited
Refer-Sports validation layout; no labels or pixels are copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_sequences(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError("sequence list must contain unique non-empty names")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--sequence-list", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sequences = read_sequences(args.sequence_list.expanduser().resolve())
    source_manifest_path = source_root.parent / "manifest.json"
    if not source_manifest_path.is_file():
        source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    metadata_by_video = {
        record["video_id"]: record
        for record in source_manifest.get("metadata", [])
        if record.get("yestrack_split") == "val"
    }
    missing = sorted(set(sequences) - set(metadata_by_video))
    if missing:
        raise ValueError(f"source validation layout does not cover: {missing}")
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shard_sequences: list[list[str]] = [[] for _ in range(args.shards)]
    for index, sequence in enumerate(sequences):
        shard_sequences[index % args.shards].append(sequence)

    for shard_index, names in enumerate(shard_sequences):
        shard_root = output_root / f"val_{shard_index:02d}"
        if shard_root.exists() and args.overwrite:
            shutil.rmtree(shard_root)
        image_root = shard_root / "KITTI" / "training" / "image_02"
        expression_root = shard_root / "expression"
        image_root.mkdir(parents=True, exist_ok=True)
        expression_root.mkdir(parents=True, exist_ok=True)
        for sequence in names:
            image_source = source_root / "KITTI" / "training" / "image_02" / sequence
            expression_source = source_root / "expression" / sequence
            if not image_source.is_dir() or not expression_source.is_dir():
                raise FileNotFoundError(f"missing source sequence tree for {sequence}")
            (image_root / sequence).symlink_to(image_source)
            (expression_root / sequence).symlink_to(expression_source)
        metadata = [metadata_by_video[sequence] for sequence in names]
        manifest = {
            "schema_version": "yestrack-sports-eval-shard-0.1",
            "source_root": str(source_root),
            "source_manifest": str(source_manifest_path),
            "sequence_count": len(names),
            "sequence_names": names,
            "metadata": metadata,
            "symlink_only": True,
        }
        (shard_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (shard_root / "sequence_list.txt").write_text(
            "\n".join(names) + ("\n" if names else ""), encoding="utf-8"
        )
    summary = {
        "output_root": str(output_root),
        "shards": args.shards,
        "sequence_count": len(sequences),
        "shard_sizes": [len(names) for names in shard_sequences],
        "symlink_only": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
