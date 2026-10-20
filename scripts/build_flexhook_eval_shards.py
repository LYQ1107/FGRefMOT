#!/usr/bin/env python3
"""Build symlink-only FlexHook validation shards for parallel inference."""

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
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    metadata_by_video = {
        record["video_id"]: record
        for record in source_manifest.get("metadata", [])
        if record.get("flexhook_split") == "val"
    }
    missing = sorted(set(sequences) - set(metadata_by_video))
    if missing:
        raise ValueError(f"source validation layout does not cover: {missing}")
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = json.loads(
        (source_root / "sequences" / "val" / "SportsMOT_labels.json").read_text(encoding="utf-8")
    )
    resolutions = json.loads(
        (source_root / "sequences" / "val" / "SportsMOT_RESOLUTION.json").read_text(encoding="utf-8")
    )
    shard_sequences: list[list[str]] = [[] for _ in range(args.shards)]
    for index, sequence in enumerate(sequences):
        shard_sequences[index % args.shards].append(sequence)

    for shard_index, names in enumerate(shard_sequences):
        shard_root = output_root / f"val_{shard_index:02d}"
        if shard_root.exists() and args.overwrite:
            shutil.rmtree(shard_root)
        sequences_root = shard_root / "sequences"
        annotations_root = shard_root / "annotations_v1"
        (sequences_root / "val" / "SportsMOT-sequences").mkdir(parents=True, exist_ok=True)
        (annotations_root / "val" / "SportsMOT").mkdir(parents=True, exist_ok=True)
        # The upstream loader constructs the train dataset before validation;
        # retain its audited train namespace as symlinks and filter only val.
        (sequences_root / "train").symlink_to(source_root / "sequences" / "train")
        (annotations_root / "train").symlink_to(source_root / "annotations_v1" / "train")
        for sequence in names:
            image_source = source_root / "sequences" / "val" / "SportsMOT-sequences" / sequence
            image_target = sequences_root / "val" / "SportsMOT-sequences" / sequence
            image_target.symlink_to(image_source)
        for expression_path in (source_root / "annotations_v1" / "val" / "SportsMOT").glob("*.json"):
            if any(expression_path.name.startswith(f"{sequence}_") for sequence in names):
                (annotations_root / "val" / "SportsMOT" / expression_path.name).symlink_to(expression_path)
        (sequences_root / "val" / "SportsMOT_labels.json").write_text(
            json.dumps({name: labels[name] for name in names}, sort_keys=True) + "\n", encoding="utf-8"
        )
        (sequences_root / "val" / "SportsMOT_RESOLUTION.json").write_text(
            json.dumps({name: resolutions[name] for name in names}, sort_keys=True) + "\n", encoding="utf-8"
        )
        # ``mydataloader_mix`` opens this auxiliary map even for SportsMOT;
        # keep the audited source file available in every shard namespace.
        (sequences_root / "val" / "TAO_frame2file.json").symlink_to(
            source_root / "sequences" / "val" / "TAO_frame2file.json"
        )
        manifest = {
            "schema_version": "flexhook-sports-eval-shard-0.1",
            "source_root": str(source_root),
            "source_manifest": str(source_manifest_path),
            "sequence_count": len(names),
            "sequence_names": names,
            "metadata": [metadata_by_video[name] for name in names],
            "symlink_only": True,
        }
        (shard_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (shard_root / "sequence_list.txt").write_text(
            "\n".join(names) + ("\n" if names else ""), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "shards": args.shards,
                "sequence_count": len(sequences),
                "shard_sizes": [len(names) for names in shard_sequences],
                "symlink_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
