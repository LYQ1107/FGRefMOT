#!/usr/bin/env python3
"""Merge expression-level YesTrack outputs without copying large result trees.

Each worker writes ``shard_XX/results/<video>/<expression>/``.  The final
evaluator expects one result tree, so this script creates symlinks for the
completed expression directories, checks that every audited expression is
present exactly once, and records the worker manifests and timing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sequences(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value and not value.startswith("#")]
    if not values or len(values) != len(set(values)):
        raise ValueError("sequence list must contain unique non-empty names")
    return values


def dirname_from_sentence(sentence: str) -> str:
    value = str(sentence or "").strip().lower()
    value = re.sub(r"[^a-z0-9,]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-") or "expr"


def source_expression_names(
    *, source_root: Path, sequence_names: list[str]
) -> dict[tuple[str, Path], str]:
    manifest_path = source_root.parent / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = set(sequence_names)
    names: dict[tuple[str, Path], str] = {}
    for record in manifest.get("metadata", []):
        video = record.get("video_id")
        if video not in wanted:
            continue
        expression_dir = source_root / "expression" / str(video)
        files = sorted(expression_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no expression files for {video}: {expression_dir}")
        used: dict[str, int] = {}
        for path in files:
            record_data = json.loads(path.read_text(encoding="utf-8"))
            base = dirname_from_sentence(record_data.get("sentence", ""))
            count = used.get(base, 0)
            used[base] = count + 1
            name = base if count == 0 else f"{base}__{count + 1}"
            names[(str(video), path.resolve())] = name
    missing_videos = wanted - {video for video, _ in names}
    if missing_videos:
        raise ValueError(f"source layout does not cover requested videos: {sorted(missing_videos)}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path, help="full YesTrack val layout")
    parser.add_argument(
        "--expression-shards-root",
        required=True,
        type=Path,
        help="expression-level shard data roots used by the workers",
    )
    parser.add_argument("--sequence-list", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_root = args.base_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    expression_shards_root = args.expression_shards_root.expanduser().resolve()
    sequence_names = read_sequences(args.sequence_list.expanduser().resolve())
    source_names = source_expression_names(source_root=source_root, sequence_names=sequence_names)
    expected = {(video, name) for (video, _), name in source_names.items()}

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    results_root = output_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    shard_records: list[dict[str, Any]] = []
    found: set[tuple[str, str]] = set()
    duplicates: list[dict[str, str]] = []
    for shard_root in sorted(base_root.glob("shard_*")):
        if not shard_root.is_dir():
            continue
        record: dict[str, Any] = {"shard_root": str(shard_root), "status": "MISSING"}
        manifest_path = shard_root / "YESTRACK_EVAL_MANIFEST.json"
        if manifest_path.is_file():
            record["manifest_path"] = str(manifest_path)
            record["manifest_sha256"] = sha256_file(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["status"] = manifest.get("status")
            record["sequence_count"] = manifest.get("sequence_count")
            record["expression_count"] = manifest.get("evaluations", [{}])[0].get("expression_count") if manifest.get("evaluations") else None
        worker_results = shard_root / "results"
        linked = 0
        if worker_results.is_dir():
            for video_root in sorted(path for path in worker_results.iterdir() if path.is_dir()):
                shard_number = shard_root.name.split("_")[-1]
                shard_expression_root = expression_shards_root / f"val_{shard_number}" / "expression" / video_root.name
                selected_json = sorted(shard_expression_root.glob("*.json"))
                selected_names: dict[str, list[str]] = {}
                for json_path in selected_json:
                    stable = source_names.get((video_root.name, json_path.resolve()))
                    if stable is None:
                        raise RuntimeError(f"worker expression is not in source layout: {json_path}")
                    selected_names.setdefault(dirname_from_sentence(json.loads(json_path.read_text(encoding="utf-8")).get("sentence", "")), []).append(stable)
                for expression_root in sorted(path for path in video_root.iterdir() if path.is_dir()):
                    base_name = expression_root.name.split("__", 1)[0]
                    candidates = selected_names.get(base_name, [])
                    if not candidates:
                        raise RuntimeError(f"cannot map worker expression directory {expression_root}")
                    if expression_root.name == base_name:
                        target_name = candidates.pop(0)
                    else:
                        try:
                            occurrence = int(expression_root.name.rsplit("__", 1)[1]) - 1
                        except ValueError as exc:
                            raise RuntimeError(f"invalid duplicate expression directory: {expression_root}") from exc
                        if occurrence < 0 or occurrence >= len(candidates):
                            raise RuntimeError(f"duplicate expression occurrence out of range: {expression_root}")
                        target_name = candidates.pop(occurrence)
                    key = (video_root.name, target_name)
                    if key in found:
                        duplicates.append({"video_id": key[0], "expression": key[1], "source": str(expression_root)})
                        continue
                    target_video = results_root / video_root.name
                    target_video.mkdir(parents=True, exist_ok=True)
                    target = target_video / target_name
                    target.symlink_to(expression_root.resolve())
                    found.add(key)
                    linked += 1
        record["linked_expression_count"] = linked
        timing_path = worker_results / "overall_infer_time.json"
        if timing_path.is_file():
            record["timing_path"] = str(timing_path)
            record["timing"] = json.loads(timing_path.read_text(encoding="utf-8"))
        shard_records.append(record)

    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if duplicates or missing or unexpected:
        raise RuntimeError(
            json.dumps(
                {"duplicates": duplicates, "missing": missing, "unexpected": unexpected},
                indent=2,
                sort_keys=True,
            )
        )

    payload: dict[str, Any] = {
        "schema_version": "yestrack-expression-shard-merge-0.1",
        "status": "SUCCEEDED",
        "base_root": str(base_root),
        "output_root": str(output_root),
        "results_root": str(results_root),
        "sequence_names": sequence_names,
        "sequence_count": len(sequence_names),
        "expression_count": len(found),
        "expected_expression_count": len(expected),
        "prediction_files": len(list(results_root.glob("*/*/predict_with_conf.txt"))),
        "symlink_only": True,
        "shards": shard_records,
    }
    manifest_path = output_root / "YESTRACK_SHARD_MERGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
