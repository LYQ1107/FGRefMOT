#!/usr/bin/env python3
"""Build a read-only efficiency report from tracker and eval manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.efficiency import (
    combine_efficiency,
    load_json,
    referring_efficiency,
    tracker_efficiency,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--timing-file", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tracker_manifest = load_json(Path(args.tracker_manifest).expanduser().resolve())
    eval_manifest = load_json(Path(args.eval_manifest).expanduser().resolve())
    timing = None
    if args.timing_file:
        timing = load_json(Path(args.timing_file).expanduser().resolve())
    elif isinstance(eval_manifest.get("inference"), dict):
        candidate = eval_manifest["inference"].get("timing")
        if isinstance(candidate, dict):
            timing = candidate
    report = combine_efficiency(
        tracker_efficiency(tracker_manifest),
        referring_efficiency(eval_manifest, timing=timing),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
