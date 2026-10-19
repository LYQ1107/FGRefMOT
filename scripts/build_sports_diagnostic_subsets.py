#!/usr/bin/env python3
"""Register auditable Refer-Sports diagnostic subset candidates.

The source manifest contains positive target intervals and explicitly marks
negative/outside-interval states as UNKNOWN.  This script only selects
expression-level candidates from the sentence and target-set metadata; it
never invents frame labels or computes a metric.  Every subset therefore
retains a label-sufficiency warning until a metric-specific annotation audit
confirms that its negatives and event boundaries are usable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


IDENTITY_RE = re.compile(r"\b(number|jerseys?|uniforms?|sportswear|shirt|blue|red|white|black|yellow|purple|green)\b")
DYNAMIC_RE = re.compile(r"\b(holding|serving|standing|running|walking|left|right|back court|front court|near|behind|front)\b")
GENERIC_GROUP_RE = re.compile(r"^(basketball players|volleyball players|soccer players|soccer goalkeepers)$")


def subset_names(record: dict[str, Any]) -> list[str]:
    sentence = str(record.get("sentence", "")).strip().lower()
    names: list[str] = []
    if IDENTITY_RE.search(sentence):
        names.append("S1_SAME_UNIFORM_IDENTITY_CANDIDATE")
    if DYNAMIC_RE.search(sentence):
        names.append("S2_TEMPORALLY_VALID_REFERRING_CANDIDATE")
    if GENERIC_GROUP_RE.fullmatch(sentence) or len(record.get("target_track_ids", [])) > 1:
        names.append("S3_GROUP_AND_SUBSET_CANDIDATE")
    return names


def build_payload(records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [record for record in records if split == "all" or record.get("original_split") == split]
    groups: dict[str, list[str]] = {}
    for record in selected:
        expression_id = str(record.get("expression_id"))
        for name in subset_names(record):
            groups.setdefault(name, []).append(expression_id)
    counts = Counter(record.get("annotation_status", "UNKNOWN") for record in selected)
    return {
        "schema_version": "rmot-sports-diagnostic-subsets-0.1",
        "source_manifest": "data/refer_sports_manifest.json",
        "split": split,
        "record_count": len(selected),
        "annotation_status_counts": dict(sorted(counts.items())),
        "subsets": {
            name: {
                "expression_count": len(expression_ids),
                "expression_ids": sorted(expression_ids),
                "label_sufficiency": "INSUFFICIENT_LABELS_UNTIL_NEGATIVE_AND_EVENT_AUDIT",
            }
            for name, expression_ids in sorted(groups.items())
        },
        "S4_OCCLUSION_ID_PERSISTENCE": {
            "expression_count": 0,
            "expression_ids": [],
            "label_sufficiency": "INSUFFICIENT_LABELS_NO_EXPLICIT_OCCLUSION_ANNOTATION",
        },
        "metric_policy": "NO_METRICS_COMPUTED; TEST_GT_NOT_USED_FOR_SELECTION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/refer_sports_manifest.json")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest records must be a list")
    result = build_payload(records, args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "record_count": result["record_count"], "subsets": {k: v["expression_count"] for k, v in result["subsets"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
