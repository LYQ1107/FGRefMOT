#!/usr/bin/env python3
"""Aggregate the four fixed-split development reports for one checkpoint.

The evaluator writes one JSON file per QIM variant.  This small, deterministic
utility combines those files only after checking that they describe the same
development protocol and frame set.  It deliberately does not call the model
or infer missing metrics.
"""

import argparse
import json
from pathlib import Path


VARIANTS = ("standard", "lstm", "temporal_attention", "mamba")
MEAN_FIELDS = (
    "hota",
    "hota_05",
    "deta",
    "assa",
    "idf1",
    "idsw_sum",
    "idsw_mean_per_sequence",
    "referring_f1",
)


def load_report(report_dir: Path, variant: str, step: int) -> dict:
    path = report_dir / f"{variant}_step{step}_dev_full.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing report: {path}")
    report = json.loads(path.read_text())
    if report.get("qim_variant") != variant:
        raise ValueError(f"{path}: qim_variant does not match filename")
    mean = report.get("mean")
    if not isinstance(mean, dict):
        raise ValueError(f"{path}: missing mean metrics")
    missing = [field for field in MEAN_FIELDS if field not in mean]
    if missing:
        raise ValueError(f"{path}: missing mean fields: {', '.join(missing)}")
    sequences = report.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"{path}: missing sequence results")
    videos = [item.get("video") for item in sequences]
    if any(video is None for video in videos):
        raise ValueError(f"{path}: a sequence has no video name")
    return {
        "source": path.name,
        "checkpoint": report.get("checkpoint"),
        "mean": {field: mean[field] for field in MEAN_FIELDS},
        "num_sequences": report.get("num_sequences", len(sequences)),
        "frames": sum(int(item.get("frames", 0)) for item in sequences),
        "videos": videos,
    }, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("reports/qim_ablation"))
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    variants = {}
    protocol = None
    video_list = None
    reference = None
    for variant in VARIANTS:
        compact, raw = load_report(args.report_dir, variant, args.step)
        if reference is None:
            reference = compact
            protocol = raw.get("protocol")
            video_list = raw.get("video_list")
        else:
            for key in ("num_sequences", "frames", "videos"):
                if compact[key] != reference[key]:
                    raise ValueError(
                        f"{variant}: {key} differs from the first variant; "
                        "refusing to mix protocols"
                    )
            if raw.get("protocol") != protocol or raw.get("video_list") != video_list:
                raise ValueError(
                    f"{variant}: protocol or video_list differs; refusing to mix reports"
                )
        variants[variant] = compact

    summary = {
        "step": args.step,
        "protocol": protocol,
        "video_list": video_list,
        "num_sequences": reference["num_sequences"],
        "frames": reference["frames"],
        "variants": variants,
    }
    output = args.output or args.report_dir / f"step{args.step}_dev_summary_20261019.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
