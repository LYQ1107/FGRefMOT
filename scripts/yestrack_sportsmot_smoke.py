#!/usr/bin/env python3
"""Smoke-test the YesTrack SportsMOT supervised sample interface.

This test uses a bounded number of real sequences from each generated split,
constructs the upstream single-frame and video datasets, and fetches one sample
from each.  It does not import or load Qwen, LoRA, CUDA model weights, or a
tracker.  The bounded subset keeps the check suitable for preflight while the
full split counts remain recorded by ``yestrack_sportsmot_adapter.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.yestrack_sportsmot_adapter import (
    make_sportsmot_loader_class,
    resolve_layout_roots,
    sequence_names,
)


SINGLE_PROMPT = "The target at <{coord}> matches: {sentence}. Answer Yes or No."
VIDEO_PROMPT = "Clip target <{coord}>: {sentence}. Answer Yes or No."


def select_sequences(root: Path, limit: int) -> list[str]:
    names = sequence_names(root)
    if limit <= 0:
        return names
    return names[:limit]


def _sample_summary(item: dict[str, Any], *, video: bool) -> dict[str, Any]:
    images = item.get("images") if video else [item.get("image")]
    return {
        "frame_count": len(images) if isinstance(images, list) else 0,
        "image_sizes": [list(image.size) for image in images if hasattr(image, "size")],
        "label": int(item["label"]),
        "text_chars": len(str(item["text"])),
        "text_prefix": str(item["text"])[:180],
    }


def run_smoke(layout_root: str | Path, sequence_limit: int = 1) -> dict[str, Any]:
    root = Path(layout_root).expanduser().resolve()
    train_root, val_root = resolve_layout_roots(root)
    upstream_root = Path(__file__).resolve().parents[1] / "third_party" / "YesTrack"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from refer_llm.data.datasets import QwenReferVideoYesNoDataset, QwenReferYesNoDataset

    loader_cls = make_sportsmot_loader_class(train_root, val_root)
    train_names = select_sequences(train_root, sequence_limit)
    val_names = select_sequences(val_root, sequence_limit)
    loader_cls._sports_train_names = train_names
    loader_cls._sports_val_names = val_names

    random.seed(42)
    torch.manual_seed(42)
    result: dict[str, Any] = {
        "policy": "SUPERVISED_SAMPLE_ONLY; NO_QWEN; NO_LORA; NO_TRACKER; NO_GT_AS_TRACKER_INPUT",
        "sequence_limit": sequence_limit,
        "splits": {},
    }
    for split, split_root, names in (
        ("train", train_root, train_names),
        ("val", val_root, val_names),
    ):
        base_ds = loader_cls(
            data_root=str(split_root),
            split=split,
            load_annotation=True,
            expression_sub_dir="expression",
            labels_with_ids_sub_dir="labels_with_ids/image_02",
        )
        single_ds = QwenReferYesNoDataset(
            refer_dataset=base_ds,
            image_size=384,
            margin_ratio=0.2,
            min_side=8,
            max_text_len=100,
            negative_downsample=1.0,
            coord_mode="xy",
            coord_decimals=3,
            prompt_single_tpl=SINGLE_PROMPT,
        )
        video_ds = QwenReferVideoYesNoDataset(
            refer_dataset=base_ds,
            image_size=384,
            margin_ratio=0.2,
            min_side=8,
            max_text_len=100,
            negative_downsample=1.0,
            video_n_frames=3,
            prompt_video_tpl=VIDEO_PROMPT,
            coord_mode="xy",
            coord_decimals=3,
        )
        if len(single_ds) == 0 or len(video_ds) == 0:
            raise RuntimeError(f"empty YesTrack sample dataset for {split}")
        result["splits"][split] = {
            "sequence_names": names,
            "single_sample_count": len(single_ds),
            "video_sample_count": len(video_ds),
            "single_first": _sample_summary(single_ds[0], video=False),
            "video_first": _sample_summary(video_ds[0], video=True),
        }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--sequence-limit", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(args.layout_root, sequence_limit=args.sequence_limit)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for split, payload in result["splits"].items():
            print(f"{split}: {json.dumps(payload, sort_keys=True)}")
        print("yestrack_sportsmot_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
