#!/usr/bin/env python3
"""Run a bounded FlexHook SportsMOT data-loader smoke test.

The smoke test imports only the pinned FlexHook data loader.  It constructs the
train and validation datasets, reads one item from each, and reports tensor
shapes.  It never builds a model, imports CUDA kernels, starts a distributed
job, or writes tracker outputs.  ``--tokenizer stub`` is an explicit structural
test for machines that do not yet have the official RoBERTa files; use
``--tokenizer local --roberta-dir ...`` after the verified asset arrives.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    from scripts.transformers_compat import install_legacy_tokenizer_alias
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    from transformers_compat import install_legacy_tokenizer_alias


def resolve_layout_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    required = (
        root / "manifest.json",
        root / "sequences" / "train" / "SportsMOT_labels.json",
        root / "sequences" / "val" / "SportsMOT_labels.json",
        root / "annotations_v1" / "train" / "SportsMOT",
        root / "annotations_v1" / "val" / "SportsMOT",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("FlexHook layout is incomplete: " + ", ".join(missing))
    return root


def build_options(layout_root: Path) -> SimpleNamespace:
    """Create the finite option object consumed by ``RMOT_Dataset_mix``."""

    return SimpleNamespace(
        DATA=SimpleNamespace(DATASET="SportsMOT", TRAIN="SportsMOT", TEST="SportsMOT"),
        TEXT="roberta",
        ROBERTA_PATH="<stub-tokenizer>",
        text_len=25,
        convert=False,
        noise2=True,
        data_root=str(layout_root),
        val_data_root=str(layout_root),
        track_root="",
        sample_frame_num=4,
        sample_frame_len=8,
        sample_frame_stride=30,
        sample_expression_num=8,
        sample_alpha=2,
        val_set_scale=5,
        val_sample_frame_stride=24,
        val_sample_expression_num=8,
        noise1=True,
        gauss_sample=False,
        img_hw=[224, 672],
        norm_mean=[0.48145466, 0.4578275, 0.40821073],
        norm_std=[0.26862954, 0.26130258, 0.27577711],
    )


class StubTokenizer:
    """Shape-compatible tokenizer used only for the no-weights smoke path."""

    @classmethod
    def from_pretrained(cls, path: str) -> "StubTokenizer":
        return cls()

    def batch_encode_plus(
        self,
        texts: list[str],
        *,
        padding: str,
        return_tensors: str,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        shape = (len(texts), max_length)
        return {
            "input_ids": torch.ones(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def run_smoke(layout_root: Path, tokenizer_mode: str, roberta_dir: str | None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    upstream_root = repo_root / "third_party" / "FlexHook"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    compat_alias_installed = install_legacy_tokenizer_alias()

    import data.mydataloader_mix as mix  # imported after the path is installed

    options = build_options(layout_root)
    if tokenizer_mode == "stub":
        mix.RobertaTokenizerFast = StubTokenizer
    else:
        if not roberta_dir:
            raise ValueError("--roberta-dir is required when --tokenizer local is selected")
        path = Path(roberta_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"RoBERTa directory does not exist: {path}")
        options.ROBERTA_PATH = str(path)

    random.seed(42)
    torch.manual_seed(42)
    result: dict[str, Any] = {
        "policy": "DATA_LOADER_ONLY; NO_MODEL; NO_GT_AS_TRACKER_INPUT",
        "tokenizer_mode": tokenizer_mode,
        "layout_root": str(layout_root),
        "transformers_legacy_tokenizer_alias": compat_alias_installed,
    }
    for mode in ("train", "val"):
        dataset = mix.RMOT_Dataset_mix(mode=mode, opt=options)
        if not dataset.data_keys:
            raise RuntimeError(f"FlexHook {mode} dataset has no sample keys")
        item = dataset[0]
        result[mode] = {
            "sample_count": len(dataset.data_keys),
            "first_item_shapes": [_shape(value) for value in item[:6]],
            "first_key": dataset.data_keys[0],
        }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--tokenizer", choices=("stub", "local"), default="stub")
    parser.add_argument("--roberta-dir", default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(
        resolve_layout_root(args.layout_root),
        tokenizer_mode=args.tokenizer,
        roberta_dir=args.roberta_dir,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for mode in ("train", "val"):
            print(f"{mode}: {json.dumps(result[mode], sort_keys=True)}")
        print("flexhook_sportsmot_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
