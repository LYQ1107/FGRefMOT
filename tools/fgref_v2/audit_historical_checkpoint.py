"""Strictly audit the historical TempRMOT checkpoint without writing assets.

The checkpoint was serialized with a RoBERTa configuration/serialization
combination that differs from the currently installed Transformers defaults.
This script reconstructs those details in-process, compares every key and
shape, and then performs a strict load. It intentionally imports the old
source from ``--source-root``; it does not alter the current research model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import RobertaConfig, RobertaModel as HFRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPooler


DEFAULT_SOURCE_ROOT = "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT"
DEFAULT_CHECKPOINT = (
    "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/"
    "exps/official_checkpoint/checkpoint_0060.pth"
)


class _OfflineTokenizer:
    pass


def _build_offline_roberta(*args, **kwargs):
    del args, kwargs
    return HFRobertaModel(
        RobertaConfig(
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=514,
            type_vocab_size=1,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    source_root = str(Path(args.source_root).resolve())
    sys.path.insert(0, source_root)

    # Import the old package only after its source root has priority.
    import models.backbone as backbone  # noqa: E402
    import models.transrmot_pro as transrmot  # noqa: E402

    backbone.is_main_process = lambda: False
    transrmot.RobertaTokenizerFast.from_pretrained = (
        lambda *unused_args, **unused_kwargs: _OfflineTokenizer()
    )
    transrmot.RobertaModel.from_pretrained = _build_offline_roberta

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["args"]
    model_args.device = "cpu"
    model_args.meta_arch = "temp_rmot"
    model_args.distributed = False
    model_args.rank = 0
    model_args.world_size = 1
    model_args.gpu = 0

    model, _, _ = transrmot.build(model_args)

    # The old source removes this unused pooler after construction. The
    # checkpoint proves that its historical run retained it.
    model.text_encoder.pooler = RobertaPooler(model.text_encoder.config)
    # Transformers 4.35 marks this buffer non-persistent; the historical file
    # serialized it, so reproduce that detail for the comparison only.
    model.text_encoder.embeddings._non_persistent_buffers_set.discard(
        "position_ids"
    )

    model_state = model.state_dict()
    checkpoint_state = checkpoint["model"]
    missing = sorted(set(model_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(model_state))
    mismatches = sorted(
        (
            key,
            tuple(model_state[key].shape),
            tuple(checkpoint_state[key].shape),
        )
        for key in set(model_state).intersection(checkpoint_state)
        if model_state[key].shape != checkpoint_state[key].shape
    )

    print(f"model_keys={len(model_state)} checkpoint_keys={len(checkpoint_state)}")
    print(f"missing_keys={len(missing)}")
    print(f"unexpected_keys={len(unexpected)}")
    print(f"shape_mismatches={len(mismatches)}")
    if missing or unexpected or mismatches:
        print("missing:", missing)
        print("unexpected:", unexpected)
        print("shape_mismatches:", mismatches)
        return 2

    model.load_state_dict(checkpoint_state, strict=True)
    print("STRICT_LOAD_OK old_TempRMOT_with_historical_RoBERTa_details")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
