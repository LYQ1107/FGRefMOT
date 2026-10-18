#!/usr/bin/env python3
"""Run a tiny TAM-only optimization smoke from a restored Mamba checkpoint.

The detector/backbone/decoder are frozen.  Only the newly initialized TAM-v2
parameters receive gradients on a real training sample.  This is deliberately
not a benchmark or a claim of useful training; it proves that the current
training loss reaches the new referring path without silently accepting
checkpoint shape mismatches.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaModel as HFModel
from transformers import RobertaTokenizerFast as HFTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_ROOT = (
    "/home/liuyeqiang/.cache/huggingface/hub/models--roberta-base/"
    "snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)
DATA_ROOT = Path(
    "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti"
)
CHECKPOINT = Path("/data1/liuyeqiang/fgref_v2_runs/checkpoints/checkpoint0089.pth")


class _OfflineTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFTokenizer.from_pretrained(HF_ROOT, local_files_only=True)


class _OfflineModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFModel.from_pretrained(HF_ROOT, local_files_only=True)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data1/liuyeqiang/fgref_v2_runs/tam_smoke/tam_seed17.pt"),
    )
    return parser.parse_args()


def _load_base_checkpoint(model: torch.nn.Module, checkpoint: dict) -> dict:
    checkpoint_state = checkpoint["model"]
    model_state = model.state_dict()
    compatible = {}
    unexpected = []
    mismatches = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(model_state[key].shape) != tuple(value.shape):
            mismatches.append(
                {
                    "key": key,
                    "checkpoint": tuple(value.shape),
                    "model": tuple(model_state[key].shape),
                }
            )
        else:
            compatible[key] = value
    if unexpected or mismatches:
        raise RuntimeError(
            "base checkpoint is not compatible with current source: "
            f"unexpected={unexpected[:5]} mismatches={mismatches[:5]}"
        )
    missing = sorted(set(model_state) - set(compatible))
    allowed = sorted(key for key in missing if key.startswith("fgref_v2_tam."))
    if missing != allowed:
        raise RuntimeError(
            "only newly initialized TAM keys may be missing; "
            f"missing_count={len(missing)} non_tam={sorted(set(missing) - set(allowed))[:10]}"
        )
    result = model.load_state_dict(compatible, strict=False)
    if sorted(result.missing_keys) != missing or result.unexpected_keys:
        raise RuntimeError(
            f"unexpected load report: missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    return {"missing_tam_keys": missing, "checkpoint_model_keys": len(checkpoint_state)}


def main() -> int:
    args = _args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(REPO_ROOT))

    import models.backbone as backbone  # noqa: WPS433
    import models.transrmot_pro as transrmot  # noqa: WPS433
    from datasets.refer_kitti import build as build_dataset  # noqa: WPS433
    from main import get_args_parser  # noqa: WPS433
    from models import build_model  # noqa: WPS433

    backbone.is_main_process = lambda: False
    transrmot.RobertaTokenizerFast = _OfflineTokenizer
    transrmot.RobertaModel = _OfflineModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args")
    model_args = get_args_parser().parse_args([])
    model_args.meta_arch = "temp_rmot"
    model_args.dataset_file = "e2e_rmot"
    model_args.device = args.device
    model_args.with_box_refine = True
    model_args.sampler_lengths = [4, 4, 4, 4]
    model_args.sampler_steps = [60, 80, 90]
    model_args.sample_mode = "fixed_interval"
    model_args.sample_interval = 1
    model_args.rmot_path = str(args.data_root)
    model_args.data_txt_path_train = str(args.data_root / "KITTI/training/refer-kitti.train")
    model_args.vis = False
    model_args.distributed = False
    model_args.rank = 0
    model_args.world_size = 1
    model_args.gpu = 0
    model_args.use_checkpoint = False
    # The checkpoint's QIM namespace depends on this historical flag.  Keep
    # it explicit so a missing positional-update branch cannot be hidden by a
    # permissive load.
    model_args.update_query_pos = bool(
        getattr(checkpoint_args, "update_query_pos", model_args.update_query_pos)
    )
    model_args.merger_dropout = float(
        getattr(checkpoint_args, "merger_dropout", model_args.merger_dropout)
    )
    model_args.fgref_v2_tam = True
    model_args.fgref_v2_tam_local_points = 4
    model_args.fgref_v2_tam_context_points = 8
    model_args.fgref_v2_tam_conditional = False
    model_args.fgref_v2_tam_heads = 8
    model_args.fgref_v2_qim = False

    model, _, _ = build_model(model_args)
    load_report = _load_base_checkpoint(model, checkpoint)
    device = torch.device(args.device)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fgref_v2_tam.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    data = build_dataset("train", model_args)[0]
    data["imgs"] = [frame.to(device, non_blocking=True) for frame in data["imgs"]]
    data["gt_instances"] = [instance.to(device) for instance in data["gt_instances"]]

    model.train()
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        losses = output["losses_dict"]
        differentiable = [value for value in losses.values() if value.requires_grad]
        if not differentiable:
            raise RuntimeError("no loss term reaches TAM-v2 parameters")
        total = torch.stack([value.float() for value in differentiable]).sum()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1e9)
        optimizer.step()
        refer_terms = {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if "loss_refer" in key
        }
        history.append(
            {
                "step": step,
                "total_differentiable_loss": float(total.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "refer_loss_terms": refer_terms,
                "loss_count": len(losses),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.fgref_v2_tam.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.steps,
            "seed": args.seed,
            "source_checkpoint": str(args.checkpoint),
            "load_report": load_report,
            "history": history,
        },
        args.output,
    )
    print("TAM_CHECKPOINT_TRAIN_SMOKE_OK")
    print(json.dumps({"load": load_report, "history": history, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
