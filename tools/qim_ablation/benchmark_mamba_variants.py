#!/usr/bin/env python
"""Benchmark the original and lightweight Mamba-QIM token paths.

This is a QIM-only engineering benchmark. It reports the exact QIM parameter
count plus CUDA mean/median/p95 latency and peak allocation for the requested
active-track counts. Accuracy and full-model image FPS remain separate
measurements and must use the SportsMOT evaluator/real-image benchmark.
"""

import argparse
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.qim_pro import build as build_qim
from models.structures import Instances


VARIANTS = (
    "standard",
    "lstm",
    "temporal_attention",
    "mamba",
    "mamba_ordered_bi",
    "mamba_trackwise",
    "mamba_reliability_hybrid",
)


def make_args(variant, dim, layers, state):
    return SimpleNamespace(
        qim_variant=variant,
        qim_depth=4,
        random_drop=0.0,
        fp_ratio=0.0,
        update_query_pos=True,
        merger_dropout=0.0,
        qim_mamba_dim=dim,
        qim_mamba_layers=layers,
        qim_mamba_state=state,
        qim_mamba_history_len=4,
        qim_mamba2_enable=False,
        qim_bidirectional_share=True,
        qim_disable_self_attention=False,
        qim_reliability_enable=True,
        qim_rmot_text_gate=False,
    )


def make_instances(number, device, variant, memory_dim=128):
    result = Instances((800, 1333))
    result.ref_pts = torch.zeros(number, 2, device=device)
    result.query_pos = torch.randn(number, 512, device=device)
    result.output_embedding = torch.randn(number, 256, device=device)
    result.obj_idxes = torch.arange(number, dtype=torch.long, device=device)
    result.iou = torch.ones(number, device=device)
    result.scores = torch.full((number,), 0.8, device=device)
    result.pred_boxes = torch.rand(number, 4, device=device)
    result.pred_logits = torch.zeros(number, 1, device=device)
    result.pred_refers = torch.zeros(number, 1, device=device)
    if variant in ("mamba_trackwise", "mamba_reliability_hybrid"):
        result.hist_embeds = torch.randn(number, 4, 256, device=device)
        result.hist_padding_masks = torch.zeros(
            number, 4, dtype=torch.bool, device=device
        )
    if variant == "mamba_reliability_hybrid":
        result.qim_long_memory = torch.randn(number, memory_dim, device=device)
        result.qim_long_memory_valid = torch.ones(
            number, dtype=torch.bool, device=device
        )
        result.qim_memory_age = torch.zeros(number, dtype=torch.long, device=device)
        result.qim_previous_box = result.pred_boxes.clone()
        result.qim_memory_write_gate = torch.zeros(number, device=device)
    return result


def measure(module, variant, number, device, warmup, repeats):
    memory_dim = getattr(module, "memory_dim", 128)
    samples = [
        {
            "track_instances": make_instances(number, device, variant, memory_dim),
            "init_track_instances": make_instances(0, device, variant, memory_dim),
        }
        for _ in range(warmup + repeats)
    ]
    module.eval()
    with torch.no_grad():
        for sample in samples[:warmup]:
            module(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    with torch.no_grad():
        for sample in samples[warmup:]:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                module(sample)
                end.record()
                end.synchronize()
                values.append(float(start.elapsed_time(end)))
            else:
                start_time = time.perf_counter()
                module(sample)
                values.append((time.perf_counter() - start_time) * 1000)
    values = torch.tensor(values, dtype=torch.float64)
    entry = {
        "mean_ms": float(values.mean()),
        "median_ms": float(values.median()),
        "p95_ms": float(torch.quantile(values, 0.95)),
        "repeats": len(values),
    }
    if device.type == "cuda":
        entry["peak_allocated_gb"] = float(torch.cuda.max_memory_allocated(device) / 2**30)
        entry["peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(device) / 2**30)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--active-tracks", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--qim-mamba-dim", type=int, default=0)
    parser.add_argument("--qim-mamba-layers", type=int, default=1)
    parser.add_argument("--qim-mamba-state", type=int, default=16)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    result = {
        "contract": {
            "device": str(device),
            "cuda_events": device.type == "cuda",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "active_tracks": args.active_tracks,
            "mamba_backend": "Mamba1:mamba_ssm.modules.mamba_simple.Mamba",
        },
        "variants": {},
    }
    for variant in args.variants:
        module = build_qim(
            make_args(
                variant,
                args.qim_mamba_dim,
                args.qim_mamba_layers,
                args.qim_mamba_state,
            ),
            "QIM",
            256,
            1024,
            512,
        ).to(device)
        entry = {
            "qim_params": sum(parameter.numel() for parameter in module.parameters()),
            "variant_name": getattr(module, "variant_name", variant),
            "active_tracks": {},
        }
        for number in args.active_tracks:
            entry["active_tracks"][str(number)] = measure(
                module, variant, number, device, args.warmup, args.repeats
            )
        result["variants"][variant] = entry
        print(json.dumps({variant: entry}, indent=2))
    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
