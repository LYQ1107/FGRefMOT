#!/usr/bin/env python3
"""Profile QIM-only latency for the four originals and six fast variants."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.qim_pro import build as build_qim
from models.structures import Instances


VARIANTS = (
    "standard", "lstm", "temporal_attention", "mamba",
    "fast_mamba_spatial", "fast_mamba_temporal", "fast_mamba_streaming",
    "fast_mamba_motion", "fast_mamba_language", "fast_mamba_reliability",
)


def make_args(variant, depth, fast_dim, fast_state, fast_layers, history_len):
    return SimpleNamespace(
        qim_variant=variant,
        qim_depth=depth,
        random_drop=0.0,
        fp_ratio=0.0,
        update_query_pos=True,
        merger_dropout=0.0,
        qim_fast_dim=fast_dim,
        qim_fast_state=fast_state,
        qim_fast_layers=fast_layers,
        qim_fast_history_len=history_len,
        qim_fast_disable_self_attention=False,
        qim_disable_self_attention=False,
        qim_mamba2_enable=False,
    )


def make_instances(number, device, history_len, fast_dim):
    tracks = Instances((800, 1333))
    tracks.ref_pts = torch.zeros(number, 2, device=device)
    tracks.query_pos = torch.randn(number, 512, device=device)
    tracks.output_embedding = torch.randn(number, 256, device=device)
    tracks.obj_idxes = torch.arange(number, dtype=torch.long, device=device)
    tracks.iou = torch.ones(number, device=device)
    tracks.scores = torch.full((number,), 0.8, device=device)
    tracks.pred_boxes = torch.rand(number, 4, device=device)
    # Give temporal variants a real bounded history instead of silently
    # reducing them to a current-token identity path.
    tracks.hist_embeds = torch.randn(number, history_len, 256, device=device)
    tracks.hist_padding_masks = torch.zeros(number, history_len, dtype=torch.bool, device=device)
    return tracks


def measure(module, number, device, warmup, repeats, history_len, fast_dim):
    module.eval()
    samples = [
        {
            "track_instances": make_instances(number, device, history_len, fast_dim),
            "init_track_instances": make_instances(0, device, history_len, fast_dim),
        }
        for _ in range(max(1, warmup + repeats))
    ]
    with torch.no_grad():
        for sample in samples[:warmup]:
            module(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    with torch.no_grad():
        for sample in samples[warmup:]:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                module(sample)
                end.record()
                end.synchronize()
                elapsed.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                module(sample)
                elapsed.append((time.perf_counter() - start) * 1000.0)
    values = torch.tensor(elapsed, dtype=torch.float64)
    report = {
        "mean_ms": float(values.mean()),
        "median_ms": float(values.median()),
        "p95_ms": float(torch.quantile(values, 0.95)),
        "repeats": len(elapsed),
    }
    if device.type == "cuda":
        report["peak_allocated_gb"] = float(torch.cuda.max_memory_allocated(device) / 2**30)
        report["peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(device) / 2**30)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--active-tracks", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--qim-depth", type=int, default=4)
    parser.add_argument("--fast-dim", type=int, default=64)
    parser.add_argument("--fast-state", type=int, default=8)
    parser.add_argument("--fast-layers", type=int, default=1)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the requested profile")
        torch.cuda.set_device(0 if device.index is None else device.index)

    report = {
        "contract": {
            "variants": list(args.variants),
            "active_tracks": args.active_tracks,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "device": str(device),
            "cuda_events": device.type == "cuda",
            "fast_dim": args.fast_dim,
            "fast_state": args.fast_state,
            "fast_layers": args.fast_layers,
            "history_len": args.history_len,
        },
        "variants": {},
    }
    for variant in args.variants:
        module = build_qim(
            make_args(variant, args.qim_depth, args.fast_dim, args.fast_state,
                      args.fast_layers, args.history_len),
            "QIM", 256, 1024, 512,
        ).to(device)
        entry = {
            "qim_params": sum(parameter.numel() for parameter in module.parameters()),
            "active_tracks": {},
        }
        for number in args.active_tracks:
            entry["active_tracks"][str(number)] = measure(
                module, number, device, args.warmup, args.repeats,
                args.history_len, args.fast_dim,
            )
        report["variants"][variant] = entry
        print(json.dumps({variant: entry}, indent=2), flush=True)
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
