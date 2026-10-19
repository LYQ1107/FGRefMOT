#!/usr/bin/env python
"""Profile the four QIMs on the original active-track token contract.

This is intentionally a QIM-only profile.  It does not convert the result into
full-model FPS.  CUDA events are used whenever CUDA is available; CPU fallback
is labelled explicitly for development-only runs.
"""

import argparse
import json
import math
import pathlib
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.qim_pro import build as build_qim
from models.structures import Instances


VARIANTS = ('standard', 'lstm', 'temporal_attention', 'mamba')


def make_args(variant, depth):
    return SimpleNamespace(
        qim_variant=variant,
        qim_depth=depth,
        random_drop=0.0,
        fp_ratio=0.0,
        update_query_pos=True,
        merger_dropout=0.0,
    )


def make_instances(number, device):
    result = Instances((800, 1333))
    result.ref_pts = torch.zeros(number, 2, device=device)
    result.query_pos = torch.randn(number, 512, device=device)
    result.output_embedding = torch.randn(number, 256, device=device)
    result.obj_idxes = torch.arange(number, dtype=torch.long, device=device)
    result.iou = torch.ones(number, device=device)
    result.scores = torch.full((number,), 0.8, device=device)
    result.pred_boxes = torch.rand(number, 4, device=device)
    return result


def _measure(module, number, device, warmup, repeats):
    module.eval()
    samples = [
        {
            'track_instances': make_instances(number, device),
            'init_track_instances': make_instances(0, device),
        }
        for _ in range(max(1, warmup + repeats))
    ]
    with torch.no_grad():
        for sample in samples[:warmup]:
            module(sample)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms = []
    with torch.no_grad():
        for sample in samples[warmup:]:
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                module(sample)
                end.record()
                end.synchronize()
                elapsed_ms.append(float(start.elapsed_time(end)))
            else:
                start_time = time.perf_counter()
                module(sample)
                elapsed_ms.append((time.perf_counter() - start_time) * 1000.0)

    values = torch.tensor(elapsed_ms, dtype=torch.float64)
    result = {
        'mean_ms': float(values.mean()),
        'median_ms': float(values.median()),
        'p95_ms': float(torch.quantile(values, 0.95)),
        'repeats': len(elapsed_ms),
    }
    if device.type == 'cuda':
        result['peak_allocated_gb'] = float(torch.cuda.max_memory_allocated(device) / 2**30)
        result['peak_reserved_gb'] = float(torch.cuda.max_memory_reserved(device) / 2**30)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--active-tracks', nargs='+', type=int, default=[5, 10, 20, 50, 100])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--qim-depth', type=int, default=4)
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA unavailable; using CPU development fallback')
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    output = {
        'contract': {
            'input': '[1, N_active, 256] for original Mamba; alternatives use the same N_active x 256 token axis',
            'warmup': args.warmup,
            'repeats': args.repeats,
            'device': str(device),
            'cuda_events': device.type == 'cuda',
        },
        'variants': {},
    }
    for variant in args.variants:
        module = build_qim(make_args(variant, args.qim_depth), 'QIM', 256, 1024, 512).to(device)
        entry = {
            'variant_label': 'ORIGINAL_MAMBA' if variant == 'mamba' else 'REIMPLEMENTED_VARIANT',
            'qim_params': sum(parameter.numel() for parameter in module.parameters()),
            'active_tracks': {},
        }
        for number in args.active_tracks:
            entry['active_tracks'][str(number)] = _measure(
                module, number, device, args.warmup, args.repeats
            )
        output['variants'][variant] = entry
        print(json.dumps({variant: entry}, indent=2))

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(output, handle, indent=2)
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

