#!/usr/bin/env python
"""Measure active-token order sensitivity without retraining."""

import argparse
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.qim_pro import QueryInteractionModule, build as build_qim
from models.structures import Instances


def make_args(variant):
    return SimpleNamespace(
        qim_variant=variant, qim_depth=4, random_drop=0.0,
        fp_ratio=0.0, update_query_pos=True, merger_dropout=0.0,
    )


def make_tracks(number, device):
    result = Instances((800, 1333))
    result.ref_pts = torch.zeros(number, 2, device=device)
    result.query_pos = torch.randn(number, 512, device=device)
    result.output_embedding = torch.randn(number, 256, device=device)
    result.obj_idxes = torch.arange(number, dtype=torch.long, device=device)
    result.iou = torch.ones(number, device=device)
    result.scores = torch.full((number,), 0.8, device=device)
    result.pred_boxes = torch.rand(number, 4, device=device)
    return result


def run(module, track_instances):
    empty = make_tracks(0, track_instances.query_pos.device)
    with torch.no_grad():
        return module({'track_instances': track_instances, 'init_track_instances': empty})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--active-tracks', type=int, default=20)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('permutation diagnostic requires CUDA for original Mamba fused norm')

    variants = ('standard', 'lstm', 'temporal_attention', 'mamba')
    permutations = {
        'original': torch.arange(args.active_tracks),
        'reverse': torch.arange(args.active_tracks - 1, -1, -1),
        'random': torch.randperm(args.active_tracks, generator=torch.Generator().manual_seed(19)),
        'track_id_sorted': torch.arange(args.active_tracks),
    }
    result = {
        'device': str(device),
        'active_tracks': args.active_tracks,
        'checkpoint': args.checkpoint or None,
        'interpretation': 'query-level diagnostic; no next-frame detector was run',
        'variants': {},
    }
    for variant in variants:
        module = build_qim(make_args(variant), 'QIM', 256, 1024, 512).to(device).eval()
        if variant == 'mamba' and args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state = {k[len('track_embed.'):]: v for k, v in checkpoint['model'].items()
                     if k.startswith('track_embed.')}
            module.load_state_dict(state, strict=True)
        base = make_tracks(args.active_tracks, device)
        original = run(module, base).query_pos
        variant_report = {}
        for label, permutation in permutations.items():
            permutation = permutation.to(device)
            reordered = base[permutation]
            inverse = torch.argsort(permutation)
            observed = run(module, reordered).query_pos[inverse]
            delta = (observed - original).abs()
            variant_report[label] = {
                'max_abs_query_delta': float(delta.max()),
                'mean_abs_query_delta': float(delta.mean()),
            }
        result['variants'][variant] = variant_report
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()

