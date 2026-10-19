#!/usr/bin/env python3
"""Measure real-image FGRefMOT inference on a fixed SportsMOT clip.

This is an engineering benchmark, not an accuracy evaluator.  It uses the
same local image tree and validation transform as the development evaluator,
keeps frames in temporal order, and reports both prepared-GPU model latency
and wall-clock latency including image decode/transform.
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.refer_sports import make_transforms_for_sports
from main import get_args_parser
from models import build_model


VARIANTS = ('standard', 'lstm', 'temporal_attention', 'mamba')


def load_model(checkpoint_path, variant, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = copy.deepcopy(checkpoint['args'])
    args.qim_variant = variant
    args.device = str(device)
    args.pretrained = None
    args.resume = ''
    args.use_checkpoint = False
    model, _, _ = build_model(args)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected = [key for key in unexpected
                  if not key.endswith('total_params') and not key.endswith('total_ops')]
    if missing or unexpected:
        raise RuntimeError(
            f'{variant} checkpoint mismatch: missing={missing}, unexpected={unexpected}'
        )
    model.to(device).eval()
    return model


def benchmark_variant(model, image_paths, sentence, transform, device, warmup_frames):
    model.clear()
    track_instances = None
    prepared_ms = []
    wall_ms = []
    active_counts = []
    for index, image_path in enumerate(image_paths):
        wall_start = time.perf_counter()
        with Image.open(image_path) as image_handle:
            image = image_handle.convert('RGB')
        raw_w, raw_h = image.size
        images, _ = transform([image], [None])
        image_tensor = images[0].to(device)
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = model.inference_single_image(
                [image_tensor], [sentence], (raw_h, raw_w), track_instances
            )
            end_event.record()
            end_event.synchronize()
            elapsed_prepared = float(start_event.elapsed_time(end_event))
        else:
            prepared_start = time.perf_counter()
            result = model.inference_single_image(
                [image_tensor], [sentence], (raw_h, raw_w), track_instances
            )
            elapsed_prepared = (time.perf_counter() - prepared_start) * 1000.0
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        track_instances = result['track_instances']
        active_counts.append(int((track_instances.obj_idxes >= 0).sum().item()))
        if index >= warmup_frames:
            prepared_ms.append(elapsed_prepared)

    prepared = np.asarray(prepared_ms, dtype=np.float64)
    wall = np.asarray(wall_ms[warmup_frames:], dtype=np.float64)
    result = {
        'frames_total': len(image_paths),
        'warmup_frames': warmup_frames,
        'timed_frames': len(prepared_ms),
        'prepared_gpu_mean_ms': float(prepared.mean()),
        'prepared_gpu_median_ms': float(np.median(prepared)),
        'prepared_gpu_p95_ms': float(np.percentile(prepared, 95)),
        'prepared_gpu_fps': float(1000.0 / prepared.mean()),
        'wall_mean_ms': float(wall.mean()),
        'wall_median_ms': float(np.median(wall)),
        'wall_p95_ms': float(np.percentile(wall, 95)),
        'wall_fps': float(1000.0 / wall.mean()),
        'active_tracks_per_frame': active_counts,
    }
    if device.type == 'cuda':
        result['peak_allocated_gb'] = float(torch.cuda.max_memory_allocated(device) / 2**30)
        result['peak_reserved_gb'] = float(torch.cuda.max_memory_reserved(device) / 2**30)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-root', required=True)
    parser.add_argument('--sportsmot-root', default='/data/DATASETS/TRACKING/JDE/SportsMOT/sportsmot_publish')
    parser.add_argument('--expression-root', default='/data1/liuyeqiang/SportsMOT/expression')
    parser.add_argument('--video', required=True)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--warmup-frames', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the requested benchmark')

    video_dir = Path(args.sportsmot_root) / 'dataset' / 'train' / args.video
    image_paths = sorted((video_dir / 'img1').glob('*'), key=lambda path: int(path.stem))
    image_paths = image_paths[:args.frames]
    if len(image_paths) <= args.warmup_frames:
        raise ValueError('need more frames than warmup-frames')
    expression_paths = sorted((Path(args.expression_root) / args.video).glob('*.json'))
    if not expression_paths:
        raise FileNotFoundError(f'no expression found for {args.video}')
    expression = json.loads(expression_paths[0].read_text())
    transform = make_transforms_for_sports('val')
    report = {
        'protocol': {
            'type': 'real SportsMOT image engineering benchmark; not an accuracy result',
            'video': args.video,
            'frames': len(image_paths),
            'warmup_frames': args.warmup_frames,
            'sentence': expression['sentence'],
            'resolution_source': 'native image files with validation transform',
            'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
            'precision': 'fp32',
        },
        'variants': {},
    }
    for variant in VARIANTS:
        checkpoint_path = Path(args.checkpoint_root) / variant / 'checkpoint_step00000200.pth'
        print('building', variant, flush=True)
        model = load_model(checkpoint_path, variant, device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        result = benchmark_variant(
            model, image_paths, expression['sentence'], transform, device, args.warmup_frames
        )
        result.update({
            'checkpoint': str(checkpoint_path),
            'qim_params': int(sum(p.numel() for p in model.track_embed.parameters())),
            'full_model_params': int(sum(p.numel() for p in model.parameters())),
        })
        report['variants'][variant] = result
        print(json.dumps({variant: result}, indent=2), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
