#!/usr/bin/env python
"""Measure full original-FGRefMOT inference with each QIM.

This tool intentionally uses a fixed synthetic tensor probe for engineering
latency only, even though the server now has the local SportsMOT image,
label-with-ID, and expression assets used by the separate accuracy evaluator.
Its output must not be used as a Refer-Sports accuracy result.
"""

import argparse
import copy
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models import build_model
import models.transrmot_pro as transrmot_pro
from models.structures import Instances
from util.misc import nested_tensor_from_tensor_list


VARIANTS = (
    'standard', 'lstm', 'temporal_attention', 'mamba',
    'mamba_ordered_bi', 'mamba_trackwise', 'mamba_reliability_hybrid',
)


class _BatchEncoding(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to(self, device):
        return _BatchEncoding({key: value.to(device) for key, value in self.items()})


class _OfflineTokenizer:
    def batch_encode_plus(self, text_queries, padding='longest', return_tensors='pt'):
        # The checkpoint contains the full RoBERTa weights; this fixed token
        # sequence only avoids an online tokenizer dependency for the latency
        # probe.  No accuracy result is derived from this input.
        batch = len(text_queries)
        length = 8
        return _BatchEncoding({
            'input_ids': torch.ones(batch, length, dtype=torch.long),
            'attention_mask': torch.ones(batch, length, dtype=torch.long),
        })


def _offline_tokenizer_loader(cls, *args, **kwargs):
    return _OfflineTokenizer()


def _offline_model_loader(cls, *args, **kwargs):
    from transformers import RobertaConfig, RobertaModel
    return RobertaModel(RobertaConfig(
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=514,
        type_vocab_size=1,
        pad_token_id=1,
    ))


def _install_offline_roberta():
    transrmot_pro.RobertaTokenizerFast.from_pretrained = classmethod(_offline_tokenizer_loader)
    transrmot_pro.RobertaModel.from_pretrained = classmethod(_offline_model_loader)


def _event_call(fn, device):
    if device.type != 'cuda':
        start = time.perf_counter()
        value = fn()
        return value, (time.perf_counter() - start) * 1000.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, float(start.elapsed_time(end))


def _load_variant(checkpoint, variant, device):
    args = copy.deepcopy(checkpoint['args'])
    args.qim_variant = variant
    args.qim_depth = 4
    args.device = str(device)
    args.pretrained = None
    args.resume = ''
    args.use_checkpoint = False
    model, _, _ = build_model(args)
    source = checkpoint['model']
    if variant == 'mamba':
        missing, unexpected = model.load_state_dict(source, strict=True)
        if missing or unexpected:
            raise RuntimeError(f'original Mamba strict load failed: {missing}, {unexpected}')
    else:
        common = {name: value for name, value in source.items()
                  if not name.startswith('track_embed.')}
        missing, unexpected = model.load_state_dict(common, strict=False)
        bad_missing = [name for name in missing if not name.startswith('track_embed.')]
        bad_unexpected = [name for name in unexpected if not name.startswith('track_embed.')]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f'common checkpoint load failed: {bad_missing}, {bad_unexpected}')
    model.to(device).eval()
    return model


def _run_clip(model, images, sentence, image_size, device, force_active_tracks=0):
    model.clear()
    state = None
    times = []
    active = []
    for image in images:
        if state is not None and force_active_tracks > 0:
            number = min(force_active_tracks, len(state))
            state.obj_idxes[:number] = torch.arange(
                number, dtype=state.obj_idxes.dtype, device=state.obj_idxes.device
            )
            state.scores[:number] = 0.8
            state.iou[:number] = 1.0
        result, elapsed = _event_call(
            lambda image=image, state=state: model.inference_single_image(
                [image], sentence, image_size, state
            ),
            device,
        )
        state = result['track_instances']
        times.append(elapsed)
        active.append(int((state.obj_idxes >= 0).sum().item()))
    return times, active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--frames', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--force-active-tracks', type=int, default=0,
                        help='engineering probe: mark this many returned slots active after frame 1')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('full-model benchmark requires CUDA')
    device = torch.device(args.device)
    _install_offline_roberta()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    images = [torch.rand(3, args.height, args.width, device=device)
              for _ in range(args.frames)]
    sentence = ['a player']

    report = {
        'protocol': {
            'input': 'synthetic fixed tensor; no SportsMOT image files are locally available',
            'resolution': [args.height, args.width],
            'frames_per_clip': args.frames,
            'warmup_clips': args.warmup,
            'timed_clips': args.repeats,
            'force_active_tracks_after_frame1': args.force_active_tracks,
            'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
            'precision': 'fp32',
            'preprocessing': 'prepared GPU tensor and NestedTensor wrapper; image decode/resize excluded',
        },
        'variants': {},
    }
    for variant in args.variants:
        print('building', variant, flush=True)
        model = _load_variant(checkpoint, variant, device)
        for _ in range(args.warmup):
            _run_clip(model, images, sentence, (args.height, args.width), device,
                      args.force_active_tracks)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        clip_times = []
        active_counts = []
        for _ in range(args.repeats):
            elapsed, active = _run_clip(model, images, sentence, (args.height, args.width), device,
                                        args.force_active_tracks)
            clip_times.append(elapsed)
            active_counts.append(active)
        tensor = torch.tensor([value for clip in clip_times for value in clip], dtype=torch.float64)
        clip_tensor = torch.tensor([sum(clip) for clip in clip_times], dtype=torch.float64)
        total_seconds = float(clip_tensor.mean() / 1000.0)
        entry = {
            'qim_label': 'ORIGINAL_MAMBA' if variant == 'mamba' else 'REIMPLEMENTED_VARIANT',
            'qim_params': sum(p.numel() for p in model.track_embed.parameters()),
            'full_model_params': sum(p.numel() for p in model.parameters()),
            'frame_mean_ms': float(tensor.mean()),
            'frame_median_ms': float(tensor.median()),
            'frame_p95_ms': float(torch.quantile(tensor, 0.95)),
            'clip_mean_ms': float(clip_tensor.mean()),
            'fps_prepared_tensor': float(args.frames / total_seconds),
            'active_tracks_per_frame': active_counts,
        }
        if device.type == 'cuda':
            entry['peak_allocated_gb'] = float(torch.cuda.max_memory_allocated(device) / 2**30)
            entry['peak_reserved_gb'] = float(torch.cuda.max_memory_reserved(device) / 2**30)
        report['variants'][variant] = entry
        print(json.dumps({variant: entry}, indent=2), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

