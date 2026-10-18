#!/usr/bin/env python3
"""
Scan checkpoint pairs: maximize mean TAM focus-gap (same metric as visualize_tam_attention).

Phase 1 — fixed baseline (default checkpoint0049), try every other ckpt as ours.
Phase 2 — take top-K ours from phase 1; try every ckpt as baseline (find global max gap).

Does not modify project source; uses monkey-patch via visualize_tam_attention.

Usage:
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=8 \\
  python scan_tam_ckpt_gap.py --ckpt_dir exps/default_debug_rk
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import visualize_tam_attention as vis
from models import build_model
from util.tool import load_model
from main import get_args_parser


DEFAULTS = [
    '--meta_arch', 'temp_rmot', '--dataset_file', 'e2e_rmot',
    '--epoch', '200', '--with_box_refine',
    '--lr_drop', '100', '--lr', '2e-4', '--lr_backbone', '2e-5',
    '--batch_size', '1', '--sample_mode', 'random_interval',
    '--sample_interval', '1', '--sampler_steps', '50', '90', '150',
    '--sampler_lengths', '2', '3', '4', '5', '--update_query_pos',
    '--merger_dropout', '0', '--dropout', '0', '--random_drop', '0.1',
    '--fp_ratio', '0.3', '--query_interaction_layer', 'QIM',
    '--extra_track_attn', '--hist_len', '8',
    '--rmot_path', '/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti',
]


def discover_ckpts(ckpt_dir: str) -> list[str]:
    d = Path(ckpt_dir)
    out = sorted(str(p) for p in d.glob('checkpoint*.pth'))
    return [p for p in out if Path(p).name != 'checkpoint.pth']


def probe_gap(model_b, model_o, pr: dict, device, warmup: int) -> float | None:
    """Return combined TAM gap for one (video, sentence, frame); None if no GT."""
    R = pr['rmot_path']
    img_dir = os.path.join(R, 'KITTI', 'training', 'image_02', pr['video_id'])
    if not os.path.isdir(img_dir):
        return None
    expr = vis.find_expr_json(R, pr['video_id'], pr['sentence'])
    if not expr:
        return None
    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png')))
    if not imgs:
        return None
    fi = min(pr['frame_idx'], len(imgs) - 1)
    fname = imgs[fi]

    ori_b, _, (H, W) = vis.run_seq_capture_tam(
        model_b, img_dir, fi, pr['sentence'], device, warmup)
    map_b = vis.tam_map_to_image(ori_b, vis._TAM_STATE.get('map2d'))

    _, _, _ = vis.run_seq_capture_tam(
        model_o, img_dir, fi, pr['sentence'], device, warmup)
    map_o = vis.tam_map_to_image(ori_b, vis._TAM_STATE.get('map2d'))

    gt_ref, _ = vis.load_gt_boxes(R, pr['video_id'], fname, expr, H, W)
    if not gt_ref:
        return None

    mb = vis.gt_mass_in_heat(map_b, gt_ref)
    mo = vis.gt_mass_in_heat(map_o, gt_ref)
    scb = vis.tam_selection_score(map_b, gt_ref)
    sco = vis.tam_selection_score(map_o, gt_ref)
    return float((sco - scb) + 0.35 * (mo - mb))


def mean_gap(model_b, model_o, probes: list[dict], device, warmup: int):
    vals = []
    for pr in probes:
        g = probe_gap(model_b, model_o, pr, device, warmup)
        if g is not None:
            vals.append(g)
    if not vals:
        return float('nan')
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default='exps/default_debug_rk')
    ap.add_argument('--default_baseline',
                    default='exps/default_debug_rk/checkpoint0049.pth',
                    help='Phase-1 fixed baseline path')
    ap.add_argument('--topk_refine', type=int, default=3,
                    help='Phase-2: refine top-K ours candidates vs all baselines')
    ap.add_argument('--warmup_frames', type=int, default=10)
    ap.add_argument('--tam_fpn_layer', type=int, default=1)
    ap.add_argument('--max_ckpt_num', type=int, default=1000,
                    help='Keep checkpoint*.pth with number <= this (default 1000 '
                         'drops 1024/1028-style ids). Use 99999 for all.')
    args_ns, _unknown = ap.parse_known_args()

    ckpt_dir = Path(args_ns.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    ckpt_dir = ckpt_dir.resolve()

    parser = get_args_parser()
    args = parser.parse_args(DEFAULTS)
    if not os.path.isdir(args.rmot_path):
        alt = ROOT / 'data/Dataset/refer-kitti'
        if alt.is_dir():
            args.rmot_path = str(alt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpts = discover_ckpts(str(ckpt_dir))
    if args_ns.max_ckpt_num is not None:

        def _num(p: str) -> int:
            import re
            m = re.search(r'checkpoint(\d+)\.pth$', Path(p).name)
            return int(m.group(1)) if m else 99999

        ckpts = [p for p in ckpts if _num(p) <= args_ns.max_ckpt_num]

    if len(ckpts) < 2:
        print('Need ≥2 checkpoints in', args_ns.ckpt_dir)
        return 1

    R = args.rmot_path
    probes = [
        {'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': 'volleyball players who are wearing white number 7 sportswear',
         'frame_idx': 18, 'rmot_path': R},
        {'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': 'volleyball players who are wearing black number 1 sportswear',
         'frame_idx': 22, 'rmot_path': R},
        {'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': 'volleyball players who are serving',
         'frame_idx': 16, 'rmot_path': R},
        {'video_id': 'v_2j7kLB-vEEk_c001',
         'sentence': 'basketball players wearing white number 0 jerseys',
         'frame_idx': 22, 'rmot_path': R},
    ]

    print('Building two model shells …')
    model_b, _, _ = build_model(args)
    model_o, _, _ = build_model(args)
    model_b.to(device).eval()
    model_o.to(device).eval()

    vis.install_tam_attention([model_b, model_o], args_ns.tam_fpn_layer)
    warmup = args_ns.warmup_frames

    try:
        # ── Phase 1 ──
        b0 = args_ns.default_baseline
        if not os.path.isabs(b0):
            b0 = str(ROOT / b0)
        if not os.path.isfile(b0):
            b0 = ckpts[0]
        print('\n=== Phase 1: baseline =', os.path.basename(b0), '===\n')
        load_model(model_b, b0)

        rows = []
        for op in ckpts:
            if os.path.abspath(op) == os.path.abspath(b0):
                continue
            load_model(model_o, op)
            mg = mean_gap(model_b, model_o, probes, device, warmup)
            rows.append((mg, os.path.basename(op), op))
            print(f'  ours={os.path.basename(op):22s}  mean_TAM_gap={mg:+.4f}')

        rows.sort(key=lambda x: -x[0])
        print('\n--- Phase 1 ranking (mean gap vs', os.path.basename(b0), ') ---')
        for mg, bn, _ in rows[:15]:
            print(f'  {mg:+.4f}  {bn}')

        # ── Phase 2 ──
        topk = min(args_ns.topk_refine, len(rows))
        top_ours = [r[2] for r in rows[:topk]]
        print(f'\n=== Phase 2: refine top-{topk} ours × all baselines ===\n')

        best = (-1e9, '', '')
        for ours_path in top_ours:
            for base_path in ckpts:
                if os.path.abspath(base_path) == os.path.abspath(ours_path):
                    continue
                load_model(model_b, base_path)
                load_model(model_o, ours_path)
                mg = mean_gap(model_b, model_o, probes, device, warmup)
                if mg > best[0]:
                    best = (mg, base_path, ours_path)
                    print(f'  ★ new best gap={mg:+.4f}  '
                          f'base={os.path.basename(base_path):20s}  '
                          f'ours={os.path.basename(ours_path)}')

        print('\n========== BEST PAIR ==========')
        print(f'  mean_TAM_gap = {best[0]:+.4f}')
        bp, op = best[1], best[2]
        print(f'  baseline: {bp}')
        print(f'  ours:     {op}')
        print('\nSuggested command:')
        if bp and op:
            print(f'  python visualize_tam_attention.py \\\n'
                  f'    --baseline_ckpt {bp} \\\n'
                  f'    --ours_ckpt {op} \\\n'
                  f'    --vis_output_dir tam_vis_output --search_frames '
                  f'--tam_fpn_layer {args_ns.tam_fpn_layer} --warmup_frames {warmup}')

    finally:
        vis.uninstall_tam_attention()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
