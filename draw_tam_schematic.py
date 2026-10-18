#!/usr/bin/env python3
"""
Schematic **blocky pixel-attention score maps** on real refer-kitti frames (no model / no ckpt).

Each pixel owns a simulated visual key vector K(x,y) built from geometry,
foreground masks, edge/texture, color and position.  A text-query proxy q is
then dotted with every pixel key and softmax-normalized, mimicking pixel-wise
cross-modal attention without running any checkpoint.  For realistic display,
the dense score is aggregated to a low-resolution feature grid and upsampled as
visible pixel blocks with mild blur, similar to projected transformer/FPN maps.

Outputs: `tam_schematic_figure.png` / `.pdf` in cwd by default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

_CAM = LinearSegmentedColormap.from_list(
    'cam',
    [
        (0.00, (0.00, 0.04, 0.35)),
        (0.20, (0.00, 0.20, 0.65)),
        (0.45, (0.00, 0.65, 0.85)),
        (0.70, (0.25, 0.92, 0.45)),
        (0.85, (0.70, 1.00, 0.05)),
        (1.00, (1.00, 1.00, 0.00)),
    ],
)


def find_expr_json(rmot_path: str, video_id: str, sentence: str) -> str | None:
    d = os.path.join(rmot_path, 'expression', video_id)
    if not os.path.isdir(d):
        return None
    for fn in os.listdir(d):
        if not fn.endswith('.json'):
            continue
        try:
            with open(os.path.join(d, fn), encoding='utf-8') as f:
                data = json.load(f)
            if data.get('sentence', '').strip() == sentence.strip():
                return os.path.join(d, fn)
        except OSError:
            pass
    return None


def load_gt_boxes(
    rmot_path: str,
    video_id: str,
    frame_filename: str,
    expr_json_path: str,
    img_h: int,
    img_w: int,
):
    frame_num = int(Path(frame_filename).stem)
    with open(expr_json_path, encoding='utf-8') as f:
        data = json.load(f)
    label_dict = data.get('label', {})

    referred_ids: set[int] = set()
    for k, v in label_dict.items():
        if int(k) == frame_num:
            referred_ids = {int(x) for x in v}
            break

    lbl = os.path.join(
        rmot_path,
        'KITTI',
        'labels_with_ids',
        'image_02',
        video_id,
        f'{frame_num:06d}.txt',
    )
    if not os.path.exists(lbl):
        return [], []

    try:
        rows = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 6)
    except OSError:
        return [], []

    ref_boxes: list[tuple[int, int, int, int]] = []
    oth_boxes: list[tuple[int, int, int, int]] = []
    for r in rows:
        tid = int(r[1])
        x1 = int(r[2] * img_w)
        y1 = int(r[3] * img_h)
        x2 = int((r[2] + r[4]) * img_w)
        y2 = int((r[3] + r[5]) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        (ref_boxes if tid in referred_ids else oth_boxes).append((x1, y1, x2, y2))
    return ref_boxes, oth_boxes


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def gt_center(ref_boxes: list) -> tuple[float, float]:
    if not ref_boxes:
        return 0.0, 0.0
    sx = sy = 0.0
    for x1, y1, x2, y2 in ref_boxes:
        sx += 0.5 * (x1 + x2)
        sy += 0.5 * (y1 + y2)
    n = len(ref_boxes)
    return sx / n, sy / n


def pick_distractor_center(
    ref_boxes: list,
    oth_boxes: list,
    hh: int,
    ww: int,
    seed: int,
) -> tuple[float, float]:
    """Center of largest non-referred box, or random non-GT point."""
    rng = np.random.default_rng(seed)
    if oth_boxes:
        best = max(oth_boxes, key=_box_area)
        return 0.5 * (best[0] + best[2]), 0.5 * (best[1] + best[3])

    # fallback: offset from GT
    cx, cy = gt_center(ref_boxes)
    cx = float(np.clip(cx + rng.uniform(-0.25, 0.25) * ww, 0, ww - 1))
    cy = float(np.clip(cy + rng.uniform(-0.20, 0.20) * hh, 0, hh - 1))
    return cx, cy


def load_frame_rgb(
    rmot_path: str,
    video_id: str,
    frame_id: int,
    *,
    by_list_index: bool = False,
) -> tuple[np.ndarray, str, int]:
    """
    Load one RGB frame. Default: `frame_id` is the KITTI-style stem (e.g. 25 →
    `000025.jpg`). Set `by_list_index=True` to use sorted-list index (legacy
    behaviour from some vis scripts).
    """
    img_dir = os.path.join(
        rmot_path, 'KITTI', 'training', 'image_02', video_id
    )
    imgs = sorted(
        f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))
    )
    if not imgs:
        raise FileNotFoundError(img_dir)
    if by_list_index:
        fi = min(int(frame_id), len(imgs) - 1)
        fname = imgs[fi]
    else:
        stem = f'{int(frame_id):06d}'
        cand = [f for f in imgs if Path(f).stem == stem]
        if not cand:
            raise FileNotFoundError(
                f'No frame {stem}.jpg/.png under {img_dir}'
            )
        fname = cand[0]
        fi = imgs.index(fname)
    path = os.path.join(img_dir, fname)
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, fname, fi


def _union_mask_from_boxes(
    boxes: list[tuple[int, int, int, int]], hh: int, ww: int
) -> np.ndarray:
    m = np.zeros((hh, ww), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 255
    return m


def _norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _distance_to_mask(fg_u8: np.ndarray) -> np.ndarray:
    """Euclidean distance (pixels) to nearest pixel where fg_u8 > 0."""
    inv = (255 - np.clip(fg_u8, 0, 255)).astype(np.uint8)
    d = cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return d.astype(np.float32)


def _distractor_fg_mask(
    oth_boxes: list,
    hh: int,
    ww: int,
    alt_x: float,
    alt_y: float,
    ref_boxes: list,
) -> np.ndarray:
    """Foreground mask for the main distractor (largest non-GT player or disk at alt)."""
    if oth_boxes:
        x1, y1, x2, y2 = max(oth_boxes, key=_box_area)
        m = np.zeros((hh, ww), dtype=np.uint8)
        m[y1:y2, x1:x2] = 255
        return m
    r = max(22, int(0.036 * min(hh, ww)))
    m = np.zeros((hh, ww), dtype=np.uint8)
    cv2.circle(m, (int(alt_x), int(alt_y)), r, 255, -1)
    # avoid identical to GT when disk overlaps
    gtm = _union_mask_from_boxes(ref_boxes, hh, ww)
    m = cv2.bitwise_and(m, cv2.bitwise_not(gtm))
    if m.max() == 0:
        cv2.circle(m, (int(alt_x), int(alt_y)), r, 255, -1)
    return m


def _image_edge_score(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return _norm01(mag)


def _box_center_map(boxes: list[tuple[int, int, int, int]], hh: int, ww: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    if not boxes:
        return xx / max(ww - 1, 1), yy / max(hh - 1, 1)
    areas = np.array([_box_area(b) for b in boxes], dtype=np.float32)
    areas = areas / max(float(areas.sum()), 1.0)
    cx = sum(float(a) * 0.5 * (b[0] + b[2]) for a, b in zip(areas, boxes))
    cy = sum(float(a) * 0.5 * (b[1] + b[3]) for a, b in zip(areas, boxes))
    return (xx - cx) / max(ww, 1), (yy - cy) / max(hh, 1)


def _rgb_feature_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = _norm01(lab[:, :, 0])
    a = _norm01(lab[:, :, 1])
    b = _norm01(lab[:, :, 2])
    return l, a, b


def _local_texture_score(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(gray, (0, 0), 3.0)
    sq_mean = cv2.GaussianBlur(gray * gray, (0, 0), 3.0)
    var = np.clip(sq_mean - mean * mean, 0.0, None)
    return _norm01(var)


def _softmax_like_score(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float32) / max(temperature, 1e-4)
    z = z - np.percentile(z, 3.0)
    z = np.clip(z, -8.0, 8.0)
    p = np.exp(z)
    return _norm01(p)


def _build_pixel_keys(
    rgb: np.ndarray,
    dist_gt: np.ndarray,
    dist_alt: np.ndarray,
    edge: np.ndarray,
    texture: np.ndarray,
    fg_gt: np.ndarray,
    alt_fg: np.ndarray,
    ref_boxes: list,
    alt_boxes: list,
    seed: int,
) -> np.ndarray:
    """
    Dense visual keys K(x,y).  This mimics pixel-wise attention computation:
    every pixel owns a feature vector with semantic, geometric, color, edge and
    positional channels, later scored by dot product with a text query vector.
    """
    rng = np.random.default_rng(seed)
    hh, ww = dist_gt.shape
    l, a, b = _rgb_feature_maps(rgb)
    gx, gy = _box_center_map(ref_boxes, hh, ww)
    ax, ay = _box_center_map(alt_boxes, hh, ww)
    gt_prox = np.exp(-dist_gt / max(0.038 * min(hh, ww), 1e-3))
    alt_prox = np.exp(-dist_alt / max(0.050 * min(hh, ww), 1e-3))
    fg = (fg_gt > 0).astype(np.float32)
    alt = (alt_fg > 0).astype(np.float32)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    xpos = xx / max(ww - 1, 1)
    ypos = yy / max(hh - 1, 1)
    noise_low = rng.normal(0.0, 1.0, size=(hh, ww)).astype(np.float32)
    noise_low = cv2.GaussianBlur(noise_low, (0, 0), 5.0)
    noise_low = _norm01(noise_low)

    feats = [
        np.ones((hh, ww), dtype=np.float32),
        _norm01(gt_prox),
        _norm01(alt_prox),
        fg,
        alt,
        edge,
        texture,
        l,
        a,
        b,
        xpos,
        ypos,
        gx,
        gy,
        ax,
        ay,
        noise_low,
    ]
    return np.stack(feats, axis=-1).astype(np.float32)


def _query_for_category(cat: str, ours: bool) -> np.ndarray:
    """Text-query proxy over the feature channels in `_build_pixel_keys`."""
    if ours:
        q = np.array([
            -0.30, 2.85, -0.22, 1.95, -0.42, 0.42, 0.24,
            0.12, 0.06, 0.06, 0.00, 0.00, -0.34, -0.34, 0.04, 0.04, 0.04,
        ], dtype=np.float32)
    else:
        q = np.array([
            -0.10, 0.95, 1.34, 0.44, 0.82, 0.52, 0.40,
            0.16, 0.10, 0.10, 0.00, 0.00, -0.10, -0.10, -0.05, -0.05, 0.30,
        ], dtype=np.float32)

    cat_l = cat.lower()
    if 'number' in cat_l or 'jersey' in cat_l:
        q[5] += 0.32
        q[6] += 0.22
        q[7] += 0.18
    elif 'action' in cat_l:
        q[5] += 0.24
        q[6] += 0.35
        q[11] += 0.08
    elif 'spatial' in cat_l:
        q[10] -= 0.34
        q[12] -= 0.18
        q[14] -= 0.12
    elif 'color' in cat_l or 'uniform' in cat_l:
        q[7] += 0.34
        q[8] += 0.22
        q[9] += 0.22
    return q


def per_pixel_attention_score(
    rgb: np.ndarray,
    dist_gt: np.ndarray,
    dist_alt: np.ndarray,
    edge: np.ndarray,
    fg_gt: np.ndarray,
    alt_fg: np.ndarray,
    ref_boxes: list,
    alt_boxes: list,
    cat: str,
    *,
    ours: bool,
    seed: int,
) -> np.ndarray:
    """
    Simulated pixel-level attention: score(x,y)=q_text^T K_pixel(x,y), followed
    by a softmax-like normalization.  Ours uses a GT-aligned query; baseline keeps
    a substantial distractor and texture/noise component.
    """
    rng = np.random.default_rng(seed)
    texture = _local_texture_score(rgb)
    keys = _build_pixel_keys(
        rgb, dist_gt, dist_alt, edge, texture, fg_gt, alt_fg, ref_boxes, alt_boxes, seed
    )
    q = _query_for_category(cat, ours=ours)
    logits = np.tensordot(keys, q, axes=([-1], [0])).astype(np.float32)
    if ours:
        logits += rng.normal(0.0, 0.060, size=logits.shape).astype(np.float32)
        logits = cv2.GaussianBlur(logits, (0, 0), 0.65)
        score = _softmax_like_score(logits, temperature=0.70)
        score = 0.88 * score + 0.12 * _norm01(edge * np.exp(-dist_gt / max(0.065 * min(logits.shape), 1e-3)))
    else:
        logits += rng.normal(0.0, 0.135, size=logits.shape).astype(np.float32)
        logits = cv2.GaussianBlur(logits, (0, 0), 1.15)
        score = _softmax_like_score(logits, temperature=1.05)
        score = 0.78 * score + 0.22 * _norm01(edge + texture)
    return _norm01(score)


def _feature_grid_attention(
    heat: np.ndarray,
    *,
    grid_short: int,
    blur_sigma: float,
    gamma: float,
) -> np.ndarray:
    """
    Mimic transformer/FPN attention maps: aggregate dense scores to a low-res
    feature grid, resize by nearest neighbor to recover visible pixel blocks,
    then apply only mild blur so block boundaries remain perceivable.
    """
    hh, ww = heat.shape
    if hh <= 0 or ww <= 0:
        return heat
    scale = grid_short / float(min(hh, ww))
    gh = max(8, int(round(hh * scale)))
    gw = max(8, int(round(ww * scale)))
    low = cv2.resize(heat.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(low, (0, 0), blur_sigma)
    block = cv2.resize(low, (ww, hh), interpolation=cv2.INTER_NEAREST)
    if blur_sigma > 0:
        block = cv2.GaussianBlur(block, (0, 0), 0.55)
    block = _norm01(block)
    return np.clip(block, 0, 1) ** gamma


def gradcam_blend(ori_rgb: np.ndarray, heat: np.ndarray, soft: bool) -> np.ndarray:
    """Blocky low-resolution attention overlay, close to feature-map projection."""
    h = _feature_grid_attention(
        heat,
        grid_short=24 if soft else 28,
        blur_sigma=1.10 if soft else 0.78,
        gamma=0.82 if soft else 1.05,
    )
    cam = (_CAM(h)[:, :, :3] * 255.0).astype(np.float32)
    o = ori_rgb.astype(np.float32)

    # Slightly dim the image first, as in many attention-map visualizations.
    dark = (o * (0.52 if soft else 0.56)).astype(np.float32)
    alpha = 0.36 if soft else 0.34
    out = cv2.addWeighted(dark, 1.0 - alpha, cam, alpha, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_gt_box(img: np.ndarray, boxes: list) -> np.ndarray:
    canvas = img.copy()
    overlay = canvas.copy()
    t = max(2, min(canvas.shape[:2]) // 120)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 220, 90), -1)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 220, 90), max(2, t * 2))
        cv2.putText(
            canvas,
            'GT',
            (x1 + 4, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            max(1, t),
            cv2.LINE_AA,
        )
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--rmot_path',
        default='/data1/liuyeqiang/TempMam/data/Dataset/refer-kitti',
        help='Root of refer-kitti (contains KITTI/, expression/).',
    )
    ap.add_argument(
        '--output',
        default='tam_schematic_figure',
        help='Output path without extension (writes .png and .pdf).',
    )
    ap.add_argument(
        '--by_list_index',
        action='store_true',
        help='Interpret column frame integers as sorted-list indices (not 000NNN stems).',
    )
    args = ap.parse_args()
    R = args.rmot_path
    out_base = Path(args.output).resolve()

    columns = [
        (
            'Number / Jersey',
            'volleyball players who are wearing white number 7 sportswear',
            25,
        ),
        (
            'Action',
            'volleyball players who are serving',
            18,
        ),
        (
            'Spatial',
            'volleyball players who are standing on the left of the net',
            20,
        ),
        (
            'Color / Uniform',
            'volleyball players who are wearing white sportswear',
            22,
        ),
    ]
    video_id = 'v_1LwtoLPw2TU_c006'

    rows_in, rows_b, rows_o = [], [], []
    meta_titles: list[tuple[str, str]] = []

    for ci, (cat, sentence, frame_idx) in enumerate(columns):
        rgb, fname, fi = load_frame_rgb(
            R, video_id, frame_idx, by_list_index=args.by_list_index
        )
        hh, ww = rgb.shape[:2]

        expr = find_expr_json(R, video_id, sentence)
        if expr is None:
            raise SystemExit(f'Missing expression JSON for: {sentence!r}')

        gt_ref, gt_oth = load_gt_boxes(R, video_id, fname, expr, hh, ww)
        if not gt_ref:
            raise SystemExit(f'No referred GT in {video_id} / {fname} for that sentence.')

        alt_x, alt_y = pick_distractor_center(gt_ref, gt_oth, hh, ww, seed=300 + ci)

        fg_gt = _union_mask_from_boxes(gt_ref, hh, ww)
        dist_gt = _distance_to_mask(fg_gt)
        alt_fg = _distractor_fg_mask(gt_oth, hh, ww, alt_x, alt_y, gt_ref)
        dist_alt = _distance_to_mask(alt_fg)
        edge = _image_edge_score(rgb)
        alt_boxes = [max(gt_oth, key=_box_area)] if gt_oth else []

        h_ours = per_pixel_attention_score(
            rgb,
            dist_gt,
            dist_alt,
            edge,
            fg_gt,
            alt_fg,
            gt_ref,
            alt_boxes,
            cat,
            ours=True,
            seed=100 + ci,
        )
        h_base = per_pixel_attention_score(
            rgb,
            dist_gt,
            dist_alt,
            edge,
            fg_gt,
            alt_fg,
            gt_ref,
            alt_boxes,
            cat,
            ours=False,
            seed=200 + ci,
        )

        short = sentence.replace('volleyball players who are ', '').strip()
        if len(short) > 48:
            short = short[:45] + '…'
        meta_titles.append((cat, short))

        rows_in.append(draw_gt_box(rgb, gt_ref))
        rows_b.append(gradcam_blend(rgb, h_base, soft=True))
        rows_o.append(gradcam_blend(rgb, h_ours, soft=False))

        print(f'  column {ci + 1} {cat}: {fname} (index {fi}) GT n={len(gt_ref)}')

    nc = len(columns)
    fig = plt.figure(figsize=(4.0 * nc + 1.3, 9.2), facecolor='white')
    gs = GridSpec(
        3,
        nc,
        figure=fig,
        left=0.08,
        right=0.88,
        top=0.91,
        bottom=0.08,
        wspace=0.05,
        hspace=0.32,
    )
    row_labels = [
        ('Input + GT', '#222222'),
        ('Baseline\n(pixel attention)', '#cc2222'),
        ('Ours\n(pixel attention)', '#1155cc'),
    ]
    axes = [[fig.add_subplot(gs[r, c]) for c in range(nc)] for r in range(3)]

    for c in range(nc):
        imgs = [rows_in[c], rows_b[c], rows_o[c]]
        for r, (im, _) in enumerate(zip(imgs, row_labels)):
            ax = axes[r][c]
            ax.imshow(im)
            ax.axis('off')
            if r == 0:
                cat, ph = meta_titles[c]
                ax.set_title(
                    f'{cat}\n{ph}',
                    fontsize=9,
                    fontweight='bold',
                    color='#111',
                    pad=6,
                )
        axes[1][c].text(
            0.02,
            0.04,
            'diffuse',
            transform=axes[1][c].transAxes,
            fontsize=8,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='#aa1111', alpha=0.85),
            va='bottom',
        )
        axes[2][c].text(
            0.02,
            0.04,
            'GT-focused',
            transform=axes[2][c].transAxes,
            fontsize=8,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='#1144bb', alpha=0.85),
            va='bottom',
        )

    for r, (lbl, col) in enumerate(row_labels):
        axes[r][0].text(
            -0.055,
            0.5,
            lbl,
            transform=axes[r][0].transAxes,
            fontsize=10.5,
            fontweight='bold',
            color=col,
            va='center',
            ha='right',
            rotation=90,
            clip_on=False,
        )

    cax = fig.add_axes([0.905, 0.18, 0.014, 0.52])
    sm = plt.cm.ScalarMappable(cmap=_CAM, norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Pixel-wise attention score (sim.)', fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=7)

    patches = [
        mpatches.Patch(fc='#00dc3c', ec='#00dc3c', label='GT (row 1, real labels)'),
        mpatches.Patch(fc='#cc2222', alpha=0.85, label='Baseline (sim. pixel attention)'),
        mpatches.Patch(fc='#1155cc', alpha=0.85, label='Ours (sim. pixel attention)'),
    ]
    fig.legend(
        handles=patches,
        loc='lower center',
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.46, -0.01),
    )

    fig.suptitle(
        f'Schematic blocky pixel-wise text-to-image attention — {video_id}\n'
        'score(x,y)=q_text^T K_pixel(x,y), projected through a low-res feature grid with mild blur.',
        fontsize=11,
        fontweight='bold',
        y=0.99,
        color='#111',
    )

    png = out_base.with_suffix('.png')
    pdf = out_base.with_suffix('.pdf')
    fig.savefig(
        str(png), dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none'
    )
    fig.savefig(pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Wrote:\n  {png}\n  {pdf}')


if __name__ == '__main__':
    main()
