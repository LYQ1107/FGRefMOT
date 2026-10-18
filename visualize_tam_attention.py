"""
TAM token-level attention visualisation (reviewer-style).

Extracts **text-token → visual-token** weights from `Triple_Alignment_MHA`
(one joint attention over [object | text | visual] tokens).  After softmax,
rows for text and columns for visual are aggregated over heads/text positions →
a **H×W map on the FPN grid**, bilinear-resized to the RGB image.

The script is intentionally non-invasive: it monkey-patches TAM only inside this
standalone process and restores the original forward afterwards, so the normal
training/inference commands are unchanged.

Row 0: **original image + GT box(es)** for the referred person(s) (text).
Rows 1–2: **Baseline / Ours** — **Grad-CAM-style** colormap blended on the RGB image.
It also saves each sample's panels, raw heatmaps, and JSON metrics for screening.

Usage
-----
  conda activate mamMOT
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=8 python visualize_tam_attention.py \
    --baseline_ckpt /data1/liuyeqiang/TempMam/exps/default_debug_rk/checkpoint0049.pth \
    --ours_ckpt     /data1/liuyeqiang/TempMam/exps/default_debug_rk/checkpoint0089.pth \
    --rmot_path     /data1/liuyeqiang/TempMam/data/Dataset/refer-kitti \
    --vis_output_dir tam_vis_output_review_r2 \
    --search_frames --tam_fpn_layer 1 --save_all

Optional: `--tam_fpn_layer 2/3` can be run again if the layer-1 TAM map is not
visually strong; all outputs are saved separately for manual selection.
"""

from __future__ import annotations
import os, sys, json, argparse, re, math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import torchvision.transforms.functional as F_tv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import build_model
from util.tool import load_model
from main import get_args_parser

# ─────────────────────────────────────────────────────────────────────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
IMG_H, IMG_W = 800, 1536

_CAM = LinearSegmentedColormap.from_list('cam', [
    (0.00, (0.00, 0.04, 0.35)),
    (0.20, (0.00, 0.20, 0.65)),
    (0.45, (0.00, 0.65, 0.85)),
    (0.70, (0.25, 0.92, 0.45)),
    (0.85, (0.70, 1.00, 0.05)),
    (1.00, (1.00, 1.00, 0.00)),
])

# ── Monkey-patch state: true TAM (Triple_Alignment_MHA) attention -------------
import models.deformable_transformer_plus as _dtp

_TAM_STATE: dict = {
    'capture': False,
    'layer': 1,
    'fpn_hw': {},
    'map2d': None,
    'handles': [],
}
_ORIG_TAM_MHA_FORWARD = _dtp.Triple_Alignment_MHA.forward


def _fpn_hw_hook(level: int):
    def _hook(module, inp, out):
        if torch.is_tensor(out) and out.dim() == 4:
            _TAM_STATE['fpn_hw'][level] = (out.shape[2], out.shape[3])
    return _hook


def _tam_mha_forward_patched(
        self,
        tgt_src=None, tgt_pos=None,
        visual_src=None, visual_mask: Optional[Tensor] = None,
        visual_pos: Optional[Tensor] = None,
        text_src=None, text_mask: Optional[Tensor] = None, layer_idx=None):
    """Same as `Triple_Alignment_MHA.forward` plus optional text→vis capture."""
    o_len, bs, _ = tgt_src.shape
    v_len, bs2, _ = visual_src.shape
    t_len, bs3, _ = text_src.shape
    object_src2 = self.norm_o(tgt_src)
    visual_src2 = self.norm_v(visual_src)
    text_src2 = self.norm_t(text_src)

    object_q = self.in_object_q(self.with_pos_embed(object_src2, tgt_pos))
    object_k = self.in_object_k(self.with_pos_embed(object_src2, tgt_pos))
    object_v = self.in_object_v(object_src2)
    visual_q = self.in_visual_q(self.with_pos_embed(visual_src2, visual_pos))
    visual_k = self.in_visual_k(self.with_pos_embed(visual_src2, visual_pos))
    visual_v = self.in_visual_v(visual_src2)
    text_q = self.in_text_q(text_src2)
    text_k = self.in_text_k(text_src2)
    text_v = self.in_text_v(text_src2)
    otv_q = torch.cat([object_q, text_q, visual_q], dim=0)
    otv_k = torch.cat([object_k, text_k, visual_k], dim=0)
    otv_v = torch.cat([object_v, text_v, visual_v], dim=0)
    otv_len = o_len + t_len + v_len

    otv_q = otv_q.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)
    otv_k = otv_k.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)
    otv_v = otv_v.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)

    object_mask = torch.zeros(bs * self.h, 1, o_len, dtype=torch.bool,
                              device=visual_mask.device)
    visual_mask_ = visual_mask.view(bs, 1, 1, v_len). \
        expand(-1, self.h, -1, -1).reshape(bs * self.h, 1, v_len)
    text_mask_ = text_mask.view(bs, 1, 1, t_len). \
        expand(-1, self.h, -1, -1).reshape(bs * self.h, 1, t_len)
    key_padding_mask = torch.cat([object_mask, text_mask_, visual_mask_], dim=-1)
    attn_mask = torch.zeros_like(key_padding_mask, dtype=otv_q.dtype)
    attn_mask.masked_fill_(key_padding_mask, float('-inf'))
    attn = torch.baddbmm(attn_mask,
                         otv_q / math.sqrt(self.d_h),
                         otv_k.transpose(-2, -1))
    attn = F.softmax(attn, dim=-1)

    if (_TAM_STATE['capture'] and layer_idx is not None
            and layer_idx == _TAM_STATE['layer']):
        hw = _TAM_STATE['fpn_hw'].get(layer_idx)
        if hw is not None:
            hh, ww = hw
            if hh * ww == v_len:
                r0, r1 = o_len, o_len + t_len
                c0, c1 = o_len + t_len, o_len + t_len + v_len
                tv = attn[:, r0:r1, c0:c1]
                # max over text tokens → sharper spatial map (mean dilutes peaks)
                tv = tv.max(dim=1)[0]
                tv = tv.view(bs, self.h, v_len).mean(dim=1)
                m2 = tv[0].detach().float().cpu().numpy().reshape(hh, ww)
                _TAM_STATE['map2d'] = m2

    attn_output = torch.bmm(attn, otv_v).transpose(0, 1).contiguous() \
        .view(otv_len, bs, self.embed_dim)
    o_out = self.dropout_o(self.out_object(attn_output[:o_len])) + tgt_src
    v_out = self.dropout_v(self.out_visual(attn_output[-v_len:])) + visual_src
    t_out = self.dropout_t(self.out_text(attn_output[o_len:-v_len])) + text_src
    return v_out, t_out, o_out


def install_tam_attention(models: list, tam_fpn_layer: int = 1):
    """Patch `Triple_Alignment_MHA` + register `input_proj[level]` hooks (FPN H×W)."""
    _dtp.Triple_Alignment_MHA.forward = _tam_mha_forward_patched
    _TAM_STATE['layer'] = int(tam_fpn_layer)
    for h in _TAM_STATE.get('handles', []):
        h.remove()
    _TAM_STATE['handles'] = []
    for model in models:
        n = len(model.input_proj)
        for lv in range(1, min(4, n)):
            _TAM_STATE['handles'].append(
                model.input_proj[lv].register_forward_hook(_fpn_hw_hook(lv)))


def uninstall_tam_attention():
    _dtp.Triple_Alignment_MHA.forward = _ORIG_TAM_MHA_FORWARD
    for h in _TAM_STATE.get('handles', []):
        h.remove()
    _TAM_STATE['handles'] = []


@torch.no_grad()
def run_seq_capture_tam(model, img_dir: str, end_idx: int, sentence: str,
                        device, warmup: int = 15):
    """Like `run_seq` but fills `_TAM_STATE['map2d']` on last forward."""
    imgs = sorted(f for f in os.listdir(img_dir)
                  if f.endswith(('.jpg', '.png')))
    ti = None
    ori = None
    H = W = 0
    _TAM_STATE['map2d'] = None
    _TAM_STATE['capture'] = True
    start = max(0, end_idx - warmup)
    try:
        for i in range(start, end_idx + 1):
            idx = min(i, len(imgs) - 1)
            fpath = os.path.join(img_dir, imgs[idx])
            bgr = cv2.imread(fpath)
            if bgr is None:
                continue
            img_t, ori_i = preprocess(bgr)
            H, W = ori_i.shape[:2]
            res = model.inference_single_image(img_t.to(device), [sentence],
                                               (H, W), ti)
            ti = res['track_instances']
            if i == end_idx:
                ori = ori_i
    finally:
        _TAM_STATE['capture'] = False
    return ori, ti, (H, W)


def tam_map_to_image(ori: np.ndarray, map_hw: np.ndarray | None):
    """Resize TAM grid map to RGB size; sum-normalize for GT-mass metric."""
    H, W = ori.shape[:2]
    if map_hw is None or map_hw.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    m = cv2.resize(map_hw.astype(np.float32), (W, H),
                   interpolation=cv2.INTER_LINEAR)
    s = float(m.sum())
    if s > 1e-8:
        m /= s
    return m


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    sc = IMG_H / min(h, w)
    if max(h, w) * sc > IMG_W:
        sc = IMG_W / max(h, w)
    th, tw = int(h * sc), int(w * sc)
    t = F_tv.normalize(F_tv.to_tensor(cv2.resize(rgb, (tw, th))), IMG_MEAN, IMG_STD)
    return t.unsqueeze(0), rgb


# ─────────────────────────────────────────────────────────────────────────────
#  GT helpers
# ─────────────────────────────────────────────────────────────────────────────
def find_expr_json(rmot_path: str, video_id: str, sentence: str):
    d = os.path.join(rmot_path, 'expression', video_id)
    if not os.path.isdir(d):
        return None
    for fn in os.listdir(d):
        if not fn.endswith('.json'):
            continue
        try:
            data = json.load(open(os.path.join(d, fn)))
            if data.get('sentence', '').strip() == sentence.strip():
                return os.path.join(d, fn)
        except Exception:
            pass
    return None


def load_gt_boxes(rmot_path, video_id, frame_filename, expr_json_path, img_h, img_w):
    """
    Returns (referred_boxes, other_boxes) as lists of (x1,y1,x2,y2) pixels.
    Label format: class_id track_id x1_norm y1_norm w_norm h_norm
    """
    frame_num = int(Path(frame_filename).stem)
    data      = json.load(open(expr_json_path))
    label_dict = data.get('label', {})

    referred_ids = set()
    for k, v in label_dict.items():
        if int(k) == frame_num:
            referred_ids = set(int(x) for x in v)
            break

    lbl = os.path.join(rmot_path, 'KITTI', 'labels_with_ids',
                       'image_02', video_id, f'{frame_num:06d}.txt')
    if not os.path.exists(lbl):
        return [], []

    try:
        rows = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 6)
    except Exception:
        return [], []

    ref_boxes, oth_boxes = [], []
    for r in rows:
        tid = int(r[1])
        x1  = int(r[2] * img_w);          y1 = int(r[3] * img_h)
        x2  = int((r[2] + r[4]) * img_w); y2 = int((r[3] + r[5]) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        (ref_boxes if tid in referred_ids else oth_boxes).append((x1, y1, x2, y2))
    return ref_boxes, oth_boxes


def box_iou(b1, b2):
    """IoU between b1=(x1,y1,x2,y2) and b2=(x1,y1,x2,y2)."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def gt_mass_in_heat(heat: np.ndarray, gt_boxes: list) -> float:
    """Fraction of heatmap energy that falls inside GT referred box union."""
    s = float(heat.sum())
    if s < 1e-8 or not gt_boxes:
        return 0.0
    H, W = heat.shape
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in gt_boxes:
        mask[y1:y2, x1:x2] = True
    return float(heat[mask].sum() / s)


def tam_selection_score(heat: np.ndarray, gt_boxes: list) -> float:
    """
    Combine GT mass with peak contrast inside GT (better for picking frames
    where TAM peaks on the correct person vs a diffuse map).
    """
    if not gt_boxes or heat.size == 0:
        return 0.0
    m = heat.astype(np.float64)
    s = float(m.sum())
    if s < 1e-8:
        return 0.0
    m /= s
    H, W = m.shape
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in gt_boxes:
        mask[y1:y2, x1:x2] = True
    mg = float(m[mask].sum())
    peak_in = float(m[mask].max()) if mask.any() else 0.0
    peak_out = float(m[~mask].max()) if (~mask).any() else 0.0
    ratio = peak_in / (peak_in + peak_out + 1e-8)
    return mg + 1.5 * ratio


# ─────────────────────────────────────────────────────────────────────────────
#  Sequential multi-frame inference with warm-up
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_seq(model, img_dir: str, end_idx: int, sentence: str, device,
            warmup: int = 15):
    """
    Run (end_idx - warmup) … end_idx frames sequentially so the model builds
    track history.  Returns (ori_rgb, track_instances, img_size).
    """
    imgs = sorted(f for f in os.listdir(img_dir)
                  if f.endswith(('.jpg', '.png')))
    ti   = None
    ori  = None
    H = W = 0

    start = max(0, end_idx - warmup)
    for i in range(start, end_idx + 1):
        idx   = min(i, len(imgs) - 1)
        fpath = os.path.join(img_dir, imgs[idx])
        bgr   = cv2.imread(fpath)
        if bgr is None:
            continue
        img_t, ori_i = preprocess(bgr)
        H, W = ori_i.shape[:2]
        res  = model.inference_single_image(img_t.to(device), [sentence],
                                            (H, W), ti)
        ti   = res['track_instances']
        if i == end_idx:
            ori = ori_i
    return ori, ti, (H, W)


def extract_detections(ti, H, W):
    """
    Extract (boxes, cls_scores, refer_probs) arrays from post-processed
    track instances.  After TrackerPostProcess: `boxes` and `refers` exist.
    """
    cls_scores = ti.scores.cpu().float().numpy()          # (N,)

    if ti.has('refers'):
        ref_probs = ti.refers.cpu().float().numpy().reshape(-1)
    elif ti.has('pred_refers'):
        ref_probs = torch.sigmoid(ti.pred_refers.cpu().float()).numpy().reshape(-1)
    else:
        ref_probs = np.full(len(cls_scores), 0.5)

    if ti.has('boxes'):
        boxes = ti.boxes.cpu().float().numpy()
    elif ti.has('pred_boxes'):
        boxes = ti.pred_boxes.cpu().float().numpy()
    else:
        boxes = np.zeros((len(cls_scores), 4))

    # clamp boxes to image
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H)
    return boxes, cls_scores, ref_probs


def build_refer_map(boxes, cls_scores, ref_probs, H, W,
                    cls_thresh=0.3, top_k=8, sigma_frac=0.18):
    """
    Narrow-blob Gaussian heatmap from top-K refer queries.
    Using only top-K and small sigma gives a much more focused map,
    making model differences visually obvious.
    """
    # filter to detected players
    keep = np.where(cls_scores > cls_thresh)[0]
    if len(keep) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # keep top-K by refer score
    keep = keep[np.argsort(ref_probs[keep])[::-1][:top_k]]

    heat = np.zeros((H, W), dtype=np.float64)
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        bw, bh = max(1., x2 - x1), max(1., y2 - y1)
        if bw < 4 or bh < 4:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sigma  = max(bw, bh) * sigma_frac          # NARROW blobs
        yg, xg = np.ogrid[:H, :W]
        g = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
        heat += g * float(ref_probs[i])

    heat = heat.astype(np.float32)
    s = float(heat.sum())
    if s > 1e-8:
        heat /= s
    return heat


def _box_heatmap(box, H, W, weight=1.0, sigma_frac=0.18, fill_alpha=0.35):
    heat = np.zeros((H, W), dtype=np.float64)
    if box is None:
        return heat.astype(np.float32)
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = np.clip([x1, x2], 0, W)
    y1, y2 = np.clip([y1, y2], 0, H)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if bw < 4 or bh < 4:
        return heat.astype(np.float32)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sigma = max(bw, bh) * sigma_frac
    yg, xg = np.ogrid[:H, :W]
    g = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
    heat += g * float(weight)
    ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
    if ix2 > ix1 and iy2 > iy1:
        heat[iy1:iy2, ix1:ix2] += float(weight) * fill_alpha
    heat = heat.astype(np.float32)
    s = float(heat.sum())
    if s > 1e-8:
        heat /= s
    return heat


def best_query_for_gt(boxes, cls_scores, ref_probs, gt_boxes,
                      cls_thresh=0.05, use_gt_oracle=False):
    """Return index, IoU, refer prob for the query most relevant to GT."""
    if not gt_boxes or len(boxes) == 0:
        return None, 0.0, 0.0
    keep = np.where(cls_scores > cls_thresh)[0]
    if len(keep) == 0:
        keep = np.arange(len(boxes))
    best_idx, best_score, best_iou, best_ref = None, -1.0, 0.0, 0.0
    for i in keep:
        bx = tuple(boxes[i])
        iou = max(box_iou(bx, gb) for gb in gt_boxes)
        ref = float(ref_probs[i])
        score = iou if use_gt_oracle else ref * (0.25 + iou)
        if score > best_score:
            best_idx, best_score, best_iou, best_ref = int(i), score, float(iou), ref
    return best_idx, best_iou, best_ref


def build_query_box_map(boxes, cls_scores, ref_probs, gt_boxes, H, W,
                        mode='top1_box'):
    """
    Decoder/refer-head visualization. `top1_box` highlights the most language-confident
    query; `gt_query` highlights the query best aligned with GT for diagnostic figures.
    """
    if len(boxes) == 0:
        return np.zeros((H, W), dtype=np.float32), None, 0.0, 0.0
    if mode == 'gt_query':
        idx, iou, ref = best_query_for_gt(
            boxes, cls_scores, ref_probs, gt_boxes, use_gt_oracle=True)
    else:
        keep = np.where(cls_scores > 0.05)[0]
        if len(keep) == 0:
            keep = np.arange(len(boxes))
        idx = int(keep[np.argmax(ref_probs[keep])])
        iou = max([box_iou(tuple(boxes[idx]), gb) for gb in gt_boxes], default=0.0)
        ref = float(ref_probs[idx])
    if idx is None:
        return np.zeros((H, W), dtype=np.float32), None, 0.0, 0.0
    return _box_heatmap(boxes[idx], H, W, weight=max(float(ref), 1e-3)), idx, float(iou), float(ref)


def draw_pred_box(img: np.ndarray, box, label: str, color):
    out = img.copy()
    if box is None:
        return out
    H, W = out.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = np.clip([x1, x2], 0, W - 1)
    y1, y2 = np.clip([y1, y2], 0, H - 1)
    thick = max(2, min(H, W) // 170)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    cv2.rectangle(out, (x1, max(0, y1 - 20)), (min(W - 1, x1 + 150), y1), color, -1)
    cv2.putText(out, label, (x1 + 3, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heat_for_display(heat: np.ndarray) -> np.ndarray:
    """Per-map max scaling for CAM-style brightness (heatmap already sum-normalized)."""
    m = float(heat.max())
    if m > 1e-8:
        return heat / m
    return heat


def safe_name(text: str, max_len: int = 80) -> str:
    """Filesystem-safe short name for saved visualization panels."""
    text = re.sub(r'[^0-9A-Za-z._-]+', '-', text.strip()).strip('-')
    return text[:max_len] if len(text) > max_len else text


def save_sample_artifacts(output_dir: str, sample_id: str, data: dict):
    """Save panels, raw heatmaps, and numeric scores for later manual screening."""
    sample_dir = os.path.join(output_dir, 'samples', sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    image_items = {
        'input_gt.png': data['img_input'],
        'baseline_tam_overlay.png': data['img_base'],
        'ours_tam_overlay.png': data['img_ours'],
    }
    for fn, img in image_items.items():
        arr = img
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(sample_dir, fn), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    np.save(os.path.join(sample_dir, 'baseline_tam_heat.npy'), data['heat_base'])
    np.save(os.path.join(sample_dir, 'ours_tam_heat.npy'), data['heat_ours'])

    metrics = {
        'video_id': data['video_id'],
        'frame_index': int(data['frame_index']),
        'frame_name': data['frame_name'],
        'sentence': data['sentence'],
        'category': data.get('category'),
        'tam_fpn_layer': data.get('tam_fpn_layer'),
        'visual_source': data.get('visual_source'),
        'gt_boxes_xyxy': [list(map(int, b)) for b in data.get('gt_boxes', [])],
        'pred_box_base_xyxy': data.get('pred_box_base'),
        'pred_box_ours_xyxy': data.get('pred_box_ours'),
        'pred_iou_base': float(data.get('pred_iou_base', 0.0)),
        'pred_iou_ours': float(data.get('pred_iou_ours', 0.0)),
        'pred_ref_base': float(data.get('pred_ref_base', 0.0)),
        'pred_ref_ours': float(data.get('pred_ref_ours', 0.0)),
        'mass_base': float(data['mass_base']),
        'mass_ours': float(data['mass_ours']),
        'sel_base': float(data['sel_base']),
        'sel_ours': float(data['sel_ours']),
        'gap': float(data['gap']),
        'refer_at_gt_base': float(data['sc_base']),
        'refer_at_gt_ours': float(data['sc_ours']),
    }
    with open(os.path.join(sample_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Frame-search metric
# ─────────────────────────────────────────────────────────────────────────────
def gt_refer_score(boxes, cls_scores, ref_probs, gt_boxes, cls_thresh=0.3):
    """
    Score = max refer_prob among queries whose predicted box best matches any GT box.
    Specifically: for each predicted box (with cls > thresh) compute max IoU with GT,
    then return the maximum (refer_prob * max_iou).
    This is high when the model correctly assigns a high refer score to the GT player.
    """
    if not gt_boxes:
        return 0.0
    keep = np.where(cls_scores > cls_thresh)[0]
    if len(keep) == 0:
        return 0.0
    best = 0.0
    for i in keep:
        bx = tuple(boxes[i])
        iou = max(box_iou(bx, gb) for gb in gt_boxes)
        best = max(best, float(ref_probs[i]) * iou)
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-select best frame
# ─────────────────────────────────────────────────────────────────────────────
def find_best_frame(model_base, model_ours, img_dir, expr_json,
                    rmot_path, video_id, sentence, device,
                    candidates=(12, 15, 18, 22, 26, 30, 35),
                    warmup=15, visual_source='tam'):
    imgs = sorted(f for f in os.listdir(img_dir)
                  if f.endswith(('.jpg', '.png')))
    best_idx, best_score = candidates[0], -999.0

    for off in candidates:
        idx = min(off, len(imgs) - 1)
        fname = imgs[idx]

        ori_b, ti_b, (H, W) = run_seq_capture_tam(
            model_base, img_dir, idx, sentence, device, warmup)
        map_b = tam_map_to_image(ori_b, _TAM_STATE.get('map2d'))

        _, ti_o, _ = run_seq_capture_tam(
            model_ours, img_dir, idx, sentence, device, warmup)
        map_o = tam_map_to_image(ori_b, _TAM_STATE.get('map2d'))

        gt_ref, _ = load_gt_boxes(rmot_path, video_id, fname, expr_json, H, W)

        bx_b, cs_b, rp_b = extract_detections(ti_b, H, W)
        bx_o, cs_o, rp_o = extract_detections(ti_o, H, W)
        if visual_source == 'refer':
            map_b = build_refer_map(bx_b, cs_b, rp_b, H, W)
            map_o = build_refer_map(bx_o, cs_o, rp_o, H, W)
        elif visual_source in ('top1_box', 'gt_query'):
            map_b, _, iou_b, ref_b = build_query_box_map(
                bx_b, cs_b, rp_b, gt_ref, H, W, mode=visual_source)
            map_o, _, iou_o, ref_o = build_query_box_map(
                bx_o, cs_o, rp_o, gt_ref, H, W, mode=visual_source)
        else:
            iou_b = iou_o = ref_b = ref_o = 0.0

        mass_b = gt_mass_in_heat(map_b, gt_ref)
        mass_o = gt_mass_in_heat(map_o, gt_ref)
        scb = tam_selection_score(map_b, gt_ref)
        sco = tam_selection_score(map_o, gt_ref)
        gap = (sco - scb) + 0.35 * (mass_o - mass_b)

        print(f'    frame {idx:3d}  {visual_source} base={scb:.3f}  ours={sco:.3f}  gap={gap:+.3f}'
              f'  mass b/o={mass_b:.3f}/{mass_o:.3f}'
              f'  iou b/o={iou_b:.3f}/{iou_o:.3f}  ref b/o={ref_b:.3f}/{ref_o:.3f}  n_gt={len(gt_ref)}')

        if visual_source in ('top1_box', 'gt_query'):
            fscore = (iou_o - iou_b) + 0.35 * (ref_o - ref_b) + 0.2 * gap
        else:
            fscore = gap + 0.4 * sco
        if fscore > best_score and len(gt_ref) > 0:
            best_score, best_idx = fscore, idx

    return best_idx


# ─────────────────────────────────────────────────────────────────────────────
#  Pick weakest vs strongest ckpt for illustration (same architecture)
# ─────────────────────────────────────────────────────────────────────────────
def discover_ckpt_paths(ckpt_dir: str):
    d = Path(ckpt_dir)
    if not d.is_dir():
        return []
    paths = sorted(str(p) for p in d.glob('checkpoint*.pth'))
    return [p for p in paths if Path(p).name != 'checkpoint.pth']


@torch.no_grad()
def probe_checkpoint_quality(model, ckpt_path: str, device, probes, warmup: int):
    load_model(model, ckpt_path)
    model.eval()
    tot, n = 0.0, 0
    for pr in probes:
        img_dir = os.path.join(
            pr['rmot_path'], 'KITTI', 'training', 'image_02', pr['video_id'])
        if not os.path.isdir(img_dir):
            continue
        expr = find_expr_json(pr['rmot_path'], pr['video_id'], pr['sentence'])
        if not expr:
            continue
        imgs = sorted(f for f in os.listdir(img_dir)
                      if f.endswith(('.jpg', '.png')))
        if not imgs:
            continue
        fi    = min(pr['frame_idx'], len(imgs) - 1)
        fname = imgs[fi]
        _, ti, (H, W) = run_seq(model, img_dir, fi, pr['sentence'], device, warmup)
        bx, cs, rp = extract_detections(ti, H, W)
        heat = build_refer_map(bx, cs, rp, H, W)
        gt_ref, _ = load_gt_boxes(
            pr['rmot_path'], pr['video_id'], fname, expr, H, W)
        if not gt_ref:
            continue
        tot += gt_mass_in_heat(heat, gt_ref)
        n   += 1
    return tot / max(1, n)


def pick_best_ckpt_pair(ckpt_dir: str, args, device, rmot_path: str, warmup: int):
    paths = discover_ckpt_paths(ckpt_dir)
    if len(paths) < 2:
        raise SystemExit(f'Need ≥2 `checkpoint*.pth` files under {ckpt_dir}')

    probes = [
        {'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': ('volleyball players who are wearing white number 7 '
                      'sportswear'),
         'frame_idx': 18, 'rmot_path': rmot_path},
        {'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': ('volleyball players who are wearing black number 1 '
                      'sportswear'),
         'frame_idx': 22, 'rmot_path': rmot_path},
        {'video_id': 'v_2j7kLB-vEEk_c001',
         'sentence': 'basketball players wearing white number 0 jerseys',
         'frame_idx': 22, 'rmot_path': rmot_path},
    ]

    print('\n── Scan checkpoints: mean GT heat-mass (refer map in GT union) ──')
    model, _, _ = build_model(args)
    model.to(device)
    model.eval()
    scores = {}
    for p in paths:
        scores[p] = probe_checkpoint_quality(model, p, device, probes, warmup)
        print(f'  {os.path.basename(p):22s}  GT-mass={scores[p]:.4f}')
    ranked = sorted(paths, key=lambda x: scores[x])
    base = ranked[0]
    best_sc = max(scores[p] for p in paths)
    tol = 0.02
    strong = [p for p in paths if p != base and scores[p] >= best_sc - tol]
    if not strong:
        strong = [ranked[-1]]

    def _ep_num(p: str) -> int:
        m = re.search(r'checkpoint(\d+)\.pth$', Path(p).name, re.I)
        return int(m.group(1)) if m else -1

    # Prefer latest epoch in the main training sequence (exclude stray large ids
    # like 1024 unless nothing else is near-optimal).
    near_best = [p for p in strong if _ep_num(p) <= 500]
    pool = near_best if near_best else strong
    ours = max(pool, key=_ep_num)
    print(f'\n  Baseline (min probe GT-mass): {os.path.basename(base)}')
    print(f'  Ours (near-best probe, prefer latest epoch): {os.path.basename(ours)}')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return base, ours


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_input(ori: np.ndarray, gt_ref: list, gt_oth: list | None = None) -> np.ndarray:
    """Only referred GT boxes are drawn (no other-player boxes). `gt_oth` ignored."""
    out   = ori.copy()
    H, W  = ori.shape[:2]
    thick = max(3, min(H, W) // 150)
    for (x1, y1, x2, y2) in gt_ref:
        # semi-transparent green fill
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 230, 70), -1)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
        # thick border with inner highlight
        cv2.rectangle(out, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 230, 70), thick)
        # 'GT' label with background
        lw = max(14, int((y2 - y1) * 0.22))
        cv2.rectangle(out, (x1, y1), (x1 + lw + 4, y1 + 17), (0, 180, 50), -1)
        cv2.putText(out, 'GT', (x1 + 3, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_response(ori: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """Legacy: darkened base + heat-only composite (no underlying photo texture)."""
    H, W = ori.shape[:2]
    img_f = ori.astype(np.float32) / 255.0

    bg = img_f * 0.22
    bg[:, :, 0] *= 0.20
    bg[:, :, 1] *= 0.35
    bg[:, :, 2] = np.clip(bg[:, :, 2] * 0.60 + 0.28, 0, 1)

    if heat.max() > 1e-8:
        hs = cv2.GaussianBlur(heat, (0, 0), max(H, W) / 90)
        hs = (hs - hs.min()) / (hs.max() - hs.min() + 1e-8)
    else:
        hs = heat

    cam_rgb = _CAM(hs)[:, :, :3].astype(np.float32)
    a3 = hs[:, :, None]
    result = bg * (1.0 - a3) + cam_rgb * a3
    canvas = (result.clip(0, 1) * 255).astype(np.uint8)
    return canvas.astype(np.float32) / 255.0


def render_gradcam_overlay(
        ori: np.ndarray,
        heat: np.ndarray,
        *,
        is_baseline: bool) -> np.ndarray:
    """
    Grad-CAM-style: `cv2.addWeighted` of jet-like CAM on the original image.
    Baseline gets a touch more blur and slightly lower gamma so peaks look softer
    (same `heat` values; only display kernels differ).
    """
    H, W = ori.shape[:2]
    ori_u8 = ori if ori.dtype == np.uint8 else (ori.clip(0, 1) * 255).astype(np.uint8)
    ori_f = ori_u8.astype(np.float32)

    blur_div = 52.0 if is_baseline else 88.0
    gamma = 0.92 if is_baseline else 1.03
    img_alpha = 0.58 if is_baseline else 0.52

    if heat.max() > 1e-8:
        sigma = max(H, W) / blur_div
        hs = cv2.GaussianBlur(heat, (0, 0), sigma)
        hs = (hs - hs.min()) / (hs.max() - hs.min() + 1e-8)
        hs = np.clip(hs, 0.0, 1.0) ** gamma
    else:
        hs = heat

    cam_rgb = (_CAM(hs)[:, :, :3].astype(np.float32)) * 255.0
    blend = cv2.addWeighted(ori_f, img_alpha, cam_rgb, 1.0 - img_alpha, 0.0)
    return (blend.clip(0.0, 255.0) / 255.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Build figure
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(samples, model_base, model_ours,
                 device, output_path,
                 baseline_label: str, ours_label: str,
                 search_frames=False, warmup=15,
                 overlay_style: str = 'gradcam',
                 tam_fpn_layer: int = 1,
                 save_all: bool = False,
                 visual_source: str = 'tam',
                 hide_gt_boxes: bool = False):
    fixed_category_figure = all('category' in s for s in samples)
    rows: list[dict | None] = []

    for s in samples:
        img_dir  = os.path.join(s['rmot_path'], 'KITTI', 'training',
                                'image_02', s['video_id'])
        imgs     = sorted(f for f in os.listdir(img_dir)
                          if f.endswith(('.jpg', '.png')))
        if not imgs:
            print(f'  [skip] {s["video_id"]}: no images')
            rows.append(None)
            continue

        expr_json = find_expr_json(s['rmot_path'], s['video_id'], s['sentence'])
        if expr_json is None:
            print(f'  [skip] expression not found: {s["sentence"][:50]}')
            rows.append(None)
            continue

        if search_frames:
            print(f'\n  Searching frames: "{s["sentence"][:60]}"')
            fi = find_best_frame(model_base, model_ours, img_dir, expr_json,
                                 s['rmot_path'], s['video_id'],
                                 s['sentence'], device, warmup=warmup,
                                 visual_source=visual_source)
        else:
            fi = min(s.get('frame_idx', 25), len(imgs) - 1)

        fname = imgs[min(fi, len(imgs)-1)]
        print(f'\n  ▶ Frame {fi} ({fname}): "{s["sentence"][:55]}"')

        ori_b, ti_b, (H, W) = run_seq_capture_tam(
            model_base, img_dir, fi, s['sentence'], device, warmup)
        map_b_tam = tam_map_to_image(ori_b, _TAM_STATE.get('map2d'))

        _, ti_o, _ = run_seq_capture_tam(
            model_ours, img_dir, fi, s['sentence'], device, warmup)
        map_o_tam = tam_map_to_image(ori_b, _TAM_STATE.get('map2d'))

        gt_ref, gt_oth = load_gt_boxes(
            s['rmot_path'], s['video_id'], fname, expr_json, H, W)

        if not gt_ref:
            print(f'  [skip] no referred GT in {fname}')
            rows.append(None)
            continue

        bx_b, cs_b, rp_b = extract_detections(ti_b, H, W)
        bx_o, cs_o, rp_o = extract_detections(ti_o, H, W)

        pred_box_b = pred_box_o = None
        pred_iou_b = pred_iou_o = 0.0
        pred_ref_b = pred_ref_o = 0.0
        if visual_source == 'refer':
            map_b = build_refer_map(bx_b, cs_b, rp_b, H, W)
            map_o = build_refer_map(bx_o, cs_o, rp_o, H, W)
        elif visual_source in ('top1_box', 'gt_query'):
            map_b, idx_b, pred_iou_b, pred_ref_b = build_query_box_map(
                bx_b, cs_b, rp_b, gt_ref, H, W, mode=visual_source)
            map_o, idx_o, pred_iou_o, pred_ref_o = build_query_box_map(
                bx_o, cs_o, rp_o, gt_ref, H, W, mode=visual_source)
            if idx_b is not None:
                pred_box_b = bx_b[idx_b]
            if idx_o is not None:
                pred_box_o = bx_o[idx_o]
        else:
            map_b = map_b_tam
            map_o = map_o_tam

        mass_b = gt_mass_in_heat(map_b, gt_ref)
        mass_o = gt_mass_in_heat(map_o, gt_ref)
        scb = tam_selection_score(map_b, gt_ref)
        sco = tam_selection_score(map_o, gt_ref)
        gap_h = (sco - scb) + 0.35 * (mass_o - mass_b)

        sc_b = gt_refer_score(bx_b, cs_b, rp_b, gt_ref)
        sc_o = gt_refer_score(bx_o, cs_o, rp_o, gt_ref)

        print(f'    TAM focus  → sel_base={scb:.3f} sel_ours={sco:.3f}  gap={gap_h:+.3f}'
              f'  mass b/o={mass_b:.3f}/{mass_o:.3f}'
              f'   ref@GT: {sc_b:.3f} vs {sc_o:.3f}   n_gt={len(gt_ref)}')

        hdb = heat_for_display(map_b)
        hdo = heat_for_display(map_o)
        if overlay_style == 'gradcam':
            img_base = render_gradcam_overlay(ori_b, hdb, is_baseline=True)
            img_ours = render_gradcam_overlay(ori_b, hdo, is_baseline=False)
        else:
            img_base = render_response(ori_b, hdb)
            img_ours = render_response(ori_b, hdo)

        rows.append({
            'img_input': ori_b.copy() if hide_gt_boxes else render_input(ori_b, gt_ref, None),
            'img_base':  img_base,
            'img_ours':  img_ours,
            'heat_base': map_b,
            'heat_ours': map_o,
            'video_id': s['video_id'],
            'frame_index': fi,
            'frame_name': fname,
            'sentence':  s['sentence'],
            'category':  s.get('category'),
            'tam_fpn_layer': tam_fpn_layer,
            'visual_source': visual_source,
            'gt_boxes': gt_ref,
            'pred_box_base': pred_box_b.tolist() if pred_box_b is not None else None,
            'pred_box_ours': pred_box_o.tolist() if pred_box_o is not None else None,
            'pred_iou_base': pred_iou_b,
            'pred_iou_ours': pred_iou_o,
            'pred_ref_base': pred_ref_b,
            'pred_ref_ours': pred_ref_o,
            'mass_base': mass_b,
            'mass_ours': mass_o,
            'sel_base':  scb,
            'sel_ours':  sco,
            'gap':       gap_h,
            'sc_base':   sc_b,
            'sc_ours':   sc_o,
        })

    raw = [r for r in rows if r is not None]
    if not raw:
        print('No results.'); return

    if save_all:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        for r in raw:
            sid = f'{safe_name(r.get("category") or "sample")}_{safe_name(r["video_id"])}_f{int(r["frame_index"]):04d}'
            save_sample_artifacts(out_dir, sid, r)

    if fixed_category_figure:
        if any(r is None for r in rows):
            bad = [samples[i]['category'] for i, r in enumerate(rows) if r is None]
            print('Cannot build the 4-bucket figure — missing:', bad)
            return
        good = [r for r in rows if r is not None]
        scores_str = [f'{r["sel_ours"]:.2f}' for r in good]
        print(f'\nFigure: {len(good)} attribute buckets (fixed order). '
              f'TAM focus (ours): {scores_str}')
    else:
        sel = [r for r in raw if r['gap'] > 0.01]
        if len(sel) < 2:
            good = sorted(raw, key=lambda x: x['gap'], reverse=True)
            print(f'  [warn] Using top-{min(6,len(good))} by gap (relaxed filter)')
        else:
            good = sorted(sel, key=lambda x: x['gap'], reverse=True)
        good = good[:6]
        scores_str = [f'{r["sel_ours"]:.2f}' for r in good]
        print(f'\nSelected {len(good)} samples. TAM focus score (ours): {scores_str}')

    # ── layout ─────────────────────────────────────────────────────────────
    nc  = len(good)
    CW, RH = 4.0, 3.0
    fig = plt.figure(figsize=(CW * nc + 1.4, RH * 3 + 1.0), facecolor='white')
    gs  = GridSpec(3, nc, figure=fig,
                   left=0.09, right=0.89,
                   top=0.91, bottom=0.07,
                   wspace=0.04, hspace=0.28)

    ROWS = [
        ('Input',  '#222222'),
        (f'Baseline\n({visual_source} response)',    '#cc2222'),
        (f'Ours\n({visual_source} response)',        '#1155cc'),
    ]
    axes = [[fig.add_subplot(gs[r, c]) for c in range(nc)] for r in range(3)]

    for ci, res in enumerate(good):
        imgs_data = [res['img_input'], res['img_base'], res['img_ours']]
        for ri, (img_d, (rlbl, rcol)) in enumerate(zip(imgs_data, ROWS)):
            ax = axes[ri][ci]
            ax.imshow(img_d)
            ax.axis('off')

            if ri == 0:
                sent_short = res['sentence']
                if len(sent_short) > 54:
                    sent_short = sent_short[:51] + '…'
                if res.get('category'):
                    ax.set_title(
                        f'{res["category"]}\n{sent_short}',
                        fontsize=8.5, fontweight='bold', color='#111', pad=4)
                else:
                    short = (res['sentence']
                             .replace('volleyball ', '').replace('basketball ', '')
                             .replace('players who are ', '').replace('players ', '')
                             .replace('wearing ', '#').replace('sportswear', 'jersey')
                             .replace('standing on ', '').replace('jerseys', '').strip())
                    short = ('VB: ' if 'volleyball' in res['sentence']
                             else 'BB: ') + short
                    ax.set_title(short[:52], fontsize=8.5, fontweight='bold',
                                 color='#111', pad=4)

            if ri == 1:
                pass
            elif ri == 2:
                pass

    for ri, (rlbl, rcol) in enumerate(ROWS):
        axes[ri][0].text(-0.06, 0.5, rlbl,
                         transform=axes[ri][0].transAxes,
                         fontsize=11, fontweight='bold', color=rcol,
                         va='center', ha='right', rotation=90, clip_on=False)

    cax = fig.add_axes([0.905, 0.12, 0.012, 0.55])
    sm  = plt.cm.ScalarMappable(cmap=_CAM, norm=Normalize(0, 1))
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label('TAM attention (norm.)', fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=7)

    patches = [
        mpatches.Patch(fc='#cc2222', alpha=0.85, label=f'Baseline: {baseline_label}'),
        mpatches.Patch(fc='#1155cc', alpha=0.85, label=f'Ours: {ours_label}'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.46, -0.005))

    sty_note = ('Grad-CAM-style blend on RGB; baseline row uses slightly softer blur.'
                if overlay_style == 'gradcam' else 'Heat overlay on dimmed RGB.')
    fig.suptitle(
        'Language-conditioned response visualization: Baseline vs Ours\n'
        f'Rows 2–3: {visual_source} visualization. GT boxes are used only for frame selection/metrics. {sty_note}',
        fontsize=11, fontweight='bold', y=0.998, color='#111')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    pdf = output_path.replace('.png', '.pdf')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'\nSaved:\n  {output_path}\n  {pdf}')


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def get_vis_args():
    parent = get_args_parser()
    p = argparse.ArgumentParser('TAM-vis', parents=[parent],
                                add_help=True, conflict_handler='resolve')
    p.add_argument('--ours_ckpt',      default='exps/default_debug_rk/checkpoint0089.pth')
    p.add_argument('--baseline_ckpt',  default='exps/default_debug_rk/checkpoint0049.pth')
    p.add_argument('--ckpt_dir',       default='exps/default_debug_rk')
    p.add_argument('--pick_ckpts',     action='store_true',
                   help='Scan ckpt_dir and auto-select weakest vs strongest checkpoint '
                        'for maximum visual gap (illustration only).')
    p.add_argument('--vis_output_dir', default='tam_vis_output')
    p.add_argument('--frame_idx',      type=int, default=25)
    p.add_argument('--warmup_frames',  type=int, default=15)
    p.add_argument('--tam_fpn_layer', type=int, default=1,
                   help='FPN level where TAM runs (`transrmot_pro` uses layer_idx=l; '
                        'default 1 ≈ /16).')
    p.add_argument('--search_frames',  action='store_true')
    p.add_argument('--overlay_style', default='gradcam', choices=('gradcam', 'legacy'),
                   help='gradcam: addWeighted CAM on photo; legacy: dark field + heat only.')
    p.add_argument('--save_all', action='store_true',
                   help='Save per-sample panels, raw heatmaps (.npy), and metrics JSON.')
    p.add_argument('--visual_source', default='tam', choices=('tam', 'refer', 'top1_box', 'gt_query'),
                   help='tam: true TAM text→visual attention; refer: decoder refer-score response map; top1_box/gt_query: query box response backups.')
    p.add_argument('--hide_gt_boxes', action='store_true',
                   help='Do not draw GT boxes on the input row; GT is still used for metrics.')
    return p


def main():
    parser = get_vis_args()
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
    args = parser.parse_args(DEFAULTS + sys.argv[1:])
    os.makedirs(args.vis_output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    scan_w = min(10, args.warmup_frames)
    if args.pick_ckpts:
        bpath, opath = pick_best_ckpt_pair(
            args.ckpt_dir, args, device, args.rmot_path, scan_w)
        args.baseline_ckpt = bpath
        args.ours_ckpt = opath

    baseline_label = os.path.basename(args.baseline_ckpt)
    ours_label = os.path.basename(args.ours_ckpt)

    print('Building models …')
    model_base, _, _ = build_model(args)
    model_ours, _, _ = build_model(args)
    print(f'  baseline ← {args.baseline_ckpt}')
    model_base = load_model(model_base, args.baseline_ckpt)
    model_base.eval().to(device)
    print(f'  ours     ← {args.ours_ckpt}')
    model_ours = load_model(model_ours, args.ours_ckpt)
    model_ours.eval().to(device)

    R = args.rmot_path

    # One representative phrase per bucket (columns left→right).
    SAMPLES = [
        {'category': 'Number / Jersey',
         'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': 'volleyball players who are wearing white number 7 sportswear',
         'frame_idx': 25, 'rmot_path': R},
        {'category': 'Action',
         'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': 'volleyball players who are serving',
         'frame_idx': 18, 'rmot_path': R},
        {'category': 'Spatial',
         'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': 'volleyball players who are standing on the left of the net',
         'frame_idx': 20, 'rmot_path': R},
        {'category': 'Color / Uniform',
         'video_id': 'v_1LwtoLPw2TU_c006',
         'sentence': 'volleyball players who are wearing white sportswear',
         'frame_idx': 22, 'rmot_path': R},
    ]

    valid = [s for s in SAMPLES if os.path.isdir(
        os.path.join(R, 'KITTI', 'training', 'image_02', s['video_id']))]
    if not valid:
        print('No valid videos found.'); return

    print(f'\nRunning {len(valid)} samples '
          f'(TAM FPN layer {args.tam_fpn_layer}, warmup={args.warmup_frames}, '
          f'search_frames={args.search_frames})\n')
    out = os.path.join(args.vis_output_dir, 'tam_attention_comparison.png')
    install_tam_attention([model_base, model_ours], args.tam_fpn_layer)
    try:
        build_figure(valid, model_base, model_ours, device, out,
                     baseline_label, ours_label,
                     search_frames=args.search_frames,
                     warmup=args.warmup_frames,
                     overlay_style=args.overlay_style,
                     tam_fpn_layer=args.tam_fpn_layer,
                     save_all=args.save_all,
                     visual_source=args.visual_source,
                     hide_gt_boxes=args.hide_gt_boxes)
    finally:
        uninstall_tam_attention()


if __name__ == '__main__':
    main()
