#!/usr/bin/env python3
"""Run a fixed development Refer-Sports tracking evaluation.

This evaluator uses the original FGRefMOT inference path and the local
TrackEval HOTA/CLEAR/Identity implementations.  It is intentionally explicit
about the protocol: expressions are evaluated separately, UNKNOWN expression
labels are not treated as negative referring labels, and the published test
split is not used because it has no local ground-truth labels.
"""
import argparse
import configparser
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# TrackEval is vendored by the original repository rather than installed in
# the experiment environment.  Prefer that pinned implementation explicitly.
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKEVAL_ROOT = REPO_ROOT / 'TrackEval'
if str(TRACKEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKEVAL_ROOT))

from trackeval.metrics import CLEAR, HOTA, Identity
from trackeval.datasets._base_dataset import _BaseDataset

from main import get_args_parser
from models import build_model
from datasets.refer_sports import make_transforms_for_sports


def read_video_info(video_dir):
    ini = configparser.ConfigParser()
    ini.read(video_dir / 'seqinfo.ini')
    section = ini['Sequence']
    return {
        'width': int(section['imWidth']),
        'height': int(section['imHeight']),
        'length': int(section['seqLength']),
    }


def read_global_labels(label_dir, width, height):
    frames = {}
    for path in label_dir.glob('*.txt'):
        frame_id = int(path.stem)
        rows = []
        with path.open() as handle:
            for line in handle:
                fields = line.replace(',', ' ').split()
                if len(fields) < 6:
                    continue
                object_id = int(float(fields[1]))
                cx, cy, box_w, box_h = (float(fields[i]) for i in range(2, 6))
                x = width * (cx - box_w / 2.0)
                y = height * (cy - box_h / 2.0)
                rows.append((object_id, np.asarray([x, y, width * box_w, height * box_h], dtype=np.float32)))
        frames[frame_id] = rows
    return frames


def compact_ids(frame_rows, key):
    ids = sorted({int(row[key]) for rows in frame_rows for row in rows})
    return {object_id: i for i, object_id in enumerate(ids)}, ids


def make_metric_data(gt_frames, pred_frames):
    gt_map, gt_ids = compact_ids(gt_frames, 0)
    pred_map, pred_ids = compact_ids(pred_frames, 0)
    gt_id_frames = []
    pred_id_frames = []
    similarities = []
    gt_dets = 0
    pred_dets = 0
    for gt_rows, pred_rows in zip(gt_frames, pred_frames):
        gt_ids_t = np.asarray([gt_map[int(row[0])] for row in gt_rows], dtype=np.int64)
        pred_ids_t = np.asarray([pred_map[int(row[0])] for row in pred_rows], dtype=np.int64)
        gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=np.float32).reshape(-1, 4)
        gt_id_frames.append(gt_ids_t)
        pred_id_frames.append(pred_ids_t)
        similarities.append(_BaseDataset._calculate_box_ious(gt_boxes, pred_boxes, box_format='xywh'))
        gt_dets += len(gt_rows)
        pred_dets += len(pred_rows)
    return {
        'gt_ids': gt_id_frames,
        'tracker_ids': pred_id_frames,
        'similarity_scores': similarities,
        'num_gt_ids': len(gt_ids),
        'num_tracker_ids': len(pred_ids),
        'num_gt_dets': gt_dets,
        'num_tracker_dets': pred_dets,
        'num_timesteps': len(gt_frames),
    }


def frame_prediction(track_instances, threshold):
    rows = []
    if track_instances is None:
        return rows
    ids = track_instances.obj_idxes.detach().cpu().tolist()
    boxes = track_instances.boxes.detach().cpu().numpy()
    scores = track_instances.scores.detach().cpu().tolist()
    refers = track_instances.refers.detach().cpu().tolist()
    for object_id, box, score, refer in zip(ids, boxes, scores, refers):
        if int(object_id) < 0 or float(score) < threshold:
            continue
        x0, y0, x1, y1 = [float(x) for x in box]
        rows.append((int(object_id), np.asarray([x0, y0, x1 - x0, y1 - y0], dtype=np.float32), float(refer)))
    return rows


def referring_counts(gt_rows, pred_rows, positive_ids, threshold=0.5, iou_threshold=0.5):
    gt_positive = [(object_id, box) for object_id, box in gt_rows if object_id in positive_ids]
    if not gt_positive:
        return 0, 0, 0
    pred_positive = [row for row in pred_rows if row[2] >= threshold]
    gt_boxes = np.asarray([box for _, box in gt_positive], dtype=np.float32).reshape(-1, 4)
    pred_boxes = np.asarray([row[1] for row in pred_positive], dtype=np.float32).reshape(-1, 4)
    ious = _BaseDataset._calculate_box_ious(gt_boxes, pred_boxes, box_format='xywh')
    if ious.size == 0:
        return 0, len(pred_positive), len(gt_positive)
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-ious)
    matches = int(sum(ious[r, c] >= iou_threshold for r, c in zip(rows, cols)))
    return matches, len(pred_positive) - matches, len(gt_positive) - matches


@torch.no_grad()
def evaluate_expression(model, transform, video_dir, expression_path, device, max_frames, track_threshold):
    info = read_video_info(video_dir)
    labels = read_global_labels(video_dir.parent.parent / 'labels_with_ids' / video_dir.name,
                                info['width'], info['height'])
    expression = json.loads(expression_path.read_text())
    image_paths = sorted((video_dir / 'img1').glob('*'), key=lambda p: int(p.stem))
    if max_frames > 0:
        image_paths = image_paths[:max_frames]

    gt_frames = []
    pred_frames = []
    ref_tp = ref_fp = ref_fn = 0
    model.clear()
    track_instances = None
    for image_path in image_paths:
        frame_id = int(image_path.stem)
        img = Image.open(image_path).convert('RGB')
        raw_w, raw_h = img.size
        images, _ = transform([img], [None])
        result = model.inference_single_image([images[0].to(device)], [expression['sentence']],
                                               (raw_h, raw_w), track_instances)
        track_instances = result['track_instances']
        pred_rows_full = frame_prediction(track_instances, track_threshold)
        pred_rows = [(object_id, box) for object_id, box, _ in pred_rows_full]
        gt_rows = labels.get(frame_id, [])
        gt_frames.append(gt_rows)
        pred_frames.append(pred_rows)
        positive_ids = {int(x) for x in expression.get('label', {}).get(str(frame_id), [])}
        tp, fp, fn = referring_counts(gt_rows, pred_rows_full, positive_ids)
        ref_tp += tp
        ref_fp += fp
        ref_fn += fn

    data = make_metric_data(gt_frames, pred_frames)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}).eval_sequence(data)
    identity = Identity({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}).eval_sequence(data)
    ref_f1 = (2.0 * ref_tp / (2 * ref_tp + ref_fp + ref_fn)) if (2 * ref_tp + ref_fp + ref_fn) else 0.0
    return {
        'expression_id': '{}:{}'.format(video_dir.name, expression_path.stem),
        'sentence': expression['sentence'],
        'video': video_dir.name,
        'frames': len(image_paths),
        'hota': float(np.mean(hota['HOTA'])),
        'hota_05': float(hota['HOTA'][9]),
        'deta': float(np.mean(hota['DetA'])),
        'assa': float(np.mean(hota['AssA'])),
        'idf1': float(identity['IDF1']),
        'idsw': int(clear['IDSW']),
        'referring_f1': ref_f1,
        'ref_tp': ref_tp,
        'ref_fp': ref_fp,
        'ref_fn': ref_fn,
        'gt_dets': data['num_gt_dets'],
        'pred_dets': data['num_tracker_dets'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--qim-variant', required=True, choices=(
        'standard', 'lstm', 'temporal_attention', 'mamba',
        'mamba_ordered_bi', 'mamba_trackwise', 'mamba_reliability_hybrid',
    ))
    parser.add_argument('--sportsmot-root', default='/data/DATASETS/TRACKING/JDE/SportsMOT/sportsmot_publish')
    parser.add_argument('--expression-root', default='/data1/liuyeqiang/SportsMOT/expression')
    parser.add_argument('--video-list', required=True)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--max-expressions-per-video', type=int, default=1)
    parser.add_argument('--track-threshold', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_args_parser().parse_args([])
    args.meta_arch = 'temp_rmot'
    args.dataset_file = 'e2e_rmot_sports'
    args.device = str(device)
    args.with_box_refine = True
    args.dropout = 0.0
    args.random_drop = 0.1
    args.fp_ratio = 0.3
    args.merger_dropout = 0.0
    args.update_query_pos = True
    args.use_checkpoint = False
    args.qim_variant = cli.qim_variant
    args.sampler_lengths = [4, 4, 4, 4]
    model, _, _ = build_model(args)
    checkpoint = torch.load(cli.checkpoint, map_location='cpu')
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected = [key for key in unexpected if not key.endswith('total_params') and not key.endswith('total_ops')]
    if missing or unexpected:
        raise RuntimeError('checkpoint mismatch missing={} unexpected={}'.format(missing, unexpected))
    model.to(device).eval()
    transform = make_transforms_for_sports('val')
    root = Path(cli.sportsmot_root) / 'dataset' / 'train'
    expression_root = Path(cli.expression_root)
    videos = [x.strip() for x in Path(cli.video_list).read_text().splitlines() if x.strip()]
    results = []
    for video in videos:
        video_dir = root / video
        expression_paths = sorted((expression_root / video).glob('*.json'))
        if cli.max_expressions_per_video > 0:
            expression_paths = expression_paths[:cli.max_expressions_per_video]
        for expression_path in expression_paths:
            print('evaluating', video, expression_path.name, flush=True)
            results.append(evaluate_expression(
                model, transform, video_dir, expression_path, device,
                cli.max_frames, cli.track_threshold,
            ))

    def mean(key):
        return float(np.mean([r[key] for r in results])) if results else 0.0

    summary = {
        'checkpoint': str(Path(cli.checkpoint)),
        'qim_variant': cli.qim_variant,
        'protocol': 'development video split; one or more fixed expressions per video; all-object MOT metrics; UNKNOWN expression negatives ignored',
        'video_list': str(Path(cli.video_list)),
        'videos': videos,
        'max_frames_per_video': cli.max_frames,
        'max_expressions_per_video': cli.max_expressions_per_video,
        'track_threshold': cli.track_threshold,
        'device': str(device),
        'num_sequences': len(results),
        'mean': {
            'hota': mean('hota'),
            'hota_05': mean('hota_05'),
            'deta': mean('deta'),
            'assa': mean('assa'),
            'idf1': mean('idf1'),
            # Keep both conventions explicit.  TrackEval reports IDSW per
            # sequence; the mean is useful for comparing fixed-length
            # development sequences, while the sum is the unambiguous total.
            'idsw_sum': int(sum(int(r['idsw']) for r in results)),
            'idsw_mean_per_sequence': mean('idsw'),
            'referring_f1': mean('referring_f1'),
        },
        'sequences': results,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary['mean'], indent=2))


if __name__ == '__main__':
    main()
