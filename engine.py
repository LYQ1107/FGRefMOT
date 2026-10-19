# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import cv2
import math
import numpy as np
import torch
from torch import nn
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from util import box_ops

from torch import Tensor
from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda

import time
import datetime


def _set_frozen_stochastic_layers_eval(model):
    """Keep frozen BN/Dropout deterministic without disabling QIM gradients."""

    trainable_top_levels = {
        name.split('.', 1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    stochastic_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    for name, module in model.named_modules():
        if isinstance(module, stochastic_types):
            top_level = name.split('.', 1)[0] if name else ''
            if top_level not in trainable_top_levels:
                module.eval()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    _set_frozen_stochastic_layers_eval(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_rmot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None,
                    max_steps=None, global_step=0, benchmark=False,
                    skip_steps=0, step_callback=None):
    model.train()
    criterion.train()
    _set_frozen_stochastic_layers_eval(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    iter_id = 0
    step_times = []
    data_times = []
    forward_times = []
    backward_times = []
    no_grad_steps = 0
    previous_step_end = time.perf_counter()
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        if skip_steps > 0:
            skip_steps -= 1
            continue
        if max_steps is not None and iter_id >= max_steps:
            break
        step_start = time.perf_counter()
        data_times.append(step_start - previous_step_end)
        
        if 'sentences' in data_dict.keys():
            sentences = data_dict['sentences']
            dataset_name = data_dict['dataset_name']
            data_dict.pop('sentences')
            data_dict.pop('dataset_name')

            data_dict = data_dict_to_cuda(data_dict, device)
            data_dict['sentences'] = sentences
            data_dict['dataset_name'] = dataset_name

        else:
            data_dict = data_dict_to_cuda(data_dict, device)
        forward_start = time.perf_counter()
        outputs = model(data_dict)
        forward_times.append(time.perf_counter() - forward_start)

        loss_dict = criterion(outputs, data_dict) # 4*7*n
        weight_dict = criterion.weight_dict # 4*7*n
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if losses.requires_grad:
            backward_start = time.perf_counter()
            losses.backward()
            backward_times.append(time.perf_counter() - backward_start)
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()
        else:
            # QIM-only/partial-freeze training can produce a valid batch with
            # no active track reaching the trainable module.  Count it as a
            # no-op cumulative step instead of calling backward on a scalar
            # without an autograd graph.
            no_grad_steps += 1
            backward_times.append(0.0)
            grad_total_norm = torch.zeros((), device=losses.device)
            if no_grad_steps <= 3:
                print('NO_GRAD_STEP', global_step + iter_id + 1,
                      'loss has no trainable QIM path; optimizer step skipped')
        if step_callback is not None:
            step_callback(global_step + iter_id + 1, iter_id + 1)

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        loss_dict_reduced_scaled_wo_aux = {k:v for k, v in loss_dict_reduced.items() if 'aux' not in k} # n*8
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled_wo_aux)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes

        if writer is not None:
            writer.add_scalar('Training/loss', loss_value,
                              global_step + iter_id)
        loss_ce,loss_bbox,loss_giou,loss_refer=0,0,0,0
        temporal_loss_ce,temporal_loss_bbox,temporal_loss_giou,temporal_loss_refer=0,0,0,0
        frame_count = len(data_dict.get('imgs', []))
        for i in range(frame_count):
            if 'frame_{}_loss_ce'.format(i) not in loss_dict_reduced_scaled_wo_aux:
                continue
            loss_ce += loss_dict_reduced_scaled_wo_aux['frame_{}_loss_ce'.format(i)]
            loss_bbox += loss_dict_reduced_scaled_wo_aux['frame_{}_loss_bbox'.format(i)]
            loss_giou += loss_dict_reduced_scaled_wo_aux['frame_{}_loss_giou'.format(i)]
            loss_refer += loss_dict_reduced_scaled_wo_aux['frame_{}_loss_refer'.format(i)]
            temporal_loss_ce += loss_dict_reduced_scaled_wo_aux['frame_{}_temporal_loss_ce'.format(i)]
            temporal_loss_bbox += loss_dict_reduced_scaled_wo_aux['frame_{}_temporal_loss_bbox'.format(i)]
            temporal_loss_giou += loss_dict_reduced_scaled_wo_aux['frame_{}_temporal_loss_giou'.format(i)]
            temporal_loss_refer += loss_dict_reduced_scaled_wo_aux['frame_{}_temporal_loss_refer'.format(i)]

        if writer is not None:
            writer.add_scalar('Training/frame_loss_ce', loss_ce, global_step + iter_id)
            writer.add_scalar('Training/frame_loss_bbox', loss_bbox, global_step + iter_id)
            writer.add_scalar('Training/frame_loss_giou', loss_giou, global_step + iter_id)
            writer.add_scalar('Training/frame_loss_refer', loss_refer, global_step + iter_id)
            writer.add_scalar('Training/frame_temporal_loss_ce', temporal_loss_ce,
                              global_step + iter_id)
            writer.add_scalar('Training/frame_temporal_loss_bbox', temporal_loss_bbox,
                              global_step + iter_id)
            writer.add_scalar('Training/frame_temporal_loss_giou', temporal_loss_giou,
                              global_step + iter_id)
            writer.add_scalar('Training/frame_temporal_loss_refer', temporal_loss_refer,
                              global_step + iter_id)

        iter_id += 1
        previous_step_end = time.perf_counter()
        step_times.append(previous_step_end - step_start)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['_steps'] = iter_id
    stats['_no_grad_steps'] = no_grad_steps
    if benchmark and step_times:
        step_array = np.asarray(step_times, dtype=np.float64)
        data_array = np.asarray(data_times, dtype=np.float64)
        forward_array = np.asarray(forward_times, dtype=np.float64)
        backward_array = np.asarray(backward_times, dtype=np.float64)
        benchmark_stats = {
            'steps': int(iter_id),
            'no_grad_steps': int(no_grad_steps),
            'step_mean_ms': float(step_array.mean() * 1000.0),
            'step_median_ms': float(np.median(step_array) * 1000.0),
            'step_p95_ms': float(np.percentile(step_array, 95) * 1000.0),
            'data_mean_ms': float(data_array.mean() * 1000.0),
            'forward_mean_ms': float(forward_array.mean() * 1000.0),
            'backward_mean_ms': float(backward_array.mean() * 1000.0),
        }
        if torch.cuda.is_available():
            benchmark_stats['peak_allocated_gb'] = float(
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )
            benchmark_stats['peak_reserved_gb'] = float(
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            )
        print('TRAIN_BENCHMARK', benchmark_stats)
        stats['_benchmark'] = benchmark_stats
    return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             )
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
