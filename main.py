# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import os

import argparse
import datetime
import hashlib
import json
import os.path
import random
import time
from pathlib import Path

import numpy as np
import torch
# torch.autograd.set_detect_anomaly = True
from torch.utils.data import DataLoader
import datasets

from util.tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch, train_one_epoch_rmot
from models import build_model

from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_temporal_names', default=["STReasoner","fusion_module","refer_embed"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--meta_arch', default='deformable_detr', type=str)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--accurate_ratio', default=False, action='store_true')


    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--nu_anchors', default=1, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--enable_fpn', action='store_true')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformerf
    parser.add_argument('--enc_layers', default=6, type=int,   # gai 1 yuan  6
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--decoder_cross_self', default=False, action='store_true')
    parser.add_argument('--sigmoid_attn', default=False, action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--cj', action='store_true')
    parser.add_argument('--extra_track_attn', action='store_true')
    parser.add_argument('--loss_normalizer', action='store_true')
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--filter_ignore', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--mix_match', action='store_true',)
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_refer', default=2, type=float,
                        help="ref score coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--refer_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--gt_file_train', type=str)
    parser.add_argument('--gt_file_val', type=str)
    parser.add_argument('--coco_path', default='/data/workspace/detectron2/datasets/coco/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--append_crowd', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # end-to-end rmot settings.
    parser.add_argument('--rmot_path', default='/data_2/zyn/refer-kitti-v2', type=str)
    parser.add_argument('--sportsmot_root', default='/data/DATASETS/TRACKING/JDE/SportsMOT/sportsmot_publish', type=str)
    parser.add_argument('--sports_expression_root', default='/data1/liuyeqiang/SportsMOT/expression', type=str)
    parser.add_argument('--sports_split', default='train', choices=('train', 'val', 'test'), type=str)
    parser.add_argument('--sports_video_list', default=None, type=str,
                        help='optional deterministic subset of video IDs for development validation')
    parser.add_argument('--sports_ref_protocol', default='unknown_mask',
                        choices=('unknown_mask', 'legacy_binary'), type=str,
                        help='Refer-Sports positive-only UNKNOWN handling or historical binary compatibility')
    parser.add_argument('--sports_manifest',
                        default='/data1/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/metadata/datasets/refer_sports_manifest_global_ids_rev2.json',
                        type=str)
    parser.add_argument('--input_video', default='figs/demo.mp4', type=str)
    parser.add_argument('--data_txt_path_train',
                        default='./datasets/data_path/refer-kitti.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--img_path', default='data/valid/JPEGImages/')

    parser.add_argument('--query_interaction_layer', default='QIM', type=str,
                        help="")
    parser.add_argument('--qim_variant', default='mamba',
                        choices=('standard', 'lstm', 'temporal_attention', 'mamba'),
                        help='QIM implementation; mamba is the untouched original implementation')
    parser.add_argument('--qim_depth', default=4, type=int,
                        help='depth for the reimplemented LSTM/attention active-token mixer')
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--update_query_pos', action='store_true')

    parser.add_argument('--sampler_steps', type=int, default=[60,80,90], nargs='*')
    parser.add_argument('--sampler_lengths', type=int, default=[4,4,4,4],nargs='*')
    parser.add_argument('--exp_name', default='submit', type=str)
    parser.add_argument('--memory_bank_score_thresh', type=float, default=0.)
    parser.add_argument('--memory_bank_len', type=int, default=4)
    parser.add_argument('--memory_bank_type', type=str, default=None)
    parser.add_argument('--memory_bank_with_self_attn', action='store_true', default=False)

    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    parser.add_argument('--visualization', action='store_true', default=False)

    # code by zyn
    parser.add_argument('--fp16',action='store_true',default=False)
    parser.add_argument('--zyn4tan',action='store_true',default=False)
    parser.add_argument('--use_history_inform',action='store_true',default=False)
    parser.add_argument('--tracking',action='store_true',default=False)
    parser.add_argument('--hist_len', type=int, default=4)
    parser.add_argument('--lr_trans', type=float, default=1e-5)
    parser.add_argument('--save_dir', default='')
    # Controlled QIM ablation and step-based training controls.
    parser.add_argument('--init_common_checkpoint', default=None, type=str,
                        help='load model weights only; never restore optimizer/epoch state')
    parser.add_argument('--reset_qim', action='store_true', default=False,
                        help='exclude track_embed.* from init_common_checkpoint')
    parser.add_argument('--train_qim_only', action='store_true', default=False)
    parser.add_argument('--train_qim_and_temporal', action='store_true', default=False)
    parser.add_argument('--max_train_steps', default=0, type=int)
    parser.add_argument('--eval_every_steps', default=0, type=int)
    parser.add_argument('--save_every_steps', default=0, type=int)
    parser.add_argument('--save_steps', default=[], type=int, nargs='*',
                        help='explicit cumulative steps to checkpoint; overrides save_every_steps when non-empty')
    parser.add_argument('--resume_exact', default='', type=str,
                        help='resume the exact same experiment including optimizer and RNG state')
    parser.add_argument('--benchmark_train', action='store_true', default=False)
    # code end
    return parser


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_hash(state_dict, prefix=None):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        if prefix is not None and not name.startswith(prefix):
            continue
        value = state_dict[name]
        if not torch.is_tensor(value):
            continue
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_init_common_checkpoint(model, checkpoint_path, qim_variant, reset_qim, qim_seed=None):
    """Load only architecture-compatible model weights for controlled adaptation."""

    checkpoint_sha = _sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    source_state = checkpoint.get('model', checkpoint)
    target_state = model.state_dict()
    skip_qim = reset_qim or qim_variant != 'mamba'
    filtered = {}
    shape_errors = []
    for name, value in source_state.items():
        if skip_qim and name.startswith('track_embed.'):
            continue
        if name not in target_state:
            if not name.startswith('track_embed.'):
                shape_errors.append(f'unexpected non-QIM key: {name}')
            continue
        if tuple(value.shape) != tuple(target_state[name].shape):
            if not name.startswith('track_embed.'):
                shape_errors.append(
                    f'{name}: checkpoint {tuple(value.shape)} != model {tuple(target_state[name].shape)}'
                )
            continue
        filtered[name] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing_non_qim = [name for name in missing if not name.startswith('track_embed.')]
    unexpected_non_qim = [name for name in unexpected if not name.startswith('track_embed.')]
    if shape_errors or missing_non_qim or unexpected_non_qim:
        details = shape_errors + [f'missing non-QIM key: {x}' for x in missing_non_qim]
        details += [f'unexpected non-QIM key: {x}' for x in unexpected_non_qim]
        raise RuntimeError('common checkpoint is not architecture-compatible:\n' + '\n'.join(details))

    shared_state = {name: value for name, value in filtered.items()
                    if not name.startswith('track_embed.')}
    print('init_common_checkpoint:', checkpoint_path)
    print('shared checkpoint SHA256:', checkpoint_sha)
    print('model missing keys:', missing)
    print('model unexpected keys:', unexpected)
    print('shared parameter hash:', _state_dict_hash(shared_state))
    print('QIM initialization seed:', qim_seed)
    return checkpoint


def _configure_trainable_parameters(model, args):
    if args.train_qim_only and args.train_qim_and_temporal:
        raise ValueError('--train_qim_only and --train_qim_and_temporal are mutually exclusive')
    if args.train_qim_only or args.train_qim_and_temporal:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('track_embed.')
            if args.train_qim_and_temporal and name.startswith('STReasoner.'):
                parameter.requires_grad = True

    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    print('train mode:', 'qim_only' if args.train_qim_only else
          'qim_plus_streasoner' if args.train_qim_and_temporal else 'full_model')
    print('trainable parameters:')
    for name, parameter in trainable:
        print('  ', name, tuple(parameter.shape))
    print('trainable parameter total:', sum(parameter.numel() for _, parameter in trainable))
    return trainable


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.init_common_checkpoint:
        if args.pretrained is not None or args.resume or args.resume_exact:
            raise ValueError('--init_common_checkpoint cannot be combined with --pretrained/--resume')
        _load_init_common_checkpoint(
            model_without_ddp,
            args.init_common_checkpoint,
            args.qim_variant,
            args.reset_qim,
            seed,
        )
    elif args.reset_qim:
        print('WARNING: --reset_qim has no effect without --init_common_checkpoint')

    _configure_trainable_parameters(model_without_ddp, args)
    print(model)
    # freeze_layers = ['deepfuse','mamba','decoder']
    # for name, param in model.named_parameters():
    #     if any(freeze_layer in name for freeze_layer in freeze_layers):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)


    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    if args.dataset_file in ['e2e_rmot', 'e2e_rmot_sports', 'e2e_dance']:
        collate_fn = utils.mot_collate_fn
    else:
        collate_fn = utils.collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    if args.eval_every_steps:
        print('NOTE: --eval_every_steps is recorded, but this original tree has no '
              'Refer-Sports validation loader; attach the frozen protocol loader before enabling it.')

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if not match_name_keywords(n, args.lr_backbone_names)
                       and not match_name_keywords(n, args.lr_linear_proj_names)
                       and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    param_dicts = [group for group in param_dicts if group['params']]
    print('optimizer parameter groups:', [
        {'lr': group['lr'], 'parameters': sum(p.numel() for p in group['params'])}
        for group in param_dicts
    ])
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    has_output_dir = bool(args.save_dir or args.output_dir)
    output_dir = Path(args.save_dir or args.output_dir or '.')
    exact_resume = None
    resume_path = args.resume_exact or args.resume
    if resume_path:
        if args.resume_exact and args.init_common_checkpoint:
            raise ValueError('--resume_exact cannot be combined with --init_common_checkpoint')
        if resume_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume_path, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if args.resume_exact:
            if 'optimizer' not in checkpoint or 'lr_scheduler' not in checkpoint:
                raise RuntimeError('--resume_exact checkpoint lacks optimizer/lr_scheduler state')
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint.get('epoch', 0)
            exact_resume = checkpoint
        elif not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    if exact_resume is not None:
        _restore_rng_state(exact_resume.get('rng_state'))
        print('exact resume global_step:', exact_resume.get('global_step', 0))

    print("Start training")
    start_time = time.time()

    writer_name = str(output_dir) if str(output_dir) else './log/fgrefmot_qim_ablation'
    writer_name = writer_name.replace('/', '_').strip('_')
    writer_name = os.path.join('./log', writer_name)
    writer = SummaryWriter(writer_name)

    train_func = train_one_epoch
    if args.dataset_file in ['e2e_rmot', 'e2e_rmot_sports', 'e2e_dance']:
        train_func = train_one_epoch_rmot
        dataset_train.set_epoch(args.start_epoch)
    global_step = 0 if exact_resume is None else int(exact_resume.get('global_step', 0))
    resume_skip_steps = 0 if exact_resume is None else int(exact_resume.get('step_in_epoch', 0))
    max_train_steps = args.max_train_steps if args.max_train_steps > 0 else None

    def save_step_checkpoint(step, step_in_epoch):
        if not has_output_dir or (not args.save_steps and args.save_every_steps <= 0):
            return
        if args.save_steps:
            if step not in set(args.save_steps):
                return
        elif step % args.save_every_steps != 0:
            return
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step_in_epoch': step_in_epoch,
            'global_step': step,
            'rng_state': _capture_rng_state(),
            'args': args,
        }, output_dir / f'checkpoint_step{step:08d}.pth')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if train_func is train_one_epoch_rmot:
            remaining_steps = None if max_train_steps is None else max_train_steps - global_step
            if remaining_steps is not None and remaining_steps <= 0:
                break
            train_stats = train_func(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, writer, max_steps=remaining_steps,
                global_step=global_step, benchmark=args.benchmark_train,
                skip_steps=resume_skip_steps if epoch == args.start_epoch else 0,
                step_callback=save_step_checkpoint)
        else:
            train_stats = train_func(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, writer)
        steps_this_epoch = int(train_stats.pop('_steps', len(data_loader_train)))
        global_step += steps_this_epoch
        resume_skip_steps = 0
        lr_scheduler.step()
        if has_output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if epoch >30 and (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'step_in_epoch': steps_this_epoch,
                    'global_step': global_step,
                    'rng_state': _capture_rng_state(),
                    'args': args,
                }, checkpoint_path)

        if args.dataset_file in ['e2e_rmot', 'e2e_rmot_sports', 'e2e_dance']:
            dataset_train.step_epoch()
        if max_train_steps is not None and global_step >= max_train_steps:
            print('Reached max_train_steps:', max_train_steps)
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    main(args)
