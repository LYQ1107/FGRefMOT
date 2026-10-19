# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import random
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List
import torchvision

from util import box_ops
from util.misc import inverse_sigmoid
from models.structures import Boxes, Instances, pairwise_iou

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from models.structures import Boxes, matched_boxlist_iou, pairwise_iou

from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy
from models.ops.modules import MSDeformAttn
from mamba_ssm.modules.mamba_simple import Mamba
from functools import partial
from timm.models.layers import DropPath, to_2tuple

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class QueryInteractionBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out,
                 ##gai+mamba---------------------------------------
                 d_state=16,
                 embed_dim=256,
                 ssm_cfg=None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = True,
                 residual_in_fp32=True,
                 fused_add_norm=True,
                 if_bimamba=False,
                 drop_path_rate=0.1,
                 depth=4,  # gai 原来24
                 device=None,
                 dtype=None,
                 **kwargs
                 ):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos





        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.mamba_layers1 = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    # bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    # if_divide_out=if_divide_out,
                    # init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.norm_f1 = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        self.mamba_layers2 = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    # bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    # if_divide_out=if_divide_out,
                    # init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.norm_f2 = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )











    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)
        self.linear_feat_mamba = nn.Linear(dim_in, dim_in)
        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.dropout_mamba1= nn.Dropout(dropout)
        self.dropout_mamba2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.norm_mamba1 = nn.LayerNorm(dim_in)
        self.norm_mamba2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.iou <= 0.5] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        
        is_pos = track_instances.scores > 0.5
        track_instances.ref_pts[is_pos] = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone()[is_pos])

        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # gai ------------------------------------------------------------------------------------------------------------------------
        tgt_expanded = tgt.unsqueeze(0)
        inference_params = None
        residual = None
        hidden_states = tgt_expanded
        for mambalayer in self.mamba_layers1:
            hidden_states, residual = mambalayer(
                hidden_states, residual, inference_params=inference_params
            )

        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f1, RMSNorm) else layer_norm_fn
        hidden_states = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f1.weight,
            self.norm_f1.bias,
            eps=self.norm_f1.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
        )
        tgt_expanded = tgt_expanded + self.dropout_mamba1(hidden_states)
        tgt_expanded = self.norm_mamba1(tgt_expanded)
        tgt = tgt_expanded.squeeze(0)
        # gai ------------------------------------------------------------------------------------------------------------------------

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)


        if self.update_query_pos:
            # gai ------------------------------------------------------------------------------------------------------------------------
            tgt_expanded = tgt.unsqueeze(0)
            inference_params = None
            residual = None
            hidden_states = tgt_expanded
            for mambalayer in self.mamba_layers2:
                hidden_states, residual = mambalayer(
                    hidden_states, residual, inference_params=inference_params
                )

            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f2, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f2.weight,
                self.norm_f2.bias,
                eps=self.norm_f2.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            tgt_expanded = tgt_expanded + self.dropout_mamba2(hidden_states)
            tgt_expanded = self.norm_mamba2(tgt_expanded)
            tgt = tgt_expanded.squeeze(0)
            # gai ------------------------------------------------------------------------------------------------------------------------

            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2][is_pos] = query_pos[is_pos]

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[:, dim//2:][is_pos] = query_feat[is_pos]

        # pred_logits = track_instances.pred_logits.squeeze(-1).sigmoid()
        # pred_boxes = track_instances.pred_boxes
        # pred_boxes0 = pred_boxes.clone()
        # pred_boxes1 = pred_boxes.clone()
        # pred_boxes1[:, 0] = pred_boxes0[:, 0] - pred_boxes0[:, 2] / 2
        # pred_boxes1[:, 1] = pred_boxes0[:, 1] - pred_boxes0[:, 3] / 2
        # pred_boxes1[:, 2] = pred_boxes0[:, 0] + pred_boxes0[:, 2] / 2
        # pred_boxes1[:, 3] = pred_boxes0[:, 1] + pred_boxes0[:, 3] / 2
        # keep = torchvision.ops.nms(pred_boxes1, pred_logits, iou_threshold=0.9)
        # # org_len = len(track_instances)
        # track_instances = track_instances[keep]
        # # print(org_len - len(track_instances))
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances

class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)





def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    # bimamba_type="none",
    # if_divide_out=False,
    # init_layer_scale=None,
):
    # if if_bimamba:
    #     bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx,
                        # bimamba_type=bimamba_type,
                        # if_divide_out=if_divide_out,
                        # init_layer_scale=init_layer_scale,
                        **ssm_cfg, **factory_kwargs)
    # norm_cls = partial(
    #     nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    # )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def build(args, layer_name, dim_in, hidden_dim, dim_out):
    # Keep the original class and its state-dict names as the default.  The
    # alternative variants live in a separate module so loading a historical
    # Mamba checkpoint remains an exact operation.
    assert layer_name == 'QIM', 'invalid query interaction layer: {}'.format(layer_name)
    variant = getattr(args, 'qim_variant', 'mamba').lower()
    if variant == 'mamba':
        return QueryInteractionModule(args, dim_in, hidden_dim, dim_out)
    from .qim_variants import build_qim_variant
    return build_qim_variant(args, variant, dim_in, hidden_dim, dim_out)
