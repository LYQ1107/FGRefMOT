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
import copy
from typing import Optional, List, Tuple
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
    # from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from einops import rearrange, repeat

# from transformers import RobertaModel, RobertaTokenizerFast
#
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,               ## yuan num_encoder_layers=6  gai num_encoder_layers=1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False,):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          sigmoid_attn=sigmoid_attn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, decoder_self_cross,
                                                          sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, sentence_embeds=None, ref_pts=None,
                ):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # flatten操作确保数组只有3维 （1，256，100，84）-》（1，8400，256）
            mask = mask.flatten(1) # (1,100,84) -> (1,8400)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) #(1,8400,256)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1) #(1,1168,256)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 计算输入掩码 mask 中每一行和每一列的有效像素占总高度和总宽度的比例，结果全为1
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:# False
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            if ref_pts is None: # what is rference_points
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
            init_reference_out = reference_points
        # language encoder
        # sentences = [sentence.replace('\'', '') for sentence in sentences]
        # memory_text, mask_text, memory_sentence = self.forward_text(sentences, src.device) # words * expressions * dimension

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index,
                                            valid_ratios, query_embed, mask_flatten, lvl_pos_embed_flatten, sentence_embeds,
                                            )

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


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


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False,

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
                 depth=4,  #gai 原来24
                 device=None,
                 dtype=None,

                 **kwargs
                 ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, inference_params=None):#gai 多添加了inference_params参数，干什么的还未知
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src
# class DeformableTransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False,
#
#                  ##gai+mamba---------------------------------------
#                  d_state=16,
#                  embed_dim=256,
#                  ssm_cfg=None,
#                  norm_epsilon: float = 1e-5,
#                  rms_norm: bool = True,
#                  residual_in_fp32=True,
#                  fused_add_norm=True,
#                  if_bimamba=False,
#                  drop_path_rate=0.1,
#                  depth=4,  #gai 原来24
#                  device=None,
#                  dtype=None,
#
#                  **kwargs
#                  ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         # add factory_kwargs into kwargs
#         kwargs.update(factory_kwargs)
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         # self attention
#         self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#
#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         # import ipdb;ipdb.set_trace()
#         inter_dpr = [0.0] + dpr
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
#
#         self.mamba_layers = nn.ModuleList(
#             [
#                 create_block(
#                     embed_dim,
#                     d_state=d_state,
#                     ssm_cfg=ssm_cfg,
#                     norm_epsilon=norm_epsilon,
#                     rms_norm=rms_norm,
#                     residual_in_fp32=residual_in_fp32,
#                     fused_add_norm=fused_add_norm,
#                     layer_idx=i,
#                     if_bimamba=if_bimamba,
#                     # bimamba_type=bimamba_type,
#                     drop_path=inter_dpr[i],
#                     # if_divide_out=if_divide_out,
#                     # init_layer_scale=init_layer_scale,
#                     **factory_kwargs,
#                 )
#                 for i in range(depth)
#             ]
#         )
#         self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
#             embed_dim, eps=norm_epsilon, **factory_kwargs
#         )
#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos
#
#     def forward_ffn(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src
#
#     def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, inference_params=None):#gai 多添加了inference_params参数，干什么的还未知
#         # self attention
#
#         src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)  # yuan --------------------
#         src = src + self.dropout1(src2)           # yuan --------------------
#
#         # gai ------------------------------------------------------------------------------------------------------------------------
#         residual = None
#         hidden_states = src
#         for mambalayer in self.mamba_layers:
#             hidden_states, residual = mambalayer(
#                 hidden_states, residual, inference_params=inference_params
#             )
#
#         fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
#         hidden_states = fused_add_norm_fn(
#             self.drop_path(hidden_states),
#             self.norm_f.weight,
#             self.norm_f.bias,
#             eps=self.norm_f.eps,
#             residual=residual,
#             prenorm=False,
#             residual_in_fp32=self.residual_in_fp32,
#         )
#         src = src + self.dropout1(hidden_states)
#         # gai ------------------------------------------------------------------------------------------------------------------------
#
#         src = self.norm1(src)
#         # ffn
#         src = self.forward_ffn(src)
#         return src
#

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False, extra_track_attn=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.text_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout22 = nn.Dropout(dropout)
        self.norm22 = nn.LayerNorm(d_model)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        if self_cross:
            print('Training with Self-Cross Attention.')
        else:
            print('Training with Cross-Self Attention.')

        # v_l fusion
        # self.vl_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.vl_dropout = nn.Dropout(dropout)
        # self.vl_norm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    # def forward_vl_fusion(self, v_features, v_pos, l_features):
    #     q = self.with_pos_embed(v_features, v_pos)
    #     tgt = self.vl_attention(q.transpose(0, 1), l_features.transpose(0, 1), l_features.transpose(0, 1))[0].transpose(0, 1)
    #     tgt = tgt + self.vl_dropout(v_features)
    #     return self.vl_norm(tgt)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        if self.extra_track_attn: # False
            tgt = self._forward_track_attn(tgt, query_pos)

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None: # False
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2) # 残差连接
        # 归一化
        return self.norm2(tgt)


    def _forward_crosstext_attn(self, tgt, query_pos, attn_mask=None):
        if self.extra_track_attn: # False
            tgt = self._forward_track_attn(tgt, query_pos)

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None: # False
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2) # 残差连接
        # 归一化
        return self.norm2(tgt)

    def _forward_track_attn(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > 300:
            tgt2 = self.update_attn(q[:, 300:].transpose(0, 1),
                                    k[:, 300:].transpose(0, 1),
                                    tgt[:, 300:].transpose(0, 1))[0].transpose(0, 1)
            tgt = torch.cat([tgt[:, :300],self.norm4(tgt[:, 300:]+self.dropout5(tgt2))], dim=1)
        return tgt

    def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None, lvl_pos_embed_flatten=None,text_word_features=None, text_word_mask=None):

        # self attention
        # tgt = tgt + sentence_embeds
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask )
        # v_l cross attention
        # src = self.forward_vl_fusion(src, lvl_pos_embed_flatten, memory_text)
        # level_extra_index = torch.as_tensor(src.shape[1], dtype=torch.long, device=level_start_index.device)
        # level_start_index = torch.cat([level_start_index, level_extra_index.view(1, )])
        # src = torch.cat([src, memory_text], dim=1)
        # src_padding_mask = torch.cat([src_padding_mask, mask_text], dim=1)
        # src = src + memory_text
        # cross attention

        tgt22 = self.text_self_attn(self.with_pos_embed(tgt, query_pos).transpose(0, 1),          #gai
                                   text_word_features,
                                   text_word_features, attn_mask=None,
                                   key_padding_mask=text_word_mask,
                                   )[0].transpose(0, 1)
        tgt22 = tgt + self.dropout22(tgt22)
        tgt22=self.norm22(tgt22)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt22, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None, lvl_pos_embed_flatten=None):
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, lvl_pos_embed_flatten=None,text_word_features=None, text_word_mask=None ):
        attn_mask = None
        if self.self_cross:# True
            return self._forward_self_cross(tgt, query_pos, reference_points, src, src_spatial_shapes,
                                            level_start_index, src_padding_mask, attn_mask, lvl_pos_embed_flatten,text_word_features, text_word_mask)
        return self._forward_cross_self(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                                        src_padding_mask, attn_mask, lvl_pos_embed_flatten)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, lvl_pos_embed_flatten=None, sentence_embeds=None):
        # b, n, c = tgt.shape
        # output = repeat(sentence_embeds, 'b c -> b n c', n=n) + tgt # rmot_4a
        # output = repeat(sentence_embeds, 'b c -> b n c', n=n) # rmot_4b

        text_word_features, text_word_mask = sentence_embeds   #gai

        output = tgt # rmot_3

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask, lvl_pos_embed_flatten,
                           text_word_features, text_word_mask  #gai
                           )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points,query_pos,


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PositionEmbeddingSine2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor, hidden_dim: int):
        """
        @param mask: a tensor of shape [B, H, W]
        @param hidden_dim: int
        @return:
        """
        num_pos_feats = hidden_dim // 2

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2)) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos


class Triple_Alignment(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads):
        super().__init__()
        self.layers = _get_clones(Triple_Alignment_MHA(embed_dim, num_heads), num_layers)

    def forward(self,
                tgt_src=None, tgt_pos=None,
                visual_src=None, visual_mask: Optional[Tensor] = None, visual_pos: Optional[Tensor] = None,
                text_src=None, text_mask: Optional[Tensor] = None, layer_idx=0):
        layer = self.layers[layer_idx]
        visual_output, text_output, tgt_src = \
            layer(tgt_src=tgt_src, tgt_pos=tgt_pos,
                  visual_src=visual_src, visual_mask=visual_mask, visual_pos=visual_pos,
                  text_src=text_src, text_mask=text_mask, layer_idx=layer_idx)
        return visual_output, text_output, tgt_src


class Triple_Alignment_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(Triple_Alignment_MHA, self).__init__()
        self.embed_dim = embed_dim
        self.d_h = embed_dim // num_heads
        self.h = num_heads
        self.d_h_t = 256 // num_heads

        self.norm_o = nn.LayerNorm(embed_dim)
        self.in_object_q = nn.Linear(embed_dim, embed_dim)
        self.in_object_k = nn.Linear(embed_dim, embed_dim)
        self.in_object_v = nn.Linear(embed_dim, embed_dim)
        self.dropout_o = nn.Dropout(dropout)

        self.norm_v = nn.LayerNorm(embed_dim)
        self.in_visual_q = nn.Linear(embed_dim, embed_dim)
        self.in_visual_k = nn.Linear(embed_dim, embed_dim)
        self.in_visual_v = nn.Linear(embed_dim, embed_dim)
        self.dropout_v = nn.Dropout(dropout)

        self.norm_t = nn.LayerNorm(256)
        self.in_text_q = nn.Linear(256, embed_dim)
        self.in_text_k = nn.Linear(256, embed_dim)
        self.in_text_v = nn.Linear(256, embed_dim)
        self.dropout_t = nn.Dropout(dropout)

        self.out_object = nn.Linear(embed_dim, embed_dim)
        self.out_visual = nn.Linear(embed_dim, embed_dim)
        self.out_text = nn.Linear(embed_dim, 256)

        # Optional: dump softmax(QK^T/sqrt(d)) for token grounding figures [B, L, L]
        self._export_attn: bool = False
        self._export_at_layer_idx: Optional[int] = None
        self.last_attn_probs: Optional[Tensor] = None
        self.last_otv_layout: Optional[Tuple[int, int, int]] = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward(self, tgt_src=None, tgt_pos=None,
                visual_src=None, visual_mask: Optional[Tensor] = None, visual_pos: Optional[Tensor] = None,
                text_src=None, text_mask: Optional[Tensor] = None, layer_idx=None):
        o_len, bs, _ = tgt_src.shape
        v_len, bs, _ = visual_src.shape
        t_len, bs, _ = text_src.shape
        object_src2 = self.norm_o(tgt_src)
        visual_src2 = self.norm_v(visual_src)
        text_src2 = self.norm_t(text_src)

        # linear projection
        object_q = self.in_object_q(self.with_pos_embed(object_src2, tgt_pos))
        object_k = self.in_object_k(self.with_pos_embed(object_src2, tgt_pos))
        object_v = self.in_object_v(object_src2)
        visual_q = self.in_visual_q(self.with_pos_embed(visual_src2, visual_pos))
        visual_k = self.in_visual_k(self.with_pos_embed(visual_src2, visual_pos))
        visual_v = self.in_visual_v(visual_src2)
        text_q = self.in_text_q(text_src2)
        text_k = self.in_text_k(text_src2)
        text_v = self.in_text_v(text_src2)
        otv_q = torch.cat([object_q, text_q, visual_q], dim=0)  # (N+L+V), bs, C
        otv_k = torch.cat([object_k, text_k, visual_k], dim=0)  # (N+L+V), bs, C
        otv_v = torch.cat([object_v, text_v, visual_v], dim=0)  # (N+L+V), bs, C
        otv_len = o_len + t_len + v_len

        # multi head
        otv_q = otv_q.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)  # bs*h, (N+L+V), C_h
        otv_k = otv_k.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)
        otv_v = otv_v.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)

        # add mask
        object_mask = torch.zeros(bs * self.h, 1, o_len, dtype=torch.bool).to(visual_mask.device)
        visual_mask = visual_mask.view(bs, 1, 1, v_len). \
            expand(-1, self.h, -1, -1).reshape(bs * self.h, 1, v_len)
        text_mask = text_mask.view(bs, 1, 1, t_len). \
            expand(-1, self.h, -1, -1).reshape(bs * self.h, 1, t_len)
        key_padding_mask = torch.cat([object_mask, text_mask, visual_mask], dim=-1)  # bs*h, 1, (N+L+V)
        attn_mask = torch.zeros_like(key_padding_mask, dtype=otv_q.dtype)
        attn_mask.masked_fill_(key_padding_mask,
                               float("-inf"))  # Fills elements of self tensor with -inf where key_padding_mask is True
        # attn
        attn = torch.baddbmm(attn_mask,
                             otv_q / math.sqrt(self.d_h), otv_k.transpose(-2, -1))  # bs*h, (L+V), (L+V)

        # DropKey
        # 原来有
        # mask_ratio = 0.1
        # m_r = torch.ones_like(attn) * mask_ratio
        # attn = attn + torch.bernoulli(m_r) * -1e-12
        attn = F.softmax(attn, dim=-1)

        # Viz-only: keep False in train / eval unless a script enables it (no overhead otherwise).
        if self._export_attn and (
                self._export_at_layer_idx is None or layer_idx == self._export_at_layer_idx):
            probs = attn.view(bs, self.h, otv_len, otv_len).float().mean(dim=1).detach()
            self.last_attn_probs = probs
            self.last_otv_layout = (int(o_len), int(t_len), int(v_len))

        # (bs*h), Len, embed/h -> Len, (bs*h), embed/h -> Len, bs, embed
        attn_output = torch.bmm(attn, otv_v).transpose(0, 1).contiguous().view(otv_len, bs, self.embed_dim)

        o_out = self.dropout_o(self.out_object(attn_output[:o_len])) + tgt_src
        v_out = self.dropout_v(self.out_visual(attn_output[-v_len:])) + visual_src
        t_out = self.dropout_t(self.out_text(attn_output[o_len:-v_len])) + text_src

        return v_out, t_out, o_out

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)# 没有任何丢弃

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        return tgt

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def  __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # self.fc1 = nn.Linear(output_feat_size, output_feat_size, bias=True)
        # self.layer_norm1 = nn.LayerNorm(output_feat_size, eps=1e-12)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        # output = self.dropout(x)
        # x = x.mean(1).unsqueeze(1)
        # x = self.fc1(x)
        # if self.do_ln:
        #     x = self.layer_norm1(x)
        return x

# 256 8 6 6 1024 0 4 4 4 False 300 True False False
def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
    )


