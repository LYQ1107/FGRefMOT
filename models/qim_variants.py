"""QIM alternatives used by the original-FGRefMOT ablation.

The repository does not contain the historical Standard/LSTM/Temporal-Attention
implementations.  These modules are therefore deliberately small, explicit
reimplementations.  They share the active-track selection, reference-point
update, track lifecycle fields, and query writer with the original QIM.  The
only variant-specific operation is the mixer applied to the active-track token
axis ``[N_active, 256]``.

This file must not be confused with the FGRefMOT-v2 QIM/TAM code.  No temporal
identity memory or new TAM is introduced here.
"""

from typing import Callable

import torch
from torch import nn

from models.structures import Boxes, Instances, pairwise_iou
from util import box_ops
from util.misc import inverse_sigmoid


def _random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    """Keep the original FGRefMOT random-drop rule for all variants."""

    if drop_probability > 0 and len(track_instances) > 0:
        keep_indices = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_indices]
    return track_instances


class _VariantQIMBase(nn.Module):
    """Common QIM contract with a replaceable active-token mixer."""

    variant_name = "REIMPLEMENTED_VARIANT"

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.random_drop = getattr(args, "random_drop", 0.0)
        self.fp_ratio = getattr(args, "fp_ratio", 0.0)
        self.update_query_pos = getattr(args, "update_query_pos", False)
        self.merger_dropout = getattr(args, "merger_dropout", 0.0)
        self.qim_depth = max(1, int(getattr(args, "qim_depth", 4)))

        self._build_common_layers()
        self._build_variant_layers()
        self._reset_parameters()

    def _build_common_layers(self):
        dropout = self.merger_dropout
        # This is the MOTR/FGRefMOT self-attention stage shared by all direct
        # replacement variants.
        self.self_attn = nn.MultiheadAttention(self.dim_in, 8, dropout)
        self.linear1 = nn.Linear(self.dim_in, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.hidden_dim, self.dim_in)

        # Keep the original query-writer names and dimensions.
        self.linear_feat_mamba = nn.Linear(self.dim_in, self.dim_in)
        self.linear_feat1 = nn.Linear(self.dim_in, self.hidden_dim)
        self.linear_feat2 = nn.Linear(self.hidden_dim, self.dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(self.dim_in)

        self.norm1 = nn.LayerNorm(self.dim_in)
        self.norm2 = nn.LayerNorm(self.dim_in)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(True)

        if self.update_query_pos:
            self.linear_pos1 = nn.Linear(self.dim_in, self.hidden_dim)
            self.linear_pos2 = nn.Linear(self.hidden_dim, self.dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(self.dim_in)

    def _build_variant_layers(self):
        # Standard QIM has exactly the shared self-attention + FFN path.
        self.mixer1 = nn.Identity()
        self.mixer2 = nn.Identity()

    def _reset_parameters(self):
        # Match QueryInteractionBase's simple initialization policy while
        # leaving LayerNorm affine parameters at their PyTorch defaults.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return _random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:
        """Retain the original false-positive augmentation rule.

        The original forward path does not call this helper by default, but it
        remains part of the QIM contract so every variant exposes the same
        training controls and can be audited without changing lifecycle rules.
        """

        inactive_instances = track_instances[track_instances.obj_idxes < 0]
        fp_probability = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active = active_track_instances[torch.bernoulli(fp_probability).bool()]

        if len(inactive_instances) > 0 and len(selected_active) > 0:
            number_fp = len(selected_active)
            if number_fp >= len(inactive_instances):
                fp_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_boxes)
                fp_indices = torch.unique(ious.max(dim=0).indices)
                fp_instances = inactive_instances[fp_indices]
            return Instances.cat([active_track_instances, fp_instances])
        return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        """Exact active-track selection used by original ``QueryInteractionModule``."""

        track_instances: Instances = data["track_instances"]
        if self.training:
            active_indices = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_track_instances = track_instances[active_indices]
            active_track_instances.obj_idxes[active_track_instances.iou <= 0.5] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]
        return active_track_instances

    @staticmethod
    def _self_attention_ffn(tgt, query_pos, self_attn, dropout1, norm1,
                            linear1, dropout, activation, linear2, dropout2, norm2):
        q = k = query_pos + tgt
        tgt2 = self_attn(q[:, None], k[:, None], value=tgt[:, None], need_weights=False)[0][:, 0]
        tgt = norm1(tgt + dropout1(tgt2))
        tgt2 = linear2(dropout(activation(linear1(tgt))))
        tgt = norm2(tgt + dropout2(tgt2))
        return tgt

    def _run_mixer(self, mixer: Callable, tgt):
        return mixer(tgt)

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances

        is_positive = track_instances.scores > 0.5
        # Preserve the original detached reference-point update exactly.
        track_instances.ref_pts[is_positive] = inverse_sigmoid(
            track_instances.pred_boxes[:, :2].detach().clone()[is_positive]
        )

        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim // 2:]

        tgt = self._self_attention_ffn(
            out_embed,
            query_pos,
            self.self_attn,
            self.dropout1,
            self.norm1,
            # The FFN is run after mixer1 below for direct replacement
            # variants, so this call is not used by the implementation.
            self.linear1,
            self.dropout,
            self.activation,
            self.linear2,
            self.dropout2,
            self.norm2,
        )
        # Standard QIM needs self-attention followed by FFN.  LSTM and
        # Temporal-Attention direct replacements override this whole core.
        return self._finish_query_writer(track_instances, is_positive, dim,
                                         query_pos, query_feat, tgt)

    def _finish_query_writer(self, track_instances, is_positive, dim,
                             query_pos, query_feat, tgt):
        if self.update_query_pos:
            query_pos2 = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt)))
            )
            query_pos = self.norm_pos(query_pos + self.dropout_pos2(query_pos2))
            track_instances.query_pos[:, :dim // 2][is_positive] = query_pos[is_positive]

        query_feat2 = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt)))
        )
        query_feat = self.norm_feat(query_feat + self.dropout_feat2(query_feat2))
        track_instances.query_pos[:, dim // 2:][is_positive] = query_feat[is_positive]
        return track_instances

    def forward(self, data: dict) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        return Instances.cat([init_track_instances, active_track_instances])


class StandardQIM(_VariantQIMBase):
    """MOTR-style self-attention + FFN QIM (reimplemented)."""

    variant_name = "REIMPLEMENTED_VARIANT:standard"


class LSTMQIM(_VariantQIMBase):
    """Track-token LSTM replacement on the original Mamba sequence axis."""

    variant_name = "REIMPLEMENTED_VARIANT:track-token-lstm"

    def _build_variant_layers(self):
        dropout = self.merger_dropout if self.qim_depth > 1 else 0.0
        self.lstm1 = nn.LSTM(
            self.dim_in, self.dim_in, num_layers=self.qim_depth,
            dropout=dropout, batch_first=False
        )
        self.lstm2 = nn.LSTM(
            self.dim_in, self.dim_in, num_layers=self.qim_depth,
            dropout=dropout, batch_first=False
        )
        self.lstm_norm1 = nn.LayerNorm(self.dim_in)
        self.lstm_norm2 = nn.LayerNorm(self.dim_in)
        self.lstm_dropout1 = nn.Dropout(self.merger_dropout)
        self.lstm_dropout2 = nn.Dropout(self.merger_dropout)

    @staticmethod
    def _run_lstm(lstm, norm, dropout, tgt):
        mixed, _ = lstm(tgt.unsqueeze(1))  # [N, 1, D], same axis as Mamba input.
        return norm(tgt + dropout(mixed[:, 0]))

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        is_positive = track_instances.scores > 0.5
        track_instances.ref_pts[is_positive] = inverse_sigmoid(
            track_instances.pred_boxes[:, :2].detach().clone()[is_positive]
        )
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim // 2:]

        q = k = query_pos + out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], out_embed[:, None], need_weights=False)[0][:, 0]
        tgt = self.norm1(out_embed + self.dropout1(tgt2))
        tgt = self._run_lstm(self.lstm1, self.lstm_norm1, self.lstm_dropout1, tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        if self.update_query_pos:
            tgt = self._run_lstm(self.lstm2, self.lstm_norm2, self.lstm_dropout2, tgt)
        return self._finish_query_writer(track_instances, is_positive, dim,
                                         query_pos, query_feat, tgt)


class TemporalAttentionQIM(_VariantQIMBase):
    """Self-attention mixer on the same active-track token axis (reimplemented)."""

    variant_name = "REIMPLEMENTED_VARIANT:active-token-attention"

    def _build_variant_layers(self):
        self.temporal_attn1 = nn.ModuleList(
            [nn.MultiheadAttention(self.dim_in, 8, self.merger_dropout)
             for _ in range(self.qim_depth)]
        )
        self.temporal_norm1 = nn.ModuleList(
            [nn.LayerNorm(self.dim_in) for _ in range(self.qim_depth)]
        )
        self.temporal_attn2 = nn.ModuleList(
            [nn.MultiheadAttention(self.dim_in, 8, self.merger_dropout)
             for _ in range(self.qim_depth)]
        )
        self.temporal_norm2 = nn.ModuleList(
            [nn.LayerNorm(self.dim_in) for _ in range(self.qim_depth)]
        )
        self.temporal_dropout1 = nn.Dropout(self.merger_dropout)
        self.temporal_dropout2 = nn.Dropout(self.merger_dropout)

    @staticmethod
    def _run_attention_stack(layers, norms, dropout, tgt):
        for attention, norm in zip(layers, norms):
            mixed = attention(
                tgt[:, None], tgt[:, None], tgt[:, None], need_weights=False
            )[0][:, 0]
            tgt = norm(tgt + dropout(mixed))
        return tgt

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        is_positive = track_instances.scores > 0.5
        track_instances.ref_pts[is_positive] = inverse_sigmoid(
            track_instances.pred_boxes[:, :2].detach().clone()[is_positive]
        )
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim // 2:]

        q = k = query_pos + out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], out_embed[:, None], need_weights=False)[0][:, 0]
        tgt = self.norm1(out_embed + self.dropout1(tgt2))
        tgt = self._run_attention_stack(
            self.temporal_attn1, self.temporal_norm1, self.temporal_dropout1, tgt
        )
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        if self.update_query_pos:
            tgt = self._run_attention_stack(
                self.temporal_attn2, self.temporal_norm2, self.temporal_dropout2, tgt
            )
        return self._finish_query_writer(track_instances, is_positive, dim,
                                         query_pos, query_feat, tgt)


def build_qim_variant(args, variant: str, dim_in: int, hidden_dim: int, dim_out: int):
    variants = {
        "standard": StandardQIM,
        "lstm": LSTMQIM,
        "temporal_attention": TemporalAttentionQIM,
    }
    key = variant.lower()
    if key not in variants:
        raise ValueError(f"unknown reimplemented QIM variant: {variant}")
    return variants[key](args, dim_in, hidden_dim, dim_out)

