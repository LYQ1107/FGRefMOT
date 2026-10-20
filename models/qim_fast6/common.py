"""Shared contract for the six lightweight Mamba-QIM variants.

Only the mixer between the original query self-attention and query writer is
variant-specific.  Active-track selection, reference-point updates, query
writer, false-positive controls, and the ``Instances`` lifecycle remain the
same across all six methods.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from models.structures import Boxes, Instances, pairwise_iou
from util import box_ops
from util.misc import inverse_sigmoid
from models.qim_mamba.shared_utils import (
    Mamba as MambaImplementation,
    MambaStack,
    masked_mean,
)


def fast_dim(args) -> int:
    value = int(getattr(args, "qim_fast_dim", 64) or 64)
    if value <= 0:
        raise ValueError("qim_fast_dim must be positive")
    return value


def fast_layers(args) -> int:
    return max(1, int(getattr(args, "qim_fast_layers", 1)))


def fast_state(args) -> int:
    return max(1, int(getattr(args, "qim_fast_state", 8)))


def fast_history_len(args) -> int:
    return max(1, int(getattr(args, "qim_fast_history_len", 4)))


def make_fast_stack(args, d_model: int) -> MambaStack:
    """Build the explicitly selected Mamba1 backend for a fast-six method."""

    if bool(getattr(args, "qim_mamba2_enable", False)):
        raise NotImplementedError(
            "fast-six uses Mamba1; Mamba2 must be screened as a separate backend"
        )
    return MambaStack(
        d_model=d_model,
        d_state=fast_state(args),
        layers=fast_layers(args),
    )


def random_drop_tracks(track_instances: Instances, probability: float) -> Instances:
    if probability > 0 and len(track_instances) > 0:
        keep = torch.rand_like(track_instances.scores) > probability
        return track_instances[keep]
    return track_instances


def _text_context(data: Optional[dict], dim: int, device, dtype) -> torch.Tensor:
    """Return a single text context from model-produced tokens.

    The language-conditioned method is allowed to read text features because
    they are available before the current frame prediction.  This helper never
    reads labels or referring scores.
    """

    if not data:
        return torch.zeros(dim, device=device, dtype=dtype)
    text = data.get("qim_text_features")
    mask = data.get("qim_text_mask")
    if text is None:
        return torch.zeros(dim, device=device, dtype=dtype)
    text = text.to(device=device, dtype=dtype)
    if text.ndim == 1:
        context = text
    elif text.ndim == 2:
        context = text.mean(dim=0)
    elif text.ndim == 3:
        if mask is not None and tuple(mask.shape) == tuple(text.shape[:2]):
            context = masked_mean(text, mask)
            context = context.reshape(-1, context.shape[-1]).mean(dim=0)
        else:
            context = text.reshape(-1, text.shape[-1]).mean(dim=0)
    else:
        context = text.reshape(-1, text.shape[-1]).mean(dim=0)
    if context.numel() != dim:
        raise ValueError(
            "language context has dimension {}, expected {}".format(context.numel(), dim)
        )
    return torch.nan_to_num(context)


class Fast6QIMBase(nn.Module):
    """Original QIM lifecycle with a low-dimensional variant mixer."""

    variant_name = "FAST6"
    requires_qim_memory = False
    requires_qim_state = False

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

        self.self_attn = nn.MultiheadAttention(dim_in, 8, self.merger_dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(self.merger_dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)
        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.dropout1 = nn.Dropout(self.merger_dropout)
        self.dropout2 = nn.Dropout(self.merger_dropout)
        self.activation = nn.ReLU(True)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(self.merger_dropout)
        self.dropout_feat2 = nn.Dropout(self.merger_dropout)
        self.norm_feat = nn.LayerNorm(dim_in)
        if self.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(self.merger_dropout)
            self.dropout_pos2 = nn.Dropout(self.merger_dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.disable_self_attention = bool(
            getattr(args, "qim_fast_disable_self_attention", False)
            or getattr(args, "qim_disable_self_attention", False)
        )
        self._build_variant_layers()
        self._reset_parameters()

    def _build_variant_layers(self):
        raise NotImplementedError

    def _reset_parameters(self):
        mamba_parameter_ids = set()
        if MambaImplementation is not None:
            for module in self.modules():
                if isinstance(module, MambaImplementation):
                    mamba_parameter_ids.update(id(p) for p in module.parameters())
        for parameter in self.parameters():
            if parameter.dim() > 1 and id(parameter) not in mamba_parameter_ids:
                nn.init.xavier_uniform_(parameter)

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances = data["track_instances"]
        if self.training:
            active = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            active_tracks = track_instances[active]
            active_tracks.obj_idxes[active_tracks.iou <= 0.5] = -1
            return active_tracks
        return track_instances[track_instances.obj_idxes >= 0]

    def _add_fp_tracks(self, track_instances, active_track_instances):
        inactive = track_instances[track_instances.obj_idxes < 0]
        probability = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected = active_track_instances[torch.bernoulli(probability).bool()]
        if len(inactive) == 0 or len(selected) == 0:
            return active_track_instances
        if len(selected) >= len(inactive):
            return Instances.cat([active_track_instances, inactive])
        inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive.pred_boxes))
        selected_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected.pred_boxes))
        ious = pairwise_iou(inactive_boxes, selected_boxes)
        indices = torch.unique(ious.max(dim=0).indices)
        return Instances.cat([active_track_instances, inactive[indices]])

    def _finish_query_writer(self, tracks, positive, dim, query_pos, query_feat, tgt):
        if self.update_query_pos:
            update = self.linear_pos2(
                self.dropout_pos1(self.activation(self.linear_pos1(tgt)))
            )
            query_pos = self.norm_pos(query_pos + self.dropout_pos2(update))
            tracks.query_pos[:, : dim // 2][positive] = query_pos[positive]
        update = self.linear_feat2(
            self.dropout_feat1(self.activation(self.linear_feat1(tgt)))
        )
        query_feat = self.norm_feat(query_feat + self.dropout_feat2(update))
        tracks.query_pos[:, dim // 2 :][positive] = query_feat[positive]
        return tracks

    def _run_shared_core(self, tracks, data=None):
        if len(tracks) == 0:
            return tracks
        positive = tracks.scores > 0.5
        tracks.ref_pts[positive] = inverse_sigmoid(
            tracks.pred_boxes[:, :2].detach().clone()[positive]
        )
        dim = tracks.query_pos.shape[1]
        out_embed = tracks.output_embedding
        query_pos = tracks.query_pos[:, : dim // 2]
        query_feat = tracks.query_pos[:, dim // 2 :]
        if self.disable_self_attention:
            tgt = out_embed
        else:
            q = k = query_pos + out_embed
            tgt2 = self.self_attn(
                q[:, None], k[:, None], out_embed[:, None], need_weights=False
            )[0][:, 0]
            tgt = self.norm1(out_embed + self.dropout1(tgt2))
        tgt = self._variant_mix(tgt, tracks, data)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        return self._finish_query_writer(
            tracks, positive, dim, query_pos, query_feat, tgt
        )

    def _variant_mix(self, tgt, tracks, data):
        raise NotImplementedError

    def forward(self, data: dict) -> Instances:
        active = self._select_active_tracks(data)
        active = self._run_shared_core(active, data=data)
        return Instances.cat([data["init_track_instances"], active])


__all__ = [
    "Fast6QIMBase",
    "fast_dim",
    "fast_layers",
    "fast_state",
    "fast_history_len",
    "make_fast_stack",
    "random_drop_tracks",
    "_text_context",
]
