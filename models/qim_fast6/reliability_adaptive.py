"""F6: reliability-adaptive causal Mamba with identity memory."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.qim_mamba.shared_utils import box_iou_xywh

from .batched_temporal import BatchedTemporalMambaMixer
from .common import Fast6QIMBase, fast_dim
from .state_manager import ensure_bool_field, ensure_box_field, ensure_long_field, ensure_tensor_field


class ReliabilityAdaptiveGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 24), nn.ReLU(inplace=True), nn.Linear(24, 1)
        )
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, signals):
        return self.network(signals).sigmoid().squeeze(-1).clamp(0.05, 0.95)


class ReliabilityAdaptiveMambaMixer(nn.Module):
    requires_qim_memory = True

    def __init__(self, args, dim_in):
        super().__init__()
        self.memory_dim = fast_dim(args)
        self.temporal = BatchedTemporalMambaMixer(args, dim_in)
        self.memory_write = nn.Linear(dim_in, self.memory_dim)
        self.memory_read = nn.Linear(self.memory_dim, dim_in)
        self.gate = ReliabilityAdaptiveGate()
        self.fuse_norm = nn.LayerNorm(dim_in)
        self.last_write_ratio = 0.0

    def forward(self, current, tracks, data=None):
        number = current.shape[0]
        history = tracks.get("hist_embeds") if tracks.has("hist_embeds") else None
        padding = tracks.get("hist_padding_masks") if tracks.has("hist_padding_masks") else None
        short = self.temporal(history, padding, current)

        memory = ensure_tensor_field(
            tracks, "qim_long_memory", (number, self.memory_dim), dtype=current.dtype
        ).to(dtype=current.dtype)
        valid = ensure_bool_field(tracks, "qim_long_memory_valid")
        age = ensure_long_field(tracks, "qim_memory_age")
        previous = ensure_box_field(tracks, "qim_previous_box").detach().float()
        boxes = tracks.pred_boxes.detach().float()
        scores = tracks.scores.detach().float().clamp(0, 1)
        appearance = F.cosine_similarity(
            current.float(), self.memory_read(memory).float(), dim=-1, eps=1e-6
        )
        appearance = ((appearance + 1) / 2).where(valid, torch.zeros_like(scores))
        overlap = box_iou_xywh(boxes, previous).where(valid, torch.zeros_like(scores))
        signals = torch.stack(
            [scores, appearance, overlap, valid.float(), age.float().clamp(0, 32) / 32, 1 - scores],
            dim=-1,
        )
        gate = self.gate(signals.to(dtype=current.dtype))
        reliability = scores * (0.5 + 0.5 * overlap)
        effective = (gate * reliability).clamp(0, 1)
        write = F.normalize(self.memory_write(current), dim=-1, eps=1e-6)
        active = tracks.obj_idxes >= 0
        birth = (~valid) & active & (scores > 0.5)
        updated = torch.where(
            birth[:, None], write,
            torch.where(valid[:, None], (1 - effective[:, None]) * memory + effective[:, None] * write, memory),
        )
        wrote = birth | (valid & (effective > 0.25))
        tracks.qim_long_memory = updated
        tracks.qim_long_memory_valid = valid | birth
        tracks.qim_memory_age = torch.where(
            tracks.qim_long_memory_valid,
            torch.where(wrote, torch.zeros_like(age), age + 1),
            torch.zeros_like(age),
        )
        tracks.qim_previous_box = boxes.to(tracks.query_pos.dtype)
        tracks.qim_memory_write_gate = effective.detach()
        self.last_write_ratio = float(effective.detach().mean().cpu()) if number else 0.0
        identity = self.memory_read(memory)
        identity = torch.where(valid[:, None], identity, current)
        fused = current + 0.5 * (short - current) + gate[:, None] * identity
        return self.fuse_norm(fused)


class ReliabilityAdaptiveMambaQIM(Fast6QIMBase):
    variant_name = "FAST6:reliability_adaptive_mamba"
    requires_qim_memory = True

    def _build_variant_layers(self):
        self.mixer = ReliabilityAdaptiveMambaMixer(self.args, self.dim_in)
        self.memory_dim = self.mixer.memory_dim

    def _variant_mix(self, tgt, tracks, data):
        return self.mixer(tgt, tracks, data=data)
