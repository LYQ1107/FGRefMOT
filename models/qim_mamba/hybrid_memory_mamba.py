"""M3: trackwise temporal Mamba with reliability-gated identity memory."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .shared_utils import box_iou_xywh, masked_mean
from .temporal_mamba import TrackwiseTemporalMixer


class ReliabilityGate(nn.Module):
    """Small learned gate over inference-available reliability signals."""

    def __init__(self, signal_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Start in the middle of the range; the bounded output prevents a
        # trivial always-open/always-closed memory at initialization.
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.network(signals).sigmoid().squeeze(-1).clamp(0.05, 0.95)


class ReliabilityHybridMambaMixer(nn.Module):
    """Short trackwise dynamics plus a long-term identity memory.

    Memory writes use score, appearance consistency, motion consistency,
    validity, age, and ambiguity only.  Referring scores/text are intentionally
    absent from the gate inputs; optional text is applied only to the short
    writer path when explicitly enabled.
    """

    requires_qim_memory = True

    def __init__(self, args, dim_in: int):
        super().__init__()
        self.temporal = TrackwiseTemporalMixer(args, dim_in)
        self.memory_dim = self.temporal.mamba_dim
        self.memory_write = nn.Linear(dim_in, self.memory_dim)
        self.memory_read = nn.Linear(self.memory_dim, dim_in)
        self.gate = ReliabilityGate()
        self.fuse_norm = nn.LayerNorm(dim_in)
        self.uses_rmot_text_gate = bool(getattr(args, "qim_rmot_text_gate", False))
        self.text_gate_projection = (
            nn.Linear(dim_in, dim_in) if self.uses_rmot_text_gate else None
        )
        self.last_write_ratio = 0.0

    @staticmethod
    def _field(track_instances, name: str, default: torch.Tensor) -> torch.Tensor:
        return track_instances.get(name) if track_instances.has(name) else default

    def _signals(self, current, track_instances, memory, valid, age):
        number = current.shape[0]
        device = current.device
        scores = self._field(
            track_instances, "scores", torch.zeros(number, device=device)
        ).detach().float().clamp(0, 1)
        previous_box = self._field(
            track_instances, "qim_previous_box", torch.zeros(number, 4, device=device)
        ).detach().float()
        boxes = self._field(
            track_instances, "pred_boxes", torch.zeros(number, 4, device=device)
        ).detach().float()
        memory_projection = self.memory_read(memory.detach())
        appearance = F.cosine_similarity(
            current.float(), memory_projection.float(), dim=-1, eps=1e-6
        )
        appearance = ((appearance + 1) / 2).where(valid, torch.zeros_like(appearance))
        previous_nonzero = previous_box.abs().sum(dim=-1) > 1e-6
        motion = box_iou_xywh(boxes, previous_box).where(
            valid & previous_nonzero, torch.zeros_like(scores)
        )
        age_signal = age.float().clamp(min=0, max=32) / 32
        ambiguity = 1 - scores
        signals = torch.stack(
            [scores, appearance, motion, valid.float(), age_signal, ambiguity], dim=-1
        )
        return signals, scores, motion, boxes

    def _update_memory(self, current, track_instances, data=None):
        number = current.shape[0]
        device = current.device
        memory = self._field(
            track_instances,
            "qim_long_memory",
            torch.zeros(number, self.memory_dim, device=device, dtype=current.dtype),
        )
        valid = self._field(
            track_instances,
            "qim_long_memory_valid",
            torch.zeros(number, device=device, dtype=torch.bool),
        ).bool()
        age = self._field(
            track_instances,
            "qim_memory_age",
            torch.zeros(number, device=device, dtype=torch.long),
        ).long()
        memory = memory.detach()
        phi = F.normalize(self.memory_write(current), dim=-1, eps=1e-6)

        signals, scores, motion, boxes = self._signals(
            current, track_instances, memory, valid, age
        )
        gate = self.gate(signals)
        reliability = scores * (0.5 + 0.5 * motion)
        effective_gate = (gate * reliability).clamp(0, 1)

        obj_idxes = self._field(
            track_instances, "obj_idxes", torch.full((number,), -1, device=device)
        )
        active = (obj_idxes >= 0) | (scores > 0.5)
        birth = (~valid) & active & (scores > 0.5)
        updated = torch.where(
            birth[:, None],
            phi,
            torch.where(
                valid[:, None],
                (1 - effective_gate[:, None]) * memory
                + effective_gate[:, None] * phi,
                memory,
            ),
        )
        new_valid = valid | birth
        wrote = birth | (valid & (effective_gate > 0.25))
        new_age = torch.where(
            new_valid,
            torch.where(wrote, torch.zeros_like(age), age + 1),
            torch.zeros_like(age),
        )

        # State fields are ordinary Instances fields, so cat/index/to and the
        # checkpoint rebuild path carry them without special serialization.
        track_instances.qim_long_memory = updated
        track_instances.qim_long_memory_valid = new_valid
        track_instances.qim_memory_age = new_age
        track_instances.qim_previous_box = boxes.detach().clone()
        track_instances.qim_memory_write_gate = effective_gate.detach()
        self.last_write_ratio = float(effective_gate.detach().mean().cpu()) if number else 0.0
        return gate, valid, memory

    def forward(self, current, track_instances, data: Optional[dict] = None):
        history = track_instances.get("hist_embeds") if track_instances.has("hist_embeds") else None
        padding = (
            track_instances.get("hist_padding_masks")
            if track_instances.has("hist_padding_masks")
            else None
        )
        short = self.temporal(history, padding, current)
        gate, valid, old_memory = self._update_memory(current, track_instances, data)
        identity = self.memory_read(old_memory)
        identity = torch.where(valid[:, None], identity, current)
        fused = current + 0.5 * (short - current) + gate[:, None] * identity

        if self.uses_rmot_text_gate and data is not None:
            text = data.get("qim_text_features")
            text_mask = data.get("qim_text_mask")
            if text is not None and self.text_gate_projection is not None:
                if text.ndim == 2:
                    text = text.unsqueeze(0)
                context = masked_mean(text, text_mask)
                if context.shape[-1] == current.shape[-1]:
                    fused = fused + 0.1 * self.text_gate_projection(context[:1]).expand_as(fused)

        return self.fuse_norm(fused)
