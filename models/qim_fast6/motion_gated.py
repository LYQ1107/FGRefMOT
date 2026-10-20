"""F4: motion-gated Tiny Spatial Mamba."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.qim_mamba.shared_utils import box_iou_xywh

from .common import Fast6QIMBase
from .state_manager import ensure_bool_field, ensure_box_field
from .tiny_spatial import TinySpatialMambaMixer


class MotionGatedMambaMixer(nn.Module):
    def __init__(self, args, dim_in):
        super().__init__()
        self.spatial = TinySpatialMambaMixer(args, dim_in)
        self.gate = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        self.last_gate = 0.0

    def forward(self, current, tracks):
        boxes = tracks.pred_boxes.detach().float()
        previous = ensure_box_field(tracks, "qim_motion_previous_box").detach().float()
        valid = ensure_bool_field(tracks, "qim_motion_previous_valid")
        delta = (boxes - previous).abs()
        overlap = box_iou_xywh(boxes, previous).where(valid, torch.zeros_like(valid, dtype=boxes.dtype))
        scores = tracks.scores.detach().float().clamp(0, 1)
        signals = torch.cat((delta, overlap[:, None], scores[:, None]), dim=-1)
        gate = self.gate(signals.to(dtype=current.dtype)).sigmoid().squeeze(-1)
        mixed = self.spatial(current, tracks)
        output = current + gate[:, None] * (mixed - current)
        tracks.qim_motion_previous_box = boxes.detach().to(tracks.query_pos.dtype)
        tracks.qim_motion_previous_valid = torch.ones_like(valid)
        self.last_gate = float(gate.detach().mean().cpu()) if len(gate) else 0.0
        return F.layer_norm(output, (output.shape[-1],))


class MotionGatedMambaQIM(Fast6QIMBase):
    variant_name = "FAST6:motion_gated_mamba"

    def _build_variant_layers(self):
        self.mixer = MotionGatedMambaMixer(self.args, self.dim_in)

    def _variant_mix(self, tgt, tracks, data):
        return self.mixer(tgt, tracks)
