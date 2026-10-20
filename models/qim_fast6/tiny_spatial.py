"""F1: geometry-ordered Tiny Spatial Mamba."""

from __future__ import annotations

import torch
from torch import nn

from models.qim_mamba.shared_utils import canonical_geometry_order

from .common import Fast6QIMBase, fast_dim, make_fast_stack


class TinySpatialMambaMixer(nn.Module):
    """One low-width causal scan over a deterministic spatial token order."""

    def __init__(self, args, dim_in):
        super().__init__()
        width = fast_dim(args)
        self.input_proj = nn.Linear(dim_in, width)
        self.mamba = make_fast_stack(args, width)
        self.output_proj = nn.Linear(width, dim_in)
        self.dropout = nn.Dropout(float(getattr(args, "merger_dropout", 0.0)))
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, features, tracks):
        if len(features) == 0:
            return features
        order = canonical_geometry_order(tracks, features)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        tokens = self.input_proj(features[order]).unsqueeze(0)
        mixed = self.output_proj(self.mamba(tokens).squeeze(0))[inverse]
        return self.norm(features + self.dropout(mixed))


class TinySpatialMambaQIM(Fast6QIMBase):
    variant_name = "FAST6:tiny_spatial_mamba"

    def _build_variant_layers(self):
        self.mixer = TinySpatialMambaMixer(self.args, self.dim_in)

    def _variant_mix(self, tgt, tracks, data):
        return self.mixer(tgt, tracks)
