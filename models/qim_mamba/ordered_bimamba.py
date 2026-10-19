"""M1: deterministic geometry-ordered bidirectional Mamba."""

from __future__ import annotations

import torch
from torch import nn

from .shared_utils import (
    canonical_geometry_order,
    make_mamba_stack,
    resolve_mamba_dim,
)


class OrderedBiMambaMixer(nn.Module):
    """Mix active-track tokens in a canonical geometry order.

    The backward scan is flipped back to the canonical token alignment before
    fusion, and then the inverse permutation restores the caller's row order.
    """

    def __init__(self, args, dim_in: int):
        super().__init__()
        self.mamba_dim = resolve_mamba_dim(args, "mamba_ordered_bi")
        self.bidirectional_share = bool(
            getattr(args, "qim_bidirectional_share", True)
        )
        self.input_proj = nn.Linear(dim_in, self.mamba_dim)
        self.forward_stack = make_mamba_stack(args, self.mamba_dim)
        self.backward_stack = (
            self.forward_stack
            if self.bidirectional_share
            else make_mamba_stack(args, self.mamba_dim)
        )
        self.output_proj = nn.Linear(self.mamba_dim, dim_in)
        self.dropout = nn.Dropout(float(getattr(args, "merger_dropout", 0.0)))
        self.norm = nn.LayerNorm(dim_in)

    def forward(self, features: torch.Tensor, track_instances):
        if features.ndim != 2:
            raise ValueError(f"OrderedBiMambaMixer expects [N,D], got {features.shape}")
        if features.shape[0] == 0:
            return features

        order = canonical_geometry_order(track_instances, features)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        ordered = features[order]
        encoded = self.input_proj(ordered).unsqueeze(0)
        forward = self.forward_stack(encoded)
        backward = torch.flip(
            self.backward_stack(torch.flip(encoded, dims=[1])), dims=[1]
        )
        mixed = self.output_proj((forward + backward).squeeze(0) / 2)
        mixed = mixed[inverse]
        return self.norm(features + self.dropout(mixed))
