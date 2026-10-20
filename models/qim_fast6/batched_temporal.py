"""F2: batched causal temporal Mamba over per-track history."""

from __future__ import annotations

import torch
from torch import nn

from .common import Fast6QIMBase, fast_dim, make_fast_stack


class BatchedTemporalMambaMixer(nn.Module):
    """Pack valid history tokens into one batch and read each last state."""

    def __init__(self, args, dim_in):
        super().__init__()
        width = fast_dim(args)
        self.input_proj = nn.Linear(dim_in, width)
        self.mamba = make_fast_stack(args, width)
        self.output_proj = nn.Linear(width, dim_in)
        self.dropout = nn.Dropout(float(getattr(args, "merger_dropout", 0.0)))
        self.norm = nn.LayerNorm(dim_in)

    @staticmethod
    def _compact(history, padding):
        valid = ~padding.bool()
        number, length, dim = history.shape
        counts = valid.sum(dim=1)
        ranks = valid.long().cumsum(dim=1) - 1
        rows, cols = valid.nonzero(as_tuple=True)
        compact = history.new_zeros(number, length, dim)
        if rows.numel():
            compact[rows, ranks[rows, cols]] = history[rows, cols]
        return compact, counts

    def forward(self, history, padding, current):
        if history is None or padding is None:
            return current
        if history.ndim != 3 or padding.ndim != 2:
            raise ValueError("BatchedTemporalMambaMixer expects [N,T,D] and [N,T]")
        if history.shape[:2] != padding.shape or history.shape[0] != current.shape[0]:
            raise ValueError("history, padding, and current counts disagree")
        if history.shape[0] == 0:
            return current
        compact, counts = self._compact(history, padding)
        encoded = self.mamba(self.input_proj(compact))
        safe_last = counts.clamp(min=1) - 1
        rows = torch.arange(history.shape[0], device=history.device)
        update = self.output_proj(encoded[rows, safe_last])
        mixed = self.norm(current + self.dropout(update))
        return torch.where((counts > 0)[:, None], mixed, current)


class BatchedTemporalMambaQIM(Fast6QIMBase):
    variant_name = "FAST6:batched_temporal_mamba"

    def _build_variant_layers(self):
        self.mixer = BatchedTemporalMambaMixer(self.args, self.dim_in)

    def _variant_mix(self, tgt, tracks, data):
        history = tracks.get("hist_embeds") if tracks.has("hist_embeds") else None
        padding = (
            tracks.get("hist_padding_masks")
            if tracks.has("hist_padding_masks")
            else None
        )
        return self.mixer(history, padding, tgt)
