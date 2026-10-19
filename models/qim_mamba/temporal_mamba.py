"""M2: one causal Mamba sequence per track."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .shared_utils import make_mamba_stack, resolve_history_len, resolve_mamba_dim


class TrackwiseTemporalMixer(nn.Module):
    """Encode each track's valid history independently.

    Rows are gathered by their actual non-padding tokens before entering the
    Mamba.  Right-padding is used only to batch rows of different lengths, and
    the state returned for each row is the last valid state, so padding tokens
    cannot change the result.
    """

    def __init__(self, args, dim_in: int):
        super().__init__()
        variant = getattr(args, "qim_variant", "mamba_trackwise")
        self.mamba_dim = resolve_mamba_dim(args, variant)
        self.history_len = resolve_history_len(args)
        self.input_proj = nn.Linear(dim_in, self.mamba_dim)
        self.stack = make_mamba_stack(args, self.mamba_dim)
        self.output_proj = nn.Linear(self.mamba_dim, dim_in)
        self.dropout = nn.Dropout(float(getattr(args, "merger_dropout", 0.0)))
        self.norm = nn.LayerNorm(dim_in)

    def _encode_valid_batch(
        self,
        history: torch.Tensor,
        padding_mask: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all non-empty histories in one batched causal pass.

        Each row keeps its last ``history_len`` valid tokens and is right
        padded to the same length.  Mamba is causal, so states before the
        padding are independent of the zero suffix; gathering each row's
        last valid state is therefore equivalent to running the original
        variable-length sequence one row at a time.
        """
        number, _, dim_in = history.shape
        lengths = valid_counts.clamp(max=self.history_len)
        valid = ~padding_mask.bool()
        ranks = valid.to(torch.long).cumsum(dim=1)
        starts = (valid_counts - lengths).unsqueeze(1)
        keep = valid & (ranks > starts)

        row_idx, time_idx = keep.nonzero(as_tuple=True)
        position = ranks[row_idx, time_idx] - starts[row_idx, 0] - 1
        flat_idx = row_idx * self.history_len + position

        selected = self.input_proj(history[row_idx, time_idx])
        batched = history.new_zeros(
            number * self.history_len,
            self.mamba_dim,
        )
        scatter_idx = flat_idx.unsqueeze(1).expand(-1, self.mamba_dim)
        batched = batched.scatter(0, scatter_idx, selected)
        encoded = self.stack(batched.view(number, self.history_len, self.mamba_dim))

        safe_last = lengths.clamp(min=1) - 1
        rows = torch.arange(number, device=history.device)
        last = encoded[rows, safe_last]
        return self.output_proj(last)

    def forward(
        self,
        history: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        current: torch.Tensor,
    ) -> torch.Tensor:
        if current.ndim != 2:
            raise ValueError(f"TrackwiseTemporalMixer expects current [N,D], got {current.shape}")
        if history is None or padding_mask is None:
            return current
        if history.ndim != 3 or padding_mask.ndim != 2:
            raise ValueError(
                "TrackwiseTemporalMixer expects history [N,T,D] and padding [N,T]"
            )
        if history.shape[:2] != padding_mask.shape or history.shape[0] != current.shape[0]:
            raise ValueError("history, padding mask, and current track counts disagree")
        if history.shape[0] == 0:
            return current

        valid_counts = (~padding_mask.bool()).sum(dim=1)
        has_history = valid_counts > 0
        if not bool(has_history.any()):
            return current

        update = self._encode_valid_batch(history, padding_mask, valid_counts)
        mixed = self.norm(current + self.dropout(update))
        return torch.where(has_history.unsqueeze(1), mixed, current)
