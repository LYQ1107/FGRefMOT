"""Causal temporal evidence encoders for one track at a time."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn


@dataclass
class TemporalEvidenceOutput:
    summary: Tensor
    encoded_history: Tensor
    valid_mask: Tensor


class TemporalEvidenceEncoder(nn.Module):
    """Encode ``[N_tracks, T_history, D]`` with an explicit time axis.

    ``valid_mask`` uses ``True`` for an observed history slot.  Online callers
    must pass history ending at ``t-1``; this module does not manufacture
    future information and every implementation masks invalid slots.
    """

    MODES = {"ema", "gru", "attention", "mamba"}

    def __init__(
        self,
        input_dim: int,
        mode: Literal["ema", "gru", "attention", "mamba"] = "ema",
        *,
        ema_decay: float = 0.8,
        attention_heads: int = 4,
        mamba_state: int = 16,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if mode == "attention" and input_dim % attention_heads != 0:
            raise ValueError("input_dim must be divisible by attention_heads")
        self.input_dim = int(input_dim)
        self.mode = mode
        self.ema_decay = float(ema_decay)
        if mode == "gru":
            self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        elif mode == "attention":
            self.attention_query = nn.Parameter(torch.zeros(1, 1, input_dim))
            nn.init.normal_(self.attention_query, std=0.02)
            self.attention = nn.MultiheadAttention(
                input_dim, attention_heads, batch_first=True
            )
        elif mode == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Temporal Mamba requires the mamba_ssm environment"
                ) from exc
            self.mamba = Mamba(
                d_model=input_dim,
                d_state=mamba_state,
                d_conv=4,
                expand=2,
            )

    @staticmethod
    def _last_valid_index(valid_mask: Tensor) -> Tensor:
        positions = torch.arange(valid_mask.shape[1], device=valid_mask.device)
        positions = positions.unsqueeze(0).expand_as(valid_mask)
        return torch.where(valid_mask, positions, torch.zeros_like(positions)).max(dim=1).values

    def _ema(self, history: Tensor, valid: Tensor) -> Tensor:
        steps = history.shape[1]
        powers = torch.arange(steps - 1, -1, -1, device=history.device, dtype=history.dtype)
        weights = self.ema_decay**powers
        weights = weights.unsqueeze(0) * valid.to(history.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(history.dtype).eps)
        return (history * weights.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, history: Tensor, valid_mask: Optional[Tensor] = None) -> TemporalEvidenceOutput:
        if history.ndim != 3:
            raise ValueError("history must have shape [N_tracks, T_history, D]")
        if history.shape[-1] != self.input_dim:
            raise ValueError(f"expected history dimension {self.input_dim}")
        tracks, steps, _ = history.shape
        if valid_mask is None:
            valid = torch.ones((tracks, steps), dtype=torch.bool, device=history.device)
        else:
            if valid_mask.shape != (tracks, steps):
                raise ValueError("valid_mask must have shape [N_tracks, T_history]")
            valid = valid_mask.to(device=history.device, dtype=torch.bool)
        if steps == 0:
            return TemporalEvidenceOutput(
                history.new_zeros((tracks, self.input_dim)),
                history.new_zeros(history.shape),
                valid,
            )
        if tracks == 0:
            empty = history.new_zeros(history.shape)
            return TemporalEvidenceOutput(history.new_zeros((0, self.input_dim)), empty, valid)
        masked_history = history * valid.to(history.dtype).unsqueeze(-1)

        if self.mode == "ema":
            encoded = masked_history
            summary = self._ema(history, valid)
        elif self.mode == "gru":
            encoded, _ = self.gru(masked_history)
            last = self._last_valid_index(valid)
            summary = encoded[torch.arange(tracks, device=history.device), last]
            summary = summary * valid.any(dim=1, keepdim=True).to(summary.dtype)
        elif self.mode == "attention":
            safe_valid = valid.clone()
            all_invalid = ~safe_valid.any(dim=1)
            safe_valid[all_invalid, 0] = True
            query = self.attention_query.to(history).expand(tracks, -1, -1)
            attended, _ = self.attention(
                query,
                masked_history,
                masked_history,
                key_padding_mask=~safe_valid,
                need_weights=False,
            )
            summary = attended[:, 0]
            summary = summary * (~all_invalid).to(summary.dtype).unsqueeze(-1)
            encoded = masked_history
        else:  # mamba
            encoded = self.mamba(masked_history)
            last = self._last_valid_index(valid)
            summary = encoded[torch.arange(tracks, device=history.device), last]
            summary = summary * valid.any(dim=1, keepdim=True).to(summary.dtype)

        encoded = encoded * valid.to(encoded.dtype).unsqueeze(-1)
        return TemporalEvidenceOutput(summary, encoded, valid)
