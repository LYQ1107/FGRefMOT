"""Temporal referring heads used by the registered FlexHook ablations."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


FLEXHOOK_VARIANTS = tuple(f"H{i}" for i in range(9))
Variant = Literal["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]


class TemporalReferringHead(nn.Module):
    """Independent per-candidate referring logits with causal evidence inputs.

    The head never applies softmax across candidates.  Its output is
    ``[batch, candidates, 1]`` so zero, one, or multiple tracks may be
    selected for one expression.
    """

    def __init__(
        self,
        feature_dim: int,
        language_dim: int | None = None,
        hidden_dim: int | None = None,
        variant: Variant = "H8",
    ) -> None:
        super().__init__()
        if variant not in FLEXHOOK_VARIANTS:
            raise ValueError(f"unknown FlexHook temporal variant: {variant}")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        hidden_dim = int(hidden_dim or feature_dim)
        language_dim = int(language_dim or feature_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = hidden_dim
        self.variant = variant

        self.current_proj = nn.Linear(feature_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.identity_proj = nn.Linear(feature_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(feature_dim, hidden_dim)
        self.capacity_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _candidate_language(language: Tensor, batch: int, candidates: int) -> Tensor:
        if language.ndim == 2:
            if language.shape[0] != batch:
                raise ValueError("language batch dimension does not match current features")
            return language[:, None, :].expand(batch, candidates, language.shape[-1])
        if language.ndim == 3:
            if language.shape[:2] != (batch, candidates):
                raise ValueError("per-candidate language shape does not match current features")
            return language
        raise ValueError("language must have shape [B, D] or [B, N, D]")

    @staticmethod
    def _optional(value: Tensor | None, reference: Tensor) -> Tensor:
        if value is None:
            return torch.zeros_like(reference)
        if value.shape != reference.shape:
            raise ValueError(f"evidence shape {tuple(value.shape)} != {tuple(reference.shape)}")
        return value.to(dtype=reference.dtype, device=reference.device)

    def forward(
        self,
        current: Tensor,
        language: Tensor,
        *,
        identity: Tensor | None = None,
        dynamic: Tensor | None = None,
        dynamic_history: Tensor | None = None,
        reliability: Tensor | None = None,
    ) -> Tensor:
        if current.ndim != 3 or current.shape[-1] != self.feature_dim:
            raise ValueError("current must have shape [B, N, feature_dim]")
        batch, candidates, _ = current.shape
        lang = self._candidate_language(language, batch, candidates)
        current_h = self.current_proj(current)
        if self.variant in {"H1", "H8"}:
            current_h = current_h + self.capacity_mlp(current)

        identity_value = self._optional(identity, current)
        dynamic_value = self._optional(dynamic, current)
        if dynamic_history is not None:
            if dynamic_history.ndim != 4 or dynamic_history.shape[:2] != (batch, candidates):
                raise ValueError("dynamic_history must have shape [B, N, T, D]")
            if dynamic_history.shape[-1] != self.feature_dim:
                raise ValueError("dynamic_history feature dimension mismatch")
            history_flat = dynamic_history.reshape(batch * candidates, dynamic_history.shape[2], self.feature_dim)
            _, hidden = self.temporal_gru(history_flat)
            dynamic_value = hidden[-1].reshape(batch, candidates, self.hidden_dim)
        else:
            dynamic_value = self.dynamic_proj(dynamic_value)

        identity_h = self.identity_proj(identity_value)
        if reliability is not None:
            rel = reliability.to(dtype=current.dtype, device=current.device)
            if rel.ndim == 2:
                rel = rel.unsqueeze(-1)
            if rel.shape != (batch, candidates, 1):
                raise ValueError("reliability must have shape [B, N] or [B, N, 1]")
            identity_h = identity_h * rel
            dynamic_value = dynamic_value * (0.5 + 0.5 * rel)

        use_identity = self.variant in {"H2", "H4", "H6", "H7", "H8"}
        use_dynamic = self.variant in {"H3", "H5", "H6", "H7", "H8"}
        if not use_identity:
            identity_h = torch.zeros_like(identity_h)
        if not use_dynamic:
            dynamic_value = torch.zeros_like(current_h)
        language_h = self.language_proj(lang)
        fused = torch.cat([current_h, language_h, identity_h, dynamic_value], dim=-1)
        return self.classifier(fused)

