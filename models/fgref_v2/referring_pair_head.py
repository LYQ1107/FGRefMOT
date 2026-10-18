"""Per-candidate language--track referring heads.

The output is one independent logit per candidate.  There is deliberately no
candidate-wise softmax: an expression may refer to zero, one, or many tracks.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _valid_mean(values: Tensor, valid: Optional[Tensor]) -> Tensor:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, N, D] or [B, L, D]")
    if values.shape[1] == 0:
        return values.new_zeros(values.shape[0], values.shape[2])
    if valid is None:
        return values.mean(dim=1)
    weights = valid.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class ReferringPairHead(nn.Module):
    """Track-conditioned cross-attention followed by an independent logit."""

    def __init__(
        self,
        query_dim: int,
        embed_dim: int,
        *,
        text_dim: Optional[int] = None,
        evidence_dim: Optional[int] = None,
        history_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim or query_dim, embed_dim)
        self.local_proj = nn.Linear(evidence_dim or query_dim, embed_dim)
        self.context_proj = nn.Linear(evidence_dim or query_dim, embed_dim)
        self.history_proj = nn.Linear(history_dim or query_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.null_evidence = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_evidence, std=0.02)
        hidden_dim = hidden_dim or 2 * embed_dim
        self.output = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _check_candidates(values: Optional[Tensor], batch: int, tracks: int, name: str) -> None:
        if values is not None and (values.ndim != 4 or values.shape[:2] != (batch, tracks)):
            raise ValueError(f"{name} must have shape [B, N, K, D]")

    def forward(
        self,
        queries: Tensor,
        local_evidence: Optional[Tensor],
        context_evidence: Optional[Tensor],
        text_tokens: Optional[Tensor],
        history_features: Optional[Tensor] = None,
        *,
        local_valid_mask: Optional[Tensor] = None,
        context_valid_mask: Optional[Tensor] = None,
        text_valid_mask: Optional[Tensor] = None,
        history_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if queries.ndim != 3:
            raise ValueError("queries must have shape [B, N, D]")
        batch, tracks = queries.shape[:2]
        self._check_candidates(local_evidence, batch, tracks, "local_evidence")
        self._check_candidates(context_evidence, batch, tracks, "context_evidence")
        self._check_candidates(history_features, batch, tracks, "history_features")
        if text_tokens is not None and (text_tokens.ndim != 3 or text_tokens.shape[0] != batch):
            raise ValueError("text_tokens must have shape [B, L, D]")
        if tracks == 0:
            return queries.new_zeros((batch, 0, 1))

        query = self.query_proj(queries)
        key_parts = []
        valid_parts = []
        if local_evidence is not None:
            local = self.local_proj(local_evidence)
            key_parts.append(local)
            if local_valid_mask is None:
                valid_parts.append(torch.ones(local.shape[:-1], dtype=torch.bool, device=local.device))
            else:
                valid_parts.append(local_valid_mask.to(device=local.device, dtype=torch.bool))
        if context_evidence is not None:
            context = self.context_proj(context_evidence)
            key_parts.append(context)
            if context_valid_mask is None:
                valid_parts.append(torch.ones(context.shape[:-1], dtype=torch.bool, device=context.device))
            else:
                valid_parts.append(context_valid_mask.to(device=context.device, dtype=torch.bool))
        if history_features is not None:
            history = self.history_proj(history_features)
            key_parts.append(history)
            if history_valid_mask is None:
                valid_parts.append(torch.ones(history.shape[:-1], dtype=torch.bool, device=history.device))
            else:
                valid_parts.append(history_valid_mask.to(device=history.device, dtype=torch.bool))
        if text_tokens is not None and text_tokens.shape[1] > 0:
            text = self.text_proj(text_tokens).unsqueeze(1).expand(-1, tracks, -1, -1)
            key_parts.append(text)
            if text_valid_mask is None:
                valid_parts.append(torch.ones(text.shape[:3], dtype=torch.bool, device=text.device))
            else:
                text_valid = text_valid_mask.to(device=text.device, dtype=torch.bool)
                if text_valid.ndim != 2 or text_valid.shape != text_tokens.shape[:2]:
                    raise ValueError("text_valid_mask must have shape [B, L]")
                valid_parts.append(text_valid.unsqueeze(1).expand(-1, tracks, -1))

        if key_parts:
            keys = torch.cat(key_parts, dim=2)
            valid = torch.cat(valid_parts, dim=2)
        else:
            keys = query.new_zeros((batch, tracks, 0, query.shape[-1]))
            valid = torch.zeros((batch, tracks, 0), dtype=torch.bool, device=query.device)
        null = self.null_evidence.to(device=query.device, dtype=query.dtype).expand(batch, tracks, -1, -1)
        keys = torch.cat([keys, null], dim=2)
        valid = torch.cat([valid, torch.ones((batch, tracks, 1), dtype=torch.bool, device=query.device)], dim=2)

        flat_query = query.reshape(batch * tracks, 1, -1)
        flat_keys = keys.reshape(batch * tracks, keys.shape[2], -1)
        flat_padding = ~valid.reshape(batch * tracks, keys.shape[2])
        attended, _ = self.cross_attention(
            flat_query,
            flat_keys,
            flat_keys,
            key_padding_mask=flat_padding,
            need_weights=False,
        )
        fused = torch.cat([flat_query[:, 0], attended[:, 0]], dim=-1)
        return self.output(fused).reshape(batch, tracks, 1)


class ReferringMLPHead(nn.Module):
    """A pooled MLP control with the same independent-logit contract."""

    def __init__(
        self,
        query_dim: int,
        embed_dim: int,
        *,
        text_dim: Optional[int] = None,
        evidence_dim: Optional[int] = None,
        history_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 2 * embed_dim
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim or query_dim, embed_dim)
        self.local_proj = nn.Linear(evidence_dim or query_dim, embed_dim)
        self.context_proj = nn.Linear(evidence_dim or query_dim, embed_dim)
        self.history_proj = nn.Linear(history_dim or query_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(5 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        queries: Tensor,
        local_evidence: Optional[Tensor],
        context_evidence: Optional[Tensor],
        text_tokens: Optional[Tensor],
        history_features: Optional[Tensor] = None,
        *,
        local_valid_mask: Optional[Tensor] = None,
        context_valid_mask: Optional[Tensor] = None,
        text_valid_mask: Optional[Tensor] = None,
        history_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if queries.ndim != 3:
            raise ValueError("queries must have shape [B, N, D]")
        batch, tracks = queries.shape[:2]
        if tracks == 0:
            return queries.new_zeros((batch, 0, 1))
        q = self.query_proj(queries)

        def pool_candidate(values: Optional[Tensor], valid: Optional[Tensor], projection: nn.Module) -> Tensor:
            if values is None:
                return q.new_zeros((batch, tracks, q.shape[-1]))
            b, n, k, d = values.shape
            flat = projection(values.reshape(b * n, k, d)).reshape(b, n, k, -1)
            flat = flat.reshape(b * n, k, -1)
            if valid is None:
                pooled = flat.mean(dim=1)
            else:
                weights = valid.to(device=flat.device, dtype=flat.dtype).reshape(b * n, k, 1)
                pooled = (flat * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return pooled.reshape(b, n, -1)

        local = pool_candidate(local_evidence, local_valid_mask, self.local_proj)
        context = pool_candidate(context_evidence, context_valid_mask, self.context_proj)
        history = pool_candidate(history_features, history_valid_mask, self.history_proj)
        if text_tokens is None or text_tokens.shape[1] == 0:
            text = q.new_zeros((batch, tracks, q.shape[-1]))
        else:
            projected = self.text_proj(text_tokens)
            if text_valid_mask is None:
                text_mean = projected.mean(dim=1)
            else:
                text_mean = _valid_mean(projected, text_valid_mask)
            text = text_mean.unsqueeze(1).expand(-1, tracks, -1)
        return self.mlp(torch.cat([q, local, context, text, history], dim=-1))


def build_parameter_matched_controls(
    query_dim: int,
    embed_dim: int,
    *,
    text_dim: Optional[int] = None,
    evidence_dim: Optional[int] = None,
    history_dim: Optional[int] = None,
    num_heads: int = 4,
    dropout: float = 0.0,
) -> Tuple[ReferringPairHead, ReferringMLPHead]:
    """Build cross-attention and pooled-MLP controls with similar capacity.

    The MLP width is selected from a deterministic candidate set by parameter
    count.  The counts are reported by callers; this helper does not silently
    add unused parameters to force a match.
    """

    pair = ReferringPairHead(
        query_dim,
        embed_dim,
        text_dim=text_dim,
        evidence_dim=evidence_dim,
        history_dim=history_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    target = sum(p.numel() for p in pair.parameters())
    candidates = sorted({max(8, embed_dim // 2), embed_dim, 2 * embed_dim, 3 * embed_dim, 4 * embed_dim, 6 * embed_dim, 8 * embed_dim})
    controls = []
    for width in candidates:
        control = ReferringMLPHead(
            query_dim,
            embed_dim,
            text_dim=text_dim,
            evidence_dim=evidence_dim,
            history_dim=history_dim,
            hidden_dim=width,
            dropout=dropout,
        )
        controls.append(control)
    return pair, min(controls, key=lambda module: abs(sum(p.numel() for p in module.parameters()) - target))
