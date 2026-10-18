"""Optional supervised objectives for the FGRefMOT-v2 research path.

These losses are intentionally standalone.  The legacy training graph still
uses its original detection/box/referring losses until reliable identity and
counterfactual annotations are available; callers must opt into this module
and pass explicit validity masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class FGRefV2LossWeights:
    identity: float = 0.0
    counterfactual: float = 0.0


def _zero_with_gradient(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def masked_binary_referring_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute independent per-candidate referring BCE with an optional mask."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    values = F.binary_cross_entropy_with_logits(
        logits, targets.to(device=logits.device, dtype=logits.dtype), reduction="none"
    )
    if valid_mask is None:
        return values.mean() if values.numel() else _zero_with_gradient(logits)
    mask = valid_mask.to(device=logits.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def identity_consistency_loss(
    embeddings: Tensor,
    track_ids: Tensor,
    valid_mask: Optional[Tensor] = None,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """Supervised contrastive identity loss over explicit ID-labelled rows.

    ``embeddings`` is ``[..., D]`` and ``track_ids``/``valid_mask`` broadcast
    over the leading dimensions.  Rows with ID ``< 0`` are ignored.  A batch
    with no positive same-ID pair returns a differentiable zero instead of
    inventing negatives or using an unrelated track as a positive.
    """

    if embeddings.ndim < 2:
        raise ValueError("embeddings must have shape [..., D]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    ids = track_ids.to(device=embeddings.device).reshape(-1)
    if ids.numel() != flat.shape[0]:
        raise ValueError("track_ids must match embeddings leading dimensions")
    valid = ids >= 0
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=embeddings.device, dtype=torch.bool).reshape(-1)
    if int(valid.sum()) < 2:
        return _zero_with_gradient(embeddings)

    rows = torch.nonzero(valid, as_tuple=False).flatten()
    values = F.normalize(flat[rows], dim=-1)
    similarities = values @ values.transpose(0, 1) / temperature
    diagonal = torch.eye(len(rows), dtype=torch.bool, device=embeddings.device)
    similarities = similarities.masked_fill(diagonal, float("-inf"))
    positive = ids[rows].unsqueeze(1).eq(ids[rows].unsqueeze(0)) & ~diagonal
    positive_count = positive.sum(dim=1)
    anchors = positive_count > 0
    if not bool(anchors.any()):
        return _zero_with_gradient(embeddings)
    log_prob = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_log_prob = torch.where(
        positive,
        log_prob,
        torch.zeros_like(log_prob),
    )
    per_anchor = -positive_log_prob.sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[anchors].mean()


def compose_fgref_v2_losses(
    base_losses: dict[str, Tensor],
    *,
    identity_embeddings: Optional[Tensor] = None,
    identity_track_ids: Optional[Tensor] = None,
    identity_valid_mask: Optional[Tensor] = None,
    counterfactual_logits: Optional[Tensor] = None,
    counterfactual_targets: Optional[Tensor] = None,
    counterfactual_valid_mask: Optional[Tensor] = None,
    weights: FGRefV2LossWeights = FGRefV2LossWeights(),
) -> dict[str, Tensor]:
    """Return base losses plus only explicitly supplied optional objectives."""

    losses = dict(base_losses)
    reference = next(iter(losses.values()), None)
    if identity_embeddings is not None or identity_track_ids is not None:
        if identity_embeddings is None or identity_track_ids is None:
            raise ValueError("identity embeddings and IDs must be supplied together")
        identity = identity_consistency_loss(
            identity_embeddings, identity_track_ids, identity_valid_mask
        )
        losses["loss_fgref_v2_identity"] = identity * float(weights.identity)
    elif weights.identity:
        if reference is None:
            raise ValueError("a base loss is required for a weighted missing identity term")
        losses["loss_fgref_v2_identity"] = _zero_with_gradient(reference)

    if counterfactual_logits is not None or counterfactual_targets is not None:
        if counterfactual_logits is None or counterfactual_targets is None:
            raise ValueError("counterfactual logits and targets must be supplied together")
        counterfactual = masked_binary_referring_loss(
            counterfactual_logits,
            counterfactual_targets,
            counterfactual_valid_mask,
        )
        losses["loss_fgref_v2_counterfactual"] = counterfactual * float(weights.counterfactual)
    elif weights.counterfactual:
        if reference is None:
            raise ValueError("a base loss is required for a weighted missing counterfactual term")
        losses["loss_fgref_v2_counterfactual"] = _zero_with_gradient(reference)
    return losses
