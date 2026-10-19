"""Causal sequence integration for the learned FlexHook temporal head.

The pinned FlexHook checkout emits independent two-class referring logits.  This
module keeps that output contract while adding a learned per-candidate residual
from causal identity and dynamic evidence.  It is deliberately an adapter: the
upstream model is not edited, and callers decide which frozen visual/query
features and language embeddings to provide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import torch
from torch import Tensor, nn

from .evidence_buffer import EvidenceBuffer
from .temporal_referring import TemporalReferringHead, Variant


@dataclass(frozen=True)
class CausalSequenceOutput:
    """Outputs from a chronological sequence pass."""

    temporal_logits: Tensor
    identity: Tensor
    dynamic: Tensor
    reliability: Tensor


class TemporalEvidenceAdapter(nn.Module):
    """Fuse a learned temporal residual with frozen FlexHook logits.

    ``current`` is a per-candidate feature tensor ``[B, N, D]`` and ``language``
    is either ``[B, L]`` or ``[B, N, L]``.  ``base_logits`` may be ``[B, N]``
    or the original FlexHook ``[B, N, 2]`` tensor.  For two-class logits the
    residual is added antisymmetrically so the output remains compatible with
    the upstream loss and decoder.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        language_dim: int,
        hidden_dim: int | None = None,
        variant: Variant = "H8",
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        self.head = TemporalReferringHead(
            feature_dim=feature_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            variant=variant,
        )
        self.residual_scale = float(residual_scale)

    def temporal_logits(
        self,
        current: Tensor,
        language: Tensor,
        *,
        identity: Tensor | None = None,
        dynamic: Tensor | None = None,
        dynamic_history: Tensor | None = None,
        reliability: Tensor | None = None,
    ) -> Tensor:
        return self.head(
            current,
            language,
            identity=identity,
            dynamic=dynamic,
            dynamic_history=dynamic_history,
            reliability=reliability,
        ).squeeze(-1)

    def forward(
        self,
        current: Tensor,
        language: Tensor,
        *,
        base_logits: Tensor | None = None,
        identity: Tensor | None = None,
        dynamic: Tensor | None = None,
        dynamic_history: Tensor | None = None,
        reliability: Tensor | None = None,
    ) -> Tensor:
        residual = self.temporal_logits(
            current,
            language,
            identity=identity,
            dynamic=dynamic,
            dynamic_history=dynamic_history,
            reliability=reliability,
        ) * self.residual_scale
        if base_logits is None:
            return residual.unsqueeze(-1)
        if base_logits.ndim == 2:
            if base_logits.shape != residual.shape:
                raise ValueError("base_logits shape must match [B, N] temporal logits")
            return base_logits + residual
        if base_logits.ndim == 3 and base_logits.shape[:2] == residual.shape and base_logits.shape[-1] == 1:
            return base_logits + residual.unsqueeze(-1)
        if base_logits.ndim == 3 and base_logits.shape[:2] == residual.shape and base_logits.shape[-1] == 2:
            delta = 0.5 * residual.unsqueeze(-1)
            return base_logits + torch.cat([-delta, delta], dim=-1)
        raise ValueError("base_logits must have shape [B, N], [B, N, 1], or [B, N, 2]")


def _at_frame(
    value: Tensor | float | None,
    frame: int,
    candidates: int,
    reference: Tensor,
    *,
    default: float,
) -> Tensor:
    """Select a scalar or per-frame/per-candidate reliability input."""

    if value is None:
        return reference.new_tensor(default)
    tensor = torch.as_tensor(value, device=reference.device)
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1:
        if tensor.shape[0] != candidates:
            raise ValueError("per-candidate reliability input has the wrong length")
        return tensor
    if tensor.ndim == 2:
        if tensor.shape[0] <= frame or tensor.shape[1] != candidates:
            raise ValueError("per-frame reliability input has the wrong shape")
        return tensor[frame]
    raise ValueError("reliability inputs must be scalar, [N], or [T, N]")


def _candidate_at_frame(
    value: Tensor | float | None,
    frame: int,
    candidate: int,
    candidates: int,
    reference: Tensor,
    *,
    default: float,
) -> Tensor:
    selected = _at_frame(value, frame, candidates, reference, default=default)
    return selected if selected.ndim == 0 else selected[candidate]


def run_causal_sequence(
    adapter: TemporalEvidenceAdapter,
    *,
    current: Tensor,
    bboxes: Tensor,
    track_ids: Tensor,
    language: Tensor,
    namespace: Hashable,
    frame_indices: Tensor | None = None,
    base_logits: Tensor | None = None,
    detection_confidence: Tensor | float | None = None,
    visibility: Tensor | float | None = None,
    local_quality: Tensor | float | None = None,
    candidate_conflict: Tensor | float | None = None,
    dynamic_window: int = 4,
) -> Tensor:
    """Run a learned head chronologically without future-frame leakage.

    Shapes are ``current=[T,N,D]``, ``bboxes=[T,N,4]``, ``track_ids=[N]`` or
    ``[T,N]``, ``language=[N,L]`` or ``[T,N,L]``, and optional base logits
    ``[T,N]``/``[T,N,2]``.  The returned tensor has the same base-logit shape,
    or ``[T,N,1]`` when no base logits are supplied.
    """

    if current.ndim != 3 or current.shape[0] <= 0 or current.shape[1] <= 0:
        raise ValueError("current must be a non-empty [T, N, D] tensor")
    if bboxes.ndim != 3 or bboxes.shape[:2] != current.shape[:2] or bboxes.shape[-1] != 4:
        raise ValueError("current must be [T, N, D] and bboxes must be [T, N, 4]")
    frames, candidates, _ = current.shape
    if track_ids.ndim not in (1, 2) or track_ids.shape[-1] != candidates:
        raise ValueError("track_ids must be [N] or [T, N]")
    if track_ids.ndim == 2 and track_ids.shape[0] != frames:
        raise ValueError("per-frame track_ids must have shape [T, N]")
    if language.ndim not in (2, 3) or language.shape[-2] != candidates:
        raise ValueError("language must be [N, L] or [T, N, L]")
    if language.ndim == 3 and language.shape[0] != frames:
        raise ValueError("per-frame language must have shape [T, N, L]")
    if base_logits is not None and (base_logits.ndim < 2 or base_logits.shape[:2] != (frames, candidates)):
        raise ValueError("base_logits must begin with [T, N]")
    if frame_indices is None:
        frame_indices = torch.arange(frames, device=current.device)
    if frame_indices.ndim != 1 or frame_indices.shape[0] != frames:
        raise ValueError("frame_indices must be [T]")

    buffer = EvidenceBuffer(feature_dim=current.shape[-1], dynamic_window=dynamic_window)
    identity_rows: list[Tensor] = []
    dynamic_rows: list[Tensor] = []
    reliability_rows: list[Tensor] = []
    history_rows: list[Tensor] = []
    for frame in range(frames):
        identity_frame: list[Tensor] = []
        dynamic_frame: list[Tensor] = []
        reliability_frame: list[Tensor] = []
        history_frame: list[Tensor] = []
        ids = track_ids if track_ids.ndim == 1 else track_ids[frame]
        for candidate in range(candidates):
            snapshot = buffer.update(
                namespace=namespace,
                track_id=int(ids[candidate].item()),
                frame_index=int(frame_indices[frame].item()),
                feature=current[frame, candidate],
                bbox=bboxes[frame, candidate],
                detection_confidence=_candidate_at_frame(
                    detection_confidence, frame, candidate, candidates, current, default=1.0,
                ),
                visibility=_candidate_at_frame(
                    visibility, frame, candidate, candidates, current, default=1.0,
                ),
                local_quality=_candidate_at_frame(
                    local_quality, frame, candidate, candidates, current, default=1.0,
                ),
                candidate_conflict=_candidate_at_frame(
                    candidate_conflict, frame, candidate, candidates, current, default=0.0,
                ),
            )
            identity_frame.append(snapshot.identity)
            dynamic_frame.append(snapshot.dynamic)
            reliability_frame.append(snapshot.reliability)
            history = snapshot.dynamic_history
            if history.shape[0] < dynamic_window:
                padding = torch.zeros(
                    dynamic_window - history.shape[0], history.shape[1],
                    dtype=history.dtype, device=history.device,
                )
                history = torch.cat([padding, history], dim=0)
            history_frame.append(history[-dynamic_window:])
        identity_rows.append(torch.stack(identity_frame))
        dynamic_rows.append(torch.stack(dynamic_frame))
        reliability_rows.append(torch.stack(reliability_frame))
        history_rows.append(torch.stack(history_frame))

    identity = torch.stack(identity_rows)
    dynamic = torch.stack(dynamic_rows)
    reliability = torch.stack(reliability_rows)
    history = torch.stack(history_rows)
    output_rows: list[Tensor] = []
    for frame in range(frames):
        lang = language if language.ndim == 2 else language[frame]
        base = None if base_logits is None else base_logits[frame]
        output_rows.append(
            adapter(
                current[frame:frame + 1],
                lang.unsqueeze(0),
                base_logits=None if base is None else base.unsqueeze(0),
                identity=identity[frame:frame + 1],
                dynamic=dynamic[frame:frame + 1],
                dynamic_history=history[frame:frame + 1],
                reliability=reliability[frame:frame + 1],
            ).squeeze(0)
        )
    return torch.stack(output_rows)
