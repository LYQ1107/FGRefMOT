"""Shared utilities for the lightweight Mamba-QIM experiments.

These helpers deliberately keep all ordering and memory decisions based on
inference-available tensors.  In particular, no target IDs or referring
labels are used to order tokens or update the long-term memory.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.structures import Instances

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError as exc:  # pragma: no cover - exercised only on unsupported installs
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


def require_mamba() -> type:
    """Return the installed Mamba1 implementation or raise a useful error."""

    if Mamba is None:
        raise RuntimeError(
            "The lightweight QIM variants require mamba_ssm.modules.mamba_simple.Mamba"
        ) from _MAMBA_IMPORT_ERROR
    return Mamba


def resolve_mamba_dim(args, variant: str) -> int:
    """Resolve the documented initial bottleneck dimensions.

    M1 uses 96 channels by default; M2/M3 use 128.  A positive CLI override
    is shared across the selected variant for controlled ablations.
    """

    override = int(getattr(args, "qim_mamba_dim", 0) or 0)
    if override > 0:
        return override
    return 96 if variant == "mamba_ordered_bi" else 128


def resolve_mamba_layers(args) -> int:
    return max(1, int(getattr(args, "qim_mamba_layers", 1)))


def resolve_mamba_state(args) -> int:
    return max(1, int(getattr(args, "qim_mamba_state", 16)))


def resolve_history_len(args) -> int:
    return max(1, int(getattr(args, "qim_mamba_history_len", 4)))


def make_mamba_stack(args, d_model: int) -> "MambaStack":
    """Build the explicitly screened Mamba1 stack.

    Mamba2 is intentionally not silently substituted here.  It is a separate
    future benchmark because its kernel/API and parameter budget must be
    verified independently on the V100 host.
    """

    if bool(getattr(args, "qim_mamba2_enable", False)):
        raise NotImplementedError(
            "qim_mamba2_enable is reserved for a separate Mamba2 compatibility "
            "screening; the current lightweight runs are explicitly Mamba1"
        )
    return MambaStack(
        d_model=d_model,
        d_state=resolve_mamba_state(args),
        layers=resolve_mamba_layers(args),
    )


class MambaStack(nn.Module):
    """Small residual Mamba1 stack operating on ``[B, T, D]`` tensors."""

    def __init__(self, d_model: int, d_state: int, layers: int):
        super().__init__()
        mamba_cls = require_mamba()
        self.layers = nn.ModuleList(
            [
                mamba_cls(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,
                    expand=1,
                    layer_idx=index,
                )
                for index in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"MambaStack expects [B,T,D], got {tuple(values.shape)}")
        for layer, norm in zip(self.layers, self.norms):
            values = norm(values + layer(values))
        return values


def canonical_geometry_order(
    track_instances: Instances,
    features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a deterministic, target-free order for active tracks.

    The primary key is predicted ``cx, cy, w, h``.  Ties use the known track
    ID when available and a deterministic projection of the current feature
    for newborn/unknown tracks.  A final stable row-order tie is harmless only
    when all observable keys are identical, in which case the tokens are
    observationally indistinguishable for this ordering decision.
    """

    number = len(track_instances)
    if number == 0:
        device = track_instances.query_pos.device
        return torch.empty(0, dtype=torch.long, device=device)

    boxes = track_instances.pred_boxes.detach().float()
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    boxes = boxes[:, :4]
    device = boxes.device
    known = (track_instances.obj_idxes.detach() >= 0).to(torch.int64)
    ids = track_instances.obj_idxes.detach().float()
    ids = torch.where(known.bool(), ids, torch.zeros_like(ids))

    if features is None:
        features = getattr(track_instances, "output_embedding", None)
    if features is None:
        fingerprint = torch.zeros(number, device=device)
    else:
        detached = features.detach().float()
        weights = torch.arange(
            1, detached.shape[-1] + 1, device=detached.device, dtype=detached.dtype
        )
        weights = weights / max(1, detached.shape[-1])
        fingerprint = torch.nan_to_num(
            (detached * weights).sum(dim=-1), nan=0.0, posinf=1e6, neginf=-1e6
        )

    # Stable lexicographic sort: the last key is applied first.  The final
    # stable sort retains row order only for truly indistinguishable rows.
    order = torch.arange(number, device=device, dtype=torch.long)
    keys = [
        fingerprint,
        ids,
        known,
        boxes[:, 3],
        boxes[:, 2],
        boxes[:, 1],
        boxes[:, 0],
    ]
    for key in keys:
        order = order[torch.argsort(key[order], stable=True)]
    return order


def box_iou_xywh(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise elementwise IoU for normalized ``cx,cy,w,h`` boxes."""

    def to_corners(boxes):
        half = boxes[:, 2:4].clamp(min=0) / 2
        return torch.cat((boxes[:, :2] - half, boxes[:, :2] + half), dim=-1)

    a = to_corners(boxes_a)
    b = to_corners(boxes_b)
    top_left = torch.maximum(a[:, :2], b[:, :2])
    bottom_right = torch.minimum(a[:, 2:], b[:, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(dim=-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(dim=-1)
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def masked_mean(values: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over non-padding tokens, with a zero-safe fallback."""

    if padding_mask is None:
        return values.mean(dim=1)
    valid = (~padding_mask).to(values.dtype).unsqueeze(-1)
    denominator = valid.sum(dim=1).clamp(min=1.0)
    return (values * valid).sum(dim=1) / denominator
