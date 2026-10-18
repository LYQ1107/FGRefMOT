"""Track-centric local/context evidence sampling.

The sampler consumes the visual pyramid *before* any TAM update.  Boxes are
normalized by default and are interpreted as ``cx, cy, w, h`` because that is
the format emitted by the current Deformable-DETR path.  ``grid_sample`` is
used instead of a custom CUDA RoIAlign dependency so the geometry can be
tested on CPU and at image boundaries.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TrackEvidence:
    local_evidence: Tensor
    context_evidence: Tensor
    valid_sample_mask: Tensor


class TrackEvidenceSampler(nn.Module):
    """Sample fixed-grid evidence around each predicted candidate box.

    Args:
        in_channels: channel count for each visual-pyramid level.
        evidence_dim: output channel dimension.
        num_local: number of points inside the candidate box.
        num_context: number of points inside an expanded context box.
        box_format: ``cxcywh`` or ``xyxy``; coordinates are normalized unless
            ``normalized_boxes`` is set to ``False`` in ``forward``.
        conditional_offsets: when true, query/text conditions predict a small
            offset for every point. This is an explicit conditional variant;
            the default is the fixed-grid T0 sampler.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        evidence_dim: int,
        num_local: int = 4,
        num_context: int = 8,
        *,
        box_format: str = "cxcywh",
        context_scale: float = 1.5,
        conditional_offsets: bool = False,
    ) -> None:
        super().__init__()
        if box_format not in {"cxcywh", "xyxy"}:
            raise ValueError("box_format must be 'cxcywh' or 'xyxy'")
        if num_local < 0 or num_context < 0 or num_local + num_context == 0:
            raise ValueError("at least one local or context sample is required")
        if context_scale < 1.0:
            raise ValueError("context_scale must be >= 1")
        self.evidence_dim = int(evidence_dim)
        self.num_local = int(num_local)
        self.num_context = int(num_context)
        self.box_format = box_format
        self.context_scale = float(context_scale)
        self.conditional_offsets = bool(conditional_offsets)
        self.projections = nn.ModuleList(
            [nn.Conv2d(int(channels), self.evidence_dim, kernel_size=1) for channels in in_channels]
        )
        if not self.projections:
            raise ValueError("in_channels must contain at least one level")
        if self.conditional_offsets:
            self.query_condition = nn.LazyLinear(self.evidence_dim)
            self.text_condition = nn.LazyLinear(self.evidence_dim)
            self.offset_predictor = nn.Sequential(
                nn.LazyLinear(self.evidence_dim),
                nn.GELU(),
                nn.Linear(self.evidence_dim, 2 * (self.num_local + self.num_context)),
            )

    @staticmethod
    def _unit_grid(count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if count == 0:
            return torch.empty((0, 2), device=device, dtype=dtype)
        side = int(math.ceil(math.sqrt(count)))
        values = torch.linspace(0.2, 0.8, side, device=device, dtype=dtype)
        grid = torch.cartesian_prod(values, values).reshape(-1, 2)
        return grid[:count]

    def _box_points(self, boxes: Tensor) -> Tensor:
        # boxes: [B, N, 4] in normalized coordinates.
        if self.box_format == "cxcywh":
            cx, cy, width, height = boxes.unbind(dim=-1)
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2
        else:
            x1, y1, x2, y2 = boxes.unbind(dim=-1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1

        local_unit = self._unit_grid(self.num_local, boxes.device, boxes.dtype)
        context_unit = self._unit_grid(self.num_context, boxes.device, boxes.dtype)
        local = torch.stack(
            [x1.unsqueeze(-1) + local_unit[:, 0] * width.unsqueeze(-1),
             y1.unsqueeze(-1) + local_unit[:, 1] * height.unsqueeze(-1)],
            dim=-1,
        )
        half_width = width.unsqueeze(-1) * self.context_scale / 2
        half_height = height.unsqueeze(-1) * self.context_scale / 2
        context_x1 = cx.unsqueeze(-1) - half_width
        context_y1 = cy.unsqueeze(-1) - half_height
        context = torch.stack(
            [context_x1 + context_unit[:, 0] * (2 * half_width),
             context_y1 + context_unit[:, 1] * (2 * half_height)],
            dim=-1,
        )
        return torch.cat([local, context], dim=2)

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("text_condition must have shape [B, L, D]")
        if tokens.shape[1] == 0:
            return tokens.new_zeros(tokens.shape[0], tokens.shape[2])
        if mask is None:
            return tokens.mean(dim=1)
        valid = ~mask.to(device=tokens.device, dtype=torch.bool)
        weights = valid.to(tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1)
        return (tokens * weights).sum(dim=1) / denom

    def _conditional_offsets(
        self,
        points: Tensor,
        query: Tensor,
        text_condition: Optional[Tensor],
        text_padding_mask: Optional[Tensor],
    ) -> Tensor:
        if not self.conditional_offsets:
            return points
        if query.ndim != 3:
            raise ValueError("query must have shape [B, N, D]")
        text = (
            query.new_zeros(query.shape[0], query.shape[-1])
            if text_condition is None
            else self._masked_mean(text_condition, text_padding_mask)
        )
        query_cond = self.query_condition(query)
        text_cond = self.text_condition(text).unsqueeze(1)
        cond = torch.tanh(query_cond + text_cond)
        offsets = self.offset_predictor(cond).view(
            query.shape[0], query.shape[1], -1, 2
        )
        # Keep the learned displacement small relative to the normalized
        # image coordinates. Validity is computed from the un-clipped points.
        return points + 0.15 * torch.tanh(offsets)

    def forward(
        self,
        visual_pyramid: Sequence[Tensor],
        boxes: Tensor,
        query: Tensor,
        text_condition: Optional[Tensor] = None,
        *,
        text_padding_mask: Optional[Tensor] = None,
        feature_masks: Optional[Sequence[Tensor]] = None,
        normalized_boxes: bool = True,
        image_size: Optional[tuple[int, int]] = None,
    ) -> TrackEvidence:
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [B, N, 4]")
        if query.ndim != 3 or query.shape[:2] != boxes.shape[:2]:
            raise ValueError("query must have shape [B, N, D] matching boxes")
        if len(visual_pyramid) != len(self.projections):
            raise ValueError("visual_pyramid length does not match in_channels")
        batch, tracks = boxes.shape[:2]
        samples = self.num_local + self.num_context
        if tracks == 0:
            empty = boxes.new_zeros((batch, 0, samples, self.evidence_dim))
            return TrackEvidence(empty[:, :, : self.num_local], empty[:, :, self.num_local :], boxes.new_zeros((batch, 0, samples), dtype=torch.bool))

        boxes_norm = boxes
        if not normalized_boxes:
            if image_size is None:
                raise ValueError("image_size is required for absolute boxes")
            height, width = image_size
            scale = boxes.new_tensor([width, height, width, height])
            boxes_norm = boxes / scale
        points = self._box_points(boxes_norm)
        points = self._conditional_offsets(
            points, query, text_condition, text_padding_mask
        )
        valid = (points >= 0).all(dim=-1) & (points <= 1).all(dim=-1)

        sampled_levels = []
        for level, (feature, projection) in enumerate(zip(visual_pyramid, self.projections)):
            if feature.ndim != 4 or feature.shape[0] != batch:
                raise ValueError("each feature level must have shape [B, C, H, W]")
            projected = projection(feature)
            grid = (points * 2 - 1).reshape(batch, tracks * samples, 1, 2)
            sampled = F.grid_sample(
                projected,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled[:, :, :, 0].transpose(1, 2).reshape(
                batch, tracks, samples, self.evidence_dim
            )
            sampled_levels.append(sampled)
            if feature_masks is not None:
                mask = feature_masks[level].to(device=feature.device)
                if mask.ndim != 3 or mask.shape[0] != batch:
                    raise ValueError("feature masks must have shape [B, H, W]")
                mask_sample = F.grid_sample(
                    mask.to(dtype=feature.dtype).unsqueeze(1),
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=True,
                )
                valid = valid & (mask_sample[:, 0, :, 0].reshape(batch, tracks, samples) < 0.5)

        evidence = torch.stack(sampled_levels, dim=0).mean(dim=0)
        local = evidence[:, :, : self.num_local]
        context = evidence[:, :, self.num_local :]
        return TrackEvidence(local, context, valid)
