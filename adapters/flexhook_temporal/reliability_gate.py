"""Simple, auditable reliability control for temporal evidence writes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReliabilityWeights:
    """Weights for the first non-learned reliability gate.

    Every input is expected in ``[0, 1]``.  Referring score is intentionally
    absent: a player not currently satisfying a changing expression can still
    provide useful identity evidence.
    """

    detection_confidence: float = 0.35
    visibility: float = 0.20
    local_quality: float = 0.20
    identity_consistency: float = 0.15
    candidate_conflict: float = 0.10

    def as_tensor(self) -> Tensor:
        values = torch.tensor(
            [
                self.detection_confidence,
                self.visibility,
                self.local_quality,
                self.identity_consistency,
                self.candidate_conflict,
            ],
            dtype=torch.float32,
        )
        if not torch.isclose(values.sum(), torch.tensor(1.0), atol=1e-6):
            raise ValueError("reliability weights must sum to one")
        if torch.any(values < 0):
            raise ValueError("reliability weights must be non-negative")
        return values


class ReliabilityGate(nn.Module):
    """Compute a reliability score and a conservative memory-write mask.

    The gate is deterministic for the first ablation.  Its weights are buffers,
    so an experiment cannot silently train the gate while claiming a fixed
    reliability rule.  A later learned-gate ablation can replace this module.
    """

    def __init__(
        self,
        weights: ReliabilityWeights | None = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.register_buffer("weights", (weights or ReliabilityWeights()).as_tensor())
        self.threshold = float(threshold)

    @staticmethod
    def _unit(value: Tensor | float, reference: Tensor) -> Tensor:
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        return tensor.clamp(0.0, 1.0)

    def forward(
        self,
        detection_confidence: Tensor | float,
        visibility: Tensor | float,
        local_quality: Tensor | float,
        identity_consistency: Tensor | float,
        candidate_conflict: Tensor | float,
    ) -> Tensor:
        """Return a score in ``[0, 1]`` with broadcast-compatible inputs."""

        reference = torch.as_tensor(detection_confidence, dtype=self.weights.dtype, device=self.weights.device)
        values = torch.stack(
            [
                self._unit(detection_confidence, reference),
                self._unit(visibility, reference),
                self._unit(local_quality, reference),
                self._unit(identity_consistency, reference),
                1.0 - self._unit(candidate_conflict, reference),
            ],
            dim=-1,
        )
        weights = self.weights.to(dtype=values.dtype, device=values.device)
        return (values * weights).sum(dim=-1).clamp(0.0, 1.0)

    def identity_consistency(self, current: Tensor, previous: Tensor | None) -> Tensor:
        """Cosine consistency against the previous identity evidence."""

        if previous is None:
            return torch.ones((), dtype=current.dtype, device=current.device)
        current_n = F.normalize(current, dim=-1)
        previous_n = F.normalize(previous.to(current), dim=-1)
        return ((current_n * previous_n).sum(dim=-1) + 1.0) * 0.5

    def write_mask(self, score: Tensor) -> Tensor:
        return score >= self.threshold

