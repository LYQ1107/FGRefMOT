"""Causal per-track identity and dynamic evidence storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Hashable

import torch
from torch import Tensor
import torch.nn.functional as F

from .reliability_gate import ReliabilityGate


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Evidence visible after one observation has been written."""

    identity: Tensor
    dynamic: Tensor
    dynamic_history: Tensor
    bbox: Tensor
    velocity: Tensor
    reliability: Tensor
    wrote_identity: bool
    generation: int


@dataclass
class _TrackState:
    identity: Tensor | None
    dynamic_features: Deque[Tensor]
    bboxes: Deque[Tensor]
    last_bbox: Tensor | None
    last_frame: int | None
    generation: int


class EvidenceBuffer:
    """Maintain causal evidence keyed by expression namespace and track ID.

    ``namespace`` should include the video and expression identity.  Keeping it
    in the key prevents one expression's history from leaking into another
    expression even when both refer to the same physical track ID.
    """

    def __init__(
        self,
        feature_dim: int,
        dynamic_window: int = 4,
        identity_momentum: float = 0.9,
        reliability_gate: ReliabilityGate | None = None,
        detach_state: bool = True,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if dynamic_window <= 0:
            raise ValueError("dynamic_window must be positive")
        if not 0.0 <= identity_momentum < 1.0:
            raise ValueError("identity_momentum must be in [0, 1)")
        self.feature_dim = int(feature_dim)
        self.dynamic_window = int(dynamic_window)
        self.identity_momentum = float(identity_momentum)
        self.reliability_gate = reliability_gate or ReliabilityGate()
        self.detach_state = bool(detach_state)
        self._states: dict[tuple[Hashable, int], _TrackState] = {}

    @staticmethod
    def _feature(value: Tensor, feature_dim: int) -> Tensor:
        feature = torch.as_tensor(value).flatten()
        if feature.numel() != feature_dim:
            raise ValueError(f"feature has {feature.numel()} values; expected {feature_dim}")
        return F.normalize(feature.float(), dim=0)

    @staticmethod
    def _bbox(value: Tensor) -> Tensor:
        bbox = torch.as_tensor(value).flatten().float()
        if bbox.numel() != 4:
            raise ValueError("bbox must contain four coordinates")
        return bbox

    def reset(self, namespace: Hashable | None = None) -> None:
        """Clear all states or only one video/expression namespace."""

        if namespace is None:
            self._states.clear()
            return
        for key in [key for key in self._states if key[0] == namespace]:
            del self._states[key]

    def _state(self, namespace: Hashable, track_id: int) -> _TrackState:
        key = (namespace, int(track_id))
        state = self._states.get(key)
        if state is None:
            state = _TrackState(
                identity=None,
                dynamic_features=deque(maxlen=self.dynamic_window),
                bboxes=deque(maxlen=self.dynamic_window),
                last_bbox=None,
                last_frame=None,
                generation=0,
            )
            self._states[key] = state
        return state

    def update(
        self,
        *,
        namespace: Hashable,
        track_id: int,
        frame_index: int,
        feature: Tensor,
        bbox: Tensor,
        detection_confidence: Tensor | float = 1.0,
        visibility: Tensor | float = 1.0,
        local_quality: Tensor | float = 1.0,
        candidate_conflict: Tensor | float = 0.0,
    ) -> EvidenceSnapshot:
        """Append one observation and return only current/past evidence.

        Frames must be supplied in strictly increasing order for a given key.
        This guards against accidental future-frame leakage in offline
        evaluation and makes the temporal protocol explicit.
        """

        state = self._state(namespace, track_id)
        frame_index = int(frame_index)
        if state.last_frame is not None and frame_index <= state.last_frame:
            raise ValueError(
                f"non-causal frame order for {namespace!r}/{track_id}: "
                f"{frame_index} after {state.last_frame}"
            )
        current = self._feature(feature, self.feature_dim)
        current_bbox = self._bbox(bbox)
        previous_identity = state.identity
        consistency = self.reliability_gate.identity_consistency(current, previous_identity)
        reliability = self.reliability_gate(
            detection_confidence,
            visibility,
            local_quality,
            consistency,
            candidate_conflict,
        )
        write = previous_identity is None or bool(self.reliability_gate.write_mask(reliability).item())

        if previous_identity is None:
            identity = current
        elif write:
            alpha = (1.0 - self.identity_momentum) * reliability.to(current)
            identity = F.normalize((1.0 - alpha) * previous_identity + alpha * current, dim=0)
        else:
            identity = previous_identity

        if state.last_bbox is None:
            velocity = torch.zeros_like(current_bbox)
        else:
            velocity = current_bbox - state.last_bbox
        state.dynamic_features.append(current.detach() if self.detach_state else current)
        state.bboxes.append(current_bbox.detach() if self.detach_state else current_bbox)
        state.identity = identity.detach() if self.detach_state else identity
        state.last_bbox = current_bbox.detach() if self.detach_state else current_bbox
        state.last_frame = frame_index

        dynamic_history = torch.stack(tuple(state.dynamic_features), dim=0)
        dynamic = F.normalize(dynamic_history.mean(dim=0), dim=0)
        return EvidenceSnapshot(
            identity=state.identity.clone(),
            dynamic=dynamic.clone(),
            dynamic_history=dynamic_history.clone(),
            bbox=current_bbox.clone(),
            velocity=velocity.clone(),
            reliability=reliability.detach().clone(),
            wrote_identity=write,
            generation=state.generation,
        )

