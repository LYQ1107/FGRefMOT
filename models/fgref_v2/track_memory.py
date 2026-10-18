"""Reliability-aware, ID-keyed track memory."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class TrackMemoryState:
    """Fixed-shape state carried by the current rows of ``Instances``.

    ``dynamic_history_mask`` uses ``True`` for an observed/valid slot.  The
    first axis is always aligned with ``track_ids``; callers must call
    ``ReliabilityAwareTrackMemory.align_state`` whenever rows are reordered or
    new IDs are born.
    """

    track_ids: Tensor
    identity_memory: Tensor
    identity_memory_valid: Tensor
    dynamic_history: Tensor
    dynamic_history_mask: Tensor
    observation_quality: Tensor
    memory_age: Tensor

    def __len__(self) -> int:
        return int(self.track_ids.shape[0])

    def to(self, *args, **kwargs) -> "TrackMemoryState":
        return TrackMemoryState(
            *(value.to(*args, **kwargs)
              for value in (
                  self.track_ids,
                  self.identity_memory,
                  self.identity_memory_valid,
                  self.dynamic_history,
                  self.dynamic_history_mask,
                  self.observation_quality,
                  self.memory_age,
              ))
        )

    def index_select(self, index: Tensor) -> "TrackMemoryState":
        return TrackMemoryState(
            self.track_ids[index],
            self.identity_memory[index],
            self.identity_memory_valid[index],
            self.dynamic_history[index],
            self.dynamic_history_mask[index],
            self.observation_quality[index],
            self.memory_age[index],
        )

    @staticmethod
    def cat(states: list["TrackMemoryState"]) -> "TrackMemoryState":
        if not states:
            raise ValueError("states must not be empty")
        return TrackMemoryState(
            *(torch.cat([getattr(state, name) for state in states], dim=0)
              for name in (
                  "track_ids",
                  "identity_memory",
                  "identity_memory_valid",
                  "dynamic_history",
                  "dynamic_history_mask",
                  "observation_quality",
                  "memory_age",
              ))
        )


class ReliabilityAwareTrackMemory(nn.Module):
    """Maintain stable identity evidence and recent dynamic history.

    Identity writes use detection quality, visibility, match confidence and a
    feature-conflict penalty.  Referring scores are intentionally absent from
    this interface, so a semantic output cannot erase identity evidence.
    """

    def __init__(
        self,
        observation_dim: int,
        memory_dim: int = 128,
        history_length: int = 8,
        *,
        identity_momentum: float = 0.9,
        write_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if history_length < 1:
            raise ValueError("history_length must be >= 1")
        if not 0 <= identity_momentum < 1:
            raise ValueError("identity_momentum must be in [0, 1)")
        if not 0 <= write_threshold <= 1:
            raise ValueError("write_threshold must be in [0, 1]")
        self.observation_dim = int(observation_dim)
        self.memory_dim = int(memory_dim)
        self.history_length = int(history_length)
        self.identity_momentum = float(identity_momentum)
        self.write_threshold = float(write_threshold)
        self.identity_projection = nn.Linear(observation_dim, memory_dim)
        self.dynamic_projection = nn.Linear(observation_dim, memory_dim)

    def initial_state(
        self,
        track_ids: Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> TrackMemoryState:
        if track_ids.ndim != 1:
            raise ValueError("track_ids must have shape [N]")
        track_ids = track_ids.to(device=device or track_ids.device, dtype=torch.long)
        count = track_ids.shape[0]
        return TrackMemoryState(
            track_ids=track_ids,
            identity_memory=torch.zeros(count, self.memory_dim, device=track_ids.device, dtype=dtype),
            identity_memory_valid=torch.zeros(count, device=track_ids.device, dtype=torch.bool),
            dynamic_history=torch.zeros(count, self.history_length, self.memory_dim, device=track_ids.device, dtype=dtype),
            dynamic_history_mask=torch.zeros(count, self.history_length, device=track_ids.device, dtype=torch.bool),
            observation_quality=torch.zeros(count, device=track_ids.device, dtype=dtype),
            memory_age=torch.zeros(count, device=track_ids.device, dtype=torch.long),
        )

    @staticmethod
    def _validate_ids(track_ids: Tensor) -> Tensor:
        track_ids = track_ids.to(dtype=torch.long)
        nonnegative = track_ids[track_ids >= 0]
        if nonnegative.numel() != torch.unique(nonnegative).numel():
            raise ValueError("physical track IDs must be unique within a frame")
        return track_ids

    def align_state(self, state: Optional[TrackMemoryState], track_ids: Tensor) -> TrackMemoryState:
        """Reindex old state by physical ID; births receive clean memory."""

        track_ids = self._validate_ids(track_ids)
        if state is None:
            return self.initial_state(track_ids)
        if state.track_ids.ndim != 1:
            raise ValueError("state.track_ids must have shape [N]")
        device = state.identity_memory.device
        track_ids = track_ids.to(device=device)
        old_ids = state.track_ids.to(device=device)
        matches = (
            (track_ids[:, None] >= 0)
            & (old_ids[None, :] >= 0)
            & (track_ids[:, None] == old_ids[None, :])
        )
        matched = matches.any(dim=1)
        old_index = matches.to(torch.int64).argmax(dim=1) if len(state) else torch.zeros_like(track_ids)
        result = self.initial_state(track_ids, device=device, dtype=state.identity_memory.dtype)
        if len(state) == 0 or not bool(matched.any()):
            return result
        new_index = torch.nonzero(matched, as_tuple=False).flatten()
        old_index = old_index[new_index]
        result.identity_memory[new_index] = state.identity_memory[old_index]
        result.identity_memory_valid[new_index] = state.identity_memory_valid[old_index]
        result.dynamic_history[new_index] = state.dynamic_history[old_index]
        result.dynamic_history_mask[new_index] = state.dynamic_history_mask[old_index]
        result.observation_quality[new_index] = state.observation_quality[old_index]
        result.memory_age[new_index] = state.memory_age[old_index]
        return result

    @staticmethod
    def _vector(value: Optional[Tensor], count: int, device: torch.device, default: float) -> Tensor:
        if value is None:
            return torch.full((count,), default, device=device, dtype=torch.float32)
        if value.ndim != 1 or value.shape[0] != count:
            raise ValueError("reliability inputs must have shape [N]")
        return value.to(device=device, dtype=torch.float32).clamp(0, 1)

    def reliability(
        self,
        detection_quality: Tensor,
        *,
        visibility: Optional[Tensor] = None,
        feature_conflict: Optional[Tensor] = None,
        match_confidence: Optional[Tensor] = None,
    ) -> Tensor:
        count = detection_quality.shape[0]
        device = detection_quality.device
        quality = self._vector(detection_quality, count, device, 0)
        visible = self._vector(visibility, count, device, 1)
        conflict = self._vector(feature_conflict, count, device, 0)
        matched = self._vector(match_confidence, count, device, 1)
        return quality * visible * (1 - conflict) * matched

    def step(
        self,
        state: Optional[TrackMemoryState],
        track_ids: Tensor,
        observation: Optional[Tensor],
        detection_quality: Tensor,
        *,
        visibility: Optional[Tensor] = None,
        feature_conflict: Optional[Tensor] = None,
        match_confidence: Optional[Tensor] = None,
        observed_mask: Optional[Tensor] = None,
    ) -> tuple[TrackMemoryState, Tensor]:
        """Read old state, then write current observations and return reliability."""

        count = track_ids.shape[0]
        if observation is not None and (observation.ndim != 2 or observation.shape[0] != count):
            raise ValueError("observation must have shape [N, observation_dim]")
        if observation is not None and observation.shape[1] != self.observation_dim:
            raise ValueError(f"expected observation dimension {self.observation_dim}")
        aligned = self.align_state(state, track_ids)
        device = aligned.identity_memory.device
        if observation is None:
            observation = torch.zeros(count, self.observation_dim, device=device, dtype=aligned.identity_memory.dtype)
            available = torch.zeros(count, device=device, dtype=torch.bool)
        else:
            observation = observation.to(device=device, dtype=aligned.identity_memory.dtype)
            available = torch.ones(count, device=device, dtype=torch.bool)
        if observed_mask is not None:
            if observed_mask.shape != (count,):
                raise ValueError("observed_mask must have shape [N]")
            available &= observed_mask.to(device=device, dtype=torch.bool)

        current_reliability = self.reliability(
            detection_quality.to(device=device),
            visibility=visibility,
            feature_conflict=feature_conflict,
            match_confidence=match_confidence,
        ).to(dtype=aligned.identity_memory.dtype)
        current_reliability = current_reliability * available.to(current_reliability.dtype)
        identity_write = available & (current_reliability >= self.write_threshold)
        dynamic_feature = self.dynamic_projection(observation)
        identity_feature = self.identity_projection(observation)

        new_identity = aligned.identity_memory.clone()
        old_valid = aligned.identity_memory_valid
        if bool(identity_write.any()):
            write_indices = torch.nonzero(identity_write, as_tuple=False).flatten()
            old_features = new_identity[write_indices]
            has_old = old_valid[write_indices].unsqueeze(-1)
            base = torch.where(has_old, old_features, identity_feature[write_indices])
            updated = self.identity_momentum * base + (1 - self.identity_momentum) * identity_feature[write_indices]
            new_identity[write_indices] = updated
        new_identity_valid = old_valid | identity_write

        if self.history_length > 1:
            new_history = torch.cat([aligned.dynamic_history[:, 1:], dynamic_feature.unsqueeze(1)], dim=1)
            new_history_mask = torch.cat([aligned.dynamic_history_mask[:, 1:], available.unsqueeze(1)], dim=1)
        else:
            new_history = dynamic_feature.unsqueeze(1)
            new_history_mask = available.unsqueeze(1)
        age = torch.where(available, torch.zeros_like(aligned.memory_age), aligned.memory_age + 1)
        new_state = TrackMemoryState(
            track_ids=aligned.track_ids,
            identity_memory=new_identity,
            identity_memory_valid=new_identity_valid,
            dynamic_history=new_history,
            dynamic_history_mask=new_history_mask,
            observation_quality=torch.where(available, current_reliability, torch.zeros_like(current_reliability)),
            memory_age=age,
        )
        return new_state, current_reliability
