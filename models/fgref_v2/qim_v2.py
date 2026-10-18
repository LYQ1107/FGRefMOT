"""Reliability-aware temporal query writer (QIM-v2)."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from models.structures import Instances

from .qim_writer import EvidenceQueryWriter
from .temporal_evidence import TemporalEvidenceEncoder, TemporalEvidenceOutput
from .track_memory import ReliabilityAwareTrackMemory, TrackMemoryState


@dataclass
class QIMV2Output:
    query_pos: Tensor
    ref_pts: Tensor
    state: TrackMemoryState
    temporal: TemporalEvidenceOutput
    reliability: Tensor


class ReliabilityAwareQIMV2(nn.Module):
    """Compose ID-keyed memory, a causal history encoder, and a query writer."""

    def __init__(
        self,
        observation_dim: int,
        query_dim: int,
        *,
        memory_dim: int = 128,
        history_length: int = 8,
        temporal_mode: Literal["ema", "gru", "attention", "mamba"] = "ema",
        update_query_pos: bool = False,
    ) -> None:
        super().__init__()
        self.memory = ReliabilityAwareTrackMemory(
            observation_dim,
            memory_dim,
            history_length,
        )
        self.temporal = TemporalEvidenceEncoder(memory_dim, temporal_mode)
        self.writer = EvidenceQueryWriter(
            query_dim,
            memory_dim,
            observation_dim=observation_dim,
            update_query_pos=update_query_pos,
        )

    def forward(
        self,
        *,
        state: Optional[TrackMemoryState],
        track_ids: Tensor,
        current_query_pos: Tensor,
        predicted_boxes: Tensor,
        current_observation: Optional[Tensor],
        detection_quality: Tensor,
        ref_pts: Optional[Tensor] = None,
        visibility: Optional[Tensor] = None,
        feature_conflict: Optional[Tensor] = None,
        match_confidence: Optional[Tensor] = None,
        observed_mask: Optional[Tensor] = None,
    ) -> QIMV2Output:
        # The read occurs before memory.step. Thus the temporal encoder sees
        # t-1 and earlier; the current observation is only written afterwards.
        aligned = self.memory.align_state(state, track_ids)
        temporal = self.temporal(aligned.dynamic_history, aligned.dynamic_history_mask)
        count = track_ids.shape[0]
        if current_observation is None:
            current_observation = current_query_pos.new_zeros((count, self.memory.observation_dim))
            writer_observed = torch.zeros(count, dtype=torch.bool, device=current_query_pos.device)
        else:
            writer_observed = (
                torch.ones(count, dtype=torch.bool, device=current_query_pos.device)
                if observed_mask is None
                else observed_mask.to(device=current_query_pos.device, dtype=torch.bool)
            )
        # ``None`` means that no current visual observation exists.  Pass the
        # effective mask into memory.step as well as the query writer so the
        # zero placeholder cannot become an identity or dynamic-memory write.
        effective_observed_mask = writer_observed
        new_state, reliability = self.memory.step(
            aligned,
            track_ids,
            current_observation,
            detection_quality,
            visibility=visibility,
            feature_conflict=feature_conflict,
            match_confidence=match_confidence,
            observed_mask=effective_observed_mask,
        )
        written = self.writer(
            current_query_pos,
            new_state.identity_memory,
            new_state.identity_memory_valid,
            current_observation,
            predicted_boxes,
            ref_pts=ref_pts,
            observation_valid=writer_observed,
            history_feature=temporal.summary,
            history_valid=aligned.dynamic_history_mask.any(dim=1),
        )
        return QIMV2Output(
            query_pos=written.query_pos,
            ref_pts=written.ref_pts,
            state=new_state,
            temporal=temporal,
            reliability=reliability,
        )


class ReliabilityAwareQueryInteraction(nn.Module):
    """Adapter that makes QIM-v2 replace the legacy row-wise QIM hook.

    The adapter keeps the existing ``Instances`` lifecycle: fresh detection
    queries are concatenated with active tracks, while the new memory fields
    stay row-aligned and are matched by ``obj_idxes`` on every call.  It does
    not create or delete physical IDs; ``RuntimeTrackerBase`` remains the
    owner of that policy.
    """

    _STATE_FIELDS = (
        "fgref_v2_identity_memory",
        "fgref_v2_identity_memory_valid",
        "fgref_v2_dynamic_history",
        "fgref_v2_dynamic_history_mask",
        "fgref_v2_observation_quality",
        "fgref_v2_memory_age",
    )

    def __init__(
        self,
        *,
        observation_dim: int,
        query_dim: int,
        memory_dim: int = 128,
        history_length: int = 8,
        temporal_mode: Literal["ema", "gru", "attention", "mamba"] = "ema",
        update_query_pos: bool = False,
    ) -> None:
        super().__init__()
        self.qim = ReliabilityAwareQIMV2(
            observation_dim,
            query_dim,
            memory_dim=memory_dim,
            history_length=history_length,
            temporal_mode=temporal_mode,
            update_query_pos=update_query_pos,
        )

    def _select_active_tracks(self, track_instances: Instances) -> Instances:
        if self.training:
            active = (track_instances.obj_idxes >= 0) | (track_instances.scores > 0.5)
            selected = track_instances[active]
            if selected.has("iou"):
                selected.obj_idxes[selected.iou <= 0.5] = -1
            return selected
        return track_instances[track_instances.obj_idxes >= 0]

    def _state_from_instances(self, tracks: Instances) -> Optional[TrackMemoryState]:
        if len(tracks) == 0 or not all(tracks.has(name) for name in self._STATE_FIELDS):
            return None
        return TrackMemoryState(
            tracks.obj_idxes,
            tracks.fgref_v2_identity_memory,
            tracks.fgref_v2_identity_memory_valid,
            tracks.fgref_v2_dynamic_history,
            tracks.fgref_v2_dynamic_history_mask,
            tracks.fgref_v2_observation_quality,
            tracks.fgref_v2_memory_age,
        )

    @staticmethod
    def _write_state(tracks: Instances, state: TrackMemoryState) -> None:
        tracks.fgref_v2_identity_memory = state.identity_memory
        tracks.fgref_v2_identity_memory_valid = state.identity_memory_valid
        tracks.fgref_v2_dynamic_history = state.dynamic_history
        tracks.fgref_v2_dynamic_history_mask = state.dynamic_history_mask
        tracks.fgref_v2_observation_quality = state.observation_quality
        tracks.fgref_v2_memory_age = state.memory_age

    def forward(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        active = self._select_active_tracks(track_instances)
        if len(active) > 0:
            state = self._state_from_instances(active)
            output = self.qim(
                state=state,
                track_ids=active.obj_idxes,
                current_query_pos=active.query_pos,
                predicted_boxes=active.pred_boxes,
                current_observation=active.output_embedding,
                detection_quality=active.scores,
                ref_pts=active.ref_pts,
                observed_mask=active.scores > 0.5,
            )
            active.query_pos = output.query_pos
            active.ref_pts = output.ref_pts
            self._write_state(active, output.state)

        init_track_instances: Instances = data["init_track_instances"]
        return Instances.cat([init_track_instances, active])
