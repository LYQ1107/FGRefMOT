"""Write reliability-aware evidence back into the original query format."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class QueryWriterOutput:
    query_pos: Tensor
    ref_pts: Tensor


class EvidenceQueryWriter(nn.Module):
    """Update query content without taking over physical track lifecycle."""

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        *,
        observation_dim: Optional[int] = None,
        update_query_pos: bool = False,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.memory_dim = int(memory_dim)
        self.update_query_pos = bool(update_query_pos)
        observation_dim = observation_dim or memory_dim
        self.identity_projection = nn.Linear(memory_dim, query_dim)
        self.observation_projection = nn.Linear(observation_dim, query_dim)
        self.history_projection = nn.Linear(memory_dim, query_dim)
        self.box_projection = nn.Linear(4, query_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * query_dim + 4, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        current_query_pos: Tensor,
        identity_memory: Tensor,
        identity_memory_valid: Tensor,
        current_observation: Tensor,
        predicted_boxes: Tensor,
        *,
        ref_pts: Optional[Tensor] = None,
        observation_valid: Optional[Tensor] = None,
        history_feature: Optional[Tensor] = None,
        history_valid: Optional[Tensor] = None,
    ) -> QueryWriterOutput:
        if current_query_pos.shape[-1] != 2 * self.query_dim:
            raise ValueError("current_query_pos must end in 2 * query_dim")
        if predicted_boxes.shape[-1] != 4:
            raise ValueError("predicted_boxes must end in 4")
        leading = current_query_pos.shape[:-1]
        for name, value in {
            "identity_memory": identity_memory,
            "identity_memory_valid": identity_memory_valid,
            "current_observation": current_observation,
            "predicted_boxes": predicted_boxes,
        }.items():
            if value.shape[: len(leading)] != leading:
                raise ValueError(f"{name} leading shape does not match query_pos")
        if identity_memory.shape[-1] != self.memory_dim:
            raise ValueError("identity_memory dimension mismatch")
        if identity_memory_valid.shape != leading:
            raise ValueError("identity_memory_valid shape mismatch")
        if observation_valid is None:
            observation_valid = torch.ones(leading, dtype=torch.bool, device=current_query_pos.device)
        else:
            observation_valid = observation_valid.to(device=current_query_pos.device, dtype=torch.bool)
        if observation_valid.shape != leading:
            raise ValueError("observation_valid shape mismatch")
        if history_feature is None:
            history_feature = identity_memory.new_zeros(*leading, self.memory_dim)
        if history_feature.shape[:-1] != leading or history_feature.shape[-1] != self.memory_dim:
            raise ValueError("history_feature shape mismatch")
        if history_valid is None:
            history_valid = torch.zeros(leading, dtype=torch.bool, device=current_query_pos.device)
        else:
            history_valid = history_valid.to(device=current_query_pos.device, dtype=torch.bool)
        if history_valid.shape != leading:
            raise ValueError("history_valid shape mismatch")

        pos, content = current_query_pos.split(self.query_dim, dim=-1)
        identity = self.identity_projection(identity_memory)
        observation = self.observation_projection(current_observation)
        boxes = predicted_boxes.to(dtype=current_query_pos.dtype)
        gate_input = torch.cat([identity, observation, boxes], dim=-1)
        write_gate = self.gate(gate_input).squeeze(-1)
        write_gate = torch.where(observation_valid, write_gate, torch.zeros_like(write_gate))
        identity = torch.where(identity_memory_valid.unsqueeze(-1), identity, torch.zeros_like(identity))
        evidence = write_gate.unsqueeze(-1) * observation + (1 - write_gate.unsqueeze(-1)) * identity
        history = self.history_projection(history_feature)
        history = torch.where(history_valid.unsqueeze(-1), history, torch.zeros_like(history))
        new_content = self.norm(content + evidence + history)
        if self.update_query_pos:
            pos = pos + self.box_projection(boxes)
        new_query_pos = torch.cat([pos, new_content], dim=-1)
        if ref_pts is None:
            new_ref_pts = boxes[..., :2]
        else:
            new_ref_pts = ref_pts
        return QueryWriterOutput(new_query_pos, new_ref_pts)

    def write_instances(
        self,
        track_instances,
        identity_memory: Tensor,
        identity_memory_valid: Tensor,
        current_observation: Tensor,
        predicted_boxes: Tensor,
        *,
        observation_valid: Optional[Tensor] = None,
    ):
        """Return a row-aligned ``Instances`` copy with query fields updated."""

        output = self(
            track_instances.query_pos,
            identity_memory,
            identity_memory_valid,
            current_observation,
            predicted_boxes,
            ref_pts=getattr(track_instances, "ref_pts", None),
            observation_valid=observation_valid,
        )
        track_instances.query_pos = output.query_pos
        track_instances.ref_pts = output.ref_pts
        return track_instances
