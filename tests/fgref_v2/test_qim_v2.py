import torch

from models.fgref_v2 import ReliabilityAwareQIMV2, ReliabilityAwareQueryInteraction
from models.structures import Instances


def test_qim_v2_reads_previous_history_before_current_write_and_reorders_by_id():
    torch.manual_seed(3)
    qim = ReliabilityAwareQIMV2(
        observation_dim=2,
        query_dim=4,
        memory_dim=4,
        history_length=2,
        temporal_mode="ema",
    )
    first = qim(
        state=None,
        track_ids=torch.tensor([5, 6]),
        current_query_pos=torch.randn(2, 8),
        predicted_boxes=torch.full((2, 4), 0.5),
        current_observation=torch.ones(2, 2),
        detection_quality=torch.ones(2),
    )
    second = qim(
        state=first.state,
        track_ids=torch.tensor([6, 5]),
        current_query_pos=torch.zeros(2, 8),
        predicted_boxes=torch.full((2, 4), 0.5),
        current_observation=None,
        detection_quality=torch.zeros(2),
        observed_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert second.temporal.valid_mask[:, -1].tolist() == [True, True]
    assert torch.allclose(second.state.identity_memory[0], first.state.identity_memory[1])
    assert torch.allclose(second.state.identity_memory[1], first.state.identity_memory[0])


def test_qim_v2_none_observation_is_not_written_without_explicit_mask():
    torch.manual_seed(4)
    qim = ReliabilityAwareQIMV2(
        observation_dim=2,
        query_dim=4,
        memory_dim=4,
        history_length=2,
        temporal_mode="ema",
    )
    first = qim(
        state=None,
        track_ids=torch.tensor([9]),
        current_query_pos=torch.randn(1, 8),
        predicted_boxes=torch.full((1, 4), 0.5),
        current_observation=torch.ones(1, 2),
        detection_quality=torch.ones(1),
    )
    second = qim(
        state=first.state,
        track_ids=torch.tensor([9]),
        current_query_pos=torch.zeros(1, 8),
        predicted_boxes=torch.full((1, 4), 0.5),
        current_observation=None,
        detection_quality=torch.ones(1),
    )
    assert torch.allclose(second.state.identity_memory, first.state.identity_memory)
    assert second.state.dynamic_history_mask[0, -1].item() is False


def test_qim_v2_instances_adapter_keeps_state_fields_row_aligned():
    def make_instances(count, ids, scores, memory_dim=3, history_length=2):
        return Instances(
            (32, 32),
            obj_idxes=torch.tensor(ids, dtype=torch.long),
            scores=torch.tensor(scores),
            iou=torch.ones(count),
            query_pos=torch.randn(count, 8),
            ref_pts=torch.full((count, 2), 0.5),
            pred_boxes=torch.full((count, 4), 0.5),
            output_embedding=torch.ones(count, 2),
            fgref_v2_identity_memory=torch.zeros(count, memory_dim),
            fgref_v2_identity_memory_valid=torch.zeros(count, dtype=torch.bool),
            fgref_v2_dynamic_history=torch.zeros(count, history_length, memory_dim),
            fgref_v2_dynamic_history_mask=torch.zeros(count, history_length, dtype=torch.bool),
            fgref_v2_observation_quality=torch.zeros(count),
            fgref_v2_memory_age=torch.zeros(count, dtype=torch.long),
        )

    adapter = ReliabilityAwareQueryInteraction(
        observation_dim=2,
        query_dim=4,
        memory_dim=3,
        history_length=2,
        temporal_mode="ema",
    )
    adapter.train()
    output = adapter(
        {
            "init_track_instances": make_instances(1, [-1], [0.0]),
            "track_instances": make_instances(2, [12, -1], [1.0, 1.0]),
        }
    )
    assert len(output) == 3
    assert output.fgref_v2_identity_memory_valid.tolist() == [False, True, True]
    assert output.fgref_v2_dynamic_history_mask[:, -1].tolist() == [False, True, True]
