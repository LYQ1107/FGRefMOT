import torch

from models.fgref_v2 import EvidenceQueryWriter, ReliabilityAwareQIMV2


def test_new_track_birth_initializes_memory_and_writes_compatible_query():
    qim = ReliabilityAwareQIMV2(
        observation_dim=3,
        query_dim=4,
        memory_dim=5,
        history_length=3,
        temporal_mode="ema",
    )
    output = qim(
        state=None,
        track_ids=torch.tensor([101, 102]),
        current_query_pos=torch.zeros(2, 8),
        predicted_boxes=torch.full((2, 4), 0.5),
        current_observation=torch.randn(2, 3),
        detection_quality=torch.ones(2),
        observed_mask=torch.ones(2, dtype=torch.bool),
    )
    assert output.query_pos.shape == (2, 8)
    assert output.ref_pts.shape == (2, 2)
    assert output.state.identity_memory_valid.tolist() == [True, True]
    assert output.state.dynamic_history_mask[:, -1].tolist() == [True, True]
