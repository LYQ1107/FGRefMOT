import torch

from models.fgref_v2 import ReliabilityAwareTrackMemory


def test_track_termination_removes_row_without_reassigning_remaining_state():
    memory = ReliabilityAwareTrackMemory(2, memory_dim=3, history_length=2)
    state, _ = memory.step(
        None,
        torch.tensor([1, 2]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.ones(2),
    )
    remaining = memory.align_state(state, torch.tensor([2]))
    assert remaining.track_ids.tolist() == [2]
    assert torch.allclose(remaining.identity_memory[0], state.identity_memory[1])
    assert remaining.dynamic_history.shape == (1, 2, 3)
