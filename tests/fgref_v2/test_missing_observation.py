import torch

from models.fgref_v2 import ReliabilityAwareTrackMemory


def test_missing_observation_does_not_write_identity_and_advances_age():
    memory = ReliabilityAwareTrackMemory(2, memory_dim=3, history_length=3)
    state, _ = memory.step(None, torch.tensor([11]), torch.ones(1, 2), torch.ones(1))
    before = state.identity_memory.clone()
    missing, reliability = memory.step(
        state,
        torch.tensor([11]),
        None,
        torch.ones(1),
        observed_mask=torch.zeros(1, dtype=torch.bool),
    )
    assert torch.allclose(missing.identity_memory, before)
    assert missing.identity_memory_valid.tolist() == [True]
    assert missing.memory_age.tolist() == [1]
    assert missing.dynamic_history_mask[0, -1].item() is False
    assert torch.equal(reliability, torch.zeros(1))
