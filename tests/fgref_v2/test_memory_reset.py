import torch

from models.fgref_v2 import ReliabilityAwareTrackMemory


def test_terminated_id_does_not_transfer_memory_to_a_reborn_id():
    memory = ReliabilityAwareTrackMemory(2, memory_dim=3, history_length=2)
    old, _ = memory.step(None, torch.tensor([7]), torch.tensor([[3.0, 0.0]]), torch.ones(1))
    terminated = memory.align_state(old, torch.tensor([9]))
    assert terminated.identity_memory_valid.tolist() == [False]
    assert torch.equal(terminated.dynamic_history_mask, torch.zeros(1, 2, dtype=torch.bool))
    reborn = memory.align_state(terminated, torch.tensor([7]))
    assert reborn.identity_memory_valid.tolist() == [False]
    assert torch.equal(reborn.identity_memory, torch.zeros_like(reborn.identity_memory))


def test_duplicate_physical_ids_are_rejected_instead_of_ambiguous_alignment():
    memory = ReliabilityAwareTrackMemory(2, memory_dim=3, history_length=2)
    try:
        memory.align_state(None, torch.tensor([2, 2]))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate physical IDs must be rejected")
