import torch

from models.fgref_v2 import ReliabilityAwareTrackMemory
from models.structures import Instances


def _memory():
    return ReliabilityAwareTrackMemory(3, memory_dim=4, history_length=3, write_threshold=0.5)


def test_memory_follows_physical_ids_after_row_reorder_and_birth():
    memory = _memory()
    ids = torch.tensor([10, 20])
    first, _ = memory.step(
        None,
        ids,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.ones(2),
        observed_mask=torch.ones(2, dtype=torch.bool),
    )
    reordered, reliability = memory.step(
        first,
        torch.tensor([20, 10, 30]),
        None,
        torch.zeros(3),
        observed_mask=torch.zeros(3, dtype=torch.bool),
    )
    assert torch.allclose(reordered.identity_memory[0], first.identity_memory[1])
    assert torch.allclose(reordered.identity_memory[1], first.identity_memory[0])
    assert not reordered.identity_memory_valid[2]
    assert torch.equal(reliability, torch.zeros(3))
    assert reordered.dynamic_history_mask[0, -1].item() is False


def test_instances_cat_slice_and_device_conversion_keep_memory_fields_aligned():
    memory = _memory()
    state_a, _ = memory.step(None, torch.tensor([4]), torch.ones(1, 3), torch.ones(1))
    state_b, _ = memory.step(None, torch.tensor([5]), torch.ones(1, 3) * 2, torch.ones(1))
    merged = Instances(
        (32, 32),
        obj_idxes=torch.cat([state_a.track_ids, state_b.track_ids]),
        identity_memory=torch.cat([state_a.identity_memory, state_b.identity_memory]),
        dynamic_history=torch.cat([state_a.dynamic_history, state_b.dynamic_history]),
    )
    selected = merged[torch.tensor([True, False])]
    moved = selected.to("cpu")
    assert moved.obj_idxes.tolist() == [4]
    assert moved.identity_memory.shape == (1, 4)
    assert moved.dynamic_history.shape == (1, 3, 4)
