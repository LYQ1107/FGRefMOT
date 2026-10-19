import pytest
import torch

from models.qim_variants import ReliabilityHybridMambaQIM

from ._mamba_helpers import make_tracks, mamba_args


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lightweight Mamba tests require CUDA"
)


def test_memory_fields_follow_instances_cat_and_track_writer():
    device = torch.device("cuda")
    module = ReliabilityHybridMambaQIM(
        mamba_args("mamba_reliability_hybrid"), 256, 1024, 512
    ).to(device).eval()
    active = make_tracks(3, device, memory_dim=module.memory_dim)
    init = make_tracks(0, device, memory_dim=module.memory_dim)
    active.obj_idxes[2] = -1
    active.scores[2] = 0.0
    with torch.no_grad():
        output = module({"track_instances": active, "init_track_instances": init})
    for field in (
        "qim_long_memory",
        "qim_long_memory_valid",
        "qim_memory_age",
        "qim_previous_box",
        "qim_memory_write_gate",
    ):
        assert output.has(field)
        assert len(getattr(output, field)) == len(output)
    assert len(output) == 2
