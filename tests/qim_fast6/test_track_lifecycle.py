import pytest
import torch

from models.qim_pro import build

from ._helpers import args, tracks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba smoke requires CUDA")


@pytest.mark.parametrize("variant", ("fast_mamba_streaming", "fast_mamba_motion", "fast_mamba_reliability"))
def test_state_fields_follow_active_track_lifecycle(variant):
    device = torch.device("cuda")
    module = build(args(variant), "QIM", 256, 1024, 512).to(device).eval()
    state = tracks(4, device)
    state.obj_idxes[1] = -1
    state.scores[1] = 0.0
    with torch.no_grad():
        output = module({"track_instances": state, "init_track_instances": tracks(0, device)})
    assert len(output) == 3
    for name in output.get_fields():
        assert len(output.get(name)) == len(output)
    expected = {
        "fast_mamba_streaming": "qim_stream_state",
        "fast_mamba_motion": "qim_motion_previous_box",
        "fast_mamba_reliability": "qim_long_memory",
    }[variant]
    assert output.has(expected)
