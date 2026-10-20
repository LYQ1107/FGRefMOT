import pytest
import torch

from models.qim_pro import build

from ._helpers import args, tracks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba smoke requires CUDA")


def test_streaming_state_is_written_and_persistent():
    device = torch.device("cuda")
    module = build(args("fast_mamba_streaming"), "QIM", 256, 1024, 512).to(device).eval()
    state = tracks(3, device)
    state.qim_stream_state_valid[:] = False
    init = tracks(0, device)
    with torch.no_grad():
        first = module({"track_instances": state, "init_track_instances": init})
    assert first.has("qim_stream_state")
    assert first.qim_stream_state_valid.all()
    before = first.qim_stream_state.clone()
    first.output_embedding = torch.randn_like(first.output_embedding)
    with torch.no_grad():
        second = module({"track_instances": first, "init_track_instances": tracks(0, device)})
    assert torch.isfinite(second.qim_stream_state).all()
    assert not torch.allclose(before, second.qim_stream_state)
