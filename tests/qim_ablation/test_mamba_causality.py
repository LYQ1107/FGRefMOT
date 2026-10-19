import pytest
import torch

from models.qim_mamba.temporal_mamba import TrackwiseTemporalMixer

from ._mamba_helpers import mamba_args


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lightweight Mamba tests require CUDA"
)


def test_future_tokens_do_not_change_a_prefix_state():
    device = torch.device("cuda")
    mixer = TrackwiseTemporalMixer(mamba_args(), 256).to(device).eval()
    prefix = torch.randn(3, 256, device=device)
    future = torch.randn(2, 256, device=device)
    with torch.no_grad():
        prefix_state = mixer.stack(mixer.input_proj(prefix.unsqueeze(0)))[0][-1]
        full_state = mixer.stack(mixer.input_proj(torch.cat([prefix, future]).unsqueeze(0)))[0][2]
    assert torch.allclose(prefix_state, full_state, atol=2e-4, rtol=2e-4)
