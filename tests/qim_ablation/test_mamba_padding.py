import pytest
import torch

from models.qim_variants import TrackwiseTemporalMambaQIM

from ._mamba_helpers import make_tracks, mamba_args


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lightweight Mamba tests require CUDA"
)


def test_qim_trackwise_variant_handles_missing_observations():
    device = torch.device("cuda")
    module = TrackwiseTemporalMambaQIM(mamba_args(), 256, 1024, 512)
    module.to(device).eval()
    tracks = make_tracks(3, device)
    tracks.hist_padding_masks[0] = torch.tensor([True, True, False, True], device=device)
    tracks.hist_padding_masks[1] = torch.tensor([True, False, True, False], device=device)
    tracks.hist_padding_masks[2] = True
    with torch.no_grad():
        output = module({"track_instances": tracks, "init_track_instances": make_tracks(0, device)})
    assert len(output) == 3
    assert torch.isfinite(output.query_pos).all()
