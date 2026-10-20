import pytest
import torch

from models.qim_pro import build

from ._helpers import VARIANTS, args, tracks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba smoke requires CUDA")


@pytest.mark.parametrize("variant", VARIANTS)
def test_fast6_interface_handles_empty_and_active_tracks(variant):
    device = torch.device("cuda")
    module = build(args(variant), "QIM", 256, 1024, 512).to(device).eval()
    with torch.no_grad():
        output = module({"track_instances": tracks(5, device), "init_track_instances": tracks(0, device)})
    assert len(output) == 5
    assert output.query_pos.shape == (5, 512)
    assert torch.isfinite(output.query_pos).all()
    with torch.no_grad():
        empty = module({"track_instances": tracks(0, device), "init_track_instances": tracks(0, device)})
    assert len(empty) == 0
