import pytest
import torch

from models.qim_pro import build

from ._helpers import VARIANTS, args, tracks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba smoke requires CUDA")


@pytest.mark.parametrize("variant", VARIANTS)
def test_track_aligned_output_is_finite_after_row_permutation(variant):
    device = torch.device("cuda")
    torch.manual_seed(17)
    module = build(args(variant), "QIM", 256, 1024, 512).to(device).eval()
    permutation = torch.tensor([2, 0, 4, 1, 3], device=device)
    inverse = torch.argsort(permutation)
    original = tracks(5, device)
    reordered = original[permutation]
    with torch.no_grad():
        expected = module({"track_instances": original, "init_track_instances": tracks(0, device)}).query_pos
        observed = module({"track_instances": reordered, "init_track_instances": tracks(0, device)}).query_pos[inverse]
    assert expected.shape == observed.shape
    assert torch.isfinite(observed).all()
