import pytest
import torch

from models.qim_pro import build

from ._helpers import VARIANTS, args, tracks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba smoke requires CUDA")


@pytest.mark.parametrize("variant", VARIANTS)
def test_fast6_has_nonzero_finite_backward(variant):
    device = torch.device("cuda")
    module = build(args(variant), "QIM", 256, 1024, 512).to(device).train()
    state = tracks(4, device)
    output = module({"track_instances": state, "init_track_instances": tracks(0, device)})
    output.query_pos.square().mean().backward()
    gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and any(g.abs().sum() > 0 for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients)
