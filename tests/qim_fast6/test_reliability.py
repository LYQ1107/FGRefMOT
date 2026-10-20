import pytest
import torch

from models.qim_pro import build

from ._helpers import args, tracks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba smoke requires CUDA")


def test_reliability_memory_birth_and_bounded_gate():
    device = torch.device("cuda")
    module = build(args("fast_mamba_reliability"), "QIM", 256, 1024, 512).to(device).eval()
    state = tracks(3, device)
    state.qim_long_memory_valid[:] = False
    with torch.no_grad():
        output = module({"track_instances": state, "init_track_instances": tracks(0, device)})
    gate = output.qim_memory_write_gate
    assert torch.isfinite(output.qim_long_memory).all()
    assert torch.all((gate >= 0) & (gate <= 1))
    assert output.qim_long_memory_valid.all()


def test_reliability_gate_has_gradient_path():
    device = torch.device("cuda")
    module = build(args("fast_mamba_reliability"), "QIM", 256, 1024, 512).to(device).train()
    state = tracks(3, device)
    output = module({"track_instances": state, "init_track_instances": tracks(0, device)})
    output.query_pos.square().mean().backward()
    gate_grads = [p.grad for p in module.mixer.gate.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in gate_grads)
