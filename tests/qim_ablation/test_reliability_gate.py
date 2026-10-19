import pytest
import torch

from models.qim_mamba.hybrid_memory_mamba import ReliabilityHybridMambaMixer

from ._mamba_helpers import make_tracks, mamba_args


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lightweight Mamba tests require CUDA"
)


def test_reliability_gate_is_bounded_and_writes_birth_memory():
    device = torch.device("cuda")
    args = mamba_args("mamba_reliability_hybrid")
    mixer = ReliabilityHybridMambaMixer(args, 256).to(device).eval()
    tracks = make_tracks(2, device, memory_dim=mixer.memory_dim)
    tracks.qim_long_memory_valid[0] = False
    tracks.qim_long_memory_valid[1] = True
    tracks.scores[0] = 0.8
    tracks.scores[1] = 0.05
    with torch.no_grad():
        output = mixer(tracks.output_embedding, tracks)
    gate = tracks.qim_memory_write_gate
    assert torch.isfinite(output).all()
    assert torch.all((gate >= 0) & (gate <= 1))
    assert tracks.qim_long_memory_valid[0]
    assert torch.isfinite(tracks.qim_long_memory).all()
    assert mixer.last_write_ratio > 0


def test_reliability_gate_has_a_real_gradient_path():
    device = torch.device("cuda")
    mixer = ReliabilityHybridMambaMixer(mamba_args("mamba_reliability_hybrid"), 256).to(device)
    tracks = make_tracks(3, device, memory_dim=mixer.memory_dim)
    current = tracks.output_embedding.detach().clone().requires_grad_(True)
    output = mixer(current, tracks)
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in mixer.gate.parameters()]
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
