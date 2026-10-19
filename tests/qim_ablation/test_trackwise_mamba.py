import pytest
import torch

from models.qim_mamba.temporal_mamba import TrackwiseTemporalMixer

from ._mamba_helpers import mamba_args


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lightweight Mamba tests require CUDA"
)


def test_trackwise_mamba_ignores_masked_padding_and_is_rowwise():
    device = torch.device("cuda")
    mixer = TrackwiseTemporalMixer(mamba_args(), 256).to(device).eval()
    current = torch.randn(2, 256, device=device)
    history = torch.randn(2, 6, 256, device=device)
    padding = torch.ones(2, 6, dtype=torch.bool, device=device)
    padding[:, 2:4] = False
    baseline = mixer(history, padding, current)

    changed_padding_values = history.clone()
    changed_padding_values[:, :2] = torch.randn(2, 2, 256, device=device)
    changed_padding_values[:, 4:] = torch.randn(2, 2, 256, device=device)
    padded_result = mixer(changed_padding_values, padding, current)
    assert torch.allclose(baseline, padded_result, atol=2e-4, rtol=2e-4)

    permutation = torch.tensor([1, 0], device=device)
    reordered = mixer(history[permutation], padding[permutation], current[permutation])
    assert torch.allclose(reordered[torch.argsort(permutation)], baseline, atol=2e-4, rtol=2e-4)


def test_trackwise_mamba_is_causal_at_each_valid_prefix():
    device = torch.device("cuda")
    mixer = TrackwiseTemporalMixer(mamba_args(), 256).to(device).eval()
    sequence = torch.randn(5, 256, device=device)
    with torch.no_grad():
        full = mixer.stack(mixer.input_proj(sequence.unsqueeze(0)))[0]
        prefix = mixer.stack(mixer.input_proj(sequence[:3].unsqueeze(0)))[0]
    assert torch.allclose(full[2], prefix[-1], atol=2e-4, rtol=2e-4)


def test_trackwise_mamba_birth_with_no_valid_history_falls_back_to_current():
    device = torch.device("cuda")
    mixer = TrackwiseTemporalMixer(mamba_args(), 256).to(device).eval()
    current = torch.randn(1, 256, device=device)
    history = torch.randn(1, 4, 256, device=device)
    padding = torch.ones(1, 4, dtype=torch.bool, device=device)
    with torch.no_grad():
        result = mixer(history, padding, current)
    assert torch.allclose(result, current, atol=1e-6, rtol=1e-6)
