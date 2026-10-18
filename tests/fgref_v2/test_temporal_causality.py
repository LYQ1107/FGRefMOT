import pytest
import torch

from models.fgref_v2 import TemporalEvidenceEncoder


@pytest.mark.parametrize("mode", ["ema", "gru", "attention"])
def test_history_summary_does_not_read_masked_future(mode):
    torch.manual_seed(7)
    history = torch.randn(2, 5, 4)
    prefix_mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    future_changed = history.clone()
    future_changed[:, 3:] += 1000
    encoder = TemporalEvidenceEncoder(4, mode=mode, attention_heads=2)
    encoder.eval()
    first = encoder(history, prefix_mask)
    second = encoder(future_changed, prefix_mask)
    assert torch.allclose(first.summary, second.summary, atol=1e-6, rtol=1e-6)
    assert torch.allclose(first.encoded_history[:, :3], second.encoded_history[:, :3], atol=1e-6, rtol=1e-6)


def test_empty_track_batch_and_invalid_history_are_safe():
    encoder = TemporalEvidenceEncoder(4, mode="ema")
    history = torch.zeros(0, 3, 4)
    output = encoder(history, torch.zeros(0, 3, dtype=torch.bool))
    assert output.summary.shape == (0, 4)
    all_invalid = encoder(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.bool))
    assert torch.equal(all_invalid.summary, torch.zeros_like(all_invalid.summary))


def test_empty_time_axis_is_safe():
    encoder = TemporalEvidenceEncoder(4, mode="attention", attention_heads=2)
    history = torch.zeros(2, 0, 4)
    output = encoder(history, torch.zeros(2, 0, dtype=torch.bool))
    assert output.summary.shape == (2, 4)
    assert output.encoded_history.shape == history.shape
    assert torch.equal(output.summary, torch.zeros_like(output.summary))
