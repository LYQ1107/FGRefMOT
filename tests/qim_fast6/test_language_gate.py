import pytest
import torch

from models.qim_pro import build

from ._helpers import args, text_data, tracks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba smoke requires CUDA")


def test_language_condition_changes_only_when_text_changes():
    device = torch.device("cuda")
    module = build(args("fast_mamba_language"), "QIM", 256, 1024, 512).to(device).eval()
    state_a = tracks(4, device)
    state_b = state_a.to(device)
    first_text = text_data(device)
    second_text = {key: value.clone() for key, value in first_text.items()}
    second_text["qim_text_features"] += 3.0
    with torch.no_grad():
        first = module({"track_instances": state_a, "init_track_instances": tracks(0, device), **first_text}).query_pos
        second = module({"track_instances": state_b, "init_track_instances": tracks(0, device), **second_text}).query_pos
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.allclose(first, second)
