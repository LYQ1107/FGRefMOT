import pytest
import torch

from models.qim_variants import OrderedBiMambaQIM

from ._mamba_helpers import make_tracks, mamba_args


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="lightweight Mamba tests require CUDA"
)


def _run(module, tracks):
    init = make_tracks(0, tracks.query_pos.device)
    with torch.no_grad():
        return module(
            {"track_instances": tracks, "init_track_instances": init}
        )


def test_ordered_bimamba_restores_track_alignment_after_permutation():
    device = torch.device("cuda")
    module = OrderedBiMambaQIM(mamba_args("mamba_ordered_bi"), 256, 1024, 512)
    module.to(device).eval()
    original = make_tracks(6, device)
    original.pred_boxes[:, :2] = torch.tensor(
        [[0.1, 0.8], [0.3, 0.2], [0.5, 0.7], [0.7, 0.4], [0.2, 0.5], [0.9, 0.1]],
        device=device,
    )
    permutation = torch.tensor([3, 0, 5, 2, 1, 4], device=device)
    expected = _run(module, original).query_pos
    observed = _run(module, original[permutation]).query_pos[
        torch.argsort(permutation)
    ]
    assert torch.allclose(observed, expected, atol=2e-4, rtol=2e-4)


def test_ordered_bimamba_has_target_free_tie_break_for_new_tracks():
    device = torch.device("cuda")
    module = OrderedBiMambaQIM(mamba_args("mamba_ordered_bi"), 256, 1024, 512)
    module.to(device).eval()
    tracks = make_tracks(5, device)
    tracks.obj_idxes.fill_(-1)
    tracks.pred_boxes[:] = tracks.pred_boxes[0]
    permutation = torch.tensor([4, 1, 3, 0, 2], device=device)
    with torch.no_grad():
        first = module.mixer(tracks.output_embedding, tracks)
        second = module.mixer(
            tracks.output_embedding[permutation], tracks[permutation]
        )[torch.argsort(permutation)]
    assert torch.allclose(first, second, atol=2e-4, rtol=2e-4)
