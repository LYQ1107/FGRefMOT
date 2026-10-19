import torch

from models.qim_variants import LSTMQIM, StandardQIM, TemporalAttentionQIM

from ._helpers import args, empty_tracks, tracks


def test_all_reimplemented_variants_handle_empty_single_and_multiple_tracks():
    torch.manual_seed(7)
    for cls in (StandardQIM, LSTMQIM, TemporalAttentionQIM):
        module = cls(args(), 256, 1024, 512).eval()
        for number in (0, 1, 5):
            output = module({
                'track_instances': tracks(number),
                'init_track_instances': empty_tracks(),
            })
            assert len(output) == number
            assert output.query_pos.shape == (number, 512)
            assert output.ref_pts.shape == (number, 2)


def test_variant_selection_keeps_original_forward_contract():
    from models.qim_pro import build

    module = build(args(), 'QIM', 256, 1024, 512)
    assert module.__class__.__name__ == 'QueryInteractionModule'
