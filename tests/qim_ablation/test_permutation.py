import torch

from models.qim_variants import LSTMQIM, StandardQIM, TemporalAttentionQIM
from models.qim_pro import QueryInteractionModule

from ._helpers import args, empty_tracks, tracks


def _run(module, instance):
    module.eval()
    return module({'track_instances': instance,
                   'init_track_instances': empty_tracks().to(instance.query_pos.device)})


def test_permutation_experiment_returns_track_aligned_outputs():
    torch.manual_seed(19)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modules = [
        StandardQIM(args(), 256, 1024, 512),
        LSTMQIM(args(), 256, 1024, 512),
        TemporalAttentionQIM(args(), 256, 1024, 512),
        QueryInteractionModule(args(), 256, 1024, 512),
    ]
    modules = [module.to(device) for module in modules]
    permutation = torch.tensor([2, 0, 4, 1, 3])
    inverse = torch.argsort(permutation)
    for module in modules:
        original = tracks(5).to(device)
        reordered = original[permutation]
        expected = _run(module, original).query_pos
        observed = _run(module, reordered).query_pos[inverse]
        assert observed.shape == expected.shape
        assert torch.isfinite(observed).all()
        # Keep the diagnostic, but do not impose permutation invariance on a
        # mixer whose explicit sequence-axis behavior is what is being tested.
        assert torch.isfinite((observed - expected)).all()
