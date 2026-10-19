import torch

from models.qim_variants import LSTMQIM, StandardQIM, TemporalAttentionQIM

from ._helpers import args, empty_tracks, tracks


def test_four_frame_query_chain_has_nonzero_qim_gradients():
    torch.manual_seed(11)
    for cls in (StandardQIM, LSTMQIM, TemporalAttentionQIM):
        module = cls(args(), 256, 1024, 512)
        module.train()
        state = tracks(3)
        initial_output = state.output_embedding
        for _ in range(4):
            state = module({
                'track_instances': state,
                'init_track_instances': empty_tracks(),
            })
            # Simulate the next decoder output consuming the QIM-updated
            # query.  This is the cross-frame edge under test.
            state.output_embedding = state.query_pos[:, 256:].clone()
            state.scores = torch.full((len(state),), 0.8)
        loss = state.query_pos.square().mean()
        loss.backward()
        nonzero = [
            p.grad for p in module.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert nonzero, cls.__name__
        assert initial_output.grad is not None
        assert torch.isfinite(initial_output.grad).all()
