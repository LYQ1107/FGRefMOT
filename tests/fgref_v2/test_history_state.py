import inspect

import torch

from models.spatial_temporal_reason import SpatialTemporalReasoner
from models.structures import Instances


def _tracks(num=2, history=3, dim=4):
    return Instances(
        (64, 64),
        query_pos=torch.zeros(num, dim * 2),
        hist_embeds=torch.zeros(num, history, dim),
        hist_padding_masks=torch.ones(num, history, dtype=torch.bool),
        output_embedding=torch.zeros(num, dim),
    )


def test_history_shift_is_per_row_and_causal():
    reasoner = object.__new__(SpatialTemporalReasoner)
    tracks = _tracks()
    tracks.output_embedding = torch.tensor([[1.0, 0, 0, 0], [2.0, 0, 0, 0]])
    reasoner.frame_shift(tracks)

    assert torch.equal(tracks.hist_embeds[:, :-1], torch.zeros(2, 2, 4))
    assert tracks.hist_embeds[:, -1, 0].tolist() == [1.0, 2.0]
    assert tracks.hist_padding_masks[:, -1].tolist() == [False, False]

    tracks.output_embedding = torch.tensor([[3.0, 0, 0, 0], [4.0, 0, 0, 0]])
    reasoner.frame_shift(tracks)
    assert tracks.hist_embeds[:, :, 0].tolist() == [[0.0, 1.0, 3.0], [0.0, 2.0, 4.0]]


def test_history_shift_handles_zero_tracks():
    reasoner = object.__new__(SpatialTemporalReasoner)
    tracks = _tracks(num=0)
    reasoner.frame_shift(tracks)
    assert tracks.hist_embeds.shape == (0, 3, 4)
    assert tracks.hist_padding_masks.shape == (0, 3)


def test_text_info_is_not_used_by_current_reasoner():
    source = inspect.getsource(SpatialTemporalReasoner.forward_history_reasoning)
    assert "text_info" not in source
