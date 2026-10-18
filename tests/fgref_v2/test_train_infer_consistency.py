import inspect

import torch
from torch import nn

from models.spatial_temporal_reason import SpatialTemporalReasoner
from models.structures import Instances
from models.transrmot_pro import TransRMOT


def _tracks():
    return Instances(
        (64, 64),
        output_embedding=torch.tensor([[0.2, -0.1, 0.4, 0.3]]),
        cache_pred_boxes=torch.full((1, 4), 0.5),
        cache_pred_logits=torch.zeros(1, 1),
        cache_pred_refers=torch.zeros(1, 1),
    )


def _reasoner_for_refine():
    reasoner = object.__new__(SpatialTemporalReasoner)
    nn.Module.__init__(reasoner)
    reasoner.track_cls = nn.Linear(4, 1, bias=False)
    reasoner.track_ref = nn.Linear(4, 1, bias=False)
    reasoner.track_box = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        reasoner.track_cls.weight.fill_(0.2)
        reasoner.track_ref.weight.fill_(0.3)
        reasoner.track_box.weight.fill_(0.1)
    return reasoner


def test_refine_logits_and_boxes_match_in_training_and_inference():
    reasoner = _reasoner_for_refine()
    train_tracks = _tracks()
    infer_tracks = _tracks()
    reasoner.forward_history_refine(train_tracks, training=True)
    reasoner.forward_history_refine(infer_tracks, training=False)

    assert torch.allclose(train_tracks.cache_pred_logits, infer_tracks.cache_pred_logits)
    assert torch.allclose(train_tracks.cache_pred_refers, infer_tracks.cache_pred_refers)
    assert torch.allclose(train_tracks.cache_pred_boxes, infer_tracks.cache_pred_boxes)


def test_inference_entrypoint_does_not_accept_gt_as_an_input():
    source = inspect.getsource(TransRMOT.inference_single_image)
    assert "gt_instances" not in source
    assert "gt_file" not in source
