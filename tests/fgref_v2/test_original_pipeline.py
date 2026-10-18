import inspect

import torch

from models.structures import Instances
from models.transrmot_pro import RuntimeTrackerBase, TransRMOT


def _instances(num=0):
    return Instances(
        (64, 64),
        obj_idxes=torch.full((num,), -1, dtype=torch.long),
        disappear_time=torch.zeros((num,), dtype=torch.long),
        scores=torch.zeros((num,), dtype=torch.float32),
        pred_refers=torch.zeros((num, 1), dtype=torch.float32),
    )


def test_runtime_tracker_handles_zero_one_and_many_tracks():
    tracker = RuntimeTrackerBase(score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=2)

    empty = _instances(0)
    tracker.update(empty)
    assert len(empty) == 0

    one = _instances(1)
    one.scores[:] = 0.9
    tracker.update(one)
    assert one.obj_idxes.tolist() == [0]

    many = _instances(3)
    many.scores[:] = torch.tensor([0.9, 0.8, 0.7])
    tracker.update(many)
    assert many.obj_idxes.tolist() == [1, 2, 3]
    assert tracker.max_obj_id == 4


def test_cache_loading_and_frame_summary_are_explicit_overwrites():
    model = object.__new__(TransRMOT)
    model.training = False
    tracks = _instances(2)
    tracks.cache_scores = torch.zeros(2)
    tracks.cache_pred_logits = torch.zeros(2, 1)
    tracks.cache_pred_boxes = torch.zeros(2, 4)
    tracks.cache_pred_refers = torch.zeros(2, 1)

    logits = torch.tensor([[[1.0], [2.0]]])
    boxes = torch.full((1, 2, 4), 0.25)
    refers = torch.tensor([[[3.0], [4.0]]])
    query_feats = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    frame = {
        "pred_logits": logits,
        "pred_boxes": boxes,
        "pred_refers": refers,
        "query_feats": query_feats,
    }

    model.load_detection_output_into_cache(tracks, frame)
    assert "query_feats" not in frame
    assert torch.equal(tracks.cache_pred_logits, logits[0])
    assert torch.equal(tracks.output_embedding, query_feats[0])

    tracks.pred_logits = torch.full((2, 1), -99.0)
    tracks.pred_boxes = torch.full((2, 4), -99.0)
    tracks.pred_refers = torch.full((2, 1), -99.0)
    model.frame_summarization(tracks, tracking=False)
    assert torch.equal(tracks.pred_logits, tracks.cache_pred_logits)
    assert torch.equal(tracks.pred_boxes, tracks.cache_pred_boxes)
    assert torch.equal(tracks.pred_refers, tracks.cache_pred_refers)


def test_referring_score_does_not_control_physical_id_assignment():
    tracker = RuntimeTrackerBase(score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=2)
    tracks = _instances(1)
    tracks.scores[:] = 0.9
    tracks.pred_refers[:] = -100.0
    tracker.update(tracks)
    first_id = tracks.obj_idxes.item()

    next_frame = _instances(1)
    next_frame.obj_idxes[:] = first_id
    next_frame.scores[:] = 0.9
    next_frame.pred_refers[:] = 100.0
    tracker.update(next_frame)
    assert next_frame.obj_idxes.item() == first_id


def test_single_image_inference_has_no_gt_dependency():
    source = inspect.getsource(TransRMOT.inference_single_image).lower()
    assert "gt_instances" not in source
    assert "gt_file" not in source
