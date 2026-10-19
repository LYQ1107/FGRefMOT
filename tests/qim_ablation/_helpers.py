from types import SimpleNamespace

import torch

from models.structures import Instances


def args(update_query_pos=True, qim_depth=2):
    return SimpleNamespace(
        random_drop=0.0,
        fp_ratio=0.0,
        update_query_pos=update_query_pos,
        merger_dropout=0.0,
        qim_depth=qim_depth,
    )


def tracks(number, positive=True):
    result = Instances((100, 100))
    result.ref_pts = torch.zeros(number, 2)
    result.query_pos = torch.randn(number, 512).clone()
    result.output_embedding = torch.randn(number, 256, requires_grad=True)
    result.obj_idxes = torch.arange(number, dtype=torch.long)
    result.matched_gt_idxes = torch.arange(number, dtype=torch.long)
    result.disappear_time = torch.zeros(number, dtype=torch.long)
    result.track_query_mask = torch.zeros(number, dtype=torch.bool)
    result.iou = torch.ones(number)
    result.scores = torch.full((number,), 0.8 if positive else 0.0)
    result.track_scores = torch.zeros(number)
    result.pred_boxes = torch.rand(number, 4)
    result.pred_logits = torch.zeros(number, 1)
    result.pred_refers = torch.zeros(number, 1)
    return result


def empty_tracks():
    result = tracks(0)
    return result
