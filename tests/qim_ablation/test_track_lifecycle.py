import torch

from models.qim_variants import LSTMQIM, StandardQIM, TemporalAttentionQIM
from models.transrmot_pro import RuntimeTrackerBase

from ._helpers import args, empty_tracks, tracks


def test_active_selection_is_identical_for_all_variants():
    base = tracks(3)
    base.obj_idxes = torch.tensor([-1, 4, 5])
    base.scores = torch.tensor([0.8, 0.2, 0.8])
    base.iou = torch.tensor([1.0, 0.4, 1.0])
    expected_training = None
    expected_eval = None
    for cls in (StandardQIM, LSTMQIM, TemporalAttentionQIM):
        module = cls(args(), 256, 1024, 512)
        module.train()
        training = module._select_active_tracks({'track_instances': base})
        module.eval()
        evaluation = module._select_active_tracks({'track_instances': base})
        values = (training.obj_idxes.tolist(), evaluation.obj_idxes.tolist())
        if expected_training is None:
            expected_training, expected_eval = values
        assert values == (expected_training, expected_eval)
    assert expected_training == [-1, -1, 5]
    assert expected_eval == [4, 5]


def test_runtime_tracker_thresholds_are_not_variant_specific():
    tracker = RuntimeTrackerBase()
    assert (tracker.score_thresh, tracker.filter_score_thresh, tracker.miss_tolerance) == (0.6, 0.5, 10)
