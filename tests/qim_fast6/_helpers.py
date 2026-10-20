from types import SimpleNamespace

import torch

from models.structures import Instances


VARIANTS = (
    "fast_mamba_spatial",
    "fast_mamba_temporal",
    "fast_mamba_streaming",
    "fast_mamba_motion",
    "fast_mamba_language",
    "fast_mamba_reliability",
)


def args(variant="fast_mamba_spatial", update_query_pos=True):
    return SimpleNamespace(
        qim_variant=variant,
        qim_depth=2,
        random_drop=0.0,
        fp_ratio=0.0,
        update_query_pos=update_query_pos,
        merger_dropout=0.0,
        qim_fast_dim=32,
        qim_fast_layers=1,
        qim_fast_state=4,
        qim_fast_history_len=4,
        qim_fast_disable_self_attention=False,
        qim_disable_self_attention=False,
        qim_mamba2_enable=False,
        qim_mamba_dim=32,
        qim_mamba_layers=1,
        qim_mamba_state=4,
        qim_mamba_history_len=4,
        qim_bidirectional_share=True,
        qim_reliability_enable=True,
        qim_rmot_text_gate=False,
    )


def tracks(number, device, state_dim=32, memory_dim=32):
    result = Instances((100, 100))
    result.ref_pts = torch.zeros(number, 2, device=device)
    result.query_pos = torch.randn(number, 512, device=device)
    result.output_embedding = torch.randn(number, 256, device=device, requires_grad=True)
    result.obj_idxes = torch.arange(number, dtype=torch.long, device=device)
    result.matched_gt_idxes = torch.arange(number, dtype=torch.long, device=device)
    result.disappear_time = torch.zeros(number, dtype=torch.long, device=device)
    result.track_query_mask = torch.zeros(number, dtype=torch.bool, device=device)
    result.iou = torch.ones(number, device=device)
    result.scores = torch.full((number,), 0.8, device=device)
    result.track_scores = torch.zeros(number, device=device)
    result.pred_boxes = torch.rand(number, 4, device=device)
    result.pred_logits = torch.zeros(number, 1, device=device)
    result.pred_refers = torch.zeros(number, 1, device=device)
    result.hist_embeds = torch.randn(number, 4, 256, device=device)
    result.hist_padding_masks = torch.zeros(number, 4, dtype=torch.bool, device=device)
    result.qim_long_memory = torch.randn(number, memory_dim, device=device)
    result.qim_long_memory_valid = torch.ones(number, dtype=torch.bool, device=device)
    result.qim_memory_age = torch.zeros(number, dtype=torch.long, device=device)
    result.qim_previous_box = result.pred_boxes.clone()
    result.qim_memory_write_gate = torch.zeros(number, device=device)
    result.qim_stream_state = torch.randn(number, state_dim, device=device)
    result.qim_stream_state_valid = torch.ones(number, dtype=torch.bool, device=device)
    result.qim_motion_previous_box = result.pred_boxes.clone()
    result.qim_motion_previous_valid = torch.ones(number, dtype=torch.bool, device=device)
    return result


def text_data(device):
    return {
        "qim_text_features": torch.randn(1, 4, 256, device=device),
        "qim_text_mask": torch.zeros(1, 4, dtype=torch.bool, device=device),
    }
