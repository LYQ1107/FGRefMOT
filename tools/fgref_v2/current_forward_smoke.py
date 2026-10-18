"""Run a one-sample current-source TAM/QIM-v2 forward smoke.

This uses random model weights and a real Refer-KITTI sample. It validates
the optional integration only; it is not a metric or a trained result.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaModel as HFModel
from transformers import RobertaTokenizerFast as HFTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_ROOT = (
    "/home/liuyeqiang/.cache/huggingface/hub/models--roberta-base/"
    "snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)
DATA_ROOT = "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti"
SPLIT = DATA_ROOT + "/KITTI/training/refer-kitti.train"


class _OfflineTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFTokenizer.from_pretrained(HF_ROOT, local_files_only=True)


class _OfflineModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFModel.from_pretrained(HF_ROOT, local_files_only=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--disable-tam", action="store_true")
    parser.add_argument("--disable-qim", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(REPO_ROOT))

    import models.backbone as backbone  # noqa: E402
    import models.transrmot_pro as transrmot  # noqa: E402
    from datasets.refer_kitti import build as build_dataset  # noqa: E402
    from main import get_args_parser  # noqa: E402
    from models import build_model  # noqa: E402

    backbone.is_main_process = lambda: False
    transrmot.RobertaTokenizerFast = _OfflineTokenizer
    transrmot.RobertaModel = _OfflineModel

    model_args = get_args_parser().parse_args([])
    model_args.meta_arch = "temp_rmot"
    model_args.dataset_file = "e2e_rmot"
    model_args.device = args.device
    model_args.with_box_refine = True
    model_args.sampler_lengths = [1, 1, 1, 1]
    model_args.sampler_steps = [60, 80, 90]
    data_root = Path(args.data_root)
    model_args.rmot_path = str(data_root)
    model_args.data_txt_path_train = str(data_root / "KITTI" / "training" / "refer-kitti.train")
    model_args.vis = False
    model_args.distributed = False
    model_args.rank = 0
    model_args.world_size = 1
    model_args.gpu = 0
    model_args.use_checkpoint = False
    model_args.fgref_v2_tam = not args.disable_tam
    model_args.fgref_v2_tam_local_points = 4
    model_args.fgref_v2_tam_context_points = 8
    model_args.fgref_v2_tam_conditional = False
    model_args.fgref_v2_tam_heads = 8
    model_args.fgref_v2_qim = not args.disable_qim
    model_args.fgref_v2_qim_memory_dim = 128
    model_args.fgref_v2_qim_history_length = 8
    model_args.fgref_v2_qim_temporal_mode = "ema"
    model_args.fgref_v2_qim_update_query_pos = False

    model, _, _ = build_model(model_args)
    device = torch.device(args.device)
    model = model.to(device).eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    tam_parameters = sum(
        parameter.numel() for parameter in model.fgref_v2_tam.parameters()
    ) if model.fgref_v2_tam is not None else 0
    qim_parameters = sum(
        parameter.numel() for parameter in model.track_embed.parameters()
    ) if args.disable_qim is False else 0
    data = build_dataset("train", model_args)[0]
    data["imgs"] = [frame.to(device, non_blocking=True) for frame in data["imgs"]]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(data)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("CURRENT_TAM_QIM_FORWARD_OK")
    print("frames", len(output["pred_logits"]))
    print("pred_logits_shape", tuple(output["pred_logits"][0].shape))
    print("pred_boxes_shape", tuple(output["pred_boxes"][0].shape))
    print("pred_refers_shape", tuple(output["pred_refers"][0].shape))
    print("tam_enabled", model.fgref_v2_tam is not None)
    print("qim_adapter", type(model.track_embed).__name__)
    print("total_parameters", total_parameters)
    print("trainable_parameters", trainable_parameters)
    print("tam_parameters", tam_parameters)
    print("qim_parameters", qim_parameters)
    print("sentence", data["sentences"][0])
    print("elapsed_seconds", round(time.perf_counter() - start, 4))
    if device.type == "cuda":
        print("peak_memory_mib", round(torch.cuda.max_memory_allocated(device) / 2**20, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
