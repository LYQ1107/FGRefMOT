"""Run a one-sample historical TempRMOT forward smoke.

This is not an evaluation script. It loads the historical checkpoint strictly,
reads one real Refer-KITTI sample, and reports output shapes. The NAS remains
read-only and no result is written by this script.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import torch
from transformers import RobertaModel as HFModel
from transformers import RobertaTokenizerFast as HFTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPooler


DEFAULT_SOURCE_ROOT = "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT"
DEFAULT_CHECKPOINT = (
    "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/"
    "exps/official_checkpoint/checkpoint_0060.pth"
)
DEFAULT_DATA_ROOT = "/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti"
DEFAULT_SPLIT = DEFAULT_DATA_ROOT + "/KITTI/training/refer-kitti.train"
DEFAULT_HF_ROOT = (
    "/home/liuyeqiang/.cache/huggingface/hub/models--roberta-base/"
    "snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)


class _OfflineTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFTokenizer.from_pretrained(DEFAULT_HF_ROOT, local_files_only=True)


class _OfflineModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return HFModel.from_pretrained(DEFAULT_HF_ROOT, local_files_only=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, args.source_root)

    import models.backbone as backbone  # noqa: E402
    import models.transrmot_pro as transrmot  # noqa: E402
    from datasets.refer_kitti import build as build_dataset  # noqa: E402

    backbone.is_main_process = lambda: False
    transrmot.RobertaTokenizerFast = _OfflineTokenizer
    transrmot.RobertaModel = _OfflineModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["args"]
    model_args.device = args.device
    model_args.meta_arch = "temp_rmot"
    model_args.use_checkpoint = False
    model_args.sampler_lengths = [1, 1, 1, 1]
    model_args.sampler_steps = [60, 80, 90]
    model_args.rmot_path = args.data_root
    model_args.data_txt_path_train = args.split
    model_args.vis = False
    model_args.distributed = False
    model_args.rank = 0
    model_args.world_size = 1
    model_args.gpu = 0

    model, _, _ = transrmot.build(model_args)
    model.text_encoder.pooler = RobertaPooler(model.text_encoder.config)
    model.text_encoder.embeddings._non_persistent_buffers_set.discard("position_ids")
    model.load_state_dict(checkpoint["model"], strict=True)

    device = torch.device(args.device)
    model = model.to(device).eval()
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
    elapsed = time.perf_counter() - start

    print("HISTORICAL_FORWARD_OK")
    print("frames", len(output["pred_logits"]))
    print("pred_logits_shape", tuple(output["pred_logits"][0].shape))
    print("pred_boxes_shape", tuple(output["pred_boxes"][0].shape))
    print("pred_refers_shape", tuple(output["pred_refers"][0].shape))
    print("sentence", data["sentences"][0])
    print("elapsed_seconds", round(elapsed, 4))
    if device.type == "cuda":
        print("peak_memory_mib", round(torch.cuda.max_memory_allocated(device) / 2**20, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
