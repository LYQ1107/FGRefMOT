#!/usr/bin/env python3
"""Smoke-test the real FlexHook feature hook used by temporal adapters.

The pinned model is loaded from a verified checkpoint, one fixed-tracker
dataset batch is evaluated, and the final ``output_ca_qkv`` representation is
reported.  No predictions, labels, or checkpoints are changed.  This keeps
the feature interface test separate from the long-running baseline jobs and
does not edit ``third_party/FlexHook``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def run_smoke(
    *,
    config_path: Path,
    layout_root: Path,
    tracker_root: Path,
    checkpoint: Path,
    runtime_root: Path,
    output_root: Path,
    gpu: int,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("FlexHook config or checkpoint is missing")
    if not layout_root.is_dir() or not tracker_root.is_dir():
        raise FileNotFoundError("FlexHook layout or tracker root is missing")
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29613")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    upstream_root = REPO_ROOT / "third_party" / "FlexHook"
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(upstream_root / "TrackEval"))

    from scripts.transformers_compat import install_legacy_tokenizer_alias

    install_legacy_tokenizer_alias()
    import torch
    import torch.distributed as dist
    from config import _C
    from data import build_loader
    from logger import create_logger
    from models import build_model
    from utils import load_ckpt_my

    runtime_root = runtime_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="flexhook_hook_", suffix=".yaml")
    os.close(fd)
    temporary_config = Path(temporary_name)
    shutil.copyfile(config_path, temporary_config)
    try:
        cfg = _C.clone()
        cfg.defrost()
        cfg.merge_from_file(str(temporary_config))
        cfg.DATA.DATASET = "mix"
        cfg.DATA.TRAIN = "SportsMOT"
        cfg.DATA.TEST = "SportsMOT"
        cfg.DATA.VAL_BATCH_SIZE = int(batch_size)
        cfg.DATA.NUM_WORKERS = int(workers)
        cfg.EVAL_MODE = True
        cfg.LOCAL_RANK = 0
        cfg.sample_frame_len = cfg.sample_frame_num
        cfg.MODEL.PRETRAINED = str(checkpoint)
        cfg.MODEL.RESUME = str(checkpoint)
        cfg.OUTPUT = str(output_root)
        cfg.data_root = str(layout_root)
        cfg.val_data_root = str(layout_root)
        cfg.track_root = str(tracker_root)
        cfg.freeze()

        dist.init_process_group(backend="gloo", init_method="env://", rank=0, world_size=1)
        os.chdir(runtime_root)
        _, dataset_val, _, data_loader_val, _ = build_loader(cfg)
        logger = create_logger(str(output_root), dist_rank=0, name="flexhook_feature_hook_smoke")
        model = build_model(cfg).cuda()
        load_ckpt_my(cfg, model, logger)
        model.eval()

        captured: list[torch.Tensor] = []

        def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not hasattr(output, "detach"):
                raise TypeError("output_ca_qkv hook returned a non-tensor value")
            captured.append(output.detach().cpu())

        handle = model.output_ca_qkv[-1].register_forward_hook(capture)
        try:
            batch = next(iter(data_loader_val))
            images, pes, bbox_gt, expressions, expid, expma, video, obj, frame_id, last = batch
            with torch.no_grad():
                outputs, regular = model(
                    images.cuda(non_blocking=True),
                    pes.cuda(non_blocking=True),
                    bbox_gt.cuda(non_blocking=True),
                    expid.cuda(non_blocking=True),
                    expma.cuda(non_blocking=True),
                )
        finally:
            handle.remove()
            if hasattr(data_loader_val, "_iterator") and data_loader_val._iterator is not None:
                data_loader_val._iterator._shutdown_workers()

        if len(captured) != 1:
            raise RuntimeError(f"expected one captured feature tensor, got {len(captured)}")
        feature = captured[0]
        result = {
            "schema_version": "flexhook-feature-hook-smoke-0.1",
            "status": "SUCCEEDED",
            "policy": "ONE_BATCH_FEATURE_INTERFACE_ONLY; NO_GT_AS_TRACKER_INPUT; NO_CHECKPOINT_WRITE",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "layout_root": str(layout_root),
            "tracker_root": str(tracker_root),
            "dataset_count": len(dataset_val),
            "batch_size": batch_size,
            "model_output_shape": _shape(outputs),
            "regularizer_shape": _shape(regular),
            "hook_module": "model.output_ca_qkv[-1]",
            "hook_feature_shape": _shape(feature),
            "feature_dtype": str(feature.dtype),
            "sample": {
                "video": [str(value) for value in video],
                "object": [str(value) for value in obj],
                "frame_id": [str(value) for value in frame_id],
                "expression_count": len(expressions[0]) if expressions else 0,
                "last_expression_count": [int(value) for value in last],
            },
            "upstream_checkout_modified": False,
        }
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        temporary_config.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(
        config_path=Path(args.config).expanduser().resolve(),
        layout_root=Path(args.layout_root).expanduser().resolve(),
        tracker_root=Path(args.tracker_root).expanduser().resolve(),
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        runtime_root=Path(args.runtime_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        gpu=args.gpu,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"flexhook_feature_hook_smoke: {result['status']}")
        print(json.dumps({key: result[key] for key in ("dataset_count", "model_output_shape", "hook_feature_shape")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
