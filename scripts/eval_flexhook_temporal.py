#!/usr/bin/env python3
"""Evaluate a learned causal temporal head on frozen FlexHook features.

The upstream FlexHook model and the physical tracker remain frozen.  This
adapter collects the model's final query features and text representations,
sorts each physical-track/expression stream by frame, applies the causal head,
and then writes the normal MOT/TrackEval tree.  The historical test split is
used only by the final offline evaluator; it is never used to train or select
the temporal checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_LIST = REPO_ROOT / "configs" / "refer_sports_eval_sequences_20261019.txt"
sys.path.insert(0, str(REPO_ROOT))

from adapters.flexhook_temporal.adapter import TemporalEvidenceAdapter, run_causal_sequence
from scripts.flexhook_feature_archive import split_data_key
from scripts.eval_flexhook_refersports import (
    _CudaTextBatchLoader,
    _load_mot_rows,
    _working_directory,
    patch_legacy_tokenizer_api,
    prepare_eval_tree,
    read_sequence_list,
    resolve_tracker_file,
    resolve_tracker_input_root,
    run_trackeval,
    sha256_file,
    write_predictions,
)


def _normalize_eval_expressions(value: Any, batch_size: int) -> list[list[str]]:
    """Normalize the two list orientations emitted by FlexHook's collate."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("expression metadata must be a list or tuple")
    if len(value) and all(isinstance(item, (list, tuple)) and len(item) == batch_size for item in value):
        return [
            [str(value[candidate][sample]) for candidate in range(len(value))]
            for sample in range(batch_size)
        ]
    if len(value) == batch_size and all(isinstance(item, (list, tuple)) for item in value):
        return [[str(item) for item in row] for row in value]
    if len(value) == batch_size and all(isinstance(item, str) for item in value):
        return [[str(item)] for item in value]
    raise ValueError("could not normalize FlexHook evaluation expressions")


def _metadata_item(value: Any, index: int) -> Any:
    item = value[index]
    if hasattr(item, "item"):
        try:
            return item.item()
        except ValueError:
            pass
    return item


def _track_id(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        import hashlib

        return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def _tracker_boxes(tracker_root: Path, videos: list[str]) -> dict[tuple[str, int, int], torch.Tensor]:
    boxes: dict[tuple[str, int, int], torch.Tensor] = {}
    for video in videos:
        path = resolve_tracker_file(tracker_root, video)
        rows = _load_mot_rows(path)
        for row in rows:
            boxes[(video, int(row[0]), int(row[1]))] = torch.tensor(row[2:6], dtype=torch.float32)
    return boxes


def collect_temporal_outputs(
    *,
    config: Any,
    data_loader: Any,
    model: torch.nn.Module,
    adapter: TemporalEvidenceAdapter,
    tracker_boxes: dict[tuple[str, int, int], torch.Tensor],
    device: torch.device,
    logger: Any,
) -> dict[str, Any]:
    """Collect frozen model outputs and apply the adapter chronologically."""

    captured: list[torch.Tensor] = []
    language_captured: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not hasattr(output, "detach"):
            raise TypeError("output_ca_qkv hook returned a non-tensor value")
        captured.append(output.detach().cpu())

    def capture_language(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", output)
        if not hasattr(hidden, "detach"):
            raise TypeError("text encoder hook returned no tensor hidden state")
        language_captured.append(hidden.detach().cpu())

    feature_handle = model.output_ca_qkv[-1].register_forward_hook(capture)
    language_handle = model.text_encoder.register_forward_hook(capture_language)
    rows: dict[tuple[str, str, str], dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(dict)
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(_CudaTextBatchLoader(data_loader)):
                images, pes, bbox_gt, expressions, expid, expma, videos, objects, frame_ids, _last = batch
                batch_size = int(images.shape[0])
                captured.clear()
                language_captured.clear()
                outputs, _regular = model(
                    images.cuda(non_blocking=True),
                    pes.cuda(non_blocking=True),
                    bbox_gt.cuda(non_blocking=True),
                    expid,
                    expma,
                )
                if len(captured) != 1 or len(language_captured) != 1:
                    raise RuntimeError(
                        f"expected one visual and text hook output, got {len(captured)} and {len(language_captured)}"
                    )
                features = captured[0]
                language_tokens = language_captured[0]
                logits = outputs[:, -1].detach().cpu()
                if features.ndim != 3 or logits.ndim != 3:
                    raise RuntimeError("FlexHook final outputs have unexpected dimensions")
                candidates = int(logits.shape[1])
                token_mask = expma.detach().cpu().float()
                if token_mask.ndim != 3 or tuple(token_mask.shape[:2]) != (batch_size, candidates):
                    raise RuntimeError("text mask shape does not match model candidates")
                if language_tokens.shape[0] != batch_size * candidates:
                    raise RuntimeError("text hook batch does not match model candidates")
                token_count = min(int(language_tokens.shape[1]), int(token_mask.shape[-1]))
                token_mask = token_mask[:, :, :token_count].reshape(batch_size * candidates, token_count, 1)
                language_tokens = language_tokens[:, :token_count]
                language = (language_tokens * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
                language = language.reshape(batch_size, candidates, language.shape[-1])
                expression_rows = _normalize_eval_expressions(expressions, batch_size)
                for sample in range(batch_size):
                    video = str(_metadata_item(videos, sample))
                    object_id = int(_metadata_item(objects, sample))
                    frame = int(_metadata_item(frame_ids, sample))
                    bbox = tracker_boxes.get((video, frame, object_id))
                    if bbox is None:
                        raise KeyError(f"tracker box missing for {video}/{frame}/{object_id}")
                    for candidate, expression in enumerate(expression_rows[sample]):
                        if candidate >= candidates:
                            break
                        key = (video, str(object_id), expression)
                        rows[key][frame] = (
                            features[sample, candidate].float(),
                            language[sample, candidate].float(),
                            logits[sample, candidate].float(),
                            bbox,
                        )
                if batch_index % int(config.PRINT_FREQ) == 0:
                    logger.info("Temporal inference batch %s/%s", batch_index, len(data_loader))
    finally:
        feature_handle.remove()
        language_handle.remove()

    cls_dict: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    adapter.eval()
    for key, by_frame in sorted(rows.items()):
        ordered = sorted(by_frame.items())
        frames = torch.tensor([item[0] for item in ordered], dtype=torch.long, device=device)
        features = torch.stack([item[1][0] for item in ordered]).to(device).unsqueeze(1)
        language = torch.stack([item[1][1] for item in ordered]).to(device).unsqueeze(1)
        logits = torch.stack([item[1][2] for item in ordered]).to(device).unsqueeze(1)
        bboxes = torch.stack([item[1][3] for item in ordered]).to(device).unsqueeze(1)
        with torch.no_grad():
            adapted = run_causal_sequence(
                adapter,
                current=features,
                bboxes=bboxes,
                track_ids=torch.tensor([_track_id(key[1])], dtype=torch.long, device=device),
                language=language,
                namespace=key,
                frame_indices=frames,
                base_logits=logits,
                dynamic_window=int(getattr(adapter, "dynamic_window", 4)),
            )
        for index, (frame, _row) in enumerate(ordered):
            cls_dict[key[0]][int(key[1])][int(frame)][key[2]] = adapted[index, 0].cpu().tolist()
    return cls_dict


def run_temporal_inference(
    *,
    config_path: Path,
    layout_root: Path,
    tracker_root: Path,
    checkpoint: Path,
    temporal_checkpoint: Path,
    output_root: Path,
    batch_size: int,
    workers: int,
    gpu: int,
    manifest: dict[str, Any],
    master_port: int | None = None,
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port or 29592)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    import torch.distributed as dist

    upstream_root = REPO_ROOT / "third_party" / "FlexHook"
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(upstream_root / "TrackEval"))
    from config import _C
    from data import build_loader
    from logger import create_logger
    from models import build_model
    from utils import load_ckpt_my
    import main as flexhook_main

    patch_legacy_tokenizer_api()
    if not torch.cuda.is_available():
        raise RuntimeError("temporal FlexHook evaluation requires CUDA")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", init_method="env://")
    temporary_config: Path | None = None
    try:
        cfg = _C.clone()
        cfg.defrost()
        merge_config = config_path
        if config_path.suffix.lower() not in {".yaml", ".yml", ".py"}:
            handle = tempfile.NamedTemporaryFile(prefix="flexhook_temporal_config_", suffix=".yaml", delete=False)
            handle.close()
            temporary_config = Path(handle.name)
            temporary_config.write_bytes(config_path.read_bytes())
            merge_config = temporary_config
        cfg.merge_from_file(str(merge_config))
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
        cfg.track_root = str(resolve_tracker_input_root(tracker_root, manifest["sequence_names"]))
        cfg.freeze()
        logger = create_logger(str(output_root), dist_rank=0, name="flexhook_temporal_eval")
        _, dataset_val, _, data_loader_val, _ = build_loader(cfg)
        with _working_directory(Path(str(cfg.ROBERTA_PATH)).expanduser().parent.parent):
            model = build_model(cfg).cuda()
        load_ckpt_my(cfg, model, logger)
        model.eval()

        payload = torch.load(temporal_checkpoint, map_location="cpu", weights_only=False)
        temporal_config = payload.get("config", {})
        adapter = TemporalEvidenceAdapter(
            feature_dim=int(temporal_config["feature_dim"]),
            language_dim=int(temporal_config["language_dim"]),
            hidden_dim=temporal_config.get("hidden_dim"),
            variant=str(temporal_config.get("variant", "H8")),
        ).cuda()
        adapter.load_state_dict(payload["state_dict"])
        adapter.eval()
        videos = list(manifest["sequence_names"])
        boxes = _tracker_boxes(tracker_root, videos)
        start = time.time()
        cls_dict = collect_temporal_outputs(
            config=cfg,
            data_loader=data_loader_val,
            model=model,
            adapter=adapter,
            tracker_boxes=boxes,
            device=torch.device("cuda"),
            logger=logger,
        )
        elapsed = time.time() - start
        written = write_predictions(
            cls_dict=cls_dict,
            manifest=manifest,
            tracker_root=tracker_root,
            output_root=output_root,
            save_scores=True,
        )
        manifest["method"] = "FlexHook_temporal"
        manifest["temporal_checkpoint"] = str(temporal_checkpoint)
        manifest["temporal_checkpoint_sha256"] = sha256_file(temporal_checkpoint)
        manifest["dataset_count"] = len(dataset_val)
        return {"dataset_count": len(dataset_val), "prediction_rows": written, "inference_seconds": elapsed}
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="frozen FlexHook task checkpoint")
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-list", default=str(DEFAULT_SEQUENCE_LIST))
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layout_root = Path(args.layout_root).expanduser().resolve()
    tracker_root = Path(args.tracker_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    temporal_checkpoint = Path(args.temporal_checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    manifest = prepare_eval_tree(
        layout_root=layout_root,
        tracker_root=tracker_root,
        output_root=output_root,
        sequence_names=read_sequence_list(args.sequence_list),
        overwrite=args.overwrite,
    )
    manifest["method"] = "FlexHook_temporal"
    manifest["temporal_checkpoint"] = str(temporal_checkpoint)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2, sort_keys=True) if args.json else f"temporal_eval: {manifest['status']}")
        return 0
    if not checkpoint.is_file() or not temporal_checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError("FlexHook or temporal checkpoint/config is missing")
    result = run_temporal_inference(
        config_path=config_path,
        layout_root=layout_root,
        tracker_root=tracker_root,
        checkpoint=checkpoint,
        temporal_checkpoint=temporal_checkpoint,
        output_root=output_root,
        batch_size=args.batch_size,
        workers=args.workers,
        gpu=args.gpu,
        manifest=manifest,
        master_port=args.master_port,
    )
    trackeval_result = run_trackeval(manifest=manifest, output_root=output_root)
    payload = {"run": result, "trackeval": trackeval_result, "manifest": str(output_root / "FLEXHOOK_EVAL_MANIFEST.json")}
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else f"temporal_eval: SUCCEEDED prediction_rows={result['prediction_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
