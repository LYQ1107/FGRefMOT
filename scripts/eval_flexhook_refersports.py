#!/usr/bin/env python3
"""Evaluate a trained FlexHook checkpoint with the fixed Refer-Sports tracker.

The upstream FlexHook ``main.py`` evaluation path assumes a LaMOT checkout and
uses hard-coded dataset paths.  This adapter keeps the upstream model and data
loader unchanged, supplies its test-mode loader with the image-only tracker
outputs, and writes a small MOT/TrackEval tree for the Refer-Sports videos.
Ground-truth boxes are created only in that offline evaluation tree; they are
never passed to the model or used to construct tracker inputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_LIST = REPO_ROOT / "configs" / "refer_sports_eval_sequences_20261019.txt"


@contextmanager
def _working_directory(path: Path):
    """Temporarily provide the relative asset root expected by FlexHook."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sequence_list(path: str | Path) -> list[str]:
    names = [line.strip() for line in Path(path).read_text().splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError("sequence list must contain unique non-empty names")
    return names


def expression_dirname(sentence: str, used: set[str]) -> str:
    """Match the upstream expression-directory convention and avoid collisions."""

    base = str(sentence).strip().replace(" ", "-") or "expr"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}__{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _load_mot_rows(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, 10), dtype=np.float64)
    rows = np.loadtxt(path, delimiter=",")
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return np.empty((0, 10), dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] < 6:
        raise ValueError(f"tracker output has fewer than six MOT fields: {path}")
    return rows


def resolve_tracker_file(tracker_root: Path, video: str) -> Path:
    """Resolve a common-tracker file from either supported output layout.

    The tracker wrapper keeps its formal files in ``tracker_outputs/`` while
    older callers pass a directory whose ``{video}.txt`` files are at the
    root.  Evaluation must accept both without copying or rewriting tracks.
    """

    candidates = (
        tracker_root / f"{video}.txt",
        tracker_root / "tracker_outputs" / f"{video}.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"tracker output missing for {video}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_tracker_input_root(tracker_root: Path, sequence_names: list[str]) -> Path:
    """Return the directory expected by the upstream FlexHook loader."""

    if all((tracker_root / f"{video}.txt").is_file() for video in sequence_names):
        return tracker_root
    nested = tracker_root / "tracker_outputs"
    if all((nested / f"{video}.txt").is_file() for video in sequence_names):
        return nested
    # Keep the detailed per-video error from resolve_tracker_file for callers
    # that need to diagnose a partial tracker tree.
    if sequence_names:
        resolve_tracker_file(tracker_root, sequence_names[0])
    raise FileNotFoundError(f"no complete tracker input root under {tracker_root}")


def patch_legacy_tokenizer_api() -> bool:
    """Restore the removed Transformers tokenizer alias for FlexHook's loader.

    The vendored FlexHook dataset calls ``batch_encode_plus`` on fast
    tokenizers.  Recent Transformers releases removed that alias while keeping
    the equivalent ``__call__`` implementation.  Patching the base class here
    keeps the upstream checkout byte-for-byte unchanged and makes the runtime
    compatibility explicit in the evaluation adapter.
    """

    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        return False

    def batch_encode_plus(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The legacy FlexHook call supplies ``padding="max_length"`` but no
        # explicit truncation flag.  Newer Transformers leaves overlong
        # examples untruncated in that combination, which prevents tensor
        # collation when expressions have different lengths.
        if kwargs.get("padding") == "max_length":
            kwargs.setdefault("truncation", True)
        return self(*args, **kwargs)

    setattr(PreTrainedTokenizerBase, "batch_encode_plus", batch_encode_plus)
    return True


class _CudaTextBatchLoader:
    """Adapt the upstream inference loader's text tensors to the active GPU."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            values = list(batch)
            # FlexHook's inference path moves image/geometry tensors itself,
            # but historically forgot expid and expma.  Keep the batch shape
            # and metadata unchanged while fixing only those two fields.
            values[4] = values[4].cuda(non_blocking=True)
            values[5] = values[5].cuda(non_blocking=True)
            yield tuple(values)

    def __len__(self) -> int:
        return len(self.loader)


def _write_rows(path: Path, rows: list[list[float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


def prepare_eval_tree(
    *,
    layout_root: Path,
    tracker_root: Path,
    output_root: Path,
    sequence_names: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    """Create GT/evaluation metadata without reading GT during model inference."""

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            existing = output_root / "FLEXHOOK_EVAL_MANIFEST.json"
            if existing.is_file():
                return json.loads(existing.read_text())
            raise FileExistsError(f"evaluation output is non-empty: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels_path = layout_root / "sequences" / "val" / "SportsMOT_labels.json"
    resolution_path = layout_root / "sequences" / "val" / "SportsMOT_RESOLUTION.json"
    if not labels_path.is_file() or not resolution_path.is_file():
        raise FileNotFoundError("FlexHook SportsMOT validation metadata is incomplete")
    labels = json.loads(labels_path.read_text())
    resolutions = json.loads(resolution_path.read_text())
    annotations_root = layout_root / "annotations_v1" / "val" / "SportsMOT"
    if not annotations_root.is_dir():
        raise FileNotFoundError(f"expression directory missing: {annotations_root}")

    records: list[dict[str, Any]] = []
    seq_info: dict[str, int] = {}
    video_lengths: dict[str, int] = {}
    for video in sequence_names:
        if video not in labels:
            raise KeyError(f"sequence missing from FlexHook labels: {video}")
        tracker_path = resolve_tracker_file(tracker_root, video)
        tracker_rows = _load_mot_rows(tracker_path)
        if tracker_rows.shape[0] == 0:
            raise ValueError(f"tracker output is empty: {tracker_path}")
        image_dir = layout_root / "sequences" / "val" / "SportsMOT-sequences" / video / "img1"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image directory missing: {image_dir}")
        video_lengths[video] = len([p for p in image_dir.iterdir() if p.is_file()])
        expression_files = sorted(
            path for path in annotations_root.glob(f"{video}_*.json") if path.is_file()
        )
        if not expression_files:
            raise FileNotFoundError(f"no expressions found for sequence: {video}")
        used_dirs: set[str] = set()
        width, height = [float(value) for value in resolutions[video]]
        for expression_path in expression_files:
            expression = str(json.loads(expression_path.read_text())["language"])
            expr_dir = expression_dirname(expression, used_dirs)
            seq_name = f"{video}+{expr_dir}"
            seq_info[seq_name] = video_lengths[video]
            gt_rows: list[list[float | int]] = []
            for object_id, frame_map in labels[video].items():
                for frame_id, frame_info in frame_map.items():
                    if expression not in frame_info.get("expression_raw", []):
                        continue
                    x, y, w, h = [float(value) for value in frame_info["bbox"]]
                    gt_rows.append(
                        [
                            int(frame_id),
                            int(object_id),
                            x * width,
                            y * height,
                            w * width,
                            h * height,
                            1,
                            1,
                            1,
                            1,
                        ]
                    )
            gt_rows.sort(key=lambda row: (int(row[0]), int(row[1])))
            sequence_dir = output_root / video / expr_dir
            _write_rows(sequence_dir / "gt.txt", gt_rows)
            records.append(
                {
                    "video": video,
                    "expression": expression,
                    "expression_file": str(expression_path),
                    "expression_dir": expr_dir,
                    "sequence_name": seq_name,
                    "gt_path": str(sequence_dir / "gt.txt"),
                    "gt_rows": len(gt_rows),
                    "tracker_path": str(tracker_path),
                    "tracker_sha256": sha256_file(tracker_path),
                }
            )

    manifest = {
        "schema_version": "flexhook-refersports-eval-0.1",
        "status": "PREPARED_NO_MODEL_RUN",
        "layout_root": str(layout_root),
        "tracker_root": str(tracker_root),
        "output_root": str(output_root),
        "sequence_names": sequence_names,
        "sequence_count": len(sequence_names),
        "seq_info": seq_info,
        "records": records,
        "policy": {
            "gt_or_labels_as_tracker_input": "REJECTED",
            "model_input": "fixed_tracker_outputs_only",
            "gt_usage": "offline_evaluation_tree_only",
        },
    }
    manifest_path = output_root / "FLEXHOOK_EVAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_predictions(
    *,
    cls_dict: Any,
    manifest: dict[str, Any],
    tracker_root: Path,
    output_root: Path,
    save_scores: bool = False,
) -> int:
    records = manifest["records"]
    # Keep records as a list: one video can contain duplicate expression text
    # (for example two separately annotated "basketball players" entries).
    # A dict keyed only by the text silently overwrites one output directory,
    # leaving TrackEval with a missing prediction file.
    by_video: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_video.setdefault(record["video"], []).append(record)
    written = 0
    for video, video_records in by_video.items():
        record_tracker_path = Path(video_records[0]["tracker_path"])
        rows = _load_mot_rows(record_tracker_path)
        tracks = {(int(row[0]), int(row[1])): row for row in rows}
        video_scores = cls_dict.get(video, {})
        expression_names = {record["expression"] for record in video_records}
        prediction_rows: dict[str, list[list[float | int]]] = {
            expression: [] for expression in expression_names
        }
        score_rows: dict[str, list[list[float | int]]] = {
            expression: [] for expression in expression_names
        }
        for object_id, frame_map in video_scores.items():
            for frame_id, expression_scores in frame_map.items():
                try:
                    object_key = int(object_id)
                    frame_key = int(frame_id)
                except (TypeError, ValueError):
                    continue
                track_row = tracks.get((frame_key, object_key))
                if track_row is None:
                    continue
                for expression, score in expression_scores.items():
                    expression_name = str(expression)
                    if expression_name not in prediction_rows:
                        continue
                    values = np.asarray(score, dtype=np.float64).reshape(-1)
                    if values.size < 2:
                        raise ValueError("FlexHook score must contain at least two class logits")
                    positive_probability = float(
                        np.exp(values[1] - np.max(values))
                        / np.exp(values - np.max(values)).sum()
                    )
                    score_row = [
                        frame_key,
                        object_key,
                        float(track_row[2]),
                        float(track_row[3]),
                        float(track_row[4]),
                        float(track_row[5]),
                        positive_probability,
                        1,
                        1,
                        1,
                    ]
                    score_rows[expression_name].append(score_row)
                    if int(np.argmax(values)) > 0:
                        prediction_rows[expression_name].append(
                            [
                                frame_key,
                                object_key,
                                float(track_row[2]),
                                float(track_row[3]),
                                float(track_row[4]),
                                float(track_row[5]),
                                1,
                                1,
                                1,
                                1,
                            ]
                        )
        for record in video_records:
            expression = record["expression"]
            values = prediction_rows[expression]
            values.sort(key=lambda row: (int(row[0]), int(row[1])))
            _write_rows(output_root / video / record["expression_dir"] / "predict.txt", values)
            if save_scores:
                all_scores = score_rows[expression]
                all_scores.sort(key=lambda row: (int(row[0]), int(row[1])))
                _write_rows(
                    output_root / video / record["expression_dir"] / "predict_scores.txt",
                    all_scores,
                )
            record["prediction_rows"] = len(values)
            if save_scores:
                record["score_rows"] = len(score_rows[expression])
            written += len(values)
    manifest["status"] = "PREDICTIONS_WRITTEN"
    manifest["prediction_rows"] = written
    manifest_path = output_root / "FLEXHOOK_EVAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return written


def run_flexhook_inference(
    *,
    config_path: Path,
    layout_root: Path,
    tracker_root: Path,
    checkpoint: Path,
    output_root: Path,
    batch_size: int,
    workers: int,
    gpu: int,
    manifest: dict[str, Any],
    save_scores: bool = False,
) -> dict[str, Any]:
    """Run upstream FlexHook inference in test mode on the fixed tracks."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    import torch
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
        raise RuntimeError("FlexHook evaluation requires a CUDA device")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", init_method="env://")
    cfg = _C.clone()
    cfg.defrost()
    # The upstream launcher writes a YAML document to ``config.json``.  YACS
    # dispatches on the suffix before parsing, so use a temporary YAML-named
    # view while preserving the immutable training artifact and its hash.
    merge_config_path = config_path
    temporary_config: Path | None = None
    if config_path.suffix.lower() not in {".yaml", ".yml", ".py"}:
        fd, temporary_name = tempfile.mkstemp(prefix="flexhook_eval_config_", suffix=".yaml")
        os.close(fd)
        temporary_config = Path(temporary_name)
        shutil.copyfile(config_path, temporary_config)
        merge_config_path = temporary_config
    try:
        cfg.merge_from_file(str(merge_config_path))
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)
    cfg.DATA.DATASET = "mix"
    cfg.DATA.TRAIN = "SportsMOT"
    cfg.DATA.TEST = "SportsMOT"
    cfg.DATA.VAL_BATCH_SIZE = int(batch_size)
    cfg.DATA.NUM_WORKERS = int(workers)
    cfg.EVAL_MODE = True
    cfg.LOCAL_RANK = 0
    # The training config keeps an 8-frame sampling pool and reduces it to
    # four frames during training.  In test mode the upstream loader indexes
    # that pool before reduction, so short fixed tracks can go out of range.
    # Evaluation uses the model's four-frame context directly, which is valid
    # for every track retained by the loader and avoids test-time randomness.
    cfg.sample_frame_len = cfg.sample_frame_num
    cfg.MODEL.PRETRAINED = str(checkpoint)
    cfg.MODEL.RESUME = str(checkpoint)
    cfg.OUTPUT = str(output_root)
    cfg.data_root = str(layout_root)
    cfg.val_data_root = str(layout_root)
    cfg.track_root = str(resolve_tracker_input_root(tracker_root, manifest["sequence_names"]))
    cfg.freeze()
    logger = create_logger(str(output_root), dist_rank=0, name="flexhook_refersports_eval")
    flexhook_main.logger = logger
    logger.info("Building test-mode SportsMOT loader from fixed tracker outputs")
    _, dataset_val, _, data_loader_val, _ = build_loader(cfg)
    logger.info("Building FlexHook model and loading %s", checkpoint)
    # The upstream Swin constructor resolves ``pretrained/...`` relative to a
    # symlink-only runtime root prepared beside the project.  Derive that root
    # from the recorded RoBERTa path instead of relying on the caller's CWD.
    # Do not resolve the RoBERTa symlink before deriving the root: the
    # symlink-only runtime directory is the relative-asset namespace, while
    # the resolved target lives on NAS under a different directory layout.
    runtime_root = Path(str(cfg.ROBERTA_PATH)).expanduser().parent.parent
    if not (runtime_root / "pretrained" / "swin_rope_mixed_tiny_patch4_window7_224").is_dir():
        raise FileNotFoundError(
            "FlexHook runtime is missing the Swin asset symlink: "
            f"{runtime_root / 'pretrained' / 'swin_rope_mixed_tiny_patch4_window7_224'}"
        )
    with _working_directory(runtime_root):
        model = build_model(cfg).cuda()
    load_ckpt_my(cfg, model, logger)
    model.eval()
    start = time.time()
    with torch.no_grad():
        cls_dict = flexhook_main.inference(cfg, _CudaTextBatchLoader(data_loader_val), model)
    elapsed = time.time() - start
    written = write_predictions(
        cls_dict=cls_dict,
        manifest=manifest,
        tracker_root=tracker_root,
        output_root=output_root,
        save_scores=save_scores,
    )
    result = {
        "dataset_count": len(dataset_val),
        "inference_seconds": elapsed,
        "prediction_rows": written,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "save_scores": bool(save_scores),
    }
    dist.destroy_process_group()
    return result


def run_trackeval(*, manifest: dict[str, Any], output_root: Path) -> dict[str, Any]:
    upstream_root = REPO_ROOT / "third_party" / "FlexHook"
    sys.path.insert(0, str(upstream_root / "TrackEval"))
    import trackeval

    from trackeval import Evaluator
    from trackeval import metrics

    metrics_dir = output_root / "trackeval"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tracker_path = str(output_root)
    eval_config = Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_FOLDER": str(metrics_dir),
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "BREAK_ON_ERROR": True,
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(output_root),
            "TRACKERS_FOLDER": str(output_root),
            "TRACKERS_TO_EVAL": [tracker_path],
            "TRACKER_DISPLAY_NAMES": ["FlexHook"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
            "SEQ_INFO": manifest["seq_info"],
            "GT_LOC_FORMAT": "{gt_folder}/{video_id}/{expression_id}/gt.txt",
            "OUTPUT_FOLDER": str(metrics_dir),
            "OUTPUT_SUB_FOLDER": "",
        }
    )
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metric_list = [metrics.HOTA({"THRESHOLD": 0.5}), metrics.CLEAR({"THRESHOLD": 0.5}), metrics.Identity({"THRESHOLD": 0.5})]
    results, messages = Evaluator(eval_config).evaluate([dataset], metric_list)
    manifest["trackeval_messages"] = _jsonable(messages)
    manifest["trackeval_results"] = _jsonable(results)
    manifest["status"] = "SUCCEEDED"
    manifest_path = output_root / "FLEXHOOK_EVAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return {"results": _jsonable(results), "messages": _jsonable(messages)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="completed FlexHook training config.json")
    parser.add_argument("--layout-root", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-list", default=str(DEFAULT_SEQUENCE_LIST))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Also write per-candidate positive probabilities to predict_scores.txt.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layout_root = Path(args.layout_root).expanduser().resolve()
    tracker_root = Path(args.tracker_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    sequence_names = read_sequence_list(args.sequence_list)
    manifest = prepare_eval_tree(
        layout_root=layout_root,
        tracker_root=tracker_root,
        output_root=output_root,
        sequence_names=sequence_names,
        overwrite=args.overwrite,
    )
    if args.prepare_only:
        if args.json:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            print(f"flexhook_refersports_eval: {manifest['status']}")
            print(f"sequences={manifest['sequence_count']} expressions={len(manifest['records'])}")
        return 0

    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError("FlexHook checkpoint/config is missing")
    run_result = run_flexhook_inference(
        config_path=config_path,
        layout_root=layout_root,
        tracker_root=tracker_root,
        checkpoint=checkpoint,
        output_root=output_root,
        batch_size=args.batch_size,
        workers=args.workers,
        gpu=args.gpu,
        manifest=manifest,
        save_scores=args.save_scores,
    )
    trackeval_result = run_trackeval(manifest=manifest, output_root=output_root)
    payload = {"run": run_result, "trackeval": trackeval_result, "manifest": str(output_root / "FLEXHOOK_EVAL_MANIFEST.json")}
    if args.json:
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    else:
        print("flexhook_refersports_eval: SUCCEEDED")
        print(f"prediction_rows={run_result['prediction_rows']} inference_seconds={run_result['inference_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
