#!/usr/bin/env python3
"""Run and evaluate YesTrack on Refer-Sports with a fixed physical tracker.

The upstream evaluator already supports a ByteTrack-style ``seq.txt`` input,
but its bundled TrackEval setup assumes Refer-KITTI sequence metadata.  This
adapter keeps the YesTrack inference code unchanged, then evaluates its nested
per-expression outputs with a small TrackEval tree whose sequence lengths are
derived from the Refer-Sports image layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_LIST = REPO_ROOT / "configs" / "refer_sports_eval_sequences_20261019.txt"


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


def _dirname_from_sentence(sentence: str) -> str:
    import re

    value = str(sentence or "").strip().lower()
    value = re.sub(r"[^a-z0-9,]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "expr"


def _load_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"YesTrack output missing: {path}")
    rows: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            raise ValueError(f"YesTrack output has fewer than seven fields: {path}")
        float(fields[6])
        rows.append(fields)
    return rows


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


def run_yestrack_inference(
    *,
    python: Path,
    model_name: Path,
    lora_path: Path,
    data_root: Path,
    tracker_root: Path,
    results_root: Path,
    gpu: int,
    re_refer_lower: float,
    re_refer_thresh: float,
    infer_every_n_frames: int,
    frame_offset: int,
) -> dict[str, Any]:
    results_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(REPO_ROOT / "third_party" / "YesTrack" / "refer_llm" / "llm_eval_from_mot.py"),
        "--model_name",
        str(model_name),
        "--lora_path",
        str(lora_path),
        "--data_root",
        str(data_root),
        "--mot_input_type",
        "bytetrack",
        "--bytetrack_dir",
        str(tracker_root),
        "--image_frame_start_one",
        "0",
        "--mot_frame_start_one",
        "1",
        "--frame_offset",
        str(frame_offset),
        "--re_refer_lower",
        str(re_refer_lower),
        "--re_refer_thresh",
        str(re_refer_thresh),
        "--infer_every_n_frames",
        str(infer_every_n_frames),
        "--preprocess_workers",
        "4",
        "--output_root",
        str(results_root),
        "--resume",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "third_party" / "YesTrack"), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(command, cwd=REPO_ROOT / "third_party" / "YesTrack", env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"YesTrack evaluator exited with {completed.returncode}")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "results_root": str(results_root),
    }


def evaluate_threshold(
    *,
    gt_manifest_root: Path,
    data_root: Path,
    results_root: Path,
    output_root: Path,
    sequence_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    manifest_path = gt_manifest_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Refer-Sports GT manifest missing: {manifest_path}")
    gt_manifest = json.loads(manifest_path.read_text())
    wanted = set(sequence_names)
    records = [record for record in gt_manifest["metadata"] if record["video_id"] in wanted]
    if len({record["video_id"] for record in records}) != len(wanted):
        missing = sorted(wanted - {record["video_id"] for record in records})
        raise ValueError(f"GT manifest does not cover requested videos: {missing}")
    tag = f"t{threshold:.2f}".replace(".", "p")
    eval_root = output_root / "trackeval" / tag
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    tracker_data = eval_root / "tracker" / "data"
    tracker_data.mkdir(parents=True, exist_ok=True)
    seq_info: dict[str, int] = {}
    prediction_files = 0
    prediction_rows = 0
    by_video: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    for video, video_records in by_video.items():
        image_dir = data_root / "KITTI" / "training" / "image_02" / video
        if not image_dir.is_dir():
            raise FileNotFoundError(f"YesTrack image directory missing: {image_dir}")
        frame_count = len([path for path in image_dir.iterdir() if path.is_file()])
        for record in video_records:
            expression_dir = _dirname_from_sentence(record["sentence"])
            sequence = record["name"]
            seq_info[sequence] = frame_count
            gt_source = gt_manifest_root / "gt" / sequence / "gt.txt"
            if not gt_source.is_file():
                raise FileNotFoundError(f"GT file missing: {gt_source}")
            gt_target = eval_root / sequence / "gt.txt"
            gt_target.parent.mkdir(parents=True, exist_ok=True)
            gt_target.symlink_to(gt_source.resolve())
            prediction_source = results_root / video / expression_dir / "predict_with_conf.txt"
            rows = _load_rows(prediction_source)
            prediction_target = tracker_data / f"{sequence}.txt"
            kept: list[str] = []
            for fields in rows:
                if float(fields[6]) >= float(threshold):
                    kept.append(",".join(fields))
            prediction_target.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            prediction_files += 1
            prediction_rows += len(kept)

    upstream_trackeval = REPO_ROOT / "third_party" / "YesTrack" / "TrackEval"
    sys.path.insert(0, str(upstream_trackeval))
    import trackeval
    from trackeval import Evaluator, metrics

    eval_config = Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_FOLDER": str(eval_root / "metrics"),
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "BREAK_ON_ERROR": True,
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(eval_root),
            "TRACKERS_FOLDER": str(eval_root),
            "TRACKERS_TO_EVAL": ["tracker"],
            "TRACKER_DISPLAY_NAMES": ["YesTrack"],
            "TRACKER_SUB_FOLDER": "data",
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "test",
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
            "SEQ_INFO": seq_info,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt.txt",
            "OUTPUT_FOLDER": str(eval_root / "metrics"),
            "OUTPUT_SUB_FOLDER": "",
        }
    )
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metric_list = [metrics.HOTA({"THRESHOLD": 0.5}), metrics.CLEAR({"THRESHOLD": 0.5}), metrics.Identity({"THRESHOLD": 0.5})]
    results, messages = Evaluator(eval_config).evaluate([dataset], metric_list)
    return {
        "threshold": threshold,
        "tag": tag,
        "sequence_count": len(seq_info),
        "expression_count": len(records),
        "prediction_files": prediction_files,
        "prediction_rows": prediction_rows,
        "results": _jsonable(results),
        "messages": _jsonable(messages),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--data-root", required=True, help="YesTrack val layout root")
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gt-manifest-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-list", default=str(DEFAULT_SEQUENCE_LIST))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--frame-offset", type=int, default=-1)
    parser.add_argument("--infer-every-n-frames", type=int, default=1)
    parser.add_argument("--re-refer-lower", type=float, default=0.2)
    parser.add_argument("--re-refer-thresh", type=float, default=0.8)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sequence_names = read_sequence_list(args.sequence_list)
    output_root = Path(args.output_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "yestrack-refersports-eval-0.1",
        "status": "RUNNING" if args.run_inference else "EVALUATING_EXISTING_RESULTS",
        "sequence_names": sequence_names,
        "sequence_count": len(sequence_names),
        "thresholds": args.thresholds,
        "tracker_root": str(Path(args.tracker_root).expanduser().resolve()),
        "results_root": str(results_root),
        "gt_manifest_root": str(Path(args.gt_manifest_root).expanduser().resolve()),
        "policy": {
            "gt_or_labels_as_tracker_input": "REJECTED",
            "model_input": "fixed_common_tracker_outputs_only",
            "gt_usage": "offline_metric_evaluation_only",
        },
    }
    if args.run_inference:
        payload["inference"] = run_yestrack_inference(
            python=Path(args.python).expanduser().resolve(),
            model_name=Path(args.model_name).expanduser().resolve(),
            lora_path=Path(args.lora_path).expanduser().resolve(),
            data_root=Path(args.data_root).expanduser().resolve(),
            tracker_root=Path(args.tracker_root).expanduser().resolve(),
            results_root=results_root,
            gpu=args.gpu,
            re_refer_lower=args.re_refer_lower,
            re_refer_thresh=args.re_refer_thresh,
            infer_every_n_frames=args.infer_every_n_frames,
            frame_offset=args.frame_offset,
        )
    evaluations = []
    for threshold in args.thresholds:
        evaluations.append(
            evaluate_threshold(
                gt_manifest_root=Path(args.gt_manifest_root).expanduser().resolve(),
                data_root=Path(args.data_root).expanduser().resolve(),
                results_root=results_root,
                output_root=output_root,
                sequence_names=sequence_names,
                threshold=float(threshold),
            )
        )
    payload["evaluations"] = evaluations
    payload["status"] = "SUCCEEDED"
    manifest_path = output_root / "YESTRACK_EVAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    if args.json:
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    else:
        print(f"yestrack_refersports_eval: {payload['status']}")
        print(f"sequences={payload['sequence_count']} thresholds={len(evaluations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
