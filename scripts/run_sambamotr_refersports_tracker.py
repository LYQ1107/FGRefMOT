#!/usr/bin/env python3
"""Generate a fixed SambaMOTR tracker set for the Refer-Sports videos.

The official SambaMOTR release contains outputs for the 150 official
SportsMOT test sequences.  Refer-Sports evaluates a separate 19-video subset
from the SportsMOT train tree, so those release files cannot be reused.  This
launcher runs the pinned SambaMOTR DAB-D-DETR checkpoint on exactly the
Refer-Sports sequence list, with a staging view containing images only.  The
staging view deliberately omits ``gt`` and label files.

The upstream checkout remains untouched by this project.  Its CUDA operator is
expected to be built in the NAS checkout after the documented PyTorch API
compatibility patch; the resulting operator is checked before a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - the dedicated env supplies PyYAML
    raise RuntimeError("PyYAML is required for the SambaMOTR launcher") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_LIST = REPO_ROOT / "configs" / "refer_sports_eval_sequences_20261019.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a manifest without exposing a partially written JSON file."""

    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_sequence_list(path: str | Path) -> list[str]:
    names = [line.strip() for line in Path(path).read_text().splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError("sequence list must contain unique non-empty names")
    return names


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def prepare_image_only_view(
    dataset_root: str | Path,
    sequence_names: Iterable[str],
    staging_root: str | Path,
) -> dict[str, Any]:
    """Create a small staging tree exposing only ``img1`` for each sequence."""

    source_train = Path(dataset_root).expanduser().resolve() / "train"
    stage = Path(staging_root).expanduser().resolve()
    split = stage / "SportsMOT" / "train"
    split.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for name in sequence_names:
        source = source_train / name
        source_images = source / "img1"
        if not source_images.is_dir():
            raise FileNotFoundError(f"SportsMOT image directory missing: {source_images}")
        destination = split / name
        _replace_path(destination)
        destination.mkdir(parents=True)
        (destination / "img1").symlink_to(source_images, target_is_directory=True)
        records.append({"sequence": name, "source_images": str(source_images)})
    return {
        "root": str(stage),
        "split": str(split),
        "sequences": records,
        "policy": "IMAGE_ONLY_STAGING; GT_AND_LABEL_FILES_OMITTED",
    }


def _source_patch_sha(source_root: Path) -> str | None:
    try:
        patch = subprocess.check_output(
            [
                "git",
                "-C",
                str(source_root),
                "diff",
                "--binary",
                "--",
                "models/ops/src/cuda/ms_deform_attn_cuda.cu",
                "train_engine.py",
                "eval_engine.py",
                "log/logger.py",
                "models/backbone.py",
            ],
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(patch).hexdigest() if patch else None


def build_runtime_config(
    template_path: Path,
    runtime_path: Path,
    *,
    output_root: Path,
    data_root: Path,
    model_link_name: str,
    physical_gpu: int,
) -> dict[str, Any]:
    config = yaml.safe_load(template_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"tracker template is not a mapping: {template_path}")
    config.update(
        {
            "AVAILABLE_GPUS": str(physical_gpu),
            "USE_DISTRIBUTED": False,
            "LAUNCHER": "none",
            "MODE": "submit",
            "DEVICE": "cuda",
            "DATA_ROOT": str(data_root),
            "DATASET": "SportsMOT",
            "SUBMIT_DIR": str(output_root),
            "SUBMIT_MODEL": model_link_name,
            "SUBMIT_DATA_SPLIT": "train",
            "EXP_NAME": "refersports_train",
            "OUTPUTS_DIR": str(output_root),
            "VISUALIZE": False,
        }
    )
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def _validate_tracker_file(path: Path) -> dict[str, Any]:
    rows = 0
    max_frame = 0
    track_ids: set[int] = set()
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 MOT fields")
            try:
                frame = int(fields[0])
                track_id = int(fields[1])
                numbers = [float(value) for value in fields[2:]]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric MOT field") from exc
            if frame < 1 or track_id < 0 or not all(math.isfinite(value) for value in numbers):
                raise ValueError(f"{path}:{line_number}: invalid MOT values")
            rows += 1
            max_frame = max(max_frame, frame)
            track_ids.add(track_id)
    if rows == 0:
        raise ValueError(f"empty tracker output: {path}")
    return {"path": str(path), "rows": rows, "max_frame": max_frame, "track_count": len(track_ids), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _check_operator(source_root: Path) -> Path:
    candidates = sorted((source_root / "models" / "ops").glob("MultiScaleDeformableAttention*.so"))
    if not candidates:
        raise FileNotFoundError("SambaMOTR CUDA operator is missing; build models/ops first")
    return candidates[-1]


def run_tracker(
    *,
    source_root: Path,
    model_root: Path,
    dataset_root: Path,
    sequence_list: Path,
    staging_root: Path,
    output_root: Path,
    physical_gpu: int,
    python_executable: str,
    run: bool,
) -> dict[str, Any]:
    names = read_sequence_list(sequence_list)
    stage_manifest = prepare_image_only_view(dataset_root, names, staging_root)
    source_commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    operator = _check_operator(source_root)
    checkpoint = model_root / "sambamotr_dab_sportsmot.pth"
    template = model_root / "source" / "sambamotr_dab_sportsmot.yaml"
    if not checkpoint.is_file() or not template.is_file():
        raise FileNotFoundError("SambaMOTR model/config asset is incomplete")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "TRACKER_RUN_MANIFEST.json"
    model_link = output_root / "model.pth"
    _replace_path(model_link)
    model_link.symlink_to(checkpoint)
    runtime_config = output_root / "train" / "config.yaml"
    config = build_runtime_config(
        template,
        runtime_config,
        output_root=output_root,
        data_root=stage_manifest["root"],
        model_link_name="model.pth",
        physical_gpu=physical_gpu,
    )
    command = [
        python_executable,
        str(source_root / "main.py"),
        "--config-path",
        str(runtime_config),
        "--mode",
        "submit",
        "--submit-dir",
        str(output_root),
        "--submit-model",
        "model.pth",
        "--submit-data-split",
        "train",
        "--exp-name",
        "refersports_train",
        "--data-root",
        str(stage_manifest["root"]),
        "--available-gpus",
        str(physical_gpu),
    ]
    result: dict[str, Any] = {
        "schema_version": "refer-sports-common-tracker-run-0.1",
        "dataset": "SportsMOT",
        "split": "refer-sports-eval-19-videos",
        "sequence_count": len(names),
        "sequence_names": names,
        "used_ground_truth": False,
        "generated_from_frames_without_ground_truth": True,
        "tracker_name": "SambaMOTR-DAB-D-DETR",
        "source_root": str(source_root),
        "source_commit": source_commit,
        "source_dirty": bool(_source_patch_sha(source_root)),
        "source_compatibility_patch_sha256": _source_patch_sha(source_root),
        "operator_path": str(operator),
        "operator_sha256": sha256_file(operator),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(template),
        "config_sha256": sha256_file(template),
        "runtime_config_path": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "physical_gpu": physical_gpu,
        "data_staging": stage_manifest,
        "command": command,
        "run_id": output_root.name,
        "launcher_pid": os.getpid(),
        "started_at": _timestamp(),
        "status": "PREPARED_NO_GPU_RUN" if not run else "RUNNING",
    }
    result["manifest_path"] = str(manifest_path)
    # Publish the RUNNING state before starting inference so a watcher can
    # distinguish this invocation from a stale terminal manifest.
    _atomic_write_json(manifest_path, result)
    if run:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        # The released tracker checkpoint contains the complete backbone.  Do
        # not let torchvision try an online ImageNet download while the model
        # is being constructed; the NAS job must be reproducible offline.
        env["SAMBAMOTR_OFFLINE_WEIGHTS"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(source_root / "models" / "ops"), str(source_root), env.get("PYTHONPATH", "")]
        )
        completed = subprocess.run(command, cwd=source_root, env=env, check=False)
        result["exit_code"] = completed.returncode
        if completed.returncode != 0:
            result["status"] = "FAILED"
            result["finished_at"] = _timestamp()
            _atomic_write_json(manifest_path, result)
            raise RuntimeError(f"SambaMOTR submit exited with {completed.returncode}")
        actual = output_root / "train" / "refersports_train" / "tracker"
        formal = output_root / "tracker_outputs"
        if formal.exists() or formal.is_symlink():
            _replace_path(formal)
        formal.symlink_to(actual, target_is_directory=True)
        result["output_root"] = str(formal)
        result["outputs"] = [_validate_tracker_file(formal / f"{name}.txt") for name in names]
        result["output_count"] = len(result["outputs"])
        result["status"] = "SUCCEEDED"
        result["finished_at"] = _timestamp()
    _atomic_write_json(manifest_path, result)
    result["manifest_sha256"] = sha256_file(manifest_path)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--dataset-root", required=True, help="SportsMOT dataset root containing train/")
    parser.add_argument("--sequence-list", default=str(DEFAULT_SEQUENCE_LIST))
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="launch GPU inference after preparing the view")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_tracker(
        source_root=Path(args.source_root).expanduser().resolve(),
        model_root=Path(args.model_root).expanduser().resolve(),
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        sequence_list=Path(args.sequence_list).expanduser().resolve(),
        staging_root=Path(args.staging_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        physical_gpu=args.gpu,
        python_executable=args.python,
        run=args.run,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"sambamotr_refersports_tracker: {result['status']}")
        print(f"sequences={result['sequence_count']} source_commit={result['source_commit']}")
        if result.get("output_root"):
            print(f"outputs={result['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
