from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_split_builder():
    path = PROJECT_ROOT / "scripts" / "build_internal_sports_split.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_internal_split_is_video_disjoint_and_reproducible(tmp_path: Path) -> None:
    module = _load_split_builder()
    manifest = tmp_path / "manifest.json"
    records = {
        "records": [
            {
                "video_id": "train-a",
                "original_split": "train",
                "sport_type": "soccer",
                "num_frames": 3,
                "expression_id": "train-a:expr-0",
            },
            {
                "video_id": "train-b",
                "original_split": "train",
                "sport_type": "basketball",
                "num_frames": 2,
                "expression_id": "train-b:expr-0",
            },
            {
                "video_id": "held-out",
                "original_split": "train",
                "sport_type": "volleyball",
                "num_frames": 4,
                "expression_id": "held-out:expr-0",
            },
            {
                "video_id": "historical-test",
                "original_split": "test",
                "sport_type": "soccer",
                "num_frames": 9,
                "expression_id": "historical-test:expr-0",
            },
        ]
    }
    manifest.write_text(json.dumps(records), encoding="utf-8")

    first = module.build_split(
        manifest_path=manifest,
        output_root=tmp_path / "split-a",
        holdout_videos=("held-out",),
    )
    second = module.build_split(
        manifest_path=manifest,
        output_root=tmp_path / "split-b",
        holdout_videos=("held-out",),
    )

    assert first["sports_train_core_videos"] == ["train-a", "train-b"]
    assert first["sports_val_videos"] == ["held-out"]
    assert set(first["sports_train_core_videos"]).isdisjoint(first["sports_val_videos"])
    assert first["frame_count"] == {"train_core": 5, "val": 4}
    assert first["expression_count"] == {"train_core": 2, "val": 1}
    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]
    assert first["lists"]["sports_train_core_images_sha256"] == second["lists"]["sports_train_core_images_sha256"]
    assert (tmp_path / "split-a" / "sports_train_core_images.txt").read_text().splitlines() == [
        "SportsMOT/train/image_02/train-a/000001.jpg",
        "SportsMOT/train/image_02/train-a/000002.jpg",
        "SportsMOT/train/image_02/train-a/000003.jpg",
        "SportsMOT/train/image_02/train-b/000001.jpg",
        "SportsMOT/train/image_02/train-b/000002.jpg",
    ]
