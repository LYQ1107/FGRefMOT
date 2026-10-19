import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.train_flexhook_temporal import build_sequences, train_temporal_head


def _make_archive(root: Path, split: str, frames: list[int]) -> None:
    shard = root / "shards" / "features_00000.pt"
    shard.parent.mkdir(parents=True)
    count = len(frames)
    payload = {
        "schema_version": "flexhook-feature-shard-0.1",
        "features": torch.randn(count, 1, 4),
        "language": torch.randn(count, 1, 3),
        "logits": torch.randn(count, 1, 2),
        "labels": torch.tensor([[index % 2] for index in range(count)]),
        "bboxes": torch.randn(count, 4),
        "clip_frame_ids": [[frame] for frame in frames],
        "video_ids": ["video_a"] * count,
        "track_ids": ["17"] * count,
        "data_keys": [f"video_a#17#{frame}#expr" for frame in frames],
        "expressions": [["expr"]] * count,
    }
    torch.save(payload, shard)
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    (root / "FEATURE_ARCHIVE_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "split": split,
                "policy": "NO_GT_AS_MODEL_OR_TRACKER_INPUT",
                "shards": [{"path": str(shard), "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )


def test_build_sequences_deduplicates_overlapping_terminal_frames() -> None:
    payload = {
        "features": torch.zeros(3, 1, 4),
        "language": torch.zeros(3, 1, 3),
        "logits": torch.zeros(3, 1, 2),
        "labels": torch.zeros(3, 1, dtype=torch.long),
        "bboxes": torch.zeros(3, 4),
        "clip_frame_ids": [[2], [1], [2]],
        "video_ids": ["v", "v", "v"],
        "track_ids": ["1", "1", "1"],
        "expressions": [["e"], ["e"], ["e"]],
    }
    sequences = build_sequences({}, [payload])
    assert len(sequences) == 1
    assert sequences[0].frames.tolist() == [1, 2]


def test_tiny_temporal_training_writes_selection_manifest() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        train_root = root / "train"
        val_root = root / "val"
        _make_archive(train_root, "train_core", [1, 2, 3, 4])
        _make_archive(val_root, "internal_val", [1, 2, 3])
        result = train_temporal_head(
            train_archive=train_root,
            val_archive=val_root,
            output_root=root / "out",
            variant="H8",
            epochs=1,
            learning_rate=1e-3,
            hidden_dim=4,
            dynamic_window=2,
            seed=42,
            device="cpu",
            max_sequences=None,
            overwrite=False,
        )
        assert result["status"] == "SUCCEEDED"
        assert (root / "out/temporal_head_best.pt").is_file()
        assert (root / "out/TEMPORAL_TRAINING_MANIFEST.json").is_file()
