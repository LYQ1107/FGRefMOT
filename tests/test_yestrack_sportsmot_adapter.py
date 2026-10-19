import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.yestrack_sportsmot_adapter import (
    _ExplicitFrameLabelMixin,
    build_upstream_argv,
    resolve_layout_roots,
    sequence_names,
)


class YesTrackSportsMOTAdapterTest(unittest.TestCase):
    def _make_split(self, root: Path, split: str, names: list[str]) -> Path:
        split_root = root / split
        image_root = split_root / "KITTI" / "training" / "image_02"
        expression_root = split_root / "expression"
        for name in names:
            (image_root / name).mkdir(parents=True)
            (expression_root / name).mkdir(parents=True)
        return split_root

    def test_sequence_names_preserve_non_numeric_sports_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = self._make_split(root, "train", ["v_z_c002", "v_a_c001"])
            self._make_split(root, "val", ["v_b_c003"])
            self.assertEqual(sequence_names(train), ["v_a_c001", "v_z_c002"])
            resolved_train, resolved_val = resolve_layout_roots(root)
            self.assertEqual(resolved_train, train.resolve())
            self.assertEqual(resolved_val, (root / "val").resolve())

    def test_adapter_owns_dataset_split_arguments(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = self._make_split(root, "train", ["v_a_c001"])
            val = self._make_split(root, "val", ["v_b_c003"])
            with self.assertRaises(ValueError):
                build_upstream_argv(["--data_root", "/tmp/other"], train)
            forwarded = build_upstream_argv(["--batch_size", "1"], train)
            self.assertEqual(forwarded[-6:], ["--data_root", str(train), "--dataset_version", "v1", "--eval_sequence", "all"])

    def test_missing_expression_frames_are_skipped_but_explicit_negatives_are_kept(self):
        class BaseDataset:
            def __init__(self, refer_dataset, *args, negative_downsample=0.0, **kwargs):
                self.ds = refer_dataset
                self.samples = []
                for sequence in refer_dataset.sequence_names:
                    for frame_idx, annotation in enumerate(refer_dataset.annotations[sequence]):
                        for sentence_idx, expression in enumerate(refer_dataset.expressions[sequence]):
                            ids = set(expression["label"].get(str(frame_idx), []))
                            for ann_idx, obj_id in enumerate(annotation["id"].tolist()):
                                if obj_id in ids:
                                    self.samples.append((sequence, frame_idx, ann_idx, sentence_idx))

        class SportsDataset(_ExplicitFrameLabelMixin, BaseDataset):
            pass

        annotation = {"id": torch.tensor([1, 2]), "bbox": torch.zeros((2, 4))}
        dataset = SimpleNamespace(
            sequence_names=["video"],
            annotations={"video": [annotation, annotation, annotation]},
            expressions={"video": [{"label": {"0": [1], "1": []}}]},
        )
        dataset._load_expressions_for_sequence = lambda sequence: dataset.expressions[sequence]

        samples = SportsDataset(dataset, negative_downsample=1.0)

        self.assertEqual(
            set(samples.samples),
            {
                ("video", 0, 0, 0),
                ("video", 0, 1, 0),
                ("video", 1, 0, 0),
                ("video", 1, 1, 0),
            },
        )
        self.assertEqual(samples.explicit_negative_samples_added, 3)
        self.assertEqual(samples.unknown_samples_skipped, 2)
        self.assertEqual(samples.sports_label_policy, "EXPLICIT_FRAME_KEYS_ONLY; MISSING_KEYS_UNKNOWN")


if __name__ == "__main__":
    unittest.main()
