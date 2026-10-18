import tempfile
import unittest
from pathlib import Path

from scripts.yestrack_sportsmot_adapter import (
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


if __name__ == "__main__":
    unittest.main()
