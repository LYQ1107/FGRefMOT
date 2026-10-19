import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.flexhook_feature_archive import _write_shard, normalize_expression_batch, split_data_key


class FlexHookFeatureArchiveTest(unittest.TestCase):
    def test_normalizes_transposed_expression_collation(self):
        value = [("a0", "a1"), ("b0", "b1"), ("c0", "c1")]
        self.assertEqual(
            normalize_expression_batch(value, 2),
            [["a0", "b0", "c0"], ["a1", "b1", "c1"]],
        )

    def test_keeps_row_major_expression_collation(self):
        value = [["a0", "b0"], ["a1", "b1"]]
        self.assertEqual(normalize_expression_batch(value, 2), value)

    def test_rejects_malformed_expression_collation(self):
        with self.assertRaises(ValueError):
            normalize_expression_batch(["a", "b", "c"], 2)

    def test_splits_video_and_track_without_touching_window_suffix(self):
        self.assertEqual(split_data_key("video_a#17#4#expr1+expr2"), ("video_a", "17"))

    def test_shard_stores_visual_and_language_features(self):
        with TemporaryDirectory() as temporary:
            record = _write_shard(
                Path(temporary),
                0,
                [
                    {
                        "features": torch.zeros(1, 8, 3),
                        "language": torch.ones(1, 8, 3),
                        "logits": torch.zeros(1, 8, 2),
                        "labels": torch.ones(1, 8, dtype=torch.long),
                        "bboxes": torch.zeros(1, 4),
                        "clip_frame_ids": [[1, 2, 3, 4]],
                        "video_ids": ["video_a"],
                        "track_ids": ["17"],
                        "data_keys": ["video_a#17#4#expr"],
                        "expressions": [["expr"]],
                    }
                ],
            )
            payload = torch.load(Path(temporary) / "shards/features_00000.pt", weights_only=False)
            self.assertEqual(record["language_shape"], [1, 8, 3])
            self.assertTrue(torch.equal(payload["language"], torch.ones(1, 8, 3)))


if __name__ == "__main__":
    unittest.main()
