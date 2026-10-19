import unittest

from scripts.flexhook_feature_archive import normalize_expression_batch, split_data_key


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


if __name__ == "__main__":
    unittest.main()
