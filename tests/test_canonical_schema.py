import unittest

from adapters.refer_sports.canonical_schema import contiguous_intervals, target_intervals


class CanonicalSchemaTest(unittest.TestCase):
    def test_contiguous_intervals(self):
        self.assertEqual(contiguous_intervals([4, 1, 2, 7, 8]), [(1, 2), (4, 4), (7, 8)])

    def test_target_intervals_preserve_track_identity(self):
        rows = target_intervals({"1": [2, 3], "2": [2], "4": [3]})
        self.assertEqual(
            [(row.track_id, row.start_frame, row.end_frame) for row in rows],
            [(2, 1, 2), (3, 1, 1), (3, 4, 4)],
        )


if __name__ == "__main__":
    unittest.main()
