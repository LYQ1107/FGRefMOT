import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManifestInvariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (ROOT / "data" / "refer_sports_manifest.json").read_text(encoding="utf-8")
        )

    def test_verified_video_and_expression_counts(self):
        self.assertEqual(self.manifest["summary"]["video_count"], 41)
        self.assertEqual(self.manifest["summary"]["expression_count"], 323)
        self.assertEqual(
            self.manifest["summary"]["id_namespace"], "SPORTSMOT_LABELS_WITH_IDS_GLOBAL"
        )
        self.assertEqual(
            self.manifest["summary"]["video_count_by_split"], {"train": 22, "test": 19}
        )

    def test_required_fields_and_hashes_are_present(self):
        required = {
            "video_id",
            "original_sportsmot_id",
            "original_split",
            "num_frames",
            "image_width",
            "image_height",
            "expression_id",
            "sentence",
            "target_track_ids",
            "temporal_intervals",
            "annotation_status",
            "source_member_sha256",
        }
        for record in self.manifest["records"]:
            self.assertTrue(required.issubset(record))
            self.assertIn(record["original_split"], {"train", "test"})
            self.assertTrue(record["source_member_sha256"])


if __name__ == "__main__":
    unittest.main()
