import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.preflight_baseline_assets import (
    INVALID,
    READY,
    check_directory,
    check_tracker_root,
)


class BaselineAssetPreflightTest(unittest.TestCase):
    def test_pretrained_directory_requires_finite_file_set(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
                (root / name).write_bytes(name.encode("ascii"))
            result = check_directory("roberta_base", root, ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.json"), hash_files=True)
            self.assertEqual(result["status"], READY)
            self.assertEqual(len(result["files"]), 5)

    def test_tracker_manifest_rejects_ground_truth(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "tracker_manifest.json").write_text(
                json.dumps(
                    {
                        "asset_role": "physical_tracker_outputs",
                        "dataset": "SportsMOT",
                        "input_type": "detector_outputs",
                        "used_ground_truth": True,
                    }
                ),
                encoding="utf-8",
            )
            result = check_tracker_root("common_sports_tracker", root)
            self.assertEqual(result["status"], INVALID)
            self.assertTrue(any("used_ground_truth" in reason for reason in result["reasons"]))

    def test_tracker_manifest_checks_output_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            detector = root / "detector.pth"
            tracker_config = root / "tracker.yaml"
            output = root / "video_001.txt"
            detector.write_bytes(b"detector")
            tracker_config.write_text("tracker: bytetrack\n", encoding="utf-8")
            output.write_text("1,1,2,3,4,0.9\n", encoding="utf-8")
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            manifest = {
                "asset_role": "physical_tracker_outputs",
                "dataset": "SportsMOT",
                "split": "test",
                "input_type": "detector_outputs",
                "used_ground_truth": False,
                "detector_checkpoint_path": detector.name,
                "tracker_config_path": tracker_config.name,
                "detector_checkpoint_sha256": hashlib.sha256(detector.read_bytes()).hexdigest(),
                "tracker_config_sha256": hashlib.sha256(tracker_config.read_bytes()).hexdigest(),
                "source_manifest_sha256": "a" * 64,
                "outputs": [{"path": output.name, "sha256": digest}],
            }
            (root / "tracker_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = check_tracker_root("common_sports_tracker", root, hash_files=True)
            self.assertEqual(result["status"], READY)
            self.assertEqual(result["manifest"]["output_count"], 1)


if __name__ == "__main__":
    unittest.main()
