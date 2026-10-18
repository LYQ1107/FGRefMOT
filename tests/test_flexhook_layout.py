import json
import tempfile
import zipfile
import unittest
from pathlib import Path

from scripts.build_flexhook_sports_layout import main


class FlexHookLayoutTest(unittest.TestCase):
    def test_layout_is_symlink_only_and_preserves_global_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = root / "dataset"
            sequence = dataset / "train" / "video_1"
            (sequence / "img1").mkdir(parents=True)
            (sequence / "seqinfo.ini").write_text("[Sequence]\nimWidth=100\nimHeight=80\n", encoding="utf-8")
            labels_root = dataset / "labels_with_ids" / "video_1"
            labels_root.mkdir(parents=True)
            (labels_root / "000001.txt").write_text("0 42 0.25 0.5 0.3 0.5\n", encoding="utf-8")
            (labels_root / "000002.txt").write_text("0 42 0.25 0.5 0.3 0.5\n", encoding="utf-8")
            (labels_root / "000010.txt").write_text("0 42 0.25 0.5 0.3 0.5\n", encoding="utf-8")
            train_list = root / "train.txt"
            train_list.write_text("a/b/c/video_1/000001.jpg\n", encoding="utf-8")
            test_list = root / "test.txt"
            test_list.write_text("", encoding="utf-8")
            archive_path = root / "expressions.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("expression/video_1/players.json", json.dumps({"sentence": "player", "label": {"1": [42]}}))
            output = root / "layout"
            old_argv = __import__("sys").argv
            try:
                __import__("sys").argv = [
                    "build_flexhook_sports_layout.py",
                    "--expression-zip", str(archive_path),
                    "--train-list", str(train_list),
                    "--test-list", str(test_list),
                    "--dataset-root", str(dataset),
                    "--labels-root", str(dataset / "labels_with_ids"),
                    "--output", str(output),
                ]
                self.assertEqual(main(), 0)
            finally:
                __import__("sys").argv = old_argv
            self.assertTrue((output / "sequences/train/SportsMOT-sequences/video_1").is_symlink())
            labels = json.loads((output / "sequences/train/SportsMOT_labels.json").read_text(encoding="utf-8"))
            self.assertEqual(labels["video_1"]["42"]["1"]["expression_raw"], ["player"])
            self.assertEqual(list(labels["video_1"]["42"]), ["1", "2", "10"])
            self.assertTrue((output / "annotations_v1/train/SportsMOT/video_1_000000.json").is_file())


if __name__ == "__main__":
    unittest.main()
