import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_yestrack_sports_layout import main


class YesTrackLayoutTest(unittest.TestCase):
    def test_frame_shift_box_conversion_and_symlinks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = root / "dataset"
            sequence = dataset / "train" / "video_1"
            (sequence / "img1").mkdir(parents=True)
            (sequence / "seqinfo.ini").write_text("[Sequence]\nimWidth=100\nimHeight=80\n", encoding="utf-8")
            (sequence / "img1" / "000001.jpg").write_bytes(b"jpg")
            labels_root = dataset / "labels_with_ids" / "video_1"
            labels_root.mkdir(parents=True)
            (labels_root / "000001.txt").write_text("0 42 0.25 0.5 0.3 0.5\n", encoding="utf-8")
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
                    "build_yestrack_sports_layout.py",
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
            image = output / "train/KITTI/training/image_02/video_1/000000.png"
            self.assertTrue(image.is_symlink())
            self.assertTrue(image.resolve().name == "000001.jpg")
            label = (output / "train/KITTI/labels_with_ids/image_02/video_1/000000.txt").read_text(encoding="utf-8")
            self.assertEqual(label, "0 42 0.10000000 0.25000000 0.30000000 0.50000000\n")
            expression = json.loads(next((output / "train/expression/video_1").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(expression["label"], {"0": [42]})


if __name__ == "__main__":
    unittest.main()
