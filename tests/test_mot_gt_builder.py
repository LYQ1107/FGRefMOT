import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_refer_sports_mot_gt import main


class MotGtBuilderTest(unittest.TestCase):
    def test_filters_source_rows_without_propagating_labels(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images = root / "dataset"
            sequence = images / "train" / "video_1"
            (sequence / "gt").mkdir(parents=True)
            (sequence / "img1").mkdir()
            (sequence / "seqinfo.ini").write_text(
                "[Sequence]\nimWidth=100\nimHeight=80\n", encoding="utf-8"
            )
            (sequence / "gt" / "gt.txt").write_text(
                "1,1,10,20,30,40,1,1,1\n"
                "1,2,11,21,31,41,1,1,1\n"
                "2,1,12,22,32,42,1,1,1\n",
                encoding="utf-8",
            )
            train_list = root / "train.txt"
            train_list.write_text("a/b/c/video_1/000001.jpg\n", encoding="utf-8")
            test_list = root / "test.txt"
            test_list.write_text("", encoding="utf-8")
            labels_root = root / "labels_with_ids" / "video_1"
            labels_root.mkdir(parents=True)
            (labels_root / "000001.txt").write_text("0 1 0.25 0.5 0.3 0.5\n", encoding="utf-8")
            archive_path = root / "expressions.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "expression/video_1/players.json",
                    json.dumps({"video_name": "video_1", "sentence": "player", "label": {"1": [1], "2": []}}),
                )
            output = root / "output"
            old_argv = __import__("sys").argv
            try:
                __import__("sys").argv = [
                    "build_refer_sports_mot_gt.py",
                    "--expression-zip", str(archive_path),
                    "--train-list", str(train_list),
                    "--test-list", str(test_list),
                    "--images-root", str(images),
                    "--labels-root", str(root / "labels_with_ids"),
                    "--output", str(output),
                    "--split", "train",
                ]
                self.assertEqual(main(), 0)
            finally:
                __import__("sys").argv = old_argv
            gt = (output / "gt" / "video_1__players" / "gt.txt").read_text(encoding="utf-8")
            self.assertEqual(gt, "1,1,10.000000,20.000000,30.000000,40.000000,1,1,1\n")
            self.assertEqual((output / "seqmaps" / "train.txt").read_text(encoding="utf-8"), "name\nvideo_1__players\n")


if __name__ == "__main__":
    unittest.main()
