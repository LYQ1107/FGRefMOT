import tempfile
import unittest
from pathlib import Path

from scripts.flexhook_sportsmot_smoke import build_options, resolve_layout_root


class FlexHookSportsMOTSmokeTest(unittest.TestCase):
    def test_layout_root_requires_loader_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaises(FileNotFoundError):
                resolve_layout_root(root)

    def test_options_pin_sports_dataset_and_protocol_shapes(self):
        options = build_options(Path("/tmp/flexhook-layout"))
        self.assertEqual(options.DATA.DATASET, "SportsMOT")
        self.assertEqual(options.DATA.TRAIN, "SportsMOT")
        self.assertEqual(options.DATA.TEST, "SportsMOT")
        self.assertEqual(options.sample_frame_num, 4)
        self.assertEqual(options.sample_frame_len, 8)
        self.assertEqual(options.text_len, 25)


if __name__ == "__main__":
    unittest.main()

