import tempfile
import unittest
from pathlib import Path

from scripts.yestrack_sportsmot_smoke import select_sequences


class YesTrackSportsMOTSmokeTest(unittest.TestCase):
    def test_sequence_limit_is_deterministic(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "KITTI" / "training" / "image_02"
            for name in ("v_z_c002", "v_a_c001", "v_b_c003"):
                (root / name).mkdir(parents=True)
            self.assertEqual(select_sequences(Path(raw), 2), ["v_a_c001", "v_b_c003"])
            self.assertEqual(select_sequences(Path(raw), 0), ["v_a_c001", "v_b_c003", "v_z_c002"])


if __name__ == "__main__":
    unittest.main()

