import json
from pathlib import Path

from tools.qim_fast6.rmot_eval import select_test_records


MANIFEST = Path(
    "/data1/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/metadata/datasets/refer_sports_manifest_global_ids_rev2.json"
)


def test_full_test_manifest_is_19_videos_142_expressions_and_141_compat_is_explicit():
    records = json.loads(MANIFEST.read_text())["records"]
    selected = select_test_records(records)
    compatible = select_test_records(records, historical_141=True)
    assert len({row["original_sportsmot_id"] for row in selected}) == 19
    assert len(selected) == 142
    assert len(compatible) == 141
    assert {row["expression_id"] for row in selected} - {row["expression_id"] for row in compatible} == {
        "v_4LXTUim5anY_c010:players"
    }
