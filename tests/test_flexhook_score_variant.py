from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_flexhook_score_variant import prepare_score_eval_tree


def test_prepare_score_eval_tree_thresholds_archived_scores_without_gt_reads(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    score = tmp_path / "score"
    expression = "video/blue-jersey"
    (archive / expression).mkdir(parents=True)
    (score / expression).mkdir(parents=True)
    (archive / expression / "gt.txt").write_text("1,1,0,0,10,20,1,1,1,1\n", encoding="utf-8")
    (score / expression / "predict_scores.txt").write_text(
        "1,1,0,0,10,20,0.75,1,1,1\n"
        "2,1,0,0,10,20,0.25,1,1,1\n",
        encoding="utf-8",
    )
    manifest = {
        "records": [{"video": "video", "expression_dir": "blue-jersey"}],
        "seq_info": {"video+blue-jersey": 2},
        "sequence_names": ["video"],
    }
    (archive / "FLEXHOOK_EVAL_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "out"
    result = prepare_score_eval_tree(
        score_root=score,
        archive_root=archive,
        output_root=output,
        variant="H0",
        threshold=0.5,
    )

    assert result["prediction_rows"] == 1
    assert (output / expression / "gt.txt").is_symlink()
    assert (output / expression / "predict.txt").read_text() == (
        "1,1,0.0,0.0,10.0,20.0,1,1,1,1\n"
    )
