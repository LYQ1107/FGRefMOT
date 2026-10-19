from __future__ import annotations

from pathlib import Path

from scripts.apply_temporal_score_variant import apply_variant


def test_apply_variant_keeps_expression_namespaces_and_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input/video/expr/predict_with_conf.txt"
    source.parent.mkdir(parents=True)
    source.write_text(
        "1,1,0,0,10,20,0.2\n"
        "1,2,2,0,10,20,0.8\n"
        "2,1,0,0,10,20,0.4\n"
        "2,2,2,0,10,20,0.6\n",
        encoding="utf-8",
    )

    output = tmp_path / "output"
    manifest = apply_variant(
        input_root=tmp_path / "input",
        output_root=output,
        variant="YB6",
    )

    assert manifest["policy"]["gt_or_labels_read"] is False
    assert len(manifest["files"]) == 1
    rows = (output / "video/expr/predict_with_conf.txt").read_text().splitlines()
    assert len(rows) == 4
    assert float(rows[-1].split(",")[6]) != 0.6
    assert (output / "TEMPORAL_SCORE_MANIFEST.json").is_file()


def test_min_score_is_applied_after_causal_update(tmp_path: Path) -> None:
    source = tmp_path / "input/video/expr/predict_with_conf.txt"
    source.parent.mkdir(parents=True)
    source.write_text("1,1,0,0,10,20,0.1\n", encoding="utf-8")

    output = tmp_path / "output"
    apply_variant(
        input_root=tmp_path / "input",
        output_root=output,
        variant="YB0",
        min_score=0.5,
    )

    assert (output / "video/expr/predict_with_conf.txt").read_text() == ""
