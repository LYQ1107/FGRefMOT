from __future__ import annotations

from pathlib import Path

import scripts.eval_yestrack_score_variant as module


def test_score_variant_skips_model_and_records_offline_evaluations(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "input"
    data_root = tmp_path / "data"
    gt_root = tmp_path / "gt"
    sequence_list = tmp_path / "seq.txt"
    for path in (input_root, data_root, gt_root):
        path.mkdir()
    sequence_list.write_text("video\n", encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_apply_variant(**kwargs):
        calls["apply"] = kwargs
        return {"status": "SUCCEEDED"}

    def fake_evaluate_threshold(**kwargs):
        calls.setdefault("evaluations", []).append(kwargs)
        return {"threshold": kwargs["threshold"], "results": {}}

    monkeypatch.setattr(module, "apply_variant", fake_apply_variant)
    monkeypatch.setattr(module, "evaluate_threshold", fake_evaluate_threshold)
    payload = module.evaluate_score_variant(
        input_results_root=input_root,
        score_root=tmp_path / "scores",
        output_root=tmp_path / "out",
        variant="YB6",
        data_root=data_root,
        gt_manifest_root=gt_root,
        sequence_list=sequence_list,
        thresholds=[0.5],
    )

    assert payload["status"] == "SUCCEEDED"
    assert payload["policy"]["model_inference"].startswith("NOT_RUN")
    assert payload["policy"]["gt_or_labels_for_score_transform"] is False
    assert calls["evaluations"][0]["sequence_names"] == ["video"]
    assert (tmp_path / "out/YESTRACK_SCORE_EVAL_MANIFEST.json").is_file()
