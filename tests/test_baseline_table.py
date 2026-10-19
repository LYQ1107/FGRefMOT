from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = PROJECT_ROOT / "scripts" / "build_baseline_table.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trackeval_payload(hota: float) -> dict:
    return {
        "MotChallenge2DBox": {
            "/tmp/tracker": {
                "COMBINED_SEQ": {
                    "all": {
                        "HOTA": {"HOTA": [hota, hota + 0.1], "DetA": [0.2], "AssA": [0.3]},
                        "CLEAR": {"IDSW": 7},
                        "Identity": {"IDF1": 0.4},
                    }
                }
            }
        }
    }


def test_formal_metrics_extract_flexhook_combined_result() -> None:
    module = _load_builder()
    metrics = module.formal_metrics({"trackeval_results": _trackeval_payload(0.5)})

    assert metrics["formal_hota"] == "55.000000"
    assert metrics["formal_deta"] == "20.000000"
    assert metrics["formal_assa"] == "30.000000"
    assert metrics["formal_idf1"] == "40.000000"
    assert metrics["formal_idsw"] == "7"
    assert metrics["formal_ref_f1"] == "NOT_RUN"


def test_formal_metrics_uses_preregistered_yestrack_threshold() -> None:
    module = _load_builder()
    payload = {
        "evaluations": [
            {"threshold": 0.2, "results": _trackeval_payload(0.1)},
            {"threshold": 0.5, "results": _trackeval_payload(0.6)},
        ]
    }

    assert module.formal_metrics(payload)["formal_hota"] == "65.000000"


def test_formal_metrics_uses_internal_selected_yestrack_threshold() -> None:
    module = _load_builder()
    payload = {
        "evaluations": [
            {"threshold": 0.5, "results": _trackeval_payload(0.6)},
            {"threshold": 0.8, "results": _trackeval_payload(0.7)},
        ]
    }

    assert module.formal_metrics(payload, selected_threshold=0.8)["formal_hota"] == "75.000000"


def test_manifest_selected_threshold_is_used_for_validation_metrics() -> None:
    module = _load_builder()
    payload = {
        "selected_threshold": 0.8,
        "evaluations": [
            {"threshold": 0.5, "results": _trackeval_payload(0.6)},
            {"threshold": 0.8, "results": _trackeval_payload(0.7)},
        ],
    }

    assert module.validation_metrics(payload)["validation_hota"] == "75.000000"


def test_validation_metrics_use_validation_namespace() -> None:
    module = _load_builder()
    metrics = module.validation_metrics({"trackeval_results": _trackeval_payload(0.4)})

    assert metrics["validation_hota"] == "45.000000"
    assert "formal_hota" not in metrics


def test_resolve_flexhook_eval_manifest_prefers_latest_success(tmp_path: Path) -> None:
    module = _load_builder()
    old = tmp_path / "runs/gpu/flexhook_f1_common_eval_20261019_v2/FLEXHOOK_EVAL_MANIFEST.json"
    new = tmp_path / "runs/gpu/flexhook_f1_common_eval_20261019_v7/FLEXHOOK_EVAL_MANIFEST.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text('{"status":"PREPARED_NO_MODEL_RUN"}\n', encoding="utf-8")
    new.write_text('{"status":"SUCCEEDED"}\n', encoding="utf-8")
    assert module.resolve_flexhook_eval_manifest(tmp_path) == new


def test_resolve_yestrack_eval_manifest_prefers_latest_success(tmp_path: Path) -> None:
    module = _load_builder()
    old = tmp_path / "runs/gpu/yestrack_y1_common_eval_20261019_v2/YESTRACK_EVAL_MANIFEST.json"
    new = tmp_path / "runs/gpu/yestrack_y1_common_eval_20261019_v3/YESTRACK_EVAL_MANIFEST.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    new.write_text('{"status":"SUCCEEDED"}\n', encoding="utf-8")
    assert module.resolve_yestrack_eval_manifest(tmp_path) == new


def test_resolve_flexhook_validation_manifest_prefers_latest_success(tmp_path: Path) -> None:
    module = _load_builder()
    old = tmp_path / "runs/gpu/flexhook_f1_internal_eval_20261019_v1/FLEXHOOK_EVAL_MANIFEST.json"
    new = tmp_path / "runs/gpu/flexhook_f1_internal_eval_20261019_v2/FLEXHOOK_EVAL_MANIFEST.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    new.write_text('{"status":"SUCCEEDED"}\n', encoding="utf-8")
    assert module.resolve_flexhook_validation_manifest(tmp_path) == new
