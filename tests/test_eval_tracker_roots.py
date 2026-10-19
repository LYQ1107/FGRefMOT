from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flexhook_resolves_nested_tracker_outputs(tmp_path: Path) -> None:
    module = _load_script("eval_flexhook_refersports.py")
    nested = tmp_path / "tracker_outputs"
    nested.mkdir()
    track = nested / "video.txt"
    track.write_text("1,7,10,20,30,40,1,1,1,1\n", encoding="utf-8")

    assert module.resolve_tracker_file(tmp_path, "video") == track
    assert module.resolve_tracker_input_root(tmp_path, ["video"]) == nested


def test_flexhook_master_port_override_is_exposed() -> None:
    module = _load_script("eval_flexhook_refersports.py")
    args = module.parse_args(
        [
            "--config", "config.yaml",
            "--layout-root", "layout",
            "--tracker-root", "tracker",
            "--checkpoint", "checkpoint.pth",
            "--output-root", "output",
            "--master-port", "29604",
        ]
    )

    assert args.master_port == 29604


def test_yestrack_prefers_root_then_nested_tracker_outputs(tmp_path: Path) -> None:
    module = _load_script("eval_yestrack_refersports.py")
    nested = tmp_path / "tracker_outputs"
    nested.mkdir()
    nested_track = nested / "video.txt"
    nested_track.write_text("1,7,10,20,30,40,1\n", encoding="utf-8")

    assert module.resolve_tracker_input_root(tmp_path) == nested

    root_track = tmp_path / "other.txt"
    root_track.write_text("1,8,1,2,3,4,1\n", encoding="utf-8")
    assert module.resolve_tracker_input_root(tmp_path) == tmp_path


def test_yestrack_expression_dirs_suffix_duplicate_sentences() -> None:
    module = _load_script("eval_yestrack_refersports.py")
    records = [{"sentence": "Basketball players"}, {"sentence": "basketball players"}]

    assert module._expression_dirnames(records) == ["basketball-players", "basketball-players__2"]


def test_yestrack_inference_forwards_temporal_controls(monkeypatch, tmp_path: Path) -> None:
    module = _load_script("eval_yestrack_refersports.py")
    tracker = tmp_path / "tracker"
    tracker.mkdir()
    (tracker / "video.txt").write_text("1,7,10,20,30,40,1\n", encoding="utf-8")
    seen: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module.run_yestrack_inference(
        python=tmp_path / "python",
        model_name=tmp_path / "model",
        lora_path=tmp_path / "lora",
        data_root=tmp_path / "data",
        tracker_root=tracker,
        results_root=tmp_path / "results",
        gpu=4,
        re_refer_lower=0.2,
        re_refer_thresh=0.8,
        infer_every_n_frames=4,
        frame_offset=-1,
        stability_enable=True,
        stability_window=6,
        stability_thresh=0.4,
        stability_boost=0.3,
        enable_video_mode=True,
        video_n_frames=4,
    )

    command = seen["command"]
    assert result["exit_code"] == 0
    assert "--stability_enable" in command
    assert command[command.index("--stability_window") + 1] == "6"
    assert "--enable_video_mode" in command
    assert command[command.index("--video_n_frames") + 1] == "4"


def test_flexhook_eval_legacy_tokenizer_compatibility(monkeypatch) -> None:
    module = _load_script("eval_flexhook_refersports.py")
    from transformers import PreTrainedTokenizerBase

    monkeypatch.delattr(PreTrainedTokenizerBase, "batch_encode_plus", raising=False)
    assert module.patch_legacy_tokenizer_api()

    class DummyTokenizer:
        def __call__(self, *args, **kwargs):
            return kwargs

    result = PreTrainedTokenizerBase.batch_encode_plus(
        DummyTokenizer(), ["short"], padding="max_length", max_length=25
    )
    assert result["truncation"] is True


def test_flexhook_eval_moves_only_text_fields_to_cuda() -> None:
    module = _load_script("eval_flexhook_refersports.py")

    class FakeTensor:
        def __init__(self):
            self.moved = False

        def cuda(self, non_blocking=False):
            self.moved = True
            return self

    expid, expma = FakeTensor(), FakeTensor()
    batch = ("images", "pes", "bbox", "exp", expid, expma, "video", "obj", "frame", "last")

    class Loader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

    wrapped = module._CudaTextBatchLoader(Loader())
    values = list(wrapped)
    assert len(wrapped) == 1
    assert values[0][4] is expid and values[0][5] is expma
    assert expid.moved and expma.moved


def test_flexhook_predictions_preserve_duplicate_expressions(tmp_path: Path) -> None:
    module = _load_script("eval_flexhook_refersports.py")
    tracker = tmp_path / "video.txt"
    tracker.write_text("1,7,10,20,30,40,1,1,1,1\n", encoding="utf-8")
    manifest = {
        "records": [
            {"video": "video", "expression": "basketball players", "expression_dir": "basketball-players", "tracker_path": str(tracker)},
            {"video": "video", "expression": "basketball players", "expression_dir": "basketball-players__2", "tracker_path": str(tracker)},
        ]
    }
    cls_dict = {"video": {"7": {"1": {"basketball players": [0.1, 0.9]}}}}

    written = module.write_predictions(
        cls_dict=cls_dict,
        manifest=manifest,
        tracker_root=tmp_path,
        output_root=tmp_path / "out",
    )

    assert written == 2
    assert (tmp_path / "out/video/basketball-players/predict.txt").read_text() == (
        "1,7,10.0,20.0,30.0,40.0,1,1,1,1\n"
    )
    assert (tmp_path / "out/video/basketball-players__2/predict.txt").read_text() == (
        "1,7,10.0,20.0,30.0,40.0,1,1,1,1\n"
    )


def test_flexhook_can_archive_candidate_probabilities_without_changing_binary_rows(tmp_path: Path) -> None:
    module = _load_script("eval_flexhook_refersports.py")
    tracker = tmp_path / "video.txt"
    tracker.write_text("1,7,10,20,30,40,1,1,1,1\n", encoding="utf-8")
    manifest = {
        "records": [
            {"video": "video", "expression": "blue jersey", "expression_dir": "blue-jersey", "tracker_path": str(tracker)},
        ]
    }
    cls_dict = {"video": {"7": {"1": {"blue jersey": [0.2, 1.2]}}}}

    module.write_predictions(
        cls_dict=cls_dict,
        manifest=manifest,
        tracker_root=tmp_path,
        output_root=tmp_path / "out",
        save_scores=True,
    )

    assert (tmp_path / "out/video/blue-jersey/predict.txt").read_text() == (
        "1,7,10.0,20.0,30.0,40.0,1,1,1,1\n"
    )
    score = (tmp_path / "out/video/blue-jersey/predict_scores.txt").read_text()
    assert score.startswith("1,7,10.0,20.0,30.0,40.0,")
    assert float(score.split(",")[6]) > 0.7
