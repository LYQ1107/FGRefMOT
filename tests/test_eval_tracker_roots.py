from __future__ import annotations

import importlib.util
from pathlib import Path


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
