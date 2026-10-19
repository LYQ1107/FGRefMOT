from pathlib import Path
import shutil
from types import SimpleNamespace

import importlib.util
import numpy as np
import torch

from scripts.launch_flexhook import _apply_flexhook_resume_patch


def test_flexhook_resume_patch_is_idempotent(tmp_path):
    source = Path(__file__).parents[1] / "third_party" / "FlexHook" / "utils.py"
    upstream = tmp_path / "FlexHook"
    upstream.mkdir()
    shutil.copyfile(source, upstream / "utils.py")

    _apply_flexhook_resume_patch(upstream)
    first = (upstream / "utils.py").read_text(encoding="utf-8")
    assert "resume_schema': 'fgrefmot-flexhook-resume-v1'" in first
    assert "'optimizer': optimizer.state_dict()" in first
    assert "'rng_state'" in first

    _apply_flexhook_resume_patch(upstream)
    second = (upstream / "utils.py").read_text(encoding="utf-8")
    assert second == first


def test_flexhook_resume_payload_restores_training_state(tmp_path):
    source = Path(__file__).parents[1] / "third_party" / "FlexHook" / "utils.py"
    upstream = tmp_path / "FlexHook"
    upstream.mkdir()
    shutil.copyfile(source, upstream / "utils.py")
    _apply_flexhook_resume_patch(upstream)

    spec = importlib.util.spec_from_file_location("patched_flexhook_utils", upstream / "utils.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config = SimpleNamespace(
        OUTPUT=str(tmp_path),
        MODEL=SimpleNamespace(RESUME="",),
        EVAL_MODE=False,
        TRAIN=SimpleNamespace(START_EPOCH=0),
    )
    model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.tensor([1.0])},
        load_state_dict=lambda state, strict=False: None,
    )
    optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {"step": 3}},
        load_state_dict=lambda state: None,
    )
    scheduler = SimpleNamespace(
        state_dict=lambda: {"last_epoch": 4},
        load_state_dict=lambda state: None,
    )
    scaler = SimpleNamespace(
        state_dict=lambda: {"scale": 128.0},
        load_state_dict=lambda state: None,
    )
    logger = SimpleNamespace(info=lambda message: None)

    random_seed = 123
    import random

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    module.save_checkpoint(config, 4, model, 0.9, optimizer, scheduler, scaler, logger, 0.8)
    checkpoint = tmp_path / "ckpt_epoch_4.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["resume_schema"] == "fgrefmot-flexhook-resume-v1"
    assert {"optimizer", "lr_scheduler", "scaler", "rng_state"} <= set(payload)

    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    config.MODEL.RESUME = str(checkpoint)
    config.defrost = lambda: None
    config.freeze = lambda: None
    module.load_checkpoint(config, model, optimizer, scheduler, scaler, logger)
    restored = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert restored == expected
    assert config.TRAIN.START_EPOCH == 5
