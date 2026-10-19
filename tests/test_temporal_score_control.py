from __future__ import annotations

import pytest

from adapters.temporal_score_control import ScoreObservation, TemporalScoreController


def _obs(frame: int, track: int, score: float, x: float = 0.0) -> ScoreObservation:
    return ScoreObservation(frame, track, (x, 0.0, 10.0, 20.0), score)


def test_raw_variants_are_pass_through_and_expression_scoped() -> None:
    controller = TemporalScoreController(variant="YB0")
    first = controller.update(("video", "identity"), _obs(0, 1, 0.2))
    other = controller.update(("video", "action"), _obs(0, 1, 0.8))
    assert first.score == pytest.approx(0.2)
    assert other.score == pytest.approx(0.8)


def test_controller_rejects_future_or_duplicate_frames() -> None:
    controller = TemporalScoreController(variant="YB3")
    controller.update("expr", _obs(4, 2, 0.5))
    with pytest.raises(ValueError):
        controller.update("expr", _obs(4, 2, 0.6))
    with pytest.raises(ValueError):
        controller.update("expr", _obs(3, 2, 0.6))


def test_frame_conflict_is_causal_and_outputs_are_independent() -> None:
    controller = TemporalScoreController(variant="YB6")
    updates = controller.process_frame(
        "expr",
        [_obs(0, 1, 0.9, 0.0), _obs(0, 2, 0.1, 2.0)],
    )
    assert len(updates) == 2
    assert all(0.0 <= update.score <= 1.0 for update in updates)
    assert updates[0].reliability >= updates[1].reliability


def test_gap_resets_track_id_memory() -> None:
    controller = TemporalScoreController(variant="YB5", max_track_gap=2)
    controller.update("expr", _obs(0, 3, 1.0))
    update = controller.update("expr", _obs(4, 3, 0.0))
    assert update.identity_score == pytest.approx(0.0)
    assert update.wrote_identity
