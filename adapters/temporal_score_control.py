"""Causal score-level controls for post-inference RMOT ablations.

The controller is deliberately independent of either upstream model.  It is a
small, auditable bridge for the first temporal ablations: it consumes the
per-track referring probability produced by a baseline and may combine it
with only current/past state.  It does not inspect ground truth and it never
normalizes scores across candidates, so group expressions remain supported.

The later learned evidence head can replace this controller without changing
the evaluation tree or the causal state contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence


TEMPORAL_VARIANTS = ("YB0", "YB1", "YB2", "YB3", "YB4", "YB5", "YB6")
FLEXHOOK_SCORE_VARIANTS = ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")
ALL_VARIANTS = TEMPORAL_VARIANTS + FLEXHOOK_SCORE_VARIANTS
Variant = Literal[
    "YB0", "YB1", "YB2", "YB3", "YB4", "YB5", "YB6",
    "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8",
]


@dataclass(frozen=True)
class ScoreObservation:
    """One tracker candidate and its baseline referring score."""

    frame_index: int
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    detection_confidence: float = 1.0
    visibility: float = 1.0
    local_quality: float = 1.0


@dataclass(frozen=True)
class ScoreUpdate:
    """Causal output and diagnostics for one observation."""

    score: float
    reliability: float
    identity_score: float
    dynamic_score: float
    wrote_identity: bool
    verification_interval: int


@dataclass
class _TrackState:
    identity_score: float | None = None
    dynamic_scores: deque[float] | None = None
    last_score: float | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    last_frame: int | None = None
    last_verified_frame: int | None = None
    last_verified_score: float | None = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(value) for value in a]
    bx, by, bw, bh = [float(value) for value in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
    return intersection / union if union > 0.0 else 0.0


class TemporalScoreController:
    """Apply a causal temporal score variant within one expression namespace.

    ``YB0`` and ``YB1`` are pass-through controls.  YB1 is paired with the
    upstream TCP/TRP inference settings, while this class stays a no-op so the
    comparison does not silently add a second smoother.  YB2--YB6 are the
    score-level adaptive-interval, EMA, reliability, identity, and complete
    causal controls registered for the first post-inference study.
    """

    def __init__(
        self,
        *,
        variant: Variant = "YB6",
        dynamic_window: int = 4,
        identity_momentum: float = 0.9,
        ema_alpha: float = 0.5,
        reliability_threshold: float = 0.5,
        max_verification_interval: int = 8,
        max_track_gap: int = 60,
    ) -> None:
        if variant not in ALL_VARIANTS:
            raise ValueError(f"unknown temporal score variant: {variant}")
        if dynamic_window <= 0:
            raise ValueError("dynamic_window must be positive")
        if not 0.0 <= identity_momentum < 1.0:
            raise ValueError("identity_momentum must be in [0, 1)")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= reliability_threshold <= 1.0:
            raise ValueError("reliability_threshold must be in [0, 1]")
        if max_verification_interval <= 0 or max_track_gap <= 0:
            raise ValueError("intervals must be positive")
        self.variant = variant
        self._effective_variant = {
            "H0": "YB0",
            "H1": "YB0",  # capacity-matched score head control is pass-through here
            "H2": "YB3",  # ordinary history EMA
            "H3": "YB2",  # adaptive temporal interval
            "H4": "YB5",  # stable identity evidence read
            "H5": "H5",
            "H6": "H6",
            "H7": "YB4",  # reliability-controlled memory write
            "H8": "YB6",
        }.get(variant, variant)
        self.dynamic_window = int(dynamic_window)
        self.identity_momentum = float(identity_momentum)
        self.ema_alpha = float(ema_alpha)
        self.reliability_threshold = float(reliability_threshold)
        self.max_verification_interval = int(max_verification_interval)
        self.max_track_gap = int(max_track_gap)
        self._states: dict[tuple[object, int], _TrackState] = {}

    def reset(self, namespace: object | None = None) -> None:
        if namespace is None:
            self._states.clear()
            return
        for key in [key for key in self._states if key[0] == namespace]:
            del self._states[key]

    @staticmethod
    def _state_feature(bbox: Sequence[float]) -> tuple[float, float, float, float]:
        x, y, w, h = [float(value) for value in bbox]
        return (x + 0.5 * w, y + 0.5 * h, max(0.0, w), max(0.0, h))

    def _reliability(
        self,
        observation: ScoreObservation,
        state: _TrackState,
        candidate_conflict: float,
    ) -> float:
        if state.last_bbox is None:
            consistency = 1.0
        else:
            consistency = 0.5 + 0.5 * _iou_xywh(observation.bbox, state.last_bbox)
        values = (
            _clamp(observation.detection_confidence),
            _clamp(observation.visibility),
            _clamp(observation.local_quality),
            _clamp(consistency),
            1.0 - _clamp(candidate_conflict),
        )
        weights = (0.35, 0.20, 0.20, 0.15, 0.10)
        return _clamp(sum(weight * value for weight, value in zip(weights, values)))

    def _get_state(self, namespace: object, observation: ScoreObservation) -> _TrackState:
        key = (namespace, int(observation.track_id))
        state = self._states.get(key)
        if state is None:
            state = _TrackState(dynamic_scores=deque(maxlen=self.dynamic_window))
            self._states[key] = state
        elif state.last_frame is not None and observation.frame_index - state.last_frame > self.max_track_gap:
            state = _TrackState(dynamic_scores=deque(maxlen=self.dynamic_window))
            self._states[key] = state
        if state.last_frame is not None and observation.frame_index <= state.last_frame:
            raise ValueError(
                f"non-causal frame order for {namespace!r}/{observation.track_id}: "
                f"{observation.frame_index} after {state.last_frame}"
            )
        return state

    def update(
        self,
        namespace: object,
        observation: ScoreObservation,
        *,
        candidate_conflict: float = 0.0,
    ) -> ScoreUpdate:
        """Update one track using current and past observations only."""

        score = _clamp(observation.score)
        state = self._get_state(namespace, observation)
        reliability = self._reliability(observation, state, candidate_conflict)
        if state.dynamic_scores is None:
            state.dynamic_scores = deque(maxlen=self.dynamic_window)
        previous = state.last_score
        state.dynamic_scores.append(score)
        dynamic = sum(state.dynamic_scores) / len(state.dynamic_scores)

        wrote_identity = state.identity_score is None or reliability >= self.reliability_threshold
        if state.identity_score is None:
            identity = score
        elif wrote_identity:
            identity = (
                self.identity_momentum * state.identity_score
                + (1.0 - self.identity_momentum) * score
            )
        else:
            identity = state.identity_score
        state.identity_score = _clamp(identity)

        interval = 1
        if self._effective_variant in {"YB2", "YB6"}:
            uncertainty = 1.0 - abs(2.0 * score - 1.0)
            interval = 1 + int(round((1.0 - uncertainty) * (self.max_verification_interval - 1)))
            if state.last_verified_frame is None or state.last_verified_score is None:
                state.last_verified_frame = observation.frame_index
                state.last_verified_score = score
            elif observation.frame_index - state.last_verified_frame >= interval:
                state.last_verified_frame = observation.frame_index
                state.last_verified_score = score
            else:
                score = state.last_verified_score

        if self._effective_variant in {"YB3"}:
            output = score if previous is None else self.ema_alpha * score + (1.0 - self.ema_alpha) * previous
        elif self._effective_variant in {"YB4"}:
            memory = state.identity_score if previous is None else previous
            output = reliability * score + (1.0 - reliability) * memory
        elif self._effective_variant in {"YB5"}:
            output = 0.6 * score + 0.4 * state.identity_score
        elif self._effective_variant == "H5":
            output = 0.5 * score + 0.5 * dynamic
        elif self._effective_variant == "H6":
            output = 0.5 * score + 0.25 * state.identity_score + 0.25 * dynamic
        elif self._effective_variant in {"YB6"}:
            output = 0.45 * score + 0.30 * state.identity_score + 0.25 * dynamic
            output = reliability * output + (1.0 - reliability) * (state.identity_score if previous is None else previous)
        else:
            output = score

        state.last_score = _clamp(output)
        state.last_bbox = tuple(float(value) for value in observation.bbox)
        state.last_frame = int(observation.frame_index)
        return ScoreUpdate(
            score=state.last_score,
            reliability=reliability,
            identity_score=state.identity_score,
            dynamic_score=_clamp(dynamic),
            wrote_identity=wrote_identity,
            verification_interval=interval,
        )

    def process_frame(
        self,
        namespace: object,
        observations: Iterable[ScoreObservation],
    ) -> list[ScoreUpdate]:
        """Process one frame and derive candidate conflict causally."""

        values = list(observations)
        if not values:
            return []
        frame = values[0].frame_index
        if any(item.frame_index != frame for item in values):
            raise ValueError("process_frame requires one frame index")
        outputs: list[ScoreUpdate] = []
        for index, item in enumerate(values):
            conflict = max(
                (_iou_xywh(item.bbox, other.bbox) for j, other in enumerate(values) if j != index),
                default=0.0,
            )
            outputs.append(self.update(namespace, item, candidate_conflict=conflict))
        return outputs


__all__ = [
    "ALL_VARIANTS",
    "FLEXHOOK_SCORE_VARIANTS",
    "TEMPORAL_VARIANTS",
    "ScoreObservation",
    "ScoreUpdate",
    "TemporalScoreController",
]
