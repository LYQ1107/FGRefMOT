"""Small dependency-free canonical records for Refer-Sports annotations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class LabelState(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TemporalInterval:
    track_id: int
    start_frame: int
    end_frame: int


def contiguous_intervals(frames: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted({int(frame) for frame in frames})
    if not values:
        return []
    intervals: list[tuple[int, int]] = []
    start = previous = values[0]
    for frame in values[1:]:
        if frame != previous + 1:
            intervals.append((start, previous))
            start = frame
        previous = frame
    intervals.append((start, previous))
    return intervals


def target_intervals(label: Mapping[str, Iterable[int]]) -> list[TemporalInterval]:
    by_track: dict[int, list[int]] = {}
    for frame_raw, track_ids in label.items():
        frame = int(frame_raw)
        for track_raw in track_ids:
            by_track.setdefault(int(track_raw), []).append(frame)
    output: list[TemporalInterval] = []
    for track_id in sorted(by_track):
        for start, end in contiguous_intervals(by_track[track_id]):
            output.append(TemporalInterval(track_id, start, end))
    return output
