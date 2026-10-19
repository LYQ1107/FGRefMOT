from __future__ import annotations

from scripts.efficiency import combine_efficiency, referring_efficiency, tracker_efficiency


def test_tracker_efficiency_uses_legacy_manifest_timestamps() -> None:
    result = tracker_efficiency(
        {
            "status": "SUCCEEDED",
            "started_at": "2026-10-20T00:00:00+00:00",
            "finished_at": "2026-10-20T00:00:10+00:00",
            "outputs": [{"max_frame": 100}, {"max_frame": 50}],
        }
    )
    assert result["wall_seconds"] == 10.0
    assert result["frame_count"] == 150
    assert result["tracker_fps"] == 15.0


def test_referring_efficiency_separates_video_and_expression_frame_rates() -> None:
    result = referring_efficiency(
        {
            "records": [
                {"video": "v1", "sequence_name": "v1+e1"},
                {"video": "v1", "sequence_name": "v1+e2"},
                {"video": "v2", "sequence_name": "v2+e1"},
            ],
            "seq_info": {"v1+e1": 10, "v1+e2": 10, "v2+e1": 20},
            "run": {"inference_seconds": 5.0},
        }
    )
    assert result["video_frame_count"] == 30
    assert result["expression_frame_count"] == 40
    assert result["video_frame_fps"] == 6.0
    assert result["referring_module_fps"] == 8.0


def test_complete_pipeline_rate_requires_both_stage_timings() -> None:
    report = combine_efficiency(
        {"wall_seconds": 10.0, "tracker_fps": 15.0},
        {"video_frame_count": 30, "inference_seconds": 5.0},
    )
    assert report["complete_pipeline"]["wall_seconds"] == 15.0
    assert report["complete_pipeline"]["complete_pipeline_fps"] == 2.0


def test_referring_efficiency_reads_recorded_yestrack_efficiency() -> None:
    result = referring_efficiency(
        {
            "efficiency": {
                "video_frame_count": 100,
                "expression_frame_count": 250,
                "inference_seconds": 10.0,
            }
        }
    )
    assert result["video_frame_count"] == 100
    assert result["expression_frame_count"] == 250
    assert result["referring_module_fps"] == 25.0
