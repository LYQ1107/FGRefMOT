from scripts.build_sports_diagnostic_subsets import build_payload, subset_names


def test_subset_candidates_are_expression_level_and_explicit() -> None:
    record = {
        "expression_id": "video:blue-number-7",
        "sentence": "basketball players wearing blue number 7 jerseys",
        "target_track_ids": [7],
        "annotation_status": "PARTIAL_POSITIVE_TRACK_LABELS",
        "original_split": "train",
    }

    assert subset_names(record) == ["S1_SAME_UNIFORM_IDENTITY_CANDIDATE"]


def test_dynamic_and_group_candidates_keep_label_warning() -> None:
    record = {
        "expression_id": "video:holding-ball",
        "sentence": "basketball players who are holding the ball",
        "target_track_ids": [1, 2],
        "annotation_status": "PARTIAL_POSITIVE_TRACK_LABELS",
        "original_split": "train",
    }

    payload = build_payload([record], "train")
    assert "S2_TEMPORALLY_VALID_REFERRING_CANDIDATE" in payload["subsets"]
    assert "S3_GROUP_AND_SUBSET_CANDIDATE" in payload["subsets"]
    assert payload["subsets"]["S2_TEMPORALLY_VALID_REFERRING_CANDIDATE"]["label_sufficiency"].startswith(
        "INSUFFICIENT_LABELS"
    )
    assert payload["S4_OCCLUSION_ID_PERSISTENCE"]["label_sufficiency"].startswith(
        "INSUFFICIENT_LABELS"
    )
