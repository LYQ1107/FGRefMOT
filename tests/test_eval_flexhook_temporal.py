from scripts.eval_flexhook_temporal import _normalize_eval_expressions, _track_id


def test_normalizes_transposed_eval_expression_metadata_for_lists_and_tuples() -> None:
    assert _normalize_eval_expressions([["a0", "a1"], ["b0", "b1"]], 2) == [
        ["a0", "b0"],
        ["a1", "b1"],
    ]
    assert _normalize_eval_expressions([("a0", "a1"), ("b0", "b1")], 2) == [
        ["a0", "b0"],
        ["a1", "b1"],
    ]


def test_non_numeric_track_ids_have_stable_evidence_namespace() -> None:
    assert _track_id("17") == 17
    assert _track_id("track-alpha") == _track_id("track-alpha")
