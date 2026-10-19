from pathlib import Path

from scripts.run_sambamotr_refersports_tracker import (
    parse_args,
    prepare_image_only_view,
    read_sequence_list,
)


def test_sambamotr_staging_view_excludes_annotations(tmp_path: Path):
    dataset = tmp_path / "dataset"
    source = dataset / "train" / "seq_a"
    (source / "img1").mkdir(parents=True)
    (source / "img1" / "000001.jpg").write_bytes(b"image")
    (source / "gt").mkdir()
    (source / "gt" / "gt.txt").write_text("1,1,0,0,1,1,1,-1,-1,-1\n")

    report = prepare_image_only_view(dataset, ["seq_a"], tmp_path / "staging")
    staged = tmp_path / "staging" / "SportsMOT" / "train" / "seq_a"
    assert report["policy"] == "IMAGE_ONLY_STAGING; GT_AND_LABEL_FILES_OMITTED"
    assert (staged / "img1").is_symlink()
    assert (staged / "img1" / "000001.jpg").is_file()
    assert not (staged / "gt").exists()


def test_sequence_list_rejects_duplicates(tmp_path: Path):
    path = tmp_path / "seq.txt"
    path.write_text("seq_a\n# comment\nseq_b\n")
    assert read_sequence_list(path) == ["seq_a", "seq_b"]
    path.write_text("seq_a\nseq_a\n")
    try:
        read_sequence_list(path)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:  # pragma: no cover - defensive assertion for the policy gate
        raise AssertionError("duplicate sequence names must be rejected")


def test_tracker_split_label_is_recorded_in_cli_contract():
    args = parse_args(["--source-root", "/src", "--model-root", "/model", "--dataset-root", "/data", "--staging-root", "/stage", "--output-root", "/out", "--split-label", "sports-val-5-videos"])
    assert args.split_label == "sports-val-5-videos"
