#!/usr/bin/env python3
"""Run or audit YesTrack against the generated SportsMOT layout.

The upstream YesTrack checkout is kept untouched.  This adapter supplies the
non-numeric SportsMOT sequence split to its existing ``ReferKittiMOT`` loader,
selects the separate train/val roots, and forwards all model/training options
to the upstream ``refer_llm.llm_train`` parser.  ``--dry-run`` only builds the
two supervised data loaders and prints bounded counts; it never loads a model
or starts a GPU job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def resolve_layout_roots(
    layout_root: str | Path,
    train_root: str | Path | None = None,
    val_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve and validate the generated ``train`` and ``val`` roots."""

    root = Path(layout_root).expanduser().resolve()
    train = Path(train_root).expanduser().resolve() if train_root else root / "train"
    val = Path(val_root).expanduser().resolve() if val_root else root / "val"
    for split, path in (("train", train), ("val", val)):
        if not path.is_dir():
            raise FileNotFoundError(f"YesTrack SportsMOT {split} root does not exist: {path}")
        image_root = path / "KITTI" / "training" / "image_02"
        expression_root = path / "expression"
        if not image_root.is_dir():
            raise FileNotFoundError(f"YesTrack SportsMOT {split} image root does not exist: {image_root}")
        if not expression_root.is_dir():
            raise FileNotFoundError(f"YesTrack SportsMOT {split} expression root does not exist: {expression_root}")
    return train, val


def sequence_names(split_root: str | Path) -> list[str]:
    """Return the finite sequence list from one generated split root."""

    image_root = Path(split_root) / "KITTI" / "training" / "image_02"
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


def _has_option(argv: Iterable[str], option: str) -> bool:
    prefix = option + "="
    return any(item == option or item.startswith(prefix) for item in argv)


def build_upstream_argv(
    passthrough: list[str],
    train_root: Path,
    *,
    reject_user_overrides: tuple[str, ...] = ("--data_root", "--dataset_version", "--eval_sequence"),
) -> list[str]:
    """Add adapter-owned upstream arguments after user options.

    Appending the values makes the adapter's split policy authoritative.  A
    user-provided override for those same options is rejected instead of being
    silently shadowed.
    """

    for option in reject_user_overrides:
        if _has_option(passthrough, option):
            raise ValueError(f"{option} is owned by the SportsMOT adapter; use its layout roots instead")
    return [
        *passthrough,
        "--data_root",
        str(train_root),
        "--dataset_version",
        "v1",
        "--eval_sequence",
        "all",
    ]


def make_sportsmot_loader_class(train_root: Path, val_root: Path):
    """Create a loader subclass after the dedicated YesTrack environment loads.

    Imports are intentionally lazy so tests that only validate path policy do
    not require Torch or Transformers.
    """

    from data.refer_kitti_mot import ReferKittiMOT

    train_names = sequence_names(train_root)
    val_names = sequence_names(val_root)

    class SportsMOTReferKittiMOT(ReferKittiMOT):
        _sports_train_root = train_root
        _sports_val_root = val_root
        _sports_train_names = train_names
        _sports_val_names = val_names

        def __init__(self, data_root: str = "", split: str = "train", **kwargs: Any):
            actual_root = self._sports_val_root if split == "val" else self._sports_train_root
            # These names are already the exact non-numeric SportsMOT IDs.
            kwargs["train_ids_override"] = list(self._sports_train_names)
            kwargs["val_ids_override"] = list(self._sports_val_names)
            super().__init__(data_root=str(actual_root), split=split, **kwargs)

        def _get_sequence_names_for_split(self):
            all_names = self._get_all_sequence_names()
            wanted = self._sports_val_names if self.split == "val" else self._sports_train_names
            wanted_set = set(wanted)
            return sorted(name for name in all_names if name in wanted_set)

    SportsMOTReferKittiMOT.__name__ = "SportsMOTReferKittiMOT"
    return SportsMOTReferKittiMOT


def install_sportsmot_adapter(llm_train_module: Any, train_root: Path, val_root: Path):
    """Patch only the imported upstream symbols used by ``llm_train.train``."""

    loader_cls = make_sportsmot_loader_class(train_root, val_root)

    def build_refer_dataset(
        data_root: str,
        split: str = "train",
        train_ids_override: list | None = None,
        val_ids_override: list | None = None,
    ):
        return loader_cls(
            data_root=data_root,
            split=split,
            load_annotation=True,
            expression_sub_dir="expression",
            labels_with_ids_sub_dir="labels_with_ids/image_02",
            train_ids_override=train_ids_override,
            val_ids_override=val_ids_override,
        )

    llm_train_module.build_refer_dataset = build_refer_dataset
    llm_train_module.ReferKittiMOT = loader_cls
    return loader_cls


def audit_layout(train_root: Path, val_root: Path) -> dict[str, Any]:
    """Build both supervised loaders and report finite structural counts."""

    # The import is intentionally local; the script can still print --help on
    # a machine without the dedicated model environment.
    loader_cls = make_sportsmot_loader_class(train_root, val_root)
    results: dict[str, Any] = {"policy": "SUPERVISED_LABELS_ONLY; NO_GT_AS_TRACKER_INPUT"}
    for split, root in (("train", train_root), ("val", val_root)):
        ds = loader_cls(
            data_root=str(root),
            split=split,
            load_annotation=True,
            expression_sub_dir="expression",
            labels_with_ids_sub_dir="labels_with_ids/image_02",
        )
        boxes = sum(int(ann["bbox"].shape[0]) for seq in ds.sequence_names for ann in ds.annotations[seq])
        expressions = sum(len(ds._load_expressions_for_sequence(seq)) for seq in ds.sequence_names)
        results[split] = {
            "root": str(root),
            "sequence_count": len(ds.sequence_names),
            "frame_count": sum(int(info["length"]) for info in ds.sequence_infos.values()),
            "box_count": boxes,
            "expression_count": expressions,
        }
    return results


def parse_adapter_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True, description=__doc__)
    parser.add_argument("--layout-root", required=True, help="runs/yestrack_sports_layout_... root containing train/ and val/")
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--val-root", default=None)
    parser.add_argument("--dry-run", action="store_true", help="audit both loaders without importing/loading a model")
    parser.add_argument("--json", action="store_true", help="print dry-run counts as JSON")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    adapter_args, passthrough = parse_adapter_args(argv)
    train_root, val_root = resolve_layout_roots(
        adapter_args.layout_root,
        train_root=adapter_args.train_root,
        val_root=adapter_args.val_root,
    )
    repo_root = Path(__file__).resolve().parents[1] / "third_party" / "YesTrack"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if adapter_args.dry_run:
        report = audit_layout(train_root, val_root)
        if adapter_args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            for split in ("train", "val"):
                print(f"{split}: {json.dumps(report[split], sort_keys=True)}")
            print("yestrack_sportsmot_adapter: DRY_RUN_OK")
        return 0

    upstream_argv = build_upstream_argv(passthrough, train_root)
    # Keep the upstream repository import and model construction out of the
    # path-only audit above.
    from refer_llm import llm_train

    install_sportsmot_adapter(llm_train, train_root, val_root)
    args = llm_train.parse_args(upstream_argv)
    # The appended arguments are authoritative, but set these explicitly as a
    # second defense against config-file defaults.
    args.data_root = str(train_root)
    args.dataset_version = "v1"
    args.eval_sequence = "all"
    llm_train.train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
