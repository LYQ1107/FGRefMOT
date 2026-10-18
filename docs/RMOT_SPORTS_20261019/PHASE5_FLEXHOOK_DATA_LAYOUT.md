# Phase 5 — FlexHook-compatible Refer-Sports layout

Audit date: 2026-10-19  
Status: **LAYOUT PREPARED / BASELINE NOT RUN**

The pinned FlexHook loader (`data/mydataloader_mix.py`) expects a LaMOT-style
directory with `sequences/<split>/SportsMOT_labels.json`, a resolution file,
MOT-style `img1` paths, and one `annotations_v1/<split>/SportsMOT/*.json`
language file per expression.  The checked-in SportsMOT frames and
`labels_with_ids` are organized differently, so
`scripts/build_flexhook_sports_layout.py` creates the adapter metadata without
copying raw frames.

The generated local artifact is:

```text
runs/flexhook_sports_layout_20261019_v2/
```

It was produced with the expression archive, historical train/test lists,
SportsMOT image root, and the global-ID `labels_with_ids` root recorded in
`configs/paths.env.example`.  The result is:

| check | value |
|---|---:|
| videos | 41 (22 train, 19 val) |
| expressions | 323 (181 train, 142 val) |
| train label tracks / boxes | 313 / 160,489 |
| val label tracks / boxes | 261 / 130,152 |
| sequence symlinks | 41 |
| warnings | 0 |

`SportsMOT_labels.json` retains numeric frame traversal order, because the
upstream loader asserts that frame IDs increase.  All object IDs come from
SportsMOT's `labels_with_ids` global namespace.  The expression archive is only
used to attach `expression_raw` sentences to matching frame/track records.
The raw `gt/gt.txt` files are not read by this adapter.

The adapter is symlink-only for frames: the 41 `SportsMOT-sequences/<video>`
entries resolve to the existing read-only dataset tree.  It writes only JSON
metadata and a manifest locally.  The manifest records the source hashes,
counts, resolved symlink targets, and the policy
`LAYOUT_METADATA_ONLY; NO_GT_AS_TRACKER_INPUT`.

This layout makes the upstream **training/validation data interface** concrete;
it does not make the inference route complete.  FlexHook test mode reads one
MOT-format tracker file per video from `track_root`, and no verified
SportsMOT-trained detector or physical-tracker output is present.  The empty
`TAO_frame2file.json` files are valid for the SportsMOT branch but do not stand
in for tracker results.  No checkpoint, pretrained backbone, or metric is
claimed by this phase.

The adapter's unit test and the workspace's full test suite pass with
`python3 -m unittest discover -s tests -v` (6 tests).  The generated local
layout is ignored by Git and can be regenerated from the command recorded in
the script; only its small manifest and report are suitable for the NAS
metadata/report trees.
