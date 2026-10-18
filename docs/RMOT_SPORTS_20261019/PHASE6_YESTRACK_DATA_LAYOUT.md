# Phase 6 — YesTrack-compatible Refer-Sports layout

Audit date: 2026-10-19
Status: **LAYOUT PREPARED / BASELINE NOT RUN**

The pinned YesTrack `ReferKittiMOT` loader expects a Refer-KITTI-shaped root:
`KITTI/training/image_02/<sequence>/<frame:06d>.png`,
`KITTI/labels_with_ids/image_02/<sequence>/<frame:06d>.txt`, and
`expression/<sequence>/*.json`.  Its image index is zero-based and it interprets
label x/y as normalized top-left coordinates.  SportsMOT uses one-based JPG
names and center-format normalized boxes, so
`scripts/build_yestrack_sports_layout.py` performs those two explicit
conversions.

The generated local artifact is:

```text
runs/yestrack_sports_layout_20261019_v1/
```

It contains separate `train/` and `val/` roots.  The expression archive's
historical train list maps to `train/` and its 19-entry test list maps to
`val/`, while all raw sequence directories resolve to the existing SportsMOT
tree.  The full result is:

| check | train | val |
|---|---:|---:|
| videos | 22 | 19 |
| frames | 14,750 | 11,900 |
| converted label boxes | 160,489 | 130,152 |
| expressions | 181 | 142 |
| frame image links | 14,750 | 11,900 |
| warnings | 0 | 0 |

Every image is a symlink to the original JPG.  Converted label files retain
the global SportsMOT track IDs and are generated only to satisfy YesTrack's
training annotation interface.  Expression frame keys are shifted from
SportsMOT's one-based convention to the loader's zero-based convention.  The
manifest records source hashes and the policy
`ANNOTATIONS_FOR_YESTRACK_DATASET_ONLY; NO_GT_AS_TRACKER_INPUT`.

The core `ReferKittiMOT` loader was exercised directly against both roots and
reported the counts above.  Importing YesTrack's umbrella `data` package in the
current incomplete environment first imports a Triton helper and fails because
that environment lacks `setuptools`; this is an environment gap, not a layout
failure.  The target loader itself has no such failure when loaded directly.

The upstream training CLI currently accepts only `v1`, `v2`, and `dancetrack`
dataset modes and its default split override assumes numeric sequence IDs.  It
must be called through a small SportsMOT wrapper that selects these explicit
layout roots and sequence names; the unmodified CLI must not be treated as a
completed SportsMOT baseline.  No YesTrack checkpoint, LoRA adapter, common
tracker output, training run, or metric is claimed by this phase.
