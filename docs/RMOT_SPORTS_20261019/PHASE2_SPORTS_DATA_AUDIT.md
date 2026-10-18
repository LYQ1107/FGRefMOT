# PHASE 2 — Refer-Sports data audit

The canonical manifest is produced by
`scripts/audit_refer_sports.py`. It reads the small historical expression ZIP,
the two historical split lists and the existing local SportsMOT frame tree.
It does not copy raw images or rewrite the source archive.

## Verified source facts

- 41 Refer-Sports video IDs are present in the expression ZIP.
- 22 video IDs are in the historical training image list.
- 19 video IDs are in `seqmap_sports.txt` and form the historical test set.
- 323 expression JSON files are present in the archive: 181 on the 22-video
  training split and 142 on the 19-video test split.
- `seqmap_sports.txt` has 141 test expression entries. One archive expression,
  `v_4LXTUim5anY_c010/players.json`, is absent from that list; it is retained
  in the manifest and flagged rather than silently dropped.
- Every expression video has a matching local SportsMOT sequence directory.
- Each expression JSON has `video_name`, `sentence`, a frame-to-positive-track
  `label` map, and an empty or optional `ignore` map.
- Expression track IDs are the global IDs in SportsMOT
  `dataset/labels_with_ids/<video>/<frame>.txt`; they are not the per-video IDs
  in `dataset/<split>/<video>/gt/gt.txt`. The manifest records this ID
  namespace and verifies every target ID against `labels_with_ids`.
- Image dimensions and frame counts are measured from the local frame files;
  FPS is `UNKNOWN` because no video container metadata is part of these
  sources.

## Label policy

For each listed frame and track ID, the canonical state is `TRUE`. An absent
track ID or a frame outside a listed temporal interval is `UNKNOWN`; the
converter never creates `FALSE` labels without an explicit source annotation.
The manifest stores per-track contiguous intervals and keeps frame indexing
one-based, matching the historical file names and labels.

## Split and count discrepancy

The attachment cites 41 videos and approximately 492 expressions as historical
statistics. The verified archive contains 323 expressions. This is recorded as
an audit warning and remains unresolved until another authoritative archive is
identified; the count is not inflated to match the paper. The 22/19 video split
is explicit and has no train/test video overlap in the checked lists.

## Outputs

- `data/refer_sports_manifest.json` — complete records and provenance.
- `data/refer_sports_manifest.csv` — tabular view for review.
- `adapters/refer_sports/canonical_schema.py` — dependency-free interval and
  label-state helpers.
- `scripts/build_refer_sports_mot_gt.py` — builds evaluation-only MOT GT from
  `labels_with_ids`, preserving the global expression ID namespace and refusing
  to use the resulting files as tracker input.

`PHASE2_DATA_AUDIT: COMPLETE` once the generator exits successfully and the
manifest hashes are recorded in the phase commit. Training and test-GT tuning
are not performed by this phase.
