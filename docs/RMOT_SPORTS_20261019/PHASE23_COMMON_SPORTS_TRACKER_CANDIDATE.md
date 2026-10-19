# Phase 23 — Common SportsMOT tracker candidate

Audit date: 2026-10-19  
Status: **OFFICIAL SPORTSMOT TEST CANDIDATE RETAINED / REFER-SPORTS COVERAGE CORRECTION RECORDED**

**Coverage correction (2026-10-19):** the 150 published files below cover the
official SportsMOT `dataset/test` tree. The Refer-Sports evaluation manifest
used by this workspace contains 19 videos from the SportsMOT `dataset/train`
tree, so there is zero filename overlap with this release. The earlier claim
that these files mapped to the 19 Refer-Sports evaluation videos was wrong and
is superseded by `PHASE36_COMMON_TRACKER_COVERAGE_CORRECTION.*`. These files
remain useful as an official asset and parser/provenance check, but they are
not activated as the Refer-Sports evaluation input.

The official SambaMOTR SportsMOT DAB-D-DETR release was downloaded through the NAS route and published atomically under:

`/home/Valadmin/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_sportsmot_dab`

The read-only server mount is:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_sportsmot_dab`

The source is pinned to the SambaMOTR repository commit `f1c139a653c00a55c0873a64d7c59a67d7dbad44` and Hugging Face revision `8c29646b16715761861a2db4cee1e877d8c63c87`. The selected variant is `sambamotr_pretrained/sportsmot/dab_detr`, with the official DAB-D-DETR checkpoint and published test tracker outputs. The repository's source tree and download manifest are retained on NAS; the download manifest hash is `8d96bfa38ff6feef0886785b50ce4d1cadc9b5f1844ae309a6d766cd0e4384b4`.

The checkpoint `sambamotr_dab_sportsmot.pth` is 591,484,713 bytes with SHA-256 `55ee27ac975311dbb2abffe016010404a843e9404105773672be41545f1f6b5c`. The retained YAML configuration is `source/sambamotr_dab_sportsmot.yaml` with SHA-256 `e1bb9d34c6b56d44db6b378bea723e0de59423b981b935ec5165be8c83cf96fe`.

The candidate contains 150 tracker files, 88,592,457 bytes in total, matching
all 150 local SportsMOT **official test** sequence directory names exactly. A
streaming audit found 998,952 non-empty rows. Every row has ten comma-separated
fields, finite numeric coordinates/scores, frame numbers starting at 1, and
non-negative track IDs; the audit reported zero malformed rows. The 19
Refer-Sports evaluation sequence names are from the SportsMOT train tree and
are absent from this candidate. No ground-truth annotations were used as
tracker input.

`tracker_manifest.json` records each output path, byte count, SHA-256, and upstream source object ID. Its SHA-256 is `a7df4f4004eb02910c5fe1b9010bd889e6be28044ed105fdcb359d2984c95018`. It records `used_ground_truth: false`, `generated_from_frames_without_ground_truth: true`, and `activation_status: CANDIDATE_ACQUIRED_NOT_YET_USED`.

The historical preflight marks the asset **READY for the official SportsMOT
test split** with 150 declared and 150 verified outputs. That status does not
prove coverage for the Refer-Sports evaluation split. A new fixed tracker set
must be generated for the 19 Refer-Sports videos before any referring metrics
are claimed.

This is an official end-to-end SambaMOTR detector-plus-tracker output release. It must not be described as an external-detector-plus-ByteTrack run. The existing protocol gate names the input `detector_outputs`; the manifest keeps that compatibility field while explicitly recording the integrated detector mode. If this candidate is activated after protocol review, every referring method must consume the same fixed output root and the resulting protocol revision must be recorded before evaluation.

Machine-readable provenance, hashes, coverage, and preflight evidence is in `PHASE23_COMMON_SPORTS_TRACKER_CANDIDATE.json`.
