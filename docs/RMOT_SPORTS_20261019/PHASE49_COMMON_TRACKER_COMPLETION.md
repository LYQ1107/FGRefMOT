# Phase 49 — Common Refer-Sports tracker completion

Status: `TRACKER_SUCCEEDED_YESTRACK_EVALUATIONS_IN_PROGRESS`

The fixed common SambaMOTR-DAB-D-DETR tracker finished on physical GPU 0. Its
manifest reports `SUCCEEDED`, exit code 0, 19 sequences, and 19 tracker-output
files. The staging tree contained image links only; the tracker manifest records
`used_ground_truth: false`, so no GT or referring labels were supplied to the
tracker.

The output manifest is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_dab_refersports_train_20261019/TRACKER_RUN_MANIFEST.json`.
It records the physical tracker checkpoint, source commit, compatibility patch,
operator hash, sequence names, row counts, and SHA-256 for every output file.

The corrected YesTrack Y1 full run completed with 90,559 optimizer steps and
exit code 0. Its formal 19-video evaluation is running on physical GPU 1. The
separate 17-video internal run also exited 0 and its five-video validation
evaluation is running on physical GPU 2. Their metric cells remain unclaimed
until the corresponding manifests succeed.

The refreshed baseline table is stored at
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/reports/intermediate/baselines_20261019`.
It marks the common tracker as succeeded, Y1 training as succeeded/evaluation
in progress, and F1 as complete. All metric cells remain `NOT_RUN` until the
corresponding formal evaluation manifests succeed.
