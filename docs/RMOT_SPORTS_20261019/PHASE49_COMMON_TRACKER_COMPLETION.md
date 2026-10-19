# Phase 49 — Common Refer-Sports tracker completion

Status: `TRACKER_SUCCEEDED_Y1_TRAINING_RUNNING`

The fixed common SambaMOTR-DAB-D-DETR tracker finished on physical GPU 0. Its
manifest reports `SUCCEEDED`, exit code 0, 19 sequences, and 19 tracker-output
files. The staging tree contained image links only; the tracker manifest records
`used_ground_truth: false`, so no GT or referring labels were supplied to the
tracker.

The output manifest is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_dab_refersports_train_20261019/TRACKER_RUN_MANIFEST.json`.
It records the physical tracker checkpoint, source commit, compatibility patch,
operator hash, sequence names, row counts, and SHA-256 for every output file.

The corrected YesTrack Y1 run remains active on physical GPU 1. The observed
DataLoader total is 90,559 steps. The dependency gate is waiting for that run's
valid LoRA checkpoint, then it will launch the YesTrack and FlexHook formal
evaluations sequentially on a free GPU.

The refreshed baseline table is stored at
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/reports/intermediate/baselines_20261019`.
It marks the common tracker as succeeded, Y1 as training, and F1 as checkpoint
ready/formal evaluation pending. All metric cells remain `NOT_RUN` until the
corresponding formal evaluation manifests succeed.
