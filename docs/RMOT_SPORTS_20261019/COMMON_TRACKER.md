# Common physical tracker

Status: `SUCCEEDED_FOR_REGISTERED_RUNS`

The controlled comparison uses the fixed SambaMOTR-DAB-D-DETR output manifest:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_dab_refersports_train_20261019/TRACKER_RUN_MANIFEST.json`

The run covers 19 evaluation sequences, records the detector/tracker
checkpoint and per-file hashes, and completed on physical GPU 0. Its staging
tree contains images only. The manifest records `used_ground_truth=false`;
the Refer-Sports expression labels and evaluation GT are not supplied to the
physical tracker. Internal-validation tracker coverage is registered
separately under `..._internal_val_20261019`.

The controlled referring-model comparison is not complete until the full
YesTrack historical manifest and the registered ablations succeed. Native
tracker results, if added later, must remain a separate table.
