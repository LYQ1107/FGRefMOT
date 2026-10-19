# Phase 55 — Internal validation tracker completed

Status: `TRACKER_SUCCEEDED_INTERNAL_EVAL_TREE_PREPARED`

The fixed SambaMOTR-DAB-D-DETR tracker has now produced image-only tracker
outputs for all five `SPORTS_VAL` videos on GPU2. The provenance-complete
manifest is:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_dab_refersports_internal_val_20261019/TRACKER_RUN_MANIFEST.json`

It records `status=SUCCEEDED`, `sequence_count=5`, `physical_gpu=2`, and
`used_ground_truth=false`. The output directory is the manifest's
`tracker_outputs/` path.

The 45-expression internal GT tree is available at
`runs/refer_sports_mot_gt_internal_val_20261019/`, and the FlexHook evaluation
tree was prepared at
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_internal_val_eval_20261019/`
with `status=PREPARED_NO_MODEL_RUN`. These artifacts are for selecting a model
trained on the 17-video core; the running full-22 YesTrack job is still a
historical-comparison run and is not evaluated here.

No internal metric is claimed until a core-trained checkpoint has run through
the corresponding referring evaluator.
