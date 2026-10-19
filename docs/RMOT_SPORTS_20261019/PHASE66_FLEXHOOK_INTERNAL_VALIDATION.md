# Phase 66 — FlexHook F1 internal validation completion

The Sports-trained FlexHook F1 checkpoint completed the fixed-tracker internal
validation protocol on the five-video, 45-expression selection split. Model
inference consumed only the fixed common SambaMOTR outputs. Ground-truth boxes
were linked into the TrackEval tree after inference and were not passed to the
model, used to build tracker inputs, or used to select the checkpoint.

The successful manifest is:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_internal_eval_20261019_v1/FLEXHOOK_EVAL_MANIFEST.json`

The evaluated checkpoint is the same F1 artifact used by the formal historical
test run:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_sports_trained_seed42/ckpt_epoch_best_0.pth`

TrackEval combined results at the registered threshold are:

| HOTA | DetA | AssA | IDF1 | IDSW |
| ---: | ---: | ---: | ---: | ---: |
| 71.241265 | 64.157324 | 79.107500 | 71.282648 | 103 |

The result contains 119,790 prediction rows across 45 expressions and has
manifest status `SUCCEEDED`. These values are internal-selection metrics; they
must not be substituted for the historical 19-video formal-test values.
