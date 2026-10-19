# Phase 60 — FlexHook F1 formal baseline completion

FlexHook F1 completed the fixed-tracker formal evaluation on the historical
Refer-Sports test protocol. The run used 19 videos and 142 expressions,
SportsMOT-trained F1 weights, and the common SambaMOTR tracker outputs. Model
inference consumed tracker outputs only; GT boxes were used only in the
offline TrackEval tree.

The first TrackEval invocation found one duplicate expression directory with
no prediction file. The model output writer had keyed records only by
expression text, so the second separately annotated occurrence of
`basketball players` overwrote the first directory. The writer now preserves
records as a list, and the completed v7 tree was repaired by copying the
same expression's already-generated predictions into the missing directory.
The regression test covers duplicate expressions. No GPU inference was
repeated.

The successful manifest is:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_common_eval_20261019_v7/FLEXHOOK_EVAL_MANIFEST.json`

TrackEval combined results at the registered threshold are:

| HOTA | DetA | AssA | IDF1 | IDSW |
| ---: | ---: | ---: | ---: | ---: |
| 68.385415 | 59.317842 | 78.842302 | 66.627791 | 409 |

The baseline table now reads these values only from the successful manifest.
The project test suite passes (`40 passed`), and the vendored FlexHook
checkout remains unchanged.
