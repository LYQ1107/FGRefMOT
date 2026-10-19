# YesTrack baseline

Status: `TRAINING_AND_INTERNAL_VALIDATION_SUCCEEDED; FORMAL_TEST_RUNNING`

The full seed42 Sports-LoRA run used the 22-video / 181-expression training
layout, completed 90,559 optimizer steps, and wrote its adapter on NAS. The
separate 17-video core run produced the adapter used for internal selection.
The fixed threshold chosen on the five-video internal split is `0.8` from the
registered grid 0.2–0.8.

| split | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal validation (5 videos) | 74.652451 | 75.683610 | 73.637193 | 77.798104 | 78.302335 | 126 |
| historical test (19 videos) | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Internal HOTA/DetA/AssA are mean-over-alpha percentages. The successful
internal manifest is
`.../runs/gpu/yestrack_y1_internal_eval_20261019_v5/YESTRACK_EVAL_MANIFEST.json`.
The full historical evaluator is still producing predictions; no test metric
is claimed until `YESTRACK_EVAL_MANIFEST.json` reports `SUCCEEDED`.

Seed43 and seed44 are independent stability trainings. Their metrics and any
stability claim remain pending until their fixed-threshold internal manifests
succeed.
