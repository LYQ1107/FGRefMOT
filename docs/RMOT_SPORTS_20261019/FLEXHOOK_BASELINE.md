# FlexHook baseline

Status: `F1_FORMAL_AND_INTERNAL_EVALUATION_SUCCEEDED`

F1 uses the pinned FlexHook implementation, Sports training, seed 42, and the
fixed common SambaMOTR tracker. The successful checkpoint is
`.../runs/gpu/flexhook_f1_sports_trained_seed42/ckpt_epoch_best_0.pth` on NAS;
its SHA-256 is recorded in `PHASE29_FLEXHOOK_F1_COMPLETION.md`.
The native training log records 248,218,728 model parameters.

| split | HOTA | DetA | AssA | IDF1 | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: |
| internal validation (5 videos) | 71.241265 | 64.157324 | 79.107500 | 71.282648 | 103 |
| historical test (19 videos) | 68.385415 | 59.317842 | 78.842302 | 66.627791 | 409 |

The values are mean-over-alpha TrackEval percentages. The internal result was
used for the registered selection route; the historical result was run only
after selection. The evaluation manifest is
`.../runs/gpu/flexhook_f1_common_eval_20261019_v7/FLEXHOOK_EVAL_MANIFEST.json`.

The strict core-only F1 run and the H8 causal temporal head are separate
artifacts. H8 is reported in `FLEXHOOK_ABLATIONS.md`; it must not be merged
into the native F1 baseline row.
