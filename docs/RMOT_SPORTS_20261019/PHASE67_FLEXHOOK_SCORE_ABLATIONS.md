# Phase 67 — FlexHook frozen-score H0–H8 historical-test ablation

The completed raw probability archive was replayed through all nine causal
score-level controls using the same Sports-trained FlexHook checkpoint, the
same common SambaMOTR tracker outputs, threshold `0.5`, and the same
historical 19-video/142-expression TrackEval tree. The controls read only
current and past prediction rows. Test ground truth was linked only after
prediction generation for offline TrackEval; it was not used by the model,
tracker, checkpoint selector, or score transform.

Archive manifest: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_common_eval_scores_20261019_v1/FLEXHOOK_EVAL_MANIFEST.json`

| Variant | Control | HOTA | DetA | AssA | IDF1 | IDSW | Prediction rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | pass-through control | 68.385415 | 59.317842 | 78.842302 | 66.627791 | 409 | 497473 |
| H1 | pass-through capacity control | 68.385415 | 59.317842 | 78.842302 | 66.627791 | 409 | 497473 |
| H2 | causal EMA | 68.512428 | 59.466877 | 78.937101 | 66.749697 | 382 | 496013 |
| H3 | causal adaptive interval | 68.063179 | 58.931499 | 78.613175 | 66.337556 | 426 | 500278 |
| H4 | stable identity score read | 68.604619 | 59.571474 | 79.010717 | 66.829675 | 375 | 495020 |
| H5 | short dynamic score history | 68.491772 | 59.441031 | 78.923806 | 66.729259 | 390 | 496244 |
| H6 | separated identity and dynamic score histories | 68.570292 | 59.528400 | 78.988776 | 66.798313 | 378 | 495400 |
| H7 | reliability-controlled memory write | 68.391700 | 59.325979 | 78.845975 | 66.633793 | 404 | 497376 |
| H8 | combined causal control | 68.632825 | 59.603905 | 79.032668 | 66.857910 | 370 | 494916 |

H0 and H1 reproduce the frozen FlexHook formal result exactly. H8 has the
best result among these controls (HOTA 68.632825, IDF1 66.857910, IDSW 370);
this is an offline score-control observation and does not constitute a
trained temporal module. H4 is the next strongest IDSW control (375).

The learned temporal-head package and any YesTrack/YB results remain separate
gates in the master plan.
