# Phase 73 — FlexHook temporal-head evaluation adapter

Status: `INTERNAL_AND_HISTORICAL_EVALUATION_SUCCEEDED`

`scripts/eval_flexhook_temporal.py` evaluates a selected temporal checkpoint
without changing the pinned FlexHook model or the fixed SambaMOTR tracker. It
collects the final query and text features, groups observations by video,
physical track and expression text, sorts them by frame, and applies the causal
head before writing the standard MOT/TrackEval tree.

The command supports internal validation for checkpoint selection and the
historical test set for one final offline evaluation. It never supplies GT to
the model or tracker.

The selected H8 checkpoint has now completed the five-video internal TrackEval:
HOTA `0.7563164568`, DetA `0.7212400259`, AssA `0.7930987775`, MOTA
`0.7125575971`, CLEAR-F1 `0.8372938579`, IDF1 `0.7243474873`, and IDSW `80`.
The run produced 66,559 prediction rows at 15.333 referring pairs/s and
1.901 unique-video-frames/s, with 12,966 MiB peak GPU memory. The manifest is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_temporal_h8_internal_eval_20261019_v1/FLEXHOOK_EVAL_MANIFEST.json`
(`98b9686ed60bc8efa6d34baf314957c54a3ee3fc8a2320673735798320a60c6b`).

The selected checkpoint then completed its one-time 19-video historical test
evaluation. The successful manifest is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_temporal_h8_common_eval_20261019_v1/FLEXHOOK_EVAL_MANIFEST.json`
(`1b50d090c1582b3b0d92e600f4dd33d6af560a01d0c2a9951d1e01d88ae95581`). It
contains 296,507 prediction rows and reports HOTA `0.719128`, DetA `0.667403`,
AssA `0.774862`, MOTA `0.625810`, CLEAR-F1 `0.799027`, IDF1 `0.677970`, and
IDSW `386`. The referring stage measured 18.301 expression-video-frame pairs/s,
2.557 unique-video-frames/s, 4,653.555 seconds of inference, 323.616 seconds
of model loading, and 12,976 MiB peak GPU memory. This is a single selected H8
result; proposed-method improvement and ablation claims remain gated on the
formal YesTrack baseline.
