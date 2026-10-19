# Phase 70 — FlexHook strict internal-core training and selection

Status: `SUCCEEDED_INTERNAL_EVAL_COMPLETE`.

The strict internal-selection FlexHook run is `flexhook_f1_internal_core_seed42`.
Its contract passed the hash and policy gate before launch. The run trained on
the 17-video/136-expression core split for 25 epochs and reserved the
5-video/45-expression internal split for checkpoint selection. The historical
19-video/142-expression test split was not read by the model, tracker, or
selector.

The job ran on physical GPU3 from `2026-10-19T23:07:28+08:00` to
`2026-10-20T01:30:24+08:00`, exited with code 0, and reported training time
`2:20:15`. The selected checkpoint is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_internal_core_seed42/ckpt_epoch_best_0.pth`
with SHA-256
`af56903c9a6da854b01fd6b0b6c5b4f25f1a263ed9309d13bb99d5ccc1199e8a`.
It is epoch 8, selected by the highest internal validation F1 of `73.702`
(`Acc@1=82.338`, precision `62.071`, recall `90.695`). The final epoch 24
diagnostic was F1 `73.103`; no later epoch displaced epoch 8.

The separate internal evaluation completed on all five reserved videos. Its
manifest is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_internal_core_eval_20261019_v1/FLEXHOOK_EVAL_MANIFEST.json`
with SHA-256
`6fdecc40da05f11090027408b1510b86402153a8de72587a579ca0916cc15a40`.
Combined TrackEval results for the selected checkpoint are HOTA `0.727836`,
CLEAR-F1 `0.784420`, MOTA `0.498165`, and IDF1 `0.695842` (TP `79,683`, FP
`36,020`, FN `7,778`). The referring-module rate was `24.967` expression
frames/s at `12,952 MiB` peak GPU memory; complete pipeline FPS still requires
tracker wall-time measurement. These are internal-selection diagnostics, not
historical-test results or a learned-method improvement claim.

The selected H8 temporal head has completed internal TrackEval. Its one-time
19-video historical evaluation is in progress; this strict-core checkpoint
itself remains an internal-selection artifact.
