# Phase 76 — Post-baseline FlexHook H0–H8 ablation watch

Status: `WAITING_FOR_Y1_FORMAL_MANIFEST`.

The first H0–H8 replay in Phase 67 was deliberately recorded as prebaseline
diagnostic evidence. A separate idempotent watcher now waits for the full
YesTrack Y1 historical evaluation manifest to become `SUCCEEDED`. Only then it
will rerun the registered causal score controls H0–H8 from the frozen
Sports-trained FlexHook score archive, using the same 19-video/142-expression
historical split and threshold 0.5. It never feeds GT or labels into score
transformation and does not reserve or modify a GPU.

The watcher writes `runs/gpu/postbaseline_flexhook_h0_h8_watch.log` and uses
versioned output roots `flexhook_h{variant}_common_eval_20261019_v2`. Existing
successful outputs are skipped, so a restart cannot duplicate completed
variants. The prebaseline reference remains
`docs/RMOT_SPORTS_20261019/PHASE67_FLEXHOOK_SCORE_ABLATIONS.json`; no postbaseline
ablation metric is claimed until the v2 manifests succeed.
