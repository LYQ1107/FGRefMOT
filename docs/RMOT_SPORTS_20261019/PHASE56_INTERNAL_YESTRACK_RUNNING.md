# Phase 56 — Internal YesTrack core training running

Status: `TRAINING_RUNNING`

The corrected YesTrack Y1 baseline has been started on the fixed 17-video
internal core split. This run is separate from the historical 22-video
comparison job and uses physical GPU2, leaving GPU1 unchanged.

Run manifest:

`runs/gpu/yestrack_y1_sports_core_lora_seed42/RUN_MANIFEST.json`

The manifest records the command, source commit, layout hash, NAS-direct
storage, and the corrected label policy
`EXPLICIT_FRAME_KEYS_ONLY; MISSING_KEYS_UNKNOWN`. The internal layout has
136 training expressions across 17 videos and 45 validation expressions
across five disjoint videos. The preflight loader count is 569,159 samples,
including 236,561 explicit negatives and 163,091 unknown-frame samples
skipped; missing expression-frame keys are never treated as No labels.

The launcher started at 2026-10-19 18:38:38 CST with batch size 8, one epoch,
seed 42, learning rate 5e-6, and negative downsample 0.5. The preflight
estimate is 71,145 optimizer steps and about 6.8–7.5 hours of training; the
actual YesTrack progress total remains authoritative.

No checkpoint or internal metric is claimed until this run finishes and the
checkpoint is evaluated on the fixed five-video internal validation split.
