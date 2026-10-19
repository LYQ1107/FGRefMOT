# Phase 56 — Internal YesTrack core training and validation

Status: `TRAINING_SUCCEEDED_EVALUATION_IN_PROGRESS`

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
seed 42, learning rate 5e-6, and negative downsample 0.5. It finished with
exit code 0 at 2026-10-20 02:15:00 CST. The resulting LoRA adapter is
`runs/20261019_184604/train/lora_epoch_0` on NAS with SHA-256
`0f67a3e5673357c4bb9341db23b94bba5fff437bb0d2caf5117e0d0d67d20d81`.

The fixed common tracker and the five-video internal validation split are now
being evaluated. The first automatic attempt stopped before inference because
the child process resolved the venv Python symlink to the shared base
interpreter; the tracked evaluator now preserves the venv path, and the
efficiency instrumentation resets CUDA peak accounting on the current device.
The retry is using the same completed adapter. No internal metric is claimed
until its evaluation manifest is `SUCCEEDED`.
