# Phase 56 — Internal YesTrack core training and validation

Status: `TRAINING_AND_INTERNAL_EVALUATION_SUCCEEDED`

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

The fixed common tracker and the five-video internal validation split have now
been evaluated with the completed adapter. The first automatic attempt stopped
before inference because the child process resolved the venv Python symlink to
the shared base interpreter. A later attempt completed inference but exposed a
split-manifest metadata bug while writing efficiency data; the adapter now
falls back to the parent layout manifest, and the completed prediction archive
was reused without retraining.

The successful validation manifest is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/yestrack_y1_internal_eval_20261019_v5/YESTRACK_EVAL_MANIFEST.json`
(`33ffa81651def7d71d74e2c291dd05b42692839ef04ca56bc99d9a91db0f7822`).
Across the registered thresholds 0.2–0.8, threshold 0.8 was selected using
internal-validation HOTA. TrackEval HOTA, DetA, and AssA are reported as the
mean over their alpha thresholds, giving HOTA `0.746525`, DetA `0.756836`,
AssA `0.736372`, MOTA `0.777981`, CLEAR-F1 `0.890872`, IDF1 `0.783023`, and
IDSW `126`. These are internal-selection metrics; the separate full 19-video
historical evaluation remains in progress.
