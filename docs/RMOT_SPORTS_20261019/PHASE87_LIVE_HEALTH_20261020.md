# Phase 87 — Live health

Status: `FORMAL_EVAL_AND_MULTI_SEED_TRAINING_HEALTHY` at 08:00:37 CST.

The formal YesTrack historical evaluator remains alive on GPU 1. It has
written 67 of the expected 142 prediction files, but the terminal
`YESTRACK_EVAL_MANIFEST.json` is still absent. These partial files are not
scored as a test result.

The isolated multi-seed trainings remain active on GPU 2 and GPU 3. Seed 43
has reached 18,091 of 90,523 optimizer steps and seed 44 has reached 17,595 of
90,528. Both logs and GPU processes are updating; neither run has produced a
terminal manifest or accepted checkpoint yet.

The timed common tracker is already terminal and recorded in Phase 86. The
train/test protocol is unchanged: training uses the registered train split,
internal validation is used for selection, and historical test ground truth is
offline-only after selection.
