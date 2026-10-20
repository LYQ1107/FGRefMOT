# Phase 100 — Live health

Status: `FORMAL_EVALUATION_AND_MULTI_SEED_TRAINING_IN_PROGRESS` at 2026-10-20T08:04:51+08:00.

The formal YesTrack historical evaluator remains alive on GPU 1 with 68/142 prediction files. Its terminal manifest is `ABSENT`; these partial files are not a test metric.

Seed 43 on GPU 2 is RUNNING at 18826/90523 steps (loss 0.0149); PID 3078707 is Sl.
Seed 44 on GPU 3 is RUNNING at 18290/90528 steps (loss 0.0176); PID 3078708 is Rl.

The timed common tracker remains terminal (`SUCCEEDED`). The registered split boundary is unchanged: full train for fitting, the video-disjoint internal split for checkpoint/threshold selection, and the 19-video historical test split for final offline scoring.

Next: wait for terminal manifests, then run fixed-threshold internal validation, timing joins, and postbaseline ablations.
