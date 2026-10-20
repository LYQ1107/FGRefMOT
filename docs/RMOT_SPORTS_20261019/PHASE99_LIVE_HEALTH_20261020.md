# Phase 99 — Live health

Status: `FORMAL_EVALUATION_AND_MULTI_SEED_TRAINING_IN_PROGRESS` at 2026-10-20T08:03:48+08:00.

The formal YesTrack is alive on GPU 1 with 67/142 prediction files; its manifest is ABSENT and no historical-test metric is claimed.

Seed 43 on GPU 2 is RUNNING at 18645/90523 steps (loss 0.1265); PID 3078707 is Rl.
Seed 44 on GPU 3 is RUNNING at 18119/90528 steps (loss 0.0223); PID 3078708 is Rl.

The timed common tracker remains terminal (`SUCCEEDED`). The protocol boundary is unchanged: the full train split is used for model fitting, the video-disjoint internal split is used for checkpoint/threshold selection, and the 19-video historical test split is reserved for final offline scoring.

Next: wait for terminal manifests, then run the registered internal validation, timing, and postbaseline ablation steps.
