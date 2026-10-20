# Phase 105 — Live health

Status: `FORMAL_EVALUATION_AND_MULTI_SEED_TRAINING_IN_PROGRESS` at 2026-10-20T08:15:17+08:00.

The formal YesTrack evaluator remains alive on GPU 1 with 72/142 prediction files. The read-only partial audit covers 481,312 non-empty rows, 1,825 unique frames (frame IDs 1–1825), and 35 track IDs with 0 format/identifier issues and 0 non-finite values. A three-second activity probe increased the evaluator child's `rchar` by 11,176,365 bytes and `syscr` by 240; its terminal manifest is still absent, so no historical-test metric is claimed.

Seed 43 on GPU 2 is RUNNING at 20,605/90,523 optimizer steps (loss 0.0213); seed 44 on GPU 3 is RUNNING at 20,008/90,528 steps (loss 0.0361). Their manifests have no `finished_at`, and no multi-seed metric or stability claim is made.

The timed common tracker remains terminal (`SUCCEEDED`). The protocol boundary is unchanged: the full train split is used for model fitting, the video-disjoint internal split is used for checkpoint/threshold selection, and the 19-video historical test split is reserved for final offline scoring.

Next: wait for terminal manifests, then run fixed-threshold internal validation, timing joins, and the registered postbaseline ablations.
