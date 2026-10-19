# Phase 73 — FlexHook temporal-head evaluation adapter

Status: `IMPLEMENTED_NOT_RUN`

`scripts/eval_flexhook_temporal.py` evaluates a selected temporal checkpoint
without changing the pinned FlexHook model or the fixed SambaMOTR tracker. It
collects the final query and text features, groups observations by video,
physical track and expression text, sorts them by frame, and applies the causal
head before writing the standard MOT/TrackEval tree.

The command supports internal validation for checkpoint selection and the
historical test set for one final offline evaluation. It never supplies GT to
the model or tracker. No real temporal checkpoint or metric has been produced;
the adapter has only passed helper tests and awaits the strict-core feature
archives.
