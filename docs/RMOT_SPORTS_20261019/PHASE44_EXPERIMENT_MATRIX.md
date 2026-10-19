# Phase 44 — Pre-registered experiment matrix

Status: `FLEXHOOK_FORMAL_COMPLETE_YESTRACK_EVALUATIONS_IN_PROGRESS`.

`configs/experiment_matrix_20261019.json` registers the common-tracker
protocol, requested metrics, seeds, baseline variants and the H0–H8 FlexHook
ablation sequence.  The validator rejects GT or label files as tracker input,
rejects test-GT tuning/checkpoint selection, and requires every proposed
variant to remain `WAIT_FOR_BASELINE` until formal FlexHook and YesTrack
baselines exist.

FlexHook F1 now has a completed formal manifest. Both registered YesTrack Y1
training runs succeeded, and the internal and 19-video formal evaluations are
in progress. This phase still does not
run a proposed method or add a proposed-method result. All incomplete entries
remain explicit statuses, and the matrix records the evidence paths for the
completed/in-progress baselines.
