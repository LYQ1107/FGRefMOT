# Phase 44 — Pre-registered experiment matrix

Status: `REGISTERED_BASELINES_IN_PROGRESS`.

`configs/experiment_matrix_20261019.json` registers the common-tracker
protocol, requested metrics, seeds, baseline variants and the H0–H8 FlexHook
ablation sequence.  The validator rejects GT or label files as tracker input,
rejects test-GT tuning/checkpoint selection, and requires every proposed
variant to remain `WAIT_FOR_BASELINE` until formal FlexHook and YesTrack
baselines exist.

This phase does not run a proposed method and does not add a result.  It makes
the later comparison auditable: all referring methods use the same 19-video
SambaMOTR output set, and all incomplete entries remain explicit statuses.
