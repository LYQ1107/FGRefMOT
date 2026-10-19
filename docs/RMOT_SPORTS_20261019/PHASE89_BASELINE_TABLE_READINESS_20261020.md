# Phase 89 — Baseline-table readiness

Status: `PASSED_PARTIAL_ROWS_EXPECTED` at 07:13:27 CST.

The conservative baseline-table builder ran successfully and produced nine
rows under the ignored live-run directory
`runs/baseline_table_live_20261020/`. Its policy requires completed manifests
for all validation and historical metrics.

The current table contains FlexHook F1's completed validation HOTA `71.241265`
and historical HOTA `68.385415`. It contains YesTrack Y1's completed internal
HOTA `74.652451` but leaves its historical HOTA as `NOT_RUN`, with status
`TRAINING_SUCCEEDED_FORMAL_EVAL_PENDING`, because the formal terminal manifest
does not exist yet.

This verifies the post-run extraction path while preserving the no-inference
boundary for partial YesTrack outputs. The table is not a final comparison.
