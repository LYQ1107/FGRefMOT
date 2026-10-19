# Phase 51 — YesTrack duplicate-expression output mapping

Status: `FIXED_TESTED_TRAINING_RUNNING`

The Refer-Sports validation layout contains two identical sentences in
`v_4LXTUim5anY_c010`. YesTrack's upstream evaluator disambiguates those output
directories by appending `__2` to the second normalized sentence. The adapter
previously reconstructed only the unsuffixed directory, which would have
failed or mis-associated one expression during formal evaluation.

The adapter now assigns duplicate sentence directory names in per-video input
order using the same suffix convention as the upstream evaluator. A regression
test covers the mapping, and the existing nested common-tracker path tests
remain green.

Validation: explicit project test glob `tests/*.py` — **27 passed**; syntax and
whitespace checks passed. No model inference or metric result is claimed in
this phase; YesTrack training remains active.
