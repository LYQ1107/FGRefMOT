# Phase 54 — Explicit evaluation-split labels

Status: `READY_FOR_UNAMBIGUOUS_METRIC_REPORTING`

The shared baseline table now includes an `evaluation_split` column. The
current full-22 training runs and their 19-video formal evaluation are labeled
`HISTORICAL_TEST_19`; the historical paper row is labeled `PAPER_REPORTED`.
This keeps the existing metric field names compatible while making it
impossible to read those values as the internal five-video validation result.

Metrics remain populated only from successful formal TrackEval manifests. The
internal `SPORTS_TRAIN_CORE`/`SPORTS_VAL` run will receive its own label when a
model trained on the 17-video core is evaluated. No threshold or checkpoint is
selected from the historical 19-video output.

Validation: targeted baseline, split, and tracker-wrapper tests — **6 passed**;
Python compilation and `git diff --check` passed.
