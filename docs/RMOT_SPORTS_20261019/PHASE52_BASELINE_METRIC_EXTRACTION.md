# Phase 52 — Conservative baseline metric extraction

Status: `READY_FOR_SUCCESSFUL_FORMAL_MANIFESTS`

The baseline-table builder now reads TrackEval's combined result only from a
successful formal evaluation manifest. It extracts HOTA, DetA, AssA, IDF1 and
IDSW while keeping Ref-F1 and FPS as `NOT_RUN` until an evaluator explicitly
records those quantities.

YesTrack emits several score thresholds. The table uses the threshold selected
on the five-video internal validation split and reuses that fixed value for the
19-video historical test. The current internal selection is `0.80` by HOTA;
the builder retains `0.50` only as a compatibility fallback for older
manifests that do not record a selection. It never selects a threshold from the
historical evaluation set. Missing, running, malformed, and prepared-only
manifests do not populate metric cells. A regression fixture covers both
FlexHook's direct TrackEval payload and YesTrack's threshold list.

Validation: the baseline-table regression tests pass (**8 passed**), the full
project suite passes (**78 passed, 2 deprecation warnings**), and syntax and
whitespace checks pass. YesTrack training and internal validation have
succeeded, while the historical evaluation remains in progress, so its final
test metric cells remain unclaimed.
