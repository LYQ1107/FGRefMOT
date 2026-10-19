# Phase 52 — Conservative baseline metric extraction

Status: `READY_FOR_FORMAL_MANIFESTS`

The baseline-table builder now reads TrackEval's combined result only from a
successful formal evaluation manifest. It extracts HOTA, DetA, AssA, IDF1 and
IDSW while keeping Ref-F1 and FPS as `NOT_RUN` until an evaluator explicitly
records those quantities.

YesTrack emits several score thresholds. The table uses the pre-registered
`0.50` threshold and never selects the best threshold from the historical
evaluation set. Missing, running, malformed, and prepared-only manifests do
not populate metric cells. A regression fixture covers both FlexHook's direct
TrackEval payload and YesTrack's threshold list.

Validation: explicit project tests `tests/*.py` — **29 passed**; syntax and
whitespace checks passed. Current real baseline metrics remain unclaimed while
YesTrack training and formal evaluation are pending.
