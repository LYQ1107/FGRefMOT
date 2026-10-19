# Phase 37 — FlexHook Refer-Sports formal-evaluation adapter

Status: `PREPARED_GPU_EVAL_PENDING`.

The upstream FlexHook evaluation entrypoint assumes a LaMOT directory and
hard-codes several paths.  `scripts/eval_flexhook_refersports.py` provides the
project-side adapter needed for the fixed Refer-Sports tracker set.  It keeps
the vendored FlexHook checkout unchanged, runs its test-mode data loader with
the 19 Refer-Sports sequence names, writes per-expression MOT files, and
invokes the pinned TrackEval implementation with HOTA, CLEAR, and Identity
metrics.

The adapter creates ground-truth rows only in an offline evaluation tree.  The
model receives image crops and boxes read from the common SambaMOTR tracker
files; labels are never used to create or filter tracker rows.  The preparation
path was exercised with one synthetic MOT row per sequence and produced 142
expression records and 142 full sequence-length entries.  The synthetic run
was deleted from the temporary directory and produced no research result.

The real run is intentionally deferred until the fixed Refer-Sports
SambaMOTR outputs exist and GPU1 is free after YesTrack Y1 training.
