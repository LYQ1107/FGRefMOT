# Phase 38 — YesTrack Refer-Sports formal-evaluation adapter

Status: `PREPARED_GPU_EVAL_PENDING`.

`scripts/eval_yestrack_refersports.py` runs the pinned YesTrack
`llm_eval_from_mot.py` in ByteTrack-input mode and evaluates its nested
per-expression outputs on the same fixed 19-video SambaMOTR tracker set.  The
adapter maps the existing Refer-Sports MOT GT manifest to the YesTrack output
directory convention, applies the pre-registered referring thresholds
`0.2..0.8`, and invokes the pinned YesTrack TrackEval checkout for HOTA, DetA,
AssA, CLEAR/ID switches, and IDF1.

Frame conversion is explicit: SambaMOTR files are 1-based while the YesTrack
image layout is 0-based, so inference uses `frame_offset=-1` and emits
1-based prediction rows for metric evaluation.  GT is only linked into the
offline metric tree; the model sees the tracker rows and image crops only.

A synthetic one-sequence evaluation with six expressions completed through
TrackEval.  No synthetic numbers are used as research results.  The real run
waits for the current YesTrack Y1 training and the fixed Refer-Sports tracker
generation.
