# Phase 50 — Formal-evaluation tracker-root compatibility

Status: `FIXED_TESTED_YESTRACK_INTERNAL_SUCCEEDED_FORMAL_IN_PROGRESS`

The completed common SambaMOTR tracker stores its 19 MOT files under
`tracker_outputs/` below the tracker run root. The first FlexHook evaluation
preflight assumed the files were directly at the root and stopped before any
model inference. The evaluation adapters now accept both layouts:

- FlexHook resolves each sequence file from either the tracker root or its
  `tracker_outputs/` child and reuses the resolved path when writing
  predictions.
- YesTrack resolves the directory passed to the upstream evaluator before
  launching it, and records that directory in the evaluation payload.

No tracker files were copied, rewritten, or used as ground truth. The
five-video internal YesTrack LoRA evaluation now has a successful manifest
after reusing the completed prediction archive; the threshold-0.8 internal
result is recorded in Phase 56. The separate 19-video formal evaluation is
still running and remains unclaimed until inference and TrackEval finish.

Validation completed in the project environment:

- `tests/*.py`: **26 passed**
- `py_compile`: both evaluation adapters and the watcher passed
- `git diff --check`: passed

Running bare `pytest` is intentionally not the project validation command: it
recursively collects vendored `third_party/` and `runs/vendor_src/` test trees
whose optional dependencies and external fixtures are not part of this
workspace. The explicit project test glob is the recorded boundary.
