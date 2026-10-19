# Phase 50 — Formal-evaluation tracker-root compatibility

Status: `FIXED_TESTED_TRAINING_RUNNING`

The completed common SambaMOTR tracker stores its 19 MOT files under
`tracker_outputs/` below the tracker run root. The first FlexHook evaluation
preflight assumed the files were directly at the root and stopped before any
model inference. The evaluation adapters now accept both layouts:

- FlexHook resolves each sequence file from either the tracker root or its
  `tracker_outputs/` child and reuses the resolved path when writing
  predictions.
- YesTrack resolves the directory passed to the upstream evaluator before
  launching it, and records that directory in the evaluation payload.

No tracker files were copied, rewritten, or used as ground truth. The formal
evaluation output directories remain empty/prepared until the valid YesTrack
checkpoint is available; the dependency gate is still waiting on training.

Validation completed in the project environment:

- `tests/*.py`: **26 passed**
- `py_compile`: both evaluation adapters and the watcher passed
- `git diff --check`: passed

Running bare `pytest` is intentionally not the project validation command: it
recursively collects vendored `third_party/` and `runs/vendor_src/` test trees
whose optional dependencies and external fixtures are not part of this
workspace. The explicit project test glob is the recorded boundary.
