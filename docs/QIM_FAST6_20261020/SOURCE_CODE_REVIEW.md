# Source-code review — fast6 run

## Frozen invariants

The six variants reuse the existing TransRMOT backbone, TAM/decoder,
STReasoner, loss construction, active-track selection, query writer, physical
track IDs, and tracker lifecycle. The new code is registered through
`--qim_variant`; the four original QIM implementations remain available.

The common initialization is loaded from the non-QIM checkpoint and QIM
parameters are reset for this fair comparison. No new backbone, TAM, decoder,
or language model is trained.

## Changed implementation surface

- `models/qim_fast6/`: six lightweight mixers, shared utilities, and state
  management.
- `models/qim_mamba/`: preserved lightweight historical Mamba modules used by
  the three previous lightweight variants.
- `models/qim_variants.py`: compatibility registration for the previous three
  variants and the six fast variants.
- `models/transrmot_pro.py`: lifecycle-safe initialization and propagation of
  streaming, motion, memory, and text fields on `Instances`.
- `main.py`: variant and fast-QIM training options.
- `scripts/` and `tools/qim_fast6/`: historical RMOT GT, inference, scoring,
  launch, and audit utilities.

## Evidence already completed

- `python -m py_compile` passed for the changed Python modules.
- `pytest -q tests/qim_fast6`: 27 passed.
- `pytest -q tests/qim_ablation`: 8 passed.
- Historical RMOT helper tests: 2 passed.
- Six variants passed forward, backward, permutation, lifecycle, language,
  reliability, and no-GT evaluation smoke tests.
- All six current training jobs entered the real optimizer loop.

## Leakage and lifecycle rules

Inference receives only the video frames and the expression text. Ground-truth
files are read only by the offline scoring process. The referring filter uses
the model's `scores`, `refers`, `boxes`, and `obj_idxes`; it does not use GT IDs
or oracle detections. New state fields are ordinary `Instances` fields so that
birth, concatenation, indexing, device movement, and termination follow the
existing lifecycle protocol.

Final metric claims remain pending until the full inference and scoring jobs
finish.
