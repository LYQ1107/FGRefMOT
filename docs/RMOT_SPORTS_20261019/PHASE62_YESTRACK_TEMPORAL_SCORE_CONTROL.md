# Phase 62 — causal YesTrack score controls

Status: `IMPLEMENTED_CPU_TESTED_WAIT_FOR_BASELINE`

The project now contains a model-independent causal score controller in
`adapters/temporal_score_control.py` and a directory adapter in
`scripts/apply_temporal_score_variant.py`. The adapter consumes only the
per-track referring score and tracker box rows, keeps each sequence/expression
in a separate namespace, rejects future or duplicate frame updates, and resets
memory after a configurable track gap. It never reads GT or labels.

The registered controls are:

- `YB0`/`YB1`: pass-through score controls; YB1 is paired with the upstream
  YesTrack TCP/TRP inference options exposed by `eval_yestrack_refersports.py`.
- `YB2`: causal adaptive verification interval.
- `YB3`: ordinary causal EMA.
- `YB4`: reliability-controlled score memory.
- `YB5`: long-term identity score memory.
- `YB6`: combined adaptive interval, identity, dynamic, and reliability score
  control.

This is a reproducible score-level ablation bridge, not a claim that a new
learned Qwen3-VL head has already been trained. It remains gated on the real
Sports-LoRA Y1 baselines. CPU tests cover namespace isolation, causal ordering,
track-gap reset, candidate conflict, and output manifests; no research metric
is claimed in this phase.
