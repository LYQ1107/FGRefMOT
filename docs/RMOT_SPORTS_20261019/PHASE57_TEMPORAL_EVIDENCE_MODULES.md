# Phase 57 — Causal temporal evidence module implementation

Status: `IMPLEMENTED_CPU_TESTED_SEQUENCE_ADAPTER_WAIT_FOR_BASELINES`

A project-level FlexHook temporal package is now available under
`adapters/flexhook_temporal/`. It keeps the pinned upstream checkout
unchanged and provides:

- a namespace-aware causal evidence buffer keyed by expression and physical
  track ID;
- separate long-term identity evidence and short-window dynamic evidence;
- a deterministic reliability gate using detection confidence, visibility,
  local evidence quality, identity consistency, and candidate conflict;
- a temporal referring head for H0–H8 with independent per-candidate logits,
  so an expression may select zero, one, or multiple tracks.
- a sequence adapter that writes expression-scoped evidence in frame order and
  fuses a learned residual back into the frozen FlexHook logits without a
  candidate softmax.

The gate deliberately does not use the current referring score as its write
signal. The buffer rejects non-increasing frame indices and prevents one
expression from inheriting another expression's history. The head accepts
optional causal history and reliability values but never normalizes across
candidate tracks.

The implementation is CPU-tested in
`tests/test_flexhook_temporal.py` (six tests), including a future-frame
prefix-invariance check for the sequence adapter. It has not been trained or
attached to a baseline checkpoint and no proposed-method metric is claimed.
Checkpoint integration and all formal H0–H8 runs remain gated on the formal
Sports-trained FlexHook and YesTrack baselines and the fixed evaluation
protocol.
