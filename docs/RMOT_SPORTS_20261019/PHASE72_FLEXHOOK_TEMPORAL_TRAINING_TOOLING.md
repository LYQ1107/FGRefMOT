# Phase 72 — FlexHook causal temporal-head training tooling

Status: `IMPLEMENTED_NOT_RUN`

`scripts/train_flexhook_temporal.py` trains the project-level causal temporal
head from the frozen `train_core` feature archive and selects its checkpoint
using only the separate `internal_val` archive. It groups each physical
track/expression chronologically, rejects historical-test archives, keeps the
FlexHook logits frozen, and records archive hashes, seed, variant, losses and
the selected checkpoint in `TEMPORAL_TRAINING_MANIFEST.json`.

The command uses the saved visual query and pooled language representations,
and the evidence buffer is updated before each prediction so a sequence cannot
see future frames. Labels are consumed as training targets only. The synthetic
unit tests pass, but no real archive or learned-method metric exists yet; the
strict core FlexHook checkpoint and its internal evaluation must finish first.
