# Phase 68 — FlexHook learned temporal sequence adapter integration path

Status: `IMPLEMENTED_CPU_TESTED_WAIT_FOR_BASELINE_CHECKPOINT`

`adapters/flexhook_temporal/adapter.py` now provides the chronological
integration path for the learned temporal head. It accepts caller-supplied
per-frame candidate features, fixed-tracker boxes, expression language
features, and optional frozen FlexHook logits. The adapter updates identity,
dynamic history, and reliability state for each expression/track key before
evaluating the current frame. A two-class upstream logit receives an
antisymmetric residual, preserving the independent Bernoulli candidate
contract and avoiding a candidate softmax.

The sequence API enforces increasing frame indices and supports static or
per-frame language and track IDs. The CPU test suite includes a prefix
invariance check: changing future frames leaves earlier outputs unchanged.
Compilation and six focused tests pass. The upstream `third_party/FlexHook`
checkout remains untouched.

This is an integration path, not a result. The adapter has not been trained
on Sports data, has not been attached to a learned checkpoint, and claims no
H0–H8 metric. After the formal YesTrack baseline and the internal selection
gate are complete, the next step is to capture real FlexHook features on the
registered internal train/core-validation splits, train the chosen temporal
variant, and only then run the historical test evaluation.
