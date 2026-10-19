# Phase 69 — FlexHook real-checkpoint temporal feature hook smoke

Status: `COMPLETED_ONE_BATCH_SMOKE`

The new `scripts/flexhook_feature_hook_smoke.py` loads the verified
Sports-trained FlexHook F1 checkpoint and runs one batch from the fixed
SambaMOTR track layout. It registers a forward hook on the final
`model.output_ca_qkv[-1]` block without changing the vendored checkout or
writing a checkpoint.

The smoke succeeded with checkpoint SHA256
`3fdc471d1fa17405ce8d57516bc6de74ba93b29f3760344bb1f0f30529818c7b`. The
loader contains 155,748 records. The captured representation has shape
`[1, 8, 768]`; the unmodified model output has shape `[1, 4, 8, 2]`. This
confirms a real feature/logit interface for the causal temporal adapter, while
the baseline's fixed tracker remains the only model-time candidate source.

This phase is an interface smoke only. It does not create the full feature
archive, train a temporal checkpoint, tune on historical test data, or claim a
learned-method metric. The next gate is a complete feature record capture on
the registered internal train/core-validation splits after the YesTrack
baseline gate.
