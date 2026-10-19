# Phase 61 — train/validation/test protocol

The Refer-Sports experiments use a strict split policy.

- **Training:** FlexHook and YesTrack task parameters are trained only on the 22-video SportsMOT training layout. The full YesTrack run has 90,559 optimizer steps and 181 expressions. A separate internal-selection run uses 17 training videos and 136 expressions.
- **Internal validation:** the separate five-video, 45-expression internal split is used only for smoke checks and selecting temporal-method settings. It is not the historical test result.
- **Formal evaluation:** the historical 19-video, 142-expression split is evaluated after training. The fixed SambaMOTR tracker outputs are the only model-time candidate input. Refer-Sports GT is linked into a TrackEval tree after inference and is never passed to the model, used to build tracker inputs, select a checkpoint, or tune a test threshold.

FlexHook F1 has completed both the internal validation and historical formal
evaluation. The internal five-video result is HOTA 71.241265, DetA 64.157324,
AssA 79.107500, IDF1 71.282648, and IDSW 103. Its historical-test result is
HOTA 68.385415, DetA 59.317842, AssA 78.842302, IDF1 66.627791, and IDSW 409.
The full 22-video YesTrack Sports-LoRA run is still training. The separate
17-video internal training run finished successfully; its fixed five-video
internal evaluation is running from the completed adapter. The historical
formal evaluation remains blocked until the full run and internal selection
are complete.

The exact sequence lists are versioned in `configs/refer_sports_eval_sequences_20261019.txt` and `configs/refer_sports_internal_val_sequences_20261019.txt`. The protocol does not use test annotations during training or model selection.
