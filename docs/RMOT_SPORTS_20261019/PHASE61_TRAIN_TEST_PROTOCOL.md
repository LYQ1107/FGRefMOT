# Phase 61 — train/validation/test protocol

The Refer-Sports experiments use a strict split policy.

- **Training:** FlexHook and YesTrack task parameters are trained only on the 22-video SportsMOT training layout. The full YesTrack run has 90,559 optimizer steps and 181 expressions. A separate internal-selection run uses 17 training videos and 136 expressions.
- **Internal validation:** the separate five-video, 45-expression internal split is used only for smoke checks and selecting temporal-method settings. It is not the historical test result.
- **Formal evaluation:** the historical 19-video, 142-expression split is evaluated after training. The fixed SambaMOTR tracker outputs are the only model-time candidate input. Refer-Sports GT is linked into a TrackEval tree after inference and is never passed to the model, used to build tracker inputs, select a checkpoint, or tune a test threshold.

FlexHook F1 has completed both the internal validation and historical formal
evaluation. The internal five-video result is HOTA 71.241265, DetA 64.157324,
AssA 79.107500, IDF1 71.282648, and IDSW 103. Its historical-test result is
HOTA 68.385415, DetA 59.317842, AssA 78.842302, IDF1 66.627791, and IDSW 409.
The full 22-video YesTrack Sports-LoRA run finished successfully with 90,559
optimizer steps and exit code 0. The separate 17-video internal training run
also finished successfully, and its fixed five-video validation evaluation
selected threshold 0.8 from the registered grid. The successful internal
manifest reports mean-over-alpha HOTA `0.746525`, DetA `0.756836`, AssA
`0.736372`, MOTA
`0.777981`, CLEAR-F1 `0.890872`, IDF1 `0.783023`, and IDSW `126`. The formal
historical evaluation has started from the full-run adapter and is recorded
under the registered output root; no full-run YesTrack test metric is claimed
until its evaluation manifest is `SUCCEEDED`. TrackEval HOTA, DetA, and AssA
above use the mean over alpha thresholds, matching the baseline table.

The selected FlexHook H8 temporal head was trained on the 17-video core and
selected on the five-video internal split. Its internal TrackEval completed with
HOTA `0.756316`, MOTA `0.712558`, CLEAR-F1 `0.837294`, IDF1 `0.724347`, and
IDSW `80`. Its one-time 19-video historical evaluation then completed with
HOTA `0.719128`, DetA `0.667403`, AssA `0.774862`, MOTA `0.625810`, CLEAR-F1
`0.799027`, IDF1 `0.677970`, and IDSW `386`. This result remains a selected
temporal-head evaluation; the proposed-method comparison gate is still open.

The exact sequence lists are versioned in `configs/refer_sports_eval_sequences_20261019.txt` and `configs/refer_sports_internal_val_sequences_20261019.txt`. The protocol does not use test annotations during training or model selection.
