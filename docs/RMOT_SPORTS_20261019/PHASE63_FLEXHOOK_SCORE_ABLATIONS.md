# Phase 63 — FlexHook score-level ablation bridge

Status: `COMPLETED_PREBASELINE_DIAGNOSTIC — see Phase 67 for the metrics`

The causal score controller now also exposes H0–H8 aliases for a reproducible
FlexHook post-inference ablation. The mappings are deliberately explicit:

- H0/H1: pass-through control;
- H2: ordinary causal EMA;
- H3: causal adaptive interval;
- H4: stable identity score read;
- H5: short dynamic score history;
- H6: separated identity and dynamic score histories;
- H7: reliability-controlled memory write;
- H8: combined causal control.

These are score-level controls over a frozen Sports-trained checkpoint. They
are useful for isolating temporal effects without changing the fixed tracker
or the model checkpoint, but they are not being reported as a learned
FlexHook temporal head. The separate `adapters/flexhook_temporal/` package
remains the learned-head integration scaffold. The archive and all H0–H8
results completed successfully as a prebaseline diagnostic; they remain gated
from final proposed-method selection until the YesTrack baseline exists. The
evidence table is recorded in
`PHASE67_FLEXHOOK_SCORE_ABLATIONS.md`.
