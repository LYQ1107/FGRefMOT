# Phase 63 — FlexHook score-level ablation bridge

Status: `IMPLEMENTED_CPU_TESTED_WAIT_FOR_SCORE_ARCHIVE`

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
remains the learned-head integration scaffold. The H runs are gated on the
raw per-candidate probability archive and will use the same historical test
TrackEval tree once that archive finishes.
