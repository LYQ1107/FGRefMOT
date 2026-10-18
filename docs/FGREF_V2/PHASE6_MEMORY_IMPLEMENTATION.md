# Phase 6 QIM-v2 implementation

Status: `IMPLEMENTED_AND_UNIT_TESTED`, `INTEGRATION_SMOKE_PASS`,
`TRAINING_NOT_RUN`.

## Components

The new temporal path is in `models/fgref_v2/`:

- `temporal_evidence.py` provides EMA, GRU, temporal attention, and optional
  Mamba encoders over `[N_tracks, T_history, D]`. The time axis is explicit;
  masked slots cannot affect the summary.
- `track_memory.py` provides
  `ReliabilityAwareTrackMemory` with fixed-shape identity memory and dynamic
  history. Writes use detection quality, visibility, feature conflict, and
  match confidence. `refer_score` is intentionally absent from the write
  gate.
- `qim_writer.py` provides `EvidenceQueryWriter`, which writes identity,
  current observation, history, and predicted-box evidence into the original
  `[pos, content]` query format.
- `qim_v2.py` composes causal read-before-write behavior and an
  `Instances` adapter, `ReliabilityAwareQueryInteraction`.

The optional model flags are:

```text
--fgref_v2_qim
--fgref_v2_qim_memory_dim 128
--fgref_v2_qim_history_length 8
--fgref_v2_qim_temporal_mode {ema,gru,attention,mamba}
--fgref_v2_qim_update_query_pos
```

When enabled, `_generate_empty_tracks()` creates fixed-size state fields:

```text
fgref_v2_identity_memory
fgref_v2_identity_memory_valid
fgref_v2_dynamic_history
fgref_v2_dynamic_history_mask
fgref_v2_observation_quality
fgref_v2_memory_age
```

The adapter replaces the legacy row-wise QIM hook and leaves
`RuntimeTrackerBase` as the owner of physical ID creation and termination.
Rows are re-aligned by `obj_idxes`; new/negative IDs receive clean state, and
deleted IDs cannot transfer their state to the next birth. In the opt-in
QIM-v2 model path the legacy STReasoner is not stacked on top; its temporal
loss keys are explicit zeros so the existing training logger does not mistake
missing supervision for a measured loss.

## Causality

At frame `t`, temporal history is read from state ending at `t-1`. The current
observation is written only afterward. A missing observation is masked in both
the dynamic history and identity write path; a zero placeholder cannot create
an identity memory update. Identity memory can remain valid while a track is
temporarily unobserved or no longer satisfies the current language query.

## Verification

The focused CPU suite passes:

`33 passed, 1 warning`

The tests include causal masking, empty histories, ID reorder, state reset,
missing observation, new-track birth, termination, and `Instances` adapter
alignment. The modules and adapter are not a performance result; no online
tracking metric or speedup is claimed before the baseline gate is cleared.

The combined current-source integration smoke can be rerun with
`tools/fgref_v2/current_forward_smoke.py` (default: TAM-v2 plus QIM-v2,
random initialization, one real Refer-KITTI sample). It completed with the
QIM-v2 adapter active and the expected `(1,300,1)`, `(1,300,4)`, and
`(1,300,1)` output shapes. The 2.0063-second/1871.44-MiB measurement is an
integration diagnostic only; QIM-v2 has no trained metric yet.
