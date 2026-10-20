# Six lightweight Mamba-QIM architectures

All six use the common QIM interface and the current run's `d_model=64`,
`d_state=8`, one Mamba1 layer, history length 4, and the existing 256-D query
input/output projections. The QIM-only trainable totals observed at startup
are below 2.2M in every case.

| variant | mechanism | persistent state | trainable QIM parameters |
|---|---|---|---:|
| F1 `fast_mamba_spatial` | geometry-ordered same-frame track-set Mamba | none | 1,890,432 |
| F2 `fast_mamba_temporal` | batched valid-history Mamba, last valid state | short history | 1,890,432 |
| F3 `fast_mamba_streaming` | per-track streaming state update | per-track stream state | 1,890,432 |
| F4 `fast_mamba_motion` | six-signal motion gate around spatial Mamba | previous predicted box | 1,890,689 |
| F5 `fast_mamba_language` | text-conditioned scale/bias modulation of temporal feature | short history; text feature is read-only input | 1,923,329 |
| F6 `fast_mamba_reliability` | short dynamics plus reliability-gated identity memory | per-track identity memory | 1,924,225 |

The parameter totals include the complete QIM module: self-attention, FFN,
projections, Mamba layers, and gates. They are not just the new SSM weights.

F1 is explicitly a lightweight same-frame track-set mixer; it is not claimed
to be temporal memory. F2 reuses a bounded history in one batched operation.
F3 keeps one logical state per physical track ID. F4 uses predicted motion
signals only, never GT. F5 reads already-produced text features and does not
instantiate a second text encoder. F6 updates identity memory with an
observation-reliability gate and handles birth/missing/termination through the
existing track fields.

The parameter gate has passed. Latency, full-model FPS, and RMOT accuracy are
not inferred from parameter count and remain pending measurement.
