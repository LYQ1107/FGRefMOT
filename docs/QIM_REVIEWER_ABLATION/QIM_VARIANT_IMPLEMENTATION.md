# QIM variant implementation

The original `QueryInteractionModule` class, constructor names, Mamba blocks, fused normalization path, active-track selection, reference-point update, and `Instances.cat([init_track_instances, active_track_instances])` output are preserved. `models/qim_pro.py::build()` only dispatches on the new `--qim_variant` flag.

All variants use the same:

- input fields and `Instances` shape;
- training/evaluation active-track selection (`obj_idxes` / score threshold and IoU rule);
- reference-point update and query position/content dimensions;
- lifecycle fields and RuntimeTrackerBase thresholds;
- call timing, Random Drop and FP helper rules.

The three missing historical implementations are explicit reimplementations:

| Variant | Mixer | Sequence axis | Label |
|---|---|---|---|
| Standard | MOTR-style self-attention + FFN | active-track token set | `REIMPLEMENTED_VARIANT` |
| LSTM | four-layer unidirectional LSTM stacks before/after the common FFN | active-track token order, matching Mamba's `[1,N_active,256]` axis | `REIMPLEMENTED_VARIANT:track-token-lstm` |
| Temporal Attention | four-layer attention stacks before/after the common FFN | active-track token set, not physical frame history | `REIMPLEMENTED_VARIANT:active-token-attention` |
| Mamba | untouched original Mamba QIM | `[1,N_active,256]` active-track sequence | `ORIGINAL_MAMBA` |

The original code creates `inference_params = None` on every Mamba pass and does not carry a persistent cross-frame Mamba state. Therefore the active-track axis is not described as a true temporal axis or long-term per-target memory.

