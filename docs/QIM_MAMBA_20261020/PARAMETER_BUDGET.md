# Parameter budget

Measured with `dim_in=256`, `hidden_dim=1024`, `qim_mamba_layers=1`,
`qim_mamba_state=16`, and the default M1/M2 bottlenecks (96/128):

| Variant | Trainable QIM parameters |
|---|---:|
| M1 `mamba_ordered_bi` | 1,992,000 |
| M2 `mamba_trackwise` | 2,032,640 |
| M3 `mamba_reliability_hybrid` | 2,099,329 |

These counts include the shared self-attention, FFN, query writer, new
projection layers, Mamba1 stack, and M3 gate/memory projections. They are below
the 2.2M preferred ceiling; final reports will record the exact checkpoint
configuration and trainable-only count.
