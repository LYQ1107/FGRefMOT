# Active-track permutation diagnostic

The diagnostic used the same 20 active tokens, applied reverse/random permutations, and inverse-permuted outputs back to the original track IDs. It ran on GPU 2 with the recovered original Mamba QIM checkpoint. This is a query-level diagnostic; no next-frame detector or SportsMOT HOTA run was possible.

| Variant | permutation | max abs query delta | mean abs query delta |
|---|---|---:|---:|
| Standard | reverse | 7.15e-7 | 8.81e-8 |
| Standard | random | 7.15e-7 | 8.87e-8 |
| LSTM | reverse | 0.04877 | 0.00369 |
| LSTM | random | 0.04917 | 0.00309 |
| Temporal Attention | reverse | 7.15e-7 | 1.32e-7 |
| Temporal Attention | random | 7.15e-7 | 1.32e-7 |
| Original Mamba | reverse | 2.00708 | 0.15710 |
| Original Mamba | random | 2.32181 | 0.16313 |

Standard and attention are effectively permutation-equivariant up to floating-point noise. The direct track-token LSTM and original Mamba are order-sensitive. This supports describing the original Mamba input as an ordered active-track sequence, not as an invariant per-object temporal memory. The effect on boxes, referring scores, ID assignment, and HOTA remains an open experiment once real SportsMOT images are available.

