# Training and inference speed profile

Hardware: `Tesla V100-DGXS-32GB`, GPU 2, FP32. The requested A100 was not present on this server; this hardware difference is recorded rather than hidden.

## Measured QIM-only profile

CUDA events, 20 warm-up calls and 50 timed calls per active-track count. The QIM input contract is the original 256-dimensional active-track token. Values below are mean latency at `N_active=20`; full raw results are in `reports/qim_ablation/qim_latency_20261019.json`.

| Variant | QIM params | mean ms | median ms | p95 ms | peak allocated GB |
|---|---:|---:|---:|---:|---:|
| Standard | 1,907,712 | 2.2608 | 2.2600 | 2.3047 | 0.0204 |
| LSTM | 6,119,424 | 4.6162 | 4.5804 | 4.8533 | 0.0742 |
| Temporal Attention | 4,017,152 | 5.7407 | 5.7242 | 5.8599 | 0.0283 |
| Original Mamba | 5,413,888 | 10.9502 | 10.8995 | 11.2000 | 0.0336 |

These are QIM-only measurements, not full-model FPS.

## Full-model engineering probe

The original Backbone, TAM/deepfuse, visual-language fusion, deformable decoder, STReasoner, RuntimeTracker, and selected QIM were executed at 360x640 for two synthetic frames. After frame 1, 20 returned slots were marked active solely to exercise the QIM path. The probe uses the checkpoint's full RoBERTa weights with an offline fixed-token adapter; image decode/resize and SportsMOT files are not included. It is an engineering measurement, not a Refer-Sports result.

| Variant | Full params | frame mean ms | prepared-tensor FPS | peak allocated GB |
|---|---:|---:|---:|---:|
| Standard | 178,730,476 | 89.0539 | 11.2292 | 0.8088 |
| LSTM | 182,942,188 | 83.1928 | 12.0203 | 0.8252 |
| Temporal Attention | 180,839,916 | 83.7305 | 11.9431 | 0.8196 |
| Original Mamba | 182,236,652 | 85.1227 | 11.7477 | 0.8248 |

Mamba is not automatically faster end-to-end on this V100 probe. The result must be repeated on the target GPU and real SportsMOT clips before making a paper efficiency claim.

## Training loop change

`engine.py` now uses `optimizer.zero_grad(set_to_none=True)` and removes unconditional `gc.collect()`/`torch.cuda.empty_cache()` from normal iterations. Frozen BN/Dropout modules are set to evaluation mode while the QIM path remains differentiable. A benchmark mode records mean/median/p95 step time, data/forward/backward time, and peak allocated/reserved memory. No BF16 result is claimed yet.

