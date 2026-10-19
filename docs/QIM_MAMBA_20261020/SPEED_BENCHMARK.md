# Speed benchmark

Initial V100 QIM-only benchmark (`CUDA_VISIBLE_DEVICES=6`, five warmups and
twenty timed repetitions per point) is recorded in
`reports/qim_mamba/speed_benchmark_v100.json`. It uses synthetic active-track
tensors and therefore is an engineering latency measurement, not full-model
image FPS.

| Variant | QIM params | N=5 mean ms | N=20 mean ms | N=50 mean ms |
|---|---:|---:|---:|---:|
| M1 Ordered Bi-Mamba | 1,992,000 | 8.59 | 7.45 | 5.48 |
| M2 Trackwise Mamba1 | 2,032,640 | 6.15 | 19.12 | 49.04 |
| M3 Reliability Hybrid | 2,099,329 | 10.08 | 24.72 | 47.18 |

The values include the shared QIM core and the variant mixer. Full-model
prepared/e2e FPS, real-image warmup, CUDA synchronization, and peak memory are
scheduled against the matched checkpoints after the 5k/20k accuracy reports.
