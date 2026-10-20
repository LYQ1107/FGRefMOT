# GPU allocation — 2026-10-20

This is the live allocation for the final FGRefMOT fast-QIM run.  Physical GPU
IDs are explicit because `CUDA_VISIBLE_DEVICES` is used for every detached
process.

| role | variant | physical GPU | process | output |
|---|---|---:|---:|---|
| F1 | `fast_mamba_spatial` | 0 | 1953 | `fg_v2_runs/qim_fast6_rmot_fulltest_20261020/fast_mamba_spatial` |
| F2 | `fast_mamba_temporal` | 2 | 1954 | `fg_v2_runs/qim_fast6_rmot_fulltest_20261020/fast_mamba_temporal` |
| F3 | `fast_mamba_streaming` | 3 | 1955 | `fg_v2_runs/qim_fast6_rmot_fulltest_20261020/fast_mamba_streaming` |
| F4 | `fast_mamba_motion` | 4 | 1956 | `fg_v2_runs/qim_fast6_rmot_fulltest_20261020/fast_mamba_motion` |
| F5 | `fast_mamba_language` | 5 | 1957 | `fg_v2_runs/qim_fast6_rmot_fulltest_20261020/fast_mamba_language` |
| F6 | `fast_mamba_reliability` | 7 | 1958 | `fg_v2_runs/qim_fast6_rmot_fulltest_20261020/fast_mamba_reliability` |
| evaluator | original `standard/lstm/temporal_attention/mamba` 5k | 8 | 2101 | `fg_v2_runs/qim_fast6_rmot_fulltest_20261020/original_5k_full_rmot` |

Configuration shared by the six training jobs:

- common non-QIM checkpoint: `/data1/liuyeqiang/fgref_v2_runs/checkpoints/checkpoint0089.pth`
- checkpoint SHA256: `adb2ad9591a4a14dcdd05718b271d6c7b391e97bd9cb20f4fd8e3bfe26e54038`
- `qim_fast_dim=64`, `qim_fast_state=8`, `qim_fast_layers=1`, `qim_fast_history_len=4`
- `train_qim_only`, `reset_qim`, 20,000 optimizer updates
- save nodes: 200, 5,000, 10,000, 20,000
- `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `num_workers=0`

GPU 1 was excluded because it contained an unrelated stale process; GPU 6 was
excluded because it contained unrelated users' work. GPU 9 remains spare. No
process is allowed to share a training GPU in this run.

This allocation was verified with `nvidia-smi` and process ownership before
launch. The process IDs above are the launch-time IDs; later monitoring records
must be treated as authoritative if a child process replaces one of them.
