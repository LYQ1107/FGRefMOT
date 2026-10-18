# Phase 5 TAM-only results

Status: `TRAINING_INTERFACE_SMOKE_PASS`, `TAM_PERFORMANCE_NOT_ESTABLISHED`.

This phase is intentionally separate from QIM-v2. The detector, decoder,
backbone, historical Mamba-QIM, and legacy temporal reasoner were not changed
by the optimization smoke below; only the newly initialized TAM-v2 parameters
were trainable.

## Training-interface smoke

Launcher:

```text
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tools/fgref_v2/tam_checkpoint_train_smoke.py --device cuda --steps 2
```

Inputs and output:

```text
base checkpoint: /data1/liuyeqiang/fgref_v2_runs/checkpoints/checkpoint0089.pth
base checkpoint SHA256: adb2ad9591a4a14dcdd05718b271d6c7b391e97bd9cb20f4fd8e3bfe26e54038
data root: /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti
split: refer-kitti.train (training sample only)
output: /data1/liuyeqiang/fgref_v2_runs/tam_smoke/tam_seed17.pt
```

The current source accepted all 1,210 checkpoint model keys. The only missing
keys were the expected 27 newly initialized `fgref_v2_tam.*` parameters; no
unexpected keys or shape mismatches were accepted. The four-frame real sample
smoke produced nonzero TAM gradients:

| Step | Differentiable loss sum | Gradient norm |
| ---: | ---: | ---: |
| 0 | 7.258664 | 28.925404 |
| 1 | 6.590466 | 27.274350 |

The repeated-frame referring loss terms decreased during these two updates,
which only verifies a live optimization path. It is not a validation gain,
does not establish a TAM contribution, and is not used in a benchmark table.

The same two-step interface smoke was repeated with the locally staged data
root for seeds 29 and 43. All three runs accepted the 1,210-key base
checkpoint with only the expected 27 new TAM keys missing and produced
nonzero gradients:

| Seed | Step-0 loss | Step-1 loss | Step-0 grad norm | Step-1 grad norm | Class |
| ---: | ---: | ---: | ---: | ---: | --- |
| 17 | 7.258664 | 6.590466 | 28.925404 | 27.274349 | `TRAINING_INTERFACE_SMOKE` |
| 29 | 8.452438 | 7.572502 | 43.669483 | 36.235447 | `TRAINING_INTERFACE_SMOKE` |
| 43 | 3.141252 | 2.719410 | 18.245129 | 16.683441 | `TRAINING_INTERFACE_SMOKE` |

These are three-seed optimization-path checks, not three-seed validation or
benchmark results; no TAM gain is inferred.

## Required controlled matrix

The following are the planned frozen-backbone/TAM-only comparisons. They must
use the same restored B0 checkpoint, training sample protocol, seed set, and
legal validation split once the split audit is complete:

| ID | Head/evidence path | Status |
| --- | --- | --- |
| T0 | Original referring head | checkpoint/online baseline pending |
| T1 | Capacity-matched pooled MLP | implementation pass; run pending |
| T2 | Ordinary cross-attention | implementation pass; run pending |
| T3 | Pair head with language/track evidence | implementation pass; run pending |
| T4 | T3 + fixed local evidence | implementation pass; run pending |
| T5 | T4 + text-conditioned sampling | implementation pass; run pending |
| T6 | T5 + semantic routing | conditional candidate; run pending |

No HOTA, IDF1, Referring F1, hard-negative FPR, or counterfactual accuracy is
reported here until these controls are trained and evaluated under a verified
protocol. The 2-step smoke is classified as `TRAINING_INTERFACE_SMOKE`, not
`NEW_EXPERIMENT`.
