# Code-change audit

Implemented in the isolated Mamba branch:

- `models/qim_mamba/`: shared Mamba1 stack, deterministic M1 ordering, M2
  valid-history processing, and M3 reliability/memory update.
- `models/qim_variants.py`: new QIM classes and registration, while preserving
  the original Mamba class and its state-dict path.
- `models/transrmot_pro.py`: M3 memory fields and optional short-path text
  handoff.
- `main.py`: variant and Mamba screening controls.
- `tools/qim_ablation/benchmark_mamba_variants.py`: QIM parameter/latency/
  memory benchmark.
- `tests/qim_ablation/`: M1 permutation, M2 padding/causality, and M3 memory
  reliability regression tests.

The initial validation command is:

```text
CUDA_VISIBLE_DEVICES=6 /home/liuyeqiang/anaconda3/envs/mamMOT/bin/python -m pytest -q tests/qim_ablation
```

At implementation time it passed 18 tests. The full-model build check also
constructed M3 and produced all five `qim_*` lifecycle fields.
