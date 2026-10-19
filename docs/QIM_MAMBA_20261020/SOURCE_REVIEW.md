# Lightweight Mamba-QIM source review

Status: implementation and screening branch `research/qim-lightweight-mamba-20261020`.

The untouched original QIM remains `models/qim_pro.py::QueryInteractionModule`
and is still selected by `--qim_variant mamba`. The three new registrations are
in `models/qim_variants.py`:

| Label | CLI variant | Backend |
|---|---|---|
| M1 | `mamba_ordered_bi` | geometry-ordered bidirectional Mamba1 |
| M2 | `mamba_trackwise` | one causal Mamba1 sequence per track |
| M3 | `mamba_reliability_hybrid` | M2 plus reliability-gated identity memory |

The initial implementation uses `mamba_ssm.modules.mamba_simple.Mamba` only.
Mamba2 is an explicit opt-in that is intentionally rejected until a separate
V100 compatibility/forward/backward benchmark is completed.

The SportsMOT source and the existing four-way fairness protocol are inherited
from the reviewer-ablation branch. Historical official Test GT remains a
separate data gap and is not used or inferred here.
