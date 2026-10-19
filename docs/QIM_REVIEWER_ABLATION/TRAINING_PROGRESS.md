# Training progress

| Stage | Status | Evidence |
|---|---|---|
| Original source isolation | COMPLETE | branch/worktree from `d651fec...` |
| Historical QIM audit | COMPLETE | `ASSET_AUDIT.md` |
| Four-way QIM interface | COMPLETE | `models/qim_pro.py`, `models/qim_variants.py` |
| N=0/N=1/N>1, lifecycle, gradient, checkpoint, permutation tests | PASS | 8 tests with `python -m pytest -q tests/qim_ablation` |
| Same-video clip sampler | PASS | `test_dataset_sampling.py` |
| Original Mamba strict load | PASS | `test_checkpoint_loading.py` and full-model probe |
| QIM-only latency | COMPLETE | `qim_latency_20261019.json` |
| Full-model engineering probe | COMPLETE | `full_model_probe_active20_20261019.json` |
| 200-step full FGRefMOT training | NOT RUN | Refer-Sports image/label root unavailable |
| 5k/20k/60k/120k training nodes | NOT RUN | Same external asset blocker |
| Refer-Sports validation/test HOTA/DetA/AssA/IDF1/IDSW | NOT RUN | Same external asset blocker |

No training result is marked `FAIR_RETRAINED` or `FINAL_TEST` in this snapshot. The code and step-based controls are ready for activation after the data and provenance-complete tracker assets are supplied.

