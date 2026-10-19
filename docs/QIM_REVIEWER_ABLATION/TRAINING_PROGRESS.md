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
| SportsMOT + expression audit | PASS | `sportsmot_audit_20261019.json`; 169,354 positive frames matched |
| 1-step real SportsMOT full-model smoke | PASS | QIM-only Standard; 2.293 s/step; 4.72 GB peak allocated |
| 200-step QIM-only four-way runs | COMPLETE | V100 GPUs 2–5; fixed 4-frame clips; all four step-200 checkpoints present |
| Step-200 development smoke | COMPLETE | fixed 4-video split, 30 frames/video, one expression/video; JSON results committed beside this report |
| Step-200 full development evaluation | COMPLETE | fixed 4-video split, all 2,621 frames, one expression/video; four JSON results committed and pushed |
| Step-5000 full development evaluation | COMPLETE | fixed 4-video split, all 2,621 frames, one expression/video; four JSON results and cross-variant summary |
| Real SportsMOT image benchmark | COMPLETE | 60-frame real clip, 5 warmup + 55 timed frames; `real_sportsmot_benchmark_step200.json` |
| 20k/60k/120k training nodes | RUNNING | all four variants resumed exactly from step 200; 5k checkpoints and full development evaluations are complete |
| Historical Test HOTA/DetA/AssA/IDF1/IDSW | BLOCKED | test images exist, corresponding local GT labels do not |

The four 200-step and four 5k benchmark logs/checkpoints are under
`/data1/liuyeqiang/fg_v2_runs/qim_ablation_sports_20261019/`. No result is
marked `FINAL_TEST`; later long-run checkpoints will be promoted only after
their actual logs and validation outputs are preserved. The development
metrics are recorded in `FINAL_REPORT.md` and `FINAL_TRACKING_RESULTS.csv`.

