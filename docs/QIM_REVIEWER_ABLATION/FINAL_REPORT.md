# FGRefMOT original QIM ablation — current report

Status: **FORMAL DEVELOPMENT RUN IN PROGRESS / HISTORICAL TEST GT STILL MISSING**.

Completed in this snapshot:

- isolated original-source branch from `d651fec...`;
- historical QIM audit and checkpoint SHA256 record;
- unified Standard, LSTM, Temporal Attention, and untouched Original Mamba interfaces;
- same-video clip sampling fix;
- QIM gradient/lifecycle/checkpoint/permutation tests (8/8 passed);
- CUDA-event QIM latency and full-model fixed-resolution engineering probes;
- step-based initialization/freezing/resume flags and normal-loop memory-cleanup fix;
- reviewer-facing implementation and limitation records.
- verified local SportsMOT image/label/expression assets and global-ID agreement;
- real SportsMOT loader, UNKNOWN-safe referring mask, fixed development split, and evaluator;
- one successful real-data full-model training smoke.
- four real-data, QIM-only 200-step runs with the same optimizer/data protocol;
- fixed development-split smoke evaluation for all four QIM variants.

Still in progress or blocked:

- completion of the 5k/20k/60k/120k checkpoints (four nodes are running continuously);
- full development validation over the frozen protocol;
- historical Test HOTA/DetA/AssA/IDF1/IDSW, because the local Test tree has no matching GT;
- multi-seed/partial-unfreeze experiments;
- formal paper-level Full-model FPS on real SportsMOT clips.

## Step-200 development smoke

These are real metrics from the frozen four-video development validation list, one expression per video, and the first 30 frames per video. They are a smoke check, not a final paper table.

| QIM | HOTA | HOTA@0.5 | DetA | AssA | IDF1 | IDSW | Referring F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.2118 | 0.2213 | 0.0617 | 0.7318 | 0.1102 | 11.5 | 0.0801 |
| LSTM | 0.1300 | 0.1464 | 0.0978 | 0.1741 | 0.0777 | 41.5 | 0.1414 |
| Temporal Attention | 0.9077 | 0.9495 | 0.8723 | 0.9480 | 0.9529 | 0.0 | 0.4739 |
| Original Mamba | 0.9338 | 0.9787 | 0.9237 | 0.9445 | 0.9855 | 0.0 | 0.4942 |

The machine-readable results are `reports/qim_ablation/*_step200_dev_smoke.json`. The original Mamba 200-step log recorded one no-gradient QIM batch as an explicit no-op; it did not alter the optimizer state or fabricate a metric.

