# FGRefMOT original QIM ablation — current report

Status: **STEP-200, STEP-5000, AND STEP-20000 FULL DEVELOPMENT EVALUATIONS COMPLETE / 60K-120K LONG RUN IN PROGRESS / HISTORICAL TEST GT STILL MISSING**.

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
- full step-200 development evaluation for all four QIM variants over 4 videos and
  2,621 frames;
- full step-5000 development evaluation for all four QIM variants over the same
  4 videos and 2,621 frames;
- full step-20000 development evaluation for all four QIM variants over the same
  4 videos and 2,621 frames;
- Standard, LSTM, and Temporal Attention 60k checkpoints with matched full
  development evaluations;
- Mamba step-200 full-development result pushed to the research branch.

Still in progress or blocked:

- completion of Original Mamba 60k and the four 120k checkpoints (the long-run
  nodes are still running continuously);
- multi-clip paper-level efficiency and training-cost aggregation;
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

## Step-200 full development evaluation

These are real metrics from the same frozen four-video development validation list,
one expression per video, all frames, and the UNKNOWN-safe referring protocol. The
four sequences contain 2,621 frames in total. They are development results, not
historical Test results; the local Test tree still has no matching ground truth.

| QIM | HOTA | HOTA@0.5 | DetA | AssA | IDF1 | IDSW sum | IDSW mean/sequence | Referring F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.0432 | 0.0454 | 0.0124 | 0.1534 | 0.0078 | 1391 | 347.75 | 0.0714 |
| LSTM | 0.0284 | 0.0309 | 0.0809 | 0.0108 | 0.0065 | 4113 | 1028.25 | 0.2360 |
| Temporal Attention | 0.6376 | 0.6704 | 0.6259 | 0.6707 | 0.6723 | 44 | 11.00 | 0.7664 |
| Original Mamba | 0.7328 | 0.7730 | 0.8695 | 0.6254 | 0.7458 | 74 | 18.50 | 0.9368 |

The exact per-sequence results are stored in
`reports/qim_ablation/{standard,lstm,temporal_attention,mamba}_step200_dev_full.json`.

## Step-5000 full development evaluation

These are the first long-run milestone results from the exact-resume four-way
training jobs. They use the same frozen four-video development list, one
expression per video, all 2,621 frames, and the UNKNOWN-safe referring protocol.
They are development results, not historical Test results.

| QIM | HOTA | HOTA@0.5 | DetA | AssA | IDF1 | IDSW sum | IDSW mean/sequence | Referring F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.6720 | 0.7058 | 0.7019 | 0.6506 | 0.7028 | 65 | 16.25 | 0.9091 |
| LSTM | 0.6519 | 0.6834 | 0.6726 | 0.6462 | 0.6754 | 62 | 15.50 | 0.8606 |
| Temporal Attention | 0.5432 | 0.5681 | 0.5008 | 0.5999 | 0.5148 | 116 | 29.00 | 0.7382 |
| Original Mamba | 0.7222 | 0.7587 | 0.8233 | 0.6378 | 0.7517 | 75 | 18.75 | 0.9695 |

The exact machine-readable results are
`reports/qim_ablation/{standard,lstm,temporal_attention,mamba}_step5000_dev_full.json`;
the cross-variant summary is `reports/qim_ablation/step5000_dev_summary_20261019.json`.
The exact machine-readable results for the next milestone are
`reports/qim_ablation/{standard,lstm,temporal_attention,mamba}_step20000_dev_full.json`;
the cross-variant summary is `reports/qim_ablation/step20000_dev_summary_20261019.json`.

## Step-20000 full development evaluation

These are exact-resume long-run results on the same frozen four-video
development list, one expression per video, all 2,621 frames, and the
UNKNOWN-safe referring protocol. They remain development results, not
historical Test results.

| QIM | HOTA | HOTA@0.5 | DetA | AssA | IDF1 | IDSW sum | IDSW mean/sequence | Referring F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 0.6878 | 0.7246 | 0.7269 | 0.6568 | 0.7172 | 59 | 14.75 | 0.9197 |
| LSTM | 0.7406 | 0.7787 | 0.8459 | 0.6550 | 0.7721 | 75 | 18.75 | 0.9669 |
| Temporal Attention | 0.5633 | 0.5902 | 0.4789 | 0.6829 | 0.5955 | 30 | 7.50 | 0.6781 |
| Original Mamba | 0.7236 | 0.7600 | 0.8054 | 0.6556 | 0.7540 | 76 | 19.00 | 0.9231 |

At this milestone LSTM has the highest development HOTA/IDF1, while Temporal
Attention has the lowest IDSW sum. These rankings are not promoted to a final
paper conclusion until the 60k/120k nodes and their matched efficiency/cost
records are complete.

## Step-60000 full development evaluation (partial milestone)

The Standard, LSTM, and Temporal Attention 60k checkpoints have completed the
same frozen four-video development evaluation (2,621 frames, one expression per
video, UNKNOWN-safe referring protocol). Standard reports HOTA 0.6770,
HOTA@0.5 0.7115, DetA 0.7318, AssA 0.6343, IDF1 0.7129, IDSW sum 58, and
Referring F1 0.8968. LSTM reports HOTA 0.6622, HOTA@0.5 0.6944, DetA 0.7371,
AssA 0.5999, IDF1 0.6966, IDSW sum 67, and Referring F1 0.8875. Temporal
Attention reports HOTA 0.5374, HOTA@0.5 0.5603, DetA 0.5040, AssA 0.5795,
IDF1 0.5518, IDSW sum 75, and Referring F1 0.7031. The exact machine-readable
results are `reports/qim_ablation/standard_step60000_dev_full.json`,
`reports/qim_ablation/lstm_step60000_dev_full.json`, and
`reports/qim_ablation/temporal_attention_step60000_dev_full.json`. Original
Mamba's 60k checkpoint/evaluation and all 120k evaluations are pending;
therefore this remains a partial milestone and is not used to declare a final
long-run ranking.

## Step-200 real SportsMOT image benchmark

This is a full-model engineering benchmark on native images from
`v_-6Os86HzwCs_c001`, with 60 frames total, 5 warmup frames, and 55 timed
frames on a Tesla V100-DGXS-32GB in FP32. `prepared GPU` excludes image
decode/validation transform; `wall` includes them. These are speed and memory
measurements, not additional accuracy claims.

| QIM | Prepared GPU mean / p95 (ms) | Prepared GPU FPS | Wall mean / p95 (ms) | Wall FPS | Peak allocated (GB) |
|---|---:|---:|---:|---:|---:|
| Standard | 132.58 / 151.02 | 7.54 | 177.38 / 203.41 | 5.64 | 2.244 |
| LSTM | 137.70 / 161.03 | 7.26 | 180.32 / 215.36 | 5.55 | 1.985 |
| Temporal Attention | 126.05 / 142.10 | 7.93 | 168.38 / 192.81 | 5.94 | 1.919 |
| Original Mamba | 130.99 / 152.51 | 7.63 | 173.35 / 197.21 | 5.77 | 1.925 |

The machine-readable benchmark is
`reports/qim_ablation/real_sportsmot_benchmark_step200.json`. Long-run
training-cost evidence is tracked in
`reports/qim_ablation/training_cost_20261019.json`. At the current snapshot, the
step-5000 and step-20000 checkpoint wall-clock observed from the step-200
checkpoint are 1.713/1.772/1.760/1.845 and 6.971/7.212/7.174/7.454 GPU-hours
for Standard, LSTM, Temporal Attention, and Original Mamba respectively. The
Standard, LSTM, and Temporal Attention 60k checkpoints are additionally
observed at 22.136, 22.990, and 23.004 GPU-hours respectively; the Original
Mamba 60k value and all 120k fields remain missing until their actual
checkpoints and logs exist. These are checkpoint-timestamp observations, not
power-meter measurements.
No ETA or extrapolated cost is presented as a measured result.

## Long-run no-gradient handling

The live long-run logs currently expose explicit QIM-only no-gradient batches at
Standard steps 1535, 3316, and 13539, and Original Mamba steps 3624 and 12472.
These are the intentional `NO_GRAD_STEP` branch in `engine.py`: the loss has no
trainable QIM path for that batch, so the optimizer step is skipped rather than
calling backward on a scalar without an autograd graph. They are recorded in
`training_cost_20261019.json` and are not treated as crashes or fabricated
metrics.

