# Phase 77 — YesTrack multi-seed stability training

The corrected seed-43 and seed-44 YesTrack Y1 runs are training in parallel on
GPU2 and GPU3. They use the same full 22-video/181-expression Sports training
layout, batch size, one-epoch LoRA schedule, label policy, and NAS-direct
output route as seed42. Each run now has an isolated NAS output root under
`runs/yestrack_y1_sports_lora_seed43_v2/` or
`runs/yestrack_y1_sports_lora_seed44_v2/` and a corresponding local launcher.

The first v1 launch was stopped after an audit found that both seeds resolved
the upstream second-resolution timestamp to the same NAS directory
`runs/20261020_055735`. Both v1 manifests are preserved as exit 143 with no
accepted checkpoint; their logs are retained as collision evidence. The v2
launches avoid this failure mode and are the only runs eligible for the
multi-seed gate.

The historical 19-video test split is not used by these jobs. After each run
finishes, the five-video internal validation split will be evaluated with the
seed42-selected threshold `0.8`; no threshold will be selected from historical
test. Metrics are not claimed until the training and fixed-threshold validation
manifests succeed.

Current status: both v2 training manifests are `RUNNING`. The next evidence
gate is seed-wise internal validation followed by a mean and spread report
across seeds 42, 43, and 44.
