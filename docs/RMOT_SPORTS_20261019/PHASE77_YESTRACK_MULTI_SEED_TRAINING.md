# Phase 77 — YesTrack multi-seed stability training

The seed-43 and seed-44 YesTrack Y1 runs are training in parallel on GPU2 and
GPU3. They use the same full 22-video/181-expression Sports training layout,
batch size, one-epoch LoRA schedule, label policy, and NAS-direct output route
as seed42. Their launch manifests and scripts are kept under
`runs/gpu/yestrack_y1_sports_lora_seed43_v1/` and
`runs/gpu/yestrack_y1_sports_lora_seed44_v1/`.

The historical 19-video test split is not used by these jobs. After each run
finishes, the five-video internal validation split will be evaluated with the
seed42-selected threshold `0.8`; no threshold will be selected from historical
test. Metrics are not claimed until the training and fixed-threshold validation
manifests succeed.

Current status: both training manifests are `RUNNING`. The next evidence gate
is seed-wise internal validation followed by a mean and spread report across
seeds 42, 43, and 44.
