# Phase 25 — Real-data smoke and GPU allocation audit

Audit date: 2026-10-19  
Status: **REAL-DATA SMOKE READY / GPU 1 IS THE ONLY IDLE CANDIDATE / NO GPU JOB STARTED**

Both method-specific layouts were exercised with the dedicated environment and
the current local/NAS assets. These were data-loader checks only; they did not
load Qwen weights, build either model, read tracker outputs as ground truth, or
write predictions.

FlexHook command:

```text
/data3/liuyeqiang/.venvs/fgrefmot-rmot-sports-20261019/bin/python scripts/flexhook_sportsmot_smoke.py --layout-root runs/flexhook_sports_layout_20261019_v2 --tokenizer local --roberta-dir /data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/flexhook/roberta-base --json
```

It passed with the Transformers compatibility alias installed. The train and
validation sample counts were `5491` and `1195`. The first item in both splits
had image shape `[4, 3, 224, 672]`, box-map shape `[4, 2, 16, 48]`, coordinate
shape `[4, 4]`, expression count `8`, and token/attention shapes `[8, 25]`.

YesTrack command:

```text
/data3/liuyeqiang/.venvs/fgrefmot-rmot-sports-20261019/bin/python scripts/yestrack_sportsmot_smoke.py --layout-root runs/yestrack_sports_layout_20261019_v1 --sequence-limit 1 --json
```

It passed in supervised-sample-only mode for one sequence per split. The first
train and validation samples each contained one 384×384 image and a valid
normalized coordinate/text record. The bounded sample counts were `48444` for
the selected train sequence and `31362` for the selected validation sequence.

The read-only `nvidia-smi` audit found ten A100 40GB cards. GPU 1 had 0% GPU
utilization, 17 MiB in use, and no compute process, so it is the only current
candidate for this study. GPUs 0, 2, 3, 4, 5, 6, 7, 8, and 9 each had an
active `/data2/yejiatong/miniconda3/envs/habitat-llm/bin/python` process and
were left untouched. The selection is provisional and must be rechecked
immediately before each launch.

The smoke entry points are pinned by SHA-256:

* `scripts/flexhook_sportsmot_smoke.py` —
  `e353c85e1f507e3f7da2fd3dcc339b89a752f27157ef2ecab9c0bb2d9f412283`;
* `scripts/yestrack_sportsmot_smoke.py` —
  `e0b48886a9f433bad454aa515a0db195eca1987fec7446731be5a146c844f725`.

The exact layout manifests used by the smoke are recorded in the JSON report.
No training, evaluation, or tracker-output mutation occurred in this phase.
