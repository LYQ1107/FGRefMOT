# Phase 58 — FlexHook formal-evaluation runtime compatibility

Status: `EVALUATION_RUNNING_NO_METRICS_CLAIMED`

The first real FlexHook SportsMOT evaluation reached the model but exposed
runtime differences between the pinned upstream code and the current
environment. The adapter now keeps the vendored checkout unchanged and
handles those differences at the evaluation boundary:

- a non-YAML training artifact named `config.json` is presented to YACS via a
  temporary YAML view;
- the fixed tracker input is resolved from its nested `tracker_outputs/`
  directory when required;
- the removed Transformers `batch_encode_plus` alias is restored through the
  tokenizer's current call API, with explicit truncation for the legacy
  `padding="max_length"` contract;
- the upstream inference loader's `expid` and `expma` text tensors are moved
  to the active CUDA device;
- evaluation sets `sample_frame_len=sample_frame_num=4`, preventing the
  upstream 8-frame sampling pool from indexing past short fixed tracks.

The current real run is on physical GPU3:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_common_eval_20261019_v7`

It uses 19 historical test videos, 142 expressions, batch size 32, four
workers, and the fixed image-only SambaMOTR tracker outputs. Ground truth is
created only in the offline TrackEval tree and is never supplied to the model.
At the time of this report the run has entered inference (`Test: [250/4868]`);
the manifest remains non-success until prediction writing and TrackEval both
finish. No metric is reported from an incomplete run.

Validation before this run: project tests `37 passed`; the evaluator compiles;
the pinned FlexHook checkout is unmodified. The checkpoint and all prediction
artifacts remain on NAS.
