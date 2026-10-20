# RMOT evaluation audit

## Protocol

- historical recovered test: 19 videos, 142 expressions;
- historical 141-compatible result: the same 19 videos, 141 expressions;
- frame/expression pairs are counted from the generated GT and prediction
  manifests, not estimated from the number of expressions;
- inference is expression-conditioned and runs the complete video from the
  first frame after resetting tracker state;
- `inference_uses_gt: false` — GT is used only by the separate scoring stage;
- thresholds are fixed at detection score `0.0` and referring score `0.4` for
  this run.

## Generated GT

- full recovered GT: `/data1/liuyeqiang/fg_v2_runs/qim_fast6_rmot_fulltest_20261020/rmot_gt_full_142`
- historical-compatible GT: `/data1/liuyeqiang/fg_v2_runs/qim_fast6_rmot_fulltest_20261020/rmot_gt_compat_141`
- generation script: `scripts/build_refer_sports_mot_gt.py`
- generation results: 142/19 and 141/19 respectively.

The evaluator filters model outputs using the model's score, referring score,
box, and physical track ID fields. It does not use GT filtering, GT ID repair,
or oracle detection. TrackEval metrics are computed offline from the saved
per-expression predictions and the corresponding expression-specific GT.

## Run status at launch

The original four-model 5k evaluator was launched sequentially on physical GPU
8. The six fast-QIM 20k jobs were launched on GPUs 0, 2, 3, 4, 5, and 7.
Metrics are deliberately `PENDING` until each inference and scoring process
completes. No training loss is reported as an RMOT metric.
