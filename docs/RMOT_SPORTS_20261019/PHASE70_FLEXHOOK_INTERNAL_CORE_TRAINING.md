# Phase 70 — FlexHook strict internal-core training launch

Status: `RUNNING`

The strict internal-selection FlexHook run is `flexhook_f1_internal_core_seed42`.
Its contract passed the hash and policy gate before launch. The run trains on
the 17-video/136-expression core split and reserves the 5-video/45-expression
internal split for checkpoint selection. The historical 19-video/142-expression
test split remains an offline TrackEval-only dataset and is not visible to the
model, tracker, selector or threshold tuner.

The job started at `2026-10-19T23:07:28+08:00` on physical GPU3. Its output
root is the NAS path
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_internal_core_seed42`;
the local run directory stores the manifest and log. The launch manifest and
contract record the exact source checkout, layout, tracker policy, seed and
hyperparameters.

An observed epoch-6 internal validation reached `F1=71.575` on the reserved
5-video/45-expression split, and candidate checkpoint
`ckpt_epoch_best_2.pth` was hashed as
`9f54677eb8eab677801720c2c9949d9f6e9c9b17a88cfdde13f2f65c179f080c`. This is
an in-training diagnostic candidate, not the final selected checkpoint or a
historical-test claim; training is continuing at epoch 7/25. The next required
evidence is a completed training manifest followed by final internal selection
without reading historical test ground truth.
