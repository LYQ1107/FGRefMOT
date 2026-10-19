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

The epoch-11 internal validation reached `F1=71.200` (`Acc@1=79.436`,
precision `57.622`, recall `93.149`) on the reserved 5-video/45-expression
split. It is below the current best epoch-8 diagnostic (`F1=73.702`), so the
candidate checkpoint remains `ckpt_epoch_best_0.pth`, hashed as
`af56903c9a6da854b01fd6b0b6c5b4f25f1a263ed9309d13bb99d5ccc1199e8a`. This is
an in-training diagnostic candidate, not the final selected checkpoint or a
historical-test claim; training is continuing at epoch 12/25. The next required
evidence is a completed training manifest followed by final internal selection
without reading historical test ground truth.
