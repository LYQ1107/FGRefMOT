# Phase 72 — FlexHook causal temporal-head training tooling

Status: `REAL_TRAINING_SUCCEEDED_INTERNAL_CHECKPOINT_SELECTED`

`scripts/train_flexhook_temporal.py` trains the project-level causal temporal
head from the frozen `train_core` feature archive and selects its checkpoint
using only the separate `internal_val` archive. It groups each physical
track/expression chronologically, rejects historical-test archives, keeps the
FlexHook logits frozen, and records archive hashes, seed, variant, losses and
the selected checkpoint in `TEMPORAL_TRAINING_MANIFEST.json`.

The command uses the saved visual query and pooled language representations,
and the evidence buffer is updated before each prediction so a sequence cannot
see future frames. Labels are consumed as training targets only. The real H8
run completed eight epochs on 10,319 train-core sequences and selected epoch 5
from 614 internal-val sequences (`best_val_loss=0.2843093469826403`). The
selected checkpoint is stored on the NAS at
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_temporal_h8_internal_core_20261019/temporal_head_best.pt`
with SHA-256
`5c3d5c39115f786390e23e86dd96763eac5d77028d43347f00f03c550d2008b9`. Its
internal TrackEval is running; no historical-test metric is claimed until that
selection/evaluation record is complete.
