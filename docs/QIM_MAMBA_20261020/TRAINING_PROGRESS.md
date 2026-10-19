# Training progress

The branch is ready for the first matched screening run. The original
four-way QIM training remains active in the separate worktree and is not
modified by this branch.

Planned matched run:

- source: `/data/DATASETS/TRACKING/JDE/SportsMOT` via the existing Refer-Sports
  adapter and `unknown_mask` protocol;
- initialization: the common non-QIM weights from the existing step-200
  controlled checkpoint, with `--reset_qim --train_qim_only`;
- variants: M1/M2/M3, one GPU per process, same train split, clip length,
  optimizer and step checkpoints;
- milestones: 200 engineering, 5k development evaluation, then 20k formal
  mid-run; promising variants continue to 60k/120k.

No accuracy result is claimed in this file until the corresponding checkpoint
and fixed Development-Val evaluator report exist. The M1, M2, and M3 step-200
reports now exist.

The 200-step engineering runs completed for all three variants. Observed
training benchmark means were 1,223.93 ms/step for M1, 1,342.28 ms/step for
M2, and 1,350.23 ms/step for M3; each saved a valid step-200 checkpoint. The
M1 full step-200 development report on the frozen four-video split records
HOTA 0.5798, AssA 0.6121, IDF1 0.5814, and IDSW 102. The three exact-resume
processes are running in parallel toward step 20,000. M2's full step-200
report records HOTA 0.0476, AssA 0.4420, IDF1 0.0074, and IDSW 673; M3 records
HOTA 0.0755, AssA 0.2999, IDF1 0.0194, and IDSW 765. Both reports contain
large false-positive counts on the four-video split, so they are real
diagnostic results rather than accepted accuracy claims. Their checkpoints
and reports are retained while score calibration/track lifecycle behavior is
diagnosed.
