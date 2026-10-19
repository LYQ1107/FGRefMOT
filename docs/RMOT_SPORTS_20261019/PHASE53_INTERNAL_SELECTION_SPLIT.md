# Phase 53 — Internal video-disjoint selection split

Status: `READY_INTERNAL_SPLIT_AND_TRACKER_PREPARED`

The historical Refer-Sports archive has 22 training videos and 19 held-out
videos. The current corrected YesTrack run was intentionally launched against
the existing 22-video training layout, so its later 19-video formal evaluation
is a historical comparison run. It must not be described as a validation
result or used to choose checkpoints, thresholds, or proposed-method settings.

This phase creates the selection split required by the study protocol without
touching the historical 19-video set:

- `SPORTS_TRAIN_CORE`: 17 videos, 136 expressions, 11,482 frames
- `SPORTS_VAL`: 5 videos, 45 expressions, 3,268 frames
- sport counts: train core basketball/soccer/volleyball = 5/6/6; validation =
  2/1/2
- the holdout is fixed and video-disjoint: `v_-6Os86HzwCs_c001`,
  `v_1LwtoLPw2TU_c006`, `v_1yHWGw8DH4A_c029`, `v_2j7kLB-vEEk_c005`, and
  `v_ApPxnw_Jffg_c001`

The deterministic lists and source manifest are recorded under
`configs/internal_split_20261019/`. FlexHook and YesTrack layouts were built
under `runs/flexhook_sports_layout_internal_20261019/` and
`runs/yestrack_sports_layout_internal_20261019/`; frames remain symlinks to the
raw dataset. The converted YesTrack labels are dataset-loader annotations only.
The common SambaMOTR tracker staging view contains images only and has no GT or
label files. Its internal-validation manifest is prepared at
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_dab_refersports_internal_val_20261019/TRACKER_RUN_MANIFEST.json`
with status `PREPARED_NO_GPU_RUN`; inference will start when an allocated GPU
is available.

Validation: `tests/*.py` — **31 passed**; targeted Python compilation and
`git diff --check` passed. No metric is claimed from the internal split yet.
