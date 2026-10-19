# Phase 46 — Offline-compatible common-tracker runtime

Status: `TRACKER_RUNNING_AFTER_RUNTIME_COMPAT_FIX`.

The image-only Refer-Sports common-tracker run is live on physical GPU 0 while
YesTrack Y1 remains live on physical GPU 1.  The tracker launch uses the 19
Refer-Sports evaluation videos staged as image symlinks only; ground-truth and
label files are omitted, and no tracker metric is claimed from this run.

Two runtime blockers found during the first launches were fixed in the NAS
SambaMOTR source and recorded for reproducibility:

* the research environment has `tensorboardX==2.6.2.2` but no optional
  `tensorboard` package, so the three SambaMOTR logging imports now fall back to
  `tensorboardX`;
* the released checkpoint contains the complete ResNet-50 backbone, so the
  tracker is launched with `SAMBAMOTR_OFFLINE_WEIGHTS=1` and the backbone skips
  torchvision's online ImageNet-weight request.  The checkpoint remains the
  source of the loaded weights.

The active retry was registered at 2026-10-19 17:18:07 Asia/Shanghai with
launcher PID `2225078` and tracker child PID `2226557` on GPU 0.  Its log shows
normal image-only submission progress.  At the phase snapshot YesTrack was at
approximately `8066/103988` steps in its one-epoch run; that training process
was not restarted or modified.

The source tree is intentionally dirty because the existing CUDA compatibility
patch and the runtime fixes are applied over pinned source commit
`f1c139a653c00a55c0873a64d7c59a67d7dbad44`.  The tracked patch file
`patches/sambamotr_runtime_compat.patch` captures the four new runtime files;
the manifest/source hash also includes the pre-existing CUDA patch.

Validation for this phase:

* offline `models.backbone.Backbone` import/build smoke test: passed;
* the active tracker passed the missing-TensorBoard and online-weight gates and
  is still running;
* final tracker success, output count, formal evaluation, and all metrics remain
  pending until the run manifest reports `SUCCEEDED`.
