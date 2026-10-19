# Phase 45 — Parallel common-tracker run and checkpoint-aware evaluation watcher

Status: `TRACKER_GPU0_RUNNING; YESTRACK_GPU1_RUNNING`.

The YesTrack Y1 one-epoch run remained healthy on physical GPU 1.  Because the
fixed SambaMOTR tracker is independent of the referring-model checkpoint, it
was started separately on physical GPU 0 after a fresh `nvidia-smi` check found
no compute process and only 16 MiB in use.  Its staging tree exposes image
symlinks only; the tracker manifest keeps `used_ground_truth: false`.

The post-tracker evaluation watcher was corrected before this parallel launch.
After a successful tracker manifest it now waits for `lora_epoch_0` to appear
and, when a training log is supplied, exits with an explicit failure if the
training wrapper reports a non-zero exit.  This prevents a valid early tracker
completion from causing the formal evaluation watcher to terminate before the
YesTrack checkpoint exists.

At registration time the training PID, tracker PID, and evaluation watcher PID
were live.  No other user's GPU process was stopped or modified.  The tracker
run and evaluation watcher write their logs and manifests under the declared
project/NAS roots.

Validation:

* `python -m py_compile scripts/post_tracker_eval_watch.py`
* `git diff --check`
* focused tests: `6 passed`

This phase adds no metric or completion claim.  Final tracker and model
evaluation statuses remain pending until their manifests report success.
