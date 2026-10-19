# Phase 40 — Dynamic post-training tracker watcher

Status: `PREPARED_TRAINING_IN_PROGRESS`.

`scripts/post_training_tracker_watch.py` replaces the earlier GPU1-only shell
watcher.  It waits for the YesTrack Y1 process to exit with the recorded
success marker, verifies that a tracker run has not already succeeded, and then
selects any physical GPU with no compute application and less than 1 GiB in use.
It invokes the pinned SambaMOTR Refer-Sports launcher with that GPU and keeps
the tracker output root on the NAS.

The watcher does not kill, pause, or reconfigure other jobs.  A busy machine is
handled by polling the GPU snapshot until a safe device is available.  The
training command and model paths remain unchanged.

Validation completed:

- Python compilation: `PASS`
- CLI help: `PASS`
- Existing 23 project tests remain the relevant scoped test suite
- Current Y1 training process is live; no tracker process has been started by
  this watcher yet
