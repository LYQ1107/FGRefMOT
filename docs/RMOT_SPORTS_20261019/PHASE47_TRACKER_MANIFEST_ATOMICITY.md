# Phase 47 — Atomic live-state manifest for the common tracker

Status: `TRACKER_RUNNING; MANIFEST_ATOMICITY_PATCH_VALIDATED`.

The active common SambaMOTR run remains on physical GPU 0 and the YesTrack Y1
run remains on physical GPU 1.  At this snapshot the tracker was at roughly
`v_-6Os86HzwCs_c009` frame `408/500`, with seven of the 19 videos observed as
complete; YesTrack was at `10231/103988` steps.  Neither run was restarted.

The tracker launcher previously wrote `TRACKER_RUN_MANIFEST.json` only after
inference ended.  That left a stale terminal status visible to a watcher while
a new invocation was already running.  The launcher now atomically publishes a
`RUNNING` manifest before starting the subprocess and adds `run_id`, launcher
PID, and UTC start/finish timestamps.  Failure and success states are also
published atomically through a same-directory temporary file and `os.replace`.
This prevents partial JSON reads and makes future watcher decisions depend on
the current invocation.

The current invocation predates this launcher patch, so its old manifest is
allowed to remain until that process finishes.  The detached evaluation gate
uses the current launcher's PID together with the success state and will start
formal evaluation only after the current tracker succeeds; it cannot be
confused by the old `FAILED` manifest.

Validation:

* `py_compile scripts/run_sambamotr_refersports_tracker.py`: passed;
* atomic manifest smoke test (temporary file, no partial remains): passed;
* full test suite: `23 passed`;
* no metrics are claimed while the tracker or YesTrack training is running.
