# Phase 41 — Post-tracker formal-evaluation watcher

Status: `PREPARED_TRAINING_IN_PROGRESS`.

`scripts/post_tracker_eval_watch.py` waits for the fixed Refer-Sports tracker
manifest to become `SUCCEEDED`, discovers the Y1 LoRA directory on NAS, and
then runs the YesTrack and FlexHook formal-evaluation adapters sequentially on
one safe GPU.  The adapters keep the same common tracker output and write their
results under NAS run roots.  Existing successful evaluation manifests are
respected, so a restarted watcher does not overwrite a completed evaluation.

The watcher polls rather than reserving a GPU.  It requires no test GT for
inference; GT is used only by each adapter's offline metric tree.  The current
training and dynamic tracker watchers were left untouched while this watcher
was prepared; it will be started only after the tracker watcher has a verified
successful manifest.

Validation completed:

- Python compilation: `PASS`
- CLI help: `PASS`
- formal GPU evaluations: `PENDING_YESTRACK_AND_TRACKER_COMPLETION`
