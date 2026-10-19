# Phase 82 — Live YesTrack progress and timed common-tracker run

Observed at `2026-10-20T06:40:13+08:00`.

The isolated YesTrack seed43 and seed44 jobs remain in the optimizer loop, at
4,547/90,523 and 4,653/90,528 live tqdm steps. Their launchers use
`--every_steps 0`, so no epoch LoRA checkpoint exists yet; the internal
watchers continue to wait for terminal training manifests and `lora_epoch_0`.

The seed42 formal historical evaluator remains alive on GPU1 and has written 58
prediction files out of the registered 142 expressions. Its formal manifest is
still absent, so these files are progress evidence only and are not a test
metric.

A separate timed common SambaMOTR run started on idle GPU0 in
`common_tracker_refersports_timed_20261020`. It uses an image-only staging view
with GT and label files omitted and writes a new NAS manifest so tracker wall
time and physical FPS can be joined to the referring-module timings. It is
independent of the existing tracker outputs and the live YesTrack jobs.

The split boundary is unchanged: the 22-video train split is used for training,
the five-video internal split selects the fixed threshold 0.8, and historical
test annotations remain unused for selection.
