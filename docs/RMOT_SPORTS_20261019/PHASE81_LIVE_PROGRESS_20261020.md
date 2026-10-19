# Phase 81 — Live progress and checkpoint boundary

Observed at `2026-10-20T06:24:21+08:00`.

The isolated YesTrack seed43 and seed44 jobs are still in the optimizer loop,
at 1,924/90,523 and 1,974/90,528 live tqdm steps. The launchers use
`--every_steps 0`, so no epoch LoRA checkpoint exists yet; the internal
watchers correctly wait for the terminal training manifests and the complete
`lora_epoch_0` output instead of evaluating a partial model.

The seed42 formal historical evaluator remains alive on GPU1 and has written 54
prediction files out of the registered 142 expressions. Its formal manifest is
still absent, so these files are not a test metric.

The split boundary is unchanged: the 22-video train split is used for training,
the five-video internal split selects the fixed threshold 0.8, and historical
test annotations remain unused for selection.
