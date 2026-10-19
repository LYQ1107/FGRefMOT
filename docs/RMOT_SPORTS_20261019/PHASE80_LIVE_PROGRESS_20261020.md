# Phase 80 — Live YesTrack training and formal-evaluation progress

Observed at `2026-10-20T06:13:40+08:00`.

The repaired seed43 and seed44 launchers completed model initialization and
entered the one-epoch optimizer loop on separate GPUs and separate NAS output
roots. The logs showed 160/90,523 and 167/90,528 optimizer steps respectively;
neither run has a terminal manifest yet. Their internal-validation watchers are
still waiting for the corresponding training manifests.

The seed42 full-model formal historical evaluator is still alive on GPU1. It
has written 50 prediction files out of the registered 142 expressions, but its
`YESTRACK_EVAL_MANIFEST.json` has not been written. These files are progress
evidence only and do not constitute a test metric.

The train/internal-selection/historical-test boundary remains unchanged:
training uses the registered 22-video train split, threshold 0.8 is inherited
from the five-video internal selection split, and historical test annotations
are not used for selection. No multi-seed or formal historical metric is
claimed from this checkpoint.
