# Research gate audit — 2026-10-20

Status: `IN_PROGRESS`.

This audit records the evidence boundary for the Refer-Sports study at the
observation time in the JSON companion. The registered protocol has 22
training videos/181 expressions, a video-disjoint 17-video/136-expression
training core plus 5-video/45-expression internal validation split, and a
19-video/142-expression historical test split. Historical test ground truth
is offline TrackEval input only after checkpoint and method selection.

The environment, NAS/data audit, split registry, fixed SambaMOTR tracker
manifests, FlexHook F1 formal baseline, reproducible temporal-module tooling,
read-only efficiency measurement tooling, and the fail-closed FlexHook
resume-state patch have evidence in the cited phase reports. The strict
FlexHook core training and five-video internal TrackEval selection are now
complete, and a real patched one-GPU resume smoke run also completed: it loaded
epoch 1, continued at epoch 2, saved a checkpoint, and exited successfully.
The feature archives are complete, and the H8 causal temporal head has
completed eight epochs with epoch 5 selected from internal validation. Its
five-video internal TrackEval succeeded (HOTA 0.7563, IDF1 0.7243), and the
selected checkpoint's one-time 19-video historical evaluation also succeeded
(HOTA 0.7191, IDF1 0.6780, IDSW 386). The full 22-video YesTrack training
finished successfully and its formal historical evaluation has started. The
separate 17-video internal YesTrack training and fixed five-video validation
evaluation also succeeded; threshold 0.8 was selected on internal HOTA
(0.7465, mean over the TrackEval alpha grid).
The FlexHook frozen-score H0-H8 replay is a diagnostic control over a frozen
checkpoint; it is not a learned temporal-head result.

The following gates remain open: YesTrack formal evaluation, proposed-method
and baseline ablations, multi-seed stability, complete online efficiency,
final-test evidence for the remaining variants, and the final reviewer report.
The internal YesTrack metric and the selected H8 historical metric are recorded
as split-specific evidence; they must not be presented as a completed full
baseline comparison or a learned-method improvement claim. Phase 76 has
registered a postbaseline H0–H8 watcher that remains idle until the full
YesTrack historical manifest succeeds.

Phase 78 added the plan-required final report entrypoints and reviewer evidence
ledger. They intentionally retain `NOT_RUN` and `IN_PROGRESS` fields for the
live YesTrack historical evaluation, postbaseline controls, multi-seed runs,
complete online efficiency, and final comparison.

The machine-readable gate states, evidence paths, live manifests and next
gates are in `RESEARCH_GATE_AUDIT_20261020.json`. This report intentionally
preserves `IN_PROGRESS`/`NOT_RUN` boundaries instead of inferring results
from partial logs or training loss.

Phase 79 records a validity repair for the multi-seed route: the first seed43
and seed44 attempts shared the upstream second-resolution NAS timestamp and
were stopped before any checkpoint was accepted. Isolated v2 roots are now
running on GPU2 and GPU3; the historical test evaluator on GPU1 was left
untouched.

Phase 80 records that both isolated jobs entered the optimizer loop after NAS
model initialization. Phase 81 records the later live checkpoint boundary:
seed43 and seed44 had reached 1,924 and 1,974 steps at observation, while the
formal evaluator had produced 54/142 prediction files without writing a
terminal manifest. Because these runs save the LoRA adapter at epoch end, the
internal watchers continue to wait for a terminal training manifest and
`lora_epoch_0`; no partial checkpoint or partial historical output is used as a
metric.

Phase 82 records the 06:40 observation: seed43 and seed44 had reached 4,547
and 4,653 steps, the formal evaluator had produced 58/142 partial prediction
files, and no terminal formal manifest existed. It also records the separate
GPU0 timed tracker run, which uses an image-only staging view and is intended to
supply the missing tracker wall-time/FPS evidence; none of these partial states
is treated as a metric.

Phase 83 records a 06:48 health check: the formal evaluator was still active
with 58/142 outputs, the timed tracker had written six sequence files, and both
seed43/seed44 logs were updating near 5,900 steps. The observed CPU/GPU activity
and fresh output mtimes support healthy long-running jobs; no restart or metric
inference is justified.

Phase 84 adds a read-only storage snapshot at 06:49:52: local `/data3` had
61G available, the NAS had 3.7T available, the local workspace used 6.5G, and
the NAS project used 23G in runs plus 5.4G in assets. This is interim evidence;
final accounting will be repeated after live jobs terminate.
