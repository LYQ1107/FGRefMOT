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

The following gates remain open: YesTrack formal evaluation and timing,
proposed-method and baseline ablations, multi-seed stability, remaining
efficiency evidence, final-test evidence for the remaining variants, and the
final reviewer report.
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

Phase 85 records the 06:53:48 increment: the formal evaluator reached 59/142
prediction files, the timed tracker reached 10/19 sequence files, and seed43 /
seed44 reached 6,744/6,690 steps. These remain progress evidence only until
terminal manifests succeed.

Phase 86 records the 07:06:25 completion of the timed common tracker and its
efficiency join. All 19 historical-test sequences produced 11,900 frames in
1,580.706636 seconds (7.5283 tracker FPS). Joining that terminal tracker
manifest with the selected FlexHook H8 historical evaluation gives 1.9088 FPS
for the complete tracker-plus-referring pipeline. YesTrack historical timing
and the remaining method-comparison gates are still open.

Phase 87 records the 07:08:00 health check: the formal YesTrack evaluator was
still alive after producing 60/142 prediction files, while seed43 and seed44
were actively training at 9,048/90,523 and 8,875/90,528 optimizer steps.
Neither partial output nor nonterminal training state is used as a metric.

Phase 88 records the 07:11:03 non-GPU regression check: the repository test
suite completed with 78 passed tests and two known deprecation warnings. This
verification does not change any live GPU run or convert partial outputs into
metrics.

Phase 89 records the 07:13:27 baseline-table readiness check. The conservative
builder produced nine rows and correctly extracted FlexHook F1's completed
validation/historical metrics while leaving the YesTrack historical row as
`NOT_RUN` because its terminal manifest is absent. The generated table is
retained under the ignored live-run directory and is not treated as a final
comparison.

Phase 96 records a fresh hash-based asset preflight and corrected
source-backed variant-specific run contracts. The full-budget F1 and Y1
contracts now match their manifests (22 training videos/181 expressions and a
19-video historical evaluation layout), while the separate core/internal-val
runs remain the selection routes. The aggregate preflight remains pending only
for optional zero-shot/task-checkpoint assets; both contracts pass
`--require-ready` with the current common tracker, layouts, explicit protocol
values and actual launch commands. This metadata update does not change any
live GPU job or promote nonterminal evaluation output into a metric.

The 08:01:17 live gate snapshot records the formal evaluator at 67/142
prediction files, with zero format issues in the read-only partial-output audit
of Phase 98. Seed43 and seed44 were still active at 18,205/90,523 and
17,701/90,528 optimizer steps. The formal manifest remains absent, so the
historical-test row and all postbaseline watchers remain pending.

Live snapshot refreshed at 2026-10-20T08:03:48+08:00 (Phase 99): the formal YesTrack evaluator remains alive with 67/142 prediction files and no terminal manifest; seed 43 is at 18645/90,523 steps and seed 44 at 18119/90,528. These are progress observations only; formal-test and multi-seed metrics remain pending terminal manifests.

Live snapshot refreshed at 2026-10-20T08:04:51+08:00 (Phase 100): formal YesTrack remains active with 68/142 partial prediction files; seed 43 is at 18826/90,523 steps and seed 44 at 18290/90,528. No terminal formal-test or multi-seed metric is claimed.

Activity probe refreshed at 2026-10-20T08:06:34+08:00 (Phase 101): PID 2708884 remained active and increased rchar by 7,147,976 over 3 seconds; formal output remains 68/142 with no terminal manifest. Seed43/44 remain in training.

Partial-output integrity refreshed at 2026-10-20T08:07:36+08:00 (Phase 102): 69/142 files and 465,077 rows were checked with 0 issues. No formal metric is claimed while the terminal manifest is absent.

Partial-output integrity refreshed at 2026-10-20T08:10:00+08:00 (Phase 103): 70/142 files and 470,786 rows were checked with 0 issues; no formal metric is claimed while the terminal manifest is absent.

Partial-output integrity refreshed at 2026-10-20T08:12:25+08:00 (Phase 104): 71/142 files and 476,495 rows were checked with 0 issues; no formal metric is claimed while the terminal manifest is absent.
