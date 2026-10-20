# Reviewer evidence ledger

Status: `IN_PROGRESS`

Evidence currently supports the following statements:

- Refer-Sports source files, global-ID label namespace, historical split, and
  video-disjoint internal selection split are audited.
- The fixed common tracker completed without GT input.
- FlexHook F1 has a real Sports training run, internal selection evaluation,
  and historical-test TrackEval result.
- YesTrack seed42 full and internal trainings succeeded; internal threshold
  selection succeeded, while its historical test evaluation is still running.
- A causal FlexHook H8 temporal head was actually trained on train-core data,
  selected on internal validation, and evaluated on the historical test.
- The timed common tracker and selected FlexHook H8 historical run have a
  measured complete-pipeline throughput of 1.908807 FPS; the analogous
  YesTrack timing remains pending.
- NAS-direct checkpoints and a real FlexHook resume smoke test are verified.

The following claims are deliberately withheld: a completed YesTrack formal
metric, a best baseline across both models, an independent proposed-method
gain, multi-seed stability, a complete cross-method speed comparison, and a
verified S1–S4 semantic improvement. The authoritative gate state is
`RESEARCH_GATE_AUDIT_20261020.json`.

Scope note (2026-10-20T09:39:03+08:00): the optional seed43/seed44 stability runs were stopped before terminal validation at the user's request. Stability remains unclaimed; this does not change the seed42 formal-test evidence boundary.
