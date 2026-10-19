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
completed eight epochs with epoch 5 selected from internal validation; its
internal TrackEval is now running. The full 22-video YesTrack training also finished successfully;
its formal historical evaluation has started. The separate 17-video internal
YesTrack training succeeded, and its fixed five-video validation evaluation is
still running against the completed adapter.
The FlexHook frozen-score H0-H8 replay is a diagnostic control over a frozen
checkpoint; it is not a learned temporal-head result.

The following gates remain open: YesTrack validation and formal evaluation,
learned temporal-head internal and historical evaluation, proposed-method
and baseline ablations, multi-seed stability, complete online efficiency,
final-test evidence, and the final reviewer report. Until those manifests exist,
the study must not claim a YesTrack metric, a learned-method improvement, a
multi-seed gain, or a complete final result.

The machine-readable gate states, evidence paths, live manifests and next
gates are in `RESEARCH_GATE_AUDIT_20261020.json`. This report intentionally
preserves `IN_PROGRESS`/`NOT_RUN` boundaries instead of inferring results
from partial logs or training loss.
