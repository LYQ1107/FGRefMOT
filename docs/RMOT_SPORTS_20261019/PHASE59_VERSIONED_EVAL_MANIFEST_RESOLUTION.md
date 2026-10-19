# Phase 59 — Versioned formal-evaluation manifest resolution

The baseline table now resolves both FlexHook and YesTrack formal-evaluation
manifests from versioned NAS output directories. It prefers the newest
successful manifest and otherwise exposes the newest pending attempt. This
keeps failed evaluator retries auditable and prevents a later retry from being
silently ignored by the table.

The YesTrack resolver covers
`runs/gpu/yestrack_y1_common_eval_20261019_v*/YESTRACK_EVAL_MANIFEST.json`.
The existing FlexHook resolver follows the same policy. No metric is emitted
unless the selected manifest is successful and contains TrackEval output.

Validation: `39 passed` from the project test suite on 2026-10-19. The change
does not touch any active training or evaluation process.
