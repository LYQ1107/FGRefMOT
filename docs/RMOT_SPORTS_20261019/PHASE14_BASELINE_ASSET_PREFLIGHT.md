# Phase 14 — baseline asset preflight and adapter revalidation

Audit date: 2026-10-19  
Status: **LAYOUT SMOKE OK / EXTERNAL ASSETS PENDING / NO GPU JOB STARTED**

The generated YesTrack-compatible SportsMOT layout was exercised again through
the actual adapter's `--dry-run --json` path.  Both supervised loaders opened
their split roots and returned the finite audited counts:

| split | videos | frames | converted boxes | expressions |
|---|---:|---:|---:|---:|
| train | 22 | 14,750 | 160,489 | 181 |
| val | 19 | 11,900 | 130,152 | 142 |

The adapter reported
`SUPERVISED_LABELS_ONLY; NO_GT_AS_TRACKER_INPUT` and exited 0.  This verifies
the data/layout boundary and the non-numeric SportsMOT sequence handling; it
does not load Qwen, LoRA, or any GPU model.

The current `--hash-files` preflight found both generated layouts `READY`, but
all six external checks remain `MISSING`: FlexHook RoBERTa, FlexHook ROPE-Swin,
FlexHook task checkpoint, YesTrack LoRA checkpoint, the matching Qwen base
model, and the provenance-complete common tracker.  It exited 2 as designed.
The full machine-readable snapshot is
`PHASE14_BASELINE_ASSET_PREFLIGHT.json`; no external file was substituted.

Consequently every formal baseline contract remains launch-blocked.  Once the
official bytes and tracker manifest arrive, rerun the same preflight, build a
contract for the chosen F0/F1/F2 or Y0/Y1/Y2/Y3 variant, and only then run the
model smoke test and GPU training.

