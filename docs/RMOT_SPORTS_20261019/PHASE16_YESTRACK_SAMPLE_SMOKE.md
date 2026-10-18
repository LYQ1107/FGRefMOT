# Phase 16 — YesTrack SportsMOT sample-interface smoke test

Audit date: 2026-10-19  
Status: **SAMPLE INTERFACE SMOKE OK / EXTERNAL ASSETS PENDING / NO GPU JOB STARTED**

The new `scripts/yestrack_sportsmot_smoke.py` adapter exercised the pinned
YesTrack supervised sample classes against one real sequence from each
generated split.  It fetched both the single-frame sample used by Y1 and the
video sample used by Y3, without loading Qwen, LoRA, CUDA model weights, or a
tracker.

| split | sequence | single samples | video samples | first crop |
|---|---|---:|---:|---|
| train | `v_-6Os86HzwCs_c001` | 48,444 | 48,444 | 384×384 |
| val | `v_-6Os86HzwCs_c007` | 31,362 | 31,362 | 384×384 |

The first video sample contains one frame because it is the first frame of its
sequence; the causal history window grows for later samples.  Coordinate
prompts, labels, image crops, and both sample interfaces were produced without
using tracker outputs or GT as tracker input.  The full machine-readable
evidence is `PHASE16_YESTRACK_SAMPLE_SMOKE.json`.

This is a bounded data-interface check, not a training or model result.  The
official Qwen base, optional public LoRA, and common physical tracker remain
outside the workspace, so the formal Y0–Y3 contracts remain blocked.

