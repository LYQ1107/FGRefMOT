# Phase 7 — baseline asset preflight

Audit date: 2026-10-19  
Status: **PENDING EXTERNAL ASSETS / GPU TRAINING NOT RUN**

The reproducible guard is `scripts/preflight_baseline_assets.py`.  It checks
only finite manifests and explicitly named files; it does not traverse the
dataset, start a GPU process, create tracker outputs, or accept labels/GT as a
tracker input.  A tracker directory is accepted only when it contains a
`tracker_manifest.json` declaring `dataset: SportsMOT`,
`input_type: detector_outputs`, `used_ground_truth: false`, detector and
tracker provenance hashes, and per-output hashes.

The checked-in result is
`PHASE7_BASELINE_ASSET_PREFLIGHT.json`.  It was generated with:

```bash
python3 scripts/preflight_baseline_assets.py \
  --hash-files \
  --output docs/RMOT_SPORTS_20261019/PHASE7_BASELINE_ASSET_PREFLIGHT.json
```

The two generated data layouts are ready and carry their no-GT policy markers:

| Gate | Result |
|---|---|
| FlexHook SportsMOT layout | READY |
| YesTrack SportsMOT layout | READY |
| FlexHook RoBERTa directory | MISSING — no verified local files supplied |
| FlexHook ROPE-Swin directory | MISSING — no verified local files supplied |
| FlexHook task checkpoint | MISSING — no verified local file supplied |
| YesTrack LoRA/task checkpoint | MISSING — no verified local file supplied |
| Common SportsMOT physical-tracker output | MISSING — no provenance-complete output supplied |

The overall status is **PENDING**.  The missing entries are intentionally not
filled from the TempRMOT checkpoint archive, Refer-KITTI materials, annotation
directories, or an unverified tracker candidate.  Training and evaluation
remain **NOT_RUN** until every missing gate is acquired, hashed, and passed by
the same preflight.

The report is a point-in-time audit.  After any asset acquisition, rerun it
with explicit paths (or the variables in `configs/paths.env.example`) and
commit the resulting report before starting a run.

The follow-up NAS candidate audit is recorded in
`PHASE8_NAS_CANDIDATE_AUDIT.md` and its JSON companion.  It found large
checkpoints under the archived TempMam/TempRMOT projects, but their code and
README identify them as TempRMOT/Refer-KITTI artifacts.  They have no
FlexHook/YesTrack provenance or SportsMOT detector-output tracker manifest and
are therefore explicitly rejected as substitutes.
