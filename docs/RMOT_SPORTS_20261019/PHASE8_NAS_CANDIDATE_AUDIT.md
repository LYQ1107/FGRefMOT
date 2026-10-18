# Phase 8 — NAS candidate asset audit

Audit date: 2026-10-19  
Status: **EVIDENCE RECORDED / TARGET ASSETS STILL MISSING**

This is a read-only audit of the currently reachable NAS mount
`/data3/liuyeqiang/nas/A100` (the NAS login path is `/home/Valadmin/A100`).
The scan was limited to the known project roots so that a network filesystem
walk would not disturb the mount.  It did not copy, rename, hash, or modify any
NAS file.

## Observed checkpoint candidates

| NAS path | observed size / count | provenance decision |
|---|---:|---|
| `codex_archive_20261014/TempMam/exps/official_checkpoint/checkpoint_0060.pth` | 1,096,317,848 bytes; 1 file in `official_checkpoint` | **REJECTED_NON_TARGET** — the adjacent README identifies the project as TempRMOT / Refer-KITTI, not FlexHook or YesTrack |
| `codex_archive_20261014/TempMam/exps/default_debug_rk/checkpoint*.pth` | 21,333,086,502 bytes; 18 files | **REJECTED_NON_TARGET** — the training and test scripts use `--meta_arch temp_rmot` and Refer-KITTI paths |
| `codex_archive_20261014/TempRMOT/exps/**/checkpoint*.pth` | representative file `encoder_mamba_4/checkpoint0060.pth`: 1,218,413,411 bytes | **REJECTED_NON_TARGET** — this is another TempRMOT experiment tree; no FlexHook/YesTrack identity or target checkpoint manifest is present |

The same roots contain `results_epoch*` and visualization directories.  They do
not provide the required `tracker_manifest.json` declaring `dataset:
SportsMOT`, `input_type: detector_outputs`, `used_ground_truth: false`,
detector/tracker provenance hashes, and per-output hashes.  They therefore
cannot be used as the common physical tracker input for this study.

## Decision

These files remain untouched and are not symlinked into the study workspace.
Their existence does not change the Phase 7 result: FlexHook RoBERTa/ROPE-Swin
assets, FlexHook task checkpoint, YesTrack LoRA/task checkpoint, and a
provenance-complete SportsMOT physical tracker remain **MISSING**.  The strict
preflight stays **PENDING**, and no GPU training or evaluation is started from
these non-target artifacts.

After target assets are obtained, run the Phase 7 preflight with explicit paths
and `--hash-files`; only a `READY` result may unlock a baseline run.
