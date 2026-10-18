# Phase 8 experiment matrix

Status: `MATRIX_DEFINED`, `CONTROLLED_141_MANIFEST_COMPLETE`,
`NEW_METHOD_METRICS_PENDING`.

All numbers must carry one of `PAPER_REPORTED`, `ARCHIVED_HISTORICAL_RESULT`,
`CHECKPOINT_REPRODUCED`, `RETRAINED_BASELINE`, or `NEW_EXPERIMENT`. A result
from one video/expression is never averaged with an archived split summary.

## Current evidence table

| Lineage | Checkpoint | Scope | HOTA | DetA | AssA | IDF1 | Class |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| TempRMOT baseline | `checkpoint0049.pth`, SHA256 `55e9b2ce...408c7e8` | NAS archive, 158 pairs | 50.641 | 39.406 | 65.241 | not listed | `ARCHIVED_HISTORICAL_RESULT` |
| TempRMOT baseline | same checkpoint | one 246-frame pair | 0.000 | 0.000 | 0.000 | 0.000 | `CHECKPOINT_REPRODUCED` |
| TempMam Mamba | `checkpoint0089.pth`, SHA256 `adb2ad95...e54038` | NAS archive, 141 pairs | 56.880 | 56.297 | 57.514 | not listed | `ARCHIVED_HISTORICAL_RESULT` |
| TempMam Mamba | same checkpoint | one 246-frame pair | 33.382 | 12.543 | 88.926 | 23.971 | `CHECKPOINT_REPRODUCED` |
| TempRMOT baseline | `checkpoint0049.pth`, SHA256 `55e9b2ce...408c7e8` | controlled 141-pair manifest, 84,640 frames | 1.597 | 0.626 | 4.176 | 1.182 | `CHECKPOINT_REPRODUCED` |
| TempMam Mamba | `checkpoint0089.pth`, SHA256 `adb2ad95...e54038` | controlled 141-pair manifest, 84,640 frames | 60.509 | 64.479 | 56.834 | 63.507 | `CHECKPOINT_REPRODUCED` |

The one-pair rows use the same local TrackEval subset runner and are included
to audit source/checkpoint/protocol alignment. They are not claims about the
Refer-KITTI split or Refer-Sports.

## Planned main matrix

| ID | Detector/tracker | TAM-v2 | QIM-v2 | Objective | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | restored TempRMOT baseline | no | legacy | L0 | n/a (checkpoint rerun) | controlled 141-pair rerun complete; paper split pending |
| B1 | same frozen B0 | T1/T2/T3/T4/T5 controls | no | L0 | 17, 29, 43 | TAM implementation pass; training/metrics pending |
| B2 | same frozen B0 | no | EMA/GRU/attention/Mamba/reliability | L0 | 17, 29, 43 | QIM-only pending |
| B3 | same training protocol | selected TAM | selected QIM | L0 | 17, 29, 43 | pending selection gates |
| B4 | B3 | selected TAM + semantic routing | selected QIM | L1 | 17, 29, 43 | requires verified IDs |
| B5 | B3 | selected TAM | selected QIM | L2 | 17, 29, 43 | requires verified counterfactuals |

## Evaluation partitions

The controlled 141-pair result is the recovered training-style manifest only
for checkpoint/implementation audit. No independent test manifest has been
verified. Refer-Sports, Refer-KITTI-V2, and the requested hard subsets remain
`INSUFFICIENT_EVIDENCE` until their source and group manifests are audited.
The historical Refer-Sports 22/19 split may be used for an archive-compatible
comparison only after its provenance is recovered; it cannot be called a new
sealed test split.

Each future run must save the full manifest, config, source commit, checkpoint
hash, seed, environment, parameter count, FPS, peak memory, HOTA, DetA, AssA,
IDF1, IDSW, referring F1, hard-negative FPR, and per-video/group outputs.
