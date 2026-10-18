# FGRefMOT-v2 research record

## Scope

This branch records an auditable continuation of FGRefMOT from the reference
commit `d651fec18d4eda54bd141d90cd934b520343b789`. Results are separated into
paper-reported, checkpoint-reproduced, retrained-baseline, and new-experiment
categories.

The first gates are asset provenance, checkpoint compatibility, and a runnable
baseline. New TAM/QIM code will only be evaluated after those gates are
recorded.

## Working locations

- Local research checkout: `/data1/liuyeqiang/FGRefMOT_v2_research`
- Branch: `research/fgrefmot-v2-temporal-grounding`
- Read-only NAS mount: `/data1/liuyeqiang/nas/A100`
- NAS code archive: `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam`
- NAS Mamba variant: `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam_frez`

No source, checkpoint, dataset, or output will be modified in the read-only
NAS archive. New code, reports, tests, and experiment outputs belong in the
local checkout or a separately named local output directory.

## Current gates

| Gate | Status | Evidence / next action |
| --- | --- | --- |
| NAS access | PASS | SSH reverse forward on local port 24557; read-only SSHFS mount is live when the forward is live. |
| Source provenance | PASS | GitHub main resolves to `d651fec18d4eda54bd141d90cd934b520343b789`; tracked source matches the NAS `TempMam` archive. |
| Checkpoint provenance | HISTORICAL STRICT PASS / BASELINE HASH PASS / MAMBA ONE-PAIR PASS / VARIANT AUDIT IN PROGRESS | The pre-Mamba official checkpoint strictly loads against the old TempRMOT namespace; TempRMOT `checkpoint0049` is byte-identical to the TempMam official copy; the Mamba-QIM checkpoint strictly loads against TempMam and reproduces one complete sequence. |
| Baseline execution | CONTROLLED_141_MANIFEST_PASS / PAPER_SPLIT_PENDING | Both checkpoints completed the same 141-pair, 84,640-frame manifest with identical pair keys; TrackEval summaries are recorded in Phase 2. This is not a sealed paper-split reproduction. |
| Dataset protocol | REFER_KITTI_BOUNDED_PASS / TEST_ISOLATION_UNVERIFIED | The staged Refer-KITTI audit found 41 expression videos, 323 expressions, zero audited frame mismatches, and no independent test manifest; Refer-KITTI-V2 still has a broken symlink. |
| TAM-v2 implementation | PASS (code gate) | Optional prediction-box TAM path, controls, manifests, and CPU tests are recorded in `PHASE4_TAM_IMPLEMENTATION.md`. Metrics remain pending. |
| QIM-v2 implementation | PASS (code gate) | Optional ID-keyed reliability-aware adapter and state lifecycle tests are recorded in `PHASE6_MEMORY_IMPLEMENTATION.md`. Training remains pending. |

## Resource policy

At the initial audit, all ten GPUs are V100 32 GB devices. GPUs 0, 1, and 3
were occupied by unrelated jobs; no training was started. The local `/data1`
volume has about 212 GB free, while `/data2` is full. Initial work therefore
uses bounded scans, CPU metadata inspection, and short smoke tests only.
