# Reviewer response matrix

Status: `EVIDENCE_TRACKING_IN_PROGRESS`. This table separates implemented
audits from unresolved experimental claims.

| Reviewer point | Current evidence / planned response | Status |
| --- | --- | --- |
| R1-1 Reproducibility and Mamba details | Historical strict-load recipe, checkpoint hashes, Mamba scan-axis audit, and a complete 141-pair same-manifest rerun are recorded. QIM-v2 makes the time axis explicit. | Controlled rerun pass; paper split still unverified |
| R1-2 Data split and statistics | Refer-KITTI bounded manifest: 41 expression videos, 323 expressions, 22-video training manifest; no test manifest recovered. | Unverified leakage/statistics |
| R1-3 Fair baseline and small gains | Old TempRMOT checkpoint is byte-hash verified; baseline and Mamba ran on identical 141 expression keys and 84,640 frames; archived and live results remain separate. | Controlled fairness pass; paper protocol unresolved |
| R1-4 Action-expression evidence | 67 Refer-KITTI expressions contain action/state lexical markers and have frame-indexed labels; action truth outside listed intervals is not established. | Insufficient for claim |
| R2-1 Decoder equations and dimensions | Phase 1 records actual `[N,L,H_lW_l,256]` paths, query/text attention, and cache overwrite behavior. | Audited |
| R2-2 Dataset difficulty | Group, color, number, spatial, and action lexical diagnostics exist; same-uniform, occlusion, and hard-negative subsets need verified labels. | Pending subset manifest |
| R2-3 Recent strong baselines | DKGTrack, FlexHook, CRMOT, Mamba, and STORM code/provenance are recorded; no external result is silently reused. | Audit recorded; fair runs pending |
| R2-4 QIM efficiency | Full controlled rerun records baseline 8.2546 weighted FPS/1054.81 MiB and Mamba 4.0855 weighted FPS/2277.49 MiB; QIM-v2 parameter/FPS comparison is not trained. | Historical runtime pass; new QIM efficiency pending |
| R3-2 Source-video/match isolation | Video IDs collapse to source-name prefixes; no authoritative match manifest or independent test manifest is available. | Unverified |
| R3-3 Independent Mamba-QIM value | Current one-pair checkpoint reproduces its archived protocol closely and the same-manifest rerun is recorded; this still does not isolate Mamba from other source/checkpoint differences. | Not proven |
| R3-4 TAM capacity and insertion point | TAM-v2 has parameter-matched MLP/cross-attention controls and samples decoder-predicted boxes from pre-fusion projected features. | Code pass; metrics pending |
| R3-5 Annotation source and LLM rewriting | Expression structure is audited, but rewrite provenance and counterfactual label correctness are not recovered. | Unverified |
| R3-10 Full runtime | Baseline and Mamba full 141-pair weighted FPS and peak memory are recorded; final TAM/QIM-v2 throughput is pending. | Controlled runtime pass; final model pending |
| R3-11 Association and lifecycle | RuntimeTrackerBase lifecycle and referring-score separation are audited; QIM-v2 state alignment tests pass. | Code/audit pass; online gain pending |

The matrix is a guardrail: unresolved items remain unresolved even if a future
single-sequence score improves.
