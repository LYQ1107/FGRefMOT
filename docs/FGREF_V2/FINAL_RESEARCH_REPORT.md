# FGRefMOT-v2 final research report

Audit date: 2026-10-18 UTC. This report is an evidence record, not a claim
that every paper number or every proposed contribution has been validated.
Each result is labelled as paper-reported, archived historical,
checkpoint-reproduced, training-interface smoke, or not proven.

## Executive conclusion

The asset and code lineage is recoverable. The old TempRMOT checkpoint and the
TempMam Mamba checkpoint were both run on an identical 141-expression,
84,640-frame manifest. That controlled rerun is useful for source, checkpoint,
and runtime auditing. It is not a sealed paper test split: the recovered
Refer-KITTI material is training-style, no independent test manifest was found,
Refer-Sports was not independently recovered, and the Refer-KITTI-V2 path is
broken.

The recovered Mamba checkpoint substantially outperformed the recovered old
TempRMOT checkpoint under this controlled manifest (HOTA 60.5086 vs 1.5970),
but this comparison does not isolate Mamba-QIM from the other source,
checkpoint, training, and wrapper differences. TAM-v2 and QIM-v2 are
implemented, unit-tested, and integrated, but no trained TAM-v2/QIM-v2
benchmark exists. Therefore the independent TAM, QIM, identity-temporal
decoupling, and cross-dataset claims remain unproven.

## A. Code and asset record

### A.1 Source and execution locations

| Item | Evidence |
| --- | --- |
| Research checkout | /data1/liuyeqiang/FGRefMOT_v2_research |
| Branch | research/fgrefmot-v2-temporal-grounding |
| Upstream source baseline | Git commit d651fec18d4eda54bd141d90cd934b520343b789; tracked source matches TempMam |
| Primary NAS archive | /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam |
| Historical upstream archive | /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT |
| Freeze variant | /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam_frez; kept separate |
| Verified staged dataset | /data1/liuyeqiang/fgref_v2_runs/datasets/refer-kitti |
| Environment | /home/liuyeqiang/anaconda3/envs/mamMOT; torch 2.0.1+cu117, transformers 4.35.2, mamba_ssm 2.2.2 |
| NAS policy | read-only SSHFS/SSH access; no NAS source, checkpoint, or result was modified |

The staged image symlinks resolve to the available SportsMOT image tree and
cv2.imread succeeds on audited images. The expression link resolves to
/data1/liuyeqiang/SportsMOT/expression.

### A.2 Checkpoints and hashes

| Lineage | Path | Size | SHA256 | Compatibility decision |
| --- | --- | ---: | --- | --- |
| Pre-Mamba official / old baseline byte identity | TempMam/exps/official_checkpoint/checkpoint_0060.pth and TempRMOT/exps/default_rk/checkpoint0049.pth | 1,096,317,848 | 55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8 | strict-loadable against historical TempRMOT after the RoBERTa shim; not strict-loadable into current TempMam |
| TempMam Mamba research | TempMam/exps/default_debug_rk/checkpoint0089.pth | 1,185,173,635 | adb2ad9591a4a14dcdd05718b271d6c7b391e97bd9cb20f4fd8e3bfe26e54038 | strict-loadable into current TempMam research source |
| Freeze variant | TempMam_frez/.../official_checkpoint/checkpoint_0060.pth | 1,096,317,848 | not used in final tables | separate lineage; no compatibility claim |

The old checkpoint has 969 state-dict keys. Against the current source it misses
current DeepFuse/Mamba/text-attention groups and has RoBERTa shape mismatches.
Against the historical TempRMOT source, reconstructing
max_position_embeddings=514, type_vocab_size=1, the retained pooler, and the
persistent position_ids buffer gives 969/969 exact keys and a strict load.
This is HISTORICAL_CHECKPOINT_STRICT_LOAD, not a current-model restoration.

### A.3 Code, configs, and reproducibility entry points

New FGRefMOT-v2 modules are under models/fgref_v2/:

    text_evidence.py          lexical/token evidence groups and masks
    track_evidence_sampler.py predicted-box local/context grid sampling
    referring_pair_head.py    independent per-candidate pair logit and controls
    tam_v2.py                 formal TAM composition
    temporal_evidence.py     explicit [N_tracks,T_history,D] encoders
    track_memory.py           ID-keyed identity/dynamic reliability state
    qim_writer.py             evidence-to-query writer
    qim_v2.py                 causal QIM-v2 and Instances adapter
    training_losses.py        optional identity/counterfactual objectives

Integration is opt-in in models/transrmot_pro.py and exposed by main.py:

    --fgref_v2_tam
    --fgref_v2_tam_local_points 4
    --fgref_v2_tam_context_points 8
    --fgref_v2_tam_conditional
    --fgref_v2_tam_heads 8
    --fgref_v2_qim
    --fgref_v2_qim_memory_dim 128
    --fgref_v2_qim_history_length 8
    --fgref_v2_qim_temporal_mode {ema,gru,attention,mamba}
    --fgref_v2_qim_update_query_pos

Declarative manifests are in configs/fgref_v2/tam_pair.yaml,
configs/fgref_v2/tam_local.yaml, and configs/fgref_v2/tam_conditional.yaml.

Important executable evidence tools are:

    tools/fgref_v2/audit_historical_checkpoint.py
    tools/fgref_v2/audit_refer_dataset.py
    tools/fgref_v2/historical_manifest_eval.py
    tools/fgref_v2/trackeval_manifest.py
    tools/fgref_v2/current_forward_smoke.py
    tools/fgref_v2/tam_checkpoint_train_smoke.py

The focused suite passes with CUDA_VISIBLE_DEVICES='' pytest -q tests:
36 passed, 1 warning. The no-GPU import and standalone TAM/QIM state-dict
round-trip checks also pass. Root-level pytest still attempts unrelated
TrackEval fixtures whose external seqmaps are absent; that collection issue is
recorded rather than hidden.

## B. Original model and dataset reproduction

### B.1 Paper values versus recovered evidence

| Dataset/protocol | Paper-reported HOTA | Actual status |
| --- | ---: | --- |
| Refer-Sports | 56.01 | not independently reproduced; source/protocol incomplete |
| Refer-KITTI | 51.64 | not reproduced on a verified independent test split |
| Refer-KITTI-V2 | 35.31 | not reproduced; recovered KITTI symlink is broken |

Archived NAS outputs are preserved as historical evidence:

| Lineage | Archived scope | HOTA | DetA | AssA | Class |
| --- | --- | ---: | ---: | ---: | --- |
| TempRMOT baseline | results_epoch49, 158 pairs | 50.641 | 39.406 | 65.241 | ARCHIVED_HISTORICAL_RESULT |
| TempMam Mamba | results_epoch89, 141 pairs | 56.880 | 56.297 | 57.514 | ARCHIVED_HISTORICAL_RESULT |

### B.2 Controlled 141-pair rerun

Both runs used the same 141 unique video/expression keys, local staged
Refer-KITTI data, thresholds det=0.6, refer=0.4, area=100, and the same
TrackEval wrapper. Each completed 141/141 with status=ok; the key sets have
zero duplicates, zero omissions, and the same 84,640 frames.

| Checkpoint lineage | HOTA | DetA | AssA | IDF1 | IDSW | Weighted FPS | Peak MiB | Result class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TempRMOT checkpoint0049.pth | 1.5970 | 0.6261 | 4.1764 | 1.1816 | 141 | 8.2546 | 1054.81 | CHECKPOINT_REPRODUCED on controlled manifest |
| TempMam Mamba checkpoint0089.pth | 60.5086 | 64.4788 | 56.8343 | 63.5075 | 1276 | 4.0855 | 2277.49 | CHECKPOINT_REPRODUCED on controlled manifest |

Summaries:

    /data1/liuyeqiang/fgref_v2_runs/historical_baseline_manifest_full2/
      tempRMOT_checkpoint0049/trackeval_manifest_summary.json
    /data1/liuyeqiang/fgref_v2_runs/historical_mamba_manifest_full2/
      checkpoint0089/trackeval_manifest_summary.json

The Mamba run averaged 4.102 FPS per sequence (weighted 4.085 FPS), while the
old baseline averaged 8.268 FPS (weighted 8.255 FPS). Peak allocation was
2,277.49 MiB versus 1,054.81 MiB. This is a runtime observation about the
recovered lineages, not a QIM-v2 efficiency result.

The auxiliary online referring counts were:

| Lineage | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TempRMOT | 2,738 | 20,267 | 338,689 | 0.1190 | 0.0080 | 0.0150 |
| TempMam Mamba | 248,176 | 12,531 | 93,251 | 0.9519 | 0.7269 | 0.8243 |

These counts are secondary diagnostics. TrackEval is the main tracking table.
The result does not prove that Mamba-QIM alone caused the gap.

### B.3 Single-pair compatibility checks

On v_ApPxnw_Jffg_c015 / volleyball-players-who-are-serving (246 frames):

| Lineage | HOTA | DetA | AssA | IDF1 | IDSW | FPS | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TempRMOT | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 7.7911 | 1054.78 |
| TempMam Mamba | 33.3818 | 12.5432 | 88.9260 | 23.9714 | 0 | 4.3966 | 2270.07 |

The nearby NAS archived Mamba pair gives HOTA 33.3363 under the local subset
runner. The close values support source/checkpoint alignment for this pair;
they do not make it a dataset-wide result.

## C. Actual architectures and contribution boundaries

### C.1 Audited original path

The active baseline path is models/__init__.py -> transrmot_pro.build; the
older transrmot.py is not the active temp_rmot implementation. The default
hidden dimension is 256, tracking queries have dimension 512 split into 256
positional and 256 content dimensions, and decoder outputs are
[B,N,1] class logits, [B,N,4] boxes, and [B,N,1] referring logits.

Triple Alignment concatenates object, text, and visual tokens at each level;
the self-cross decoder then performs query-to-text attention followed by
multi-scale deformable visual attention. SpatialTemporalReasoner receives
text_info but does not read it. The historical Mamba-QIM call sends
[1,N_active,D] with fresh inference_params=None on every update, so its scan
axis is active-track row order rather than persistent time. No physical
track-ID-keyed Mamba state is stored.

RuntimeTrackerBase owns physical ID creation and termination. Referring scores
are applied later as an output filter and do not keep a physical track alive.
These observations motivated, but do not themselves validate, QIM-v2.

### C.2 TAM-v2

The formal TAM path consumes decoder-predicted boxes, never GT boxes. In compact
notation, for candidate i, projected visual pyramid F_l, text tokens X, and
query q_i:

    E_i^local   = mean_l Sample(P_l F_l, grid_local(b_i))
    E_i^context = mean_l Sample(P_l F_l, grid_context(b_i))
    K_i = [W_l E_i^local, W_c E_i^context, W_t X, W_h H_i]
    A_i = MHA(W_q q_i, K_i, K_i; valid_sample/text/history_mask)
    r_i = MLP([W_q q_i, A_i])

b_i is the predicted normalized cxcywh box. The sampler uses fixed local and
expanded-context grids; the conditional variant learns bounded offsets from
query/text conditions. Text evidence is routed into identity, action,
relation, unknown, and negation masks by an inspectable lexical router. The
pair head emits an independent logit per candidate, with no candidate-wise
softmax, so zero, one, or multiple referred objects remain representable.

The original decoder/detection path and legacy referring score are retained;
TAM is an opt-in post-decoder evidence path. A separate diagnostic API can
accept boxes only with explicit diagnostic flags. use_checkpoint with TAM is
rejected explicitly because the legacy activation-checkpoint tuple does not
contain the new named evidence fields.

### C.3 QIM-v2

For track i at time t, the reliability gate is:

    rho_i,t = detection_quality_i,t * visibility_i,t
              * (1 - feature_conflict_i,t) * match_confidence_i,t

refer_score is deliberately absent from this gate. Identity memory is updated
only when an observation is available and rho >= write_threshold:

    I_i,t = I_i,t-1                       if no reliable write
    I_i,t = alpha I_i,t-1 + (1-alpha) W_I o_i,t  otherwise

Dynamic history shifts with an explicit validity mask. The temporal encoder
reads state ending at t-1; only afterward does the current observation write to
state. The query writer combines identity, current observation, temporal
summary, and predicted box:

    g_i = sigmoid(MLP([W_I I_i, W_o o_i, b_i]))
    c'_i = LN(c_i + g_i W_o o_i + (1-g_i) W_I I_i + W_h h_i)

The positional half is unchanged unless update_query_pos is explicitly enabled.
ReliabilityAwareQueryInteraction aligns state by physical obj_idxes; births
get clean state, reordering preserves state, and deleted IDs cannot transfer
memory to a new birth. RuntimeTrackerBase remains the physical lifecycle
owner. In the opt-in QIM-v2 path the legacy STReasoner is not stacked on top;
temporal-loss keys are explicit zeros for logger compatibility.

### C.4 Retained, borrowed, proposed, and validated

| Category | Decision |
| --- | --- |
| Retained | Deformable detector/backbone, decoder, query format, legacy losses, physical tracker lifecycle, and independent referring-logit contract |
| Replaced/opt-in | TAM branch is an optional evidence path; QIM-v2 adapter replaces the legacy row-wise QIM hook when enabled |
| Borrowed conceptually | DKGTrack language routing, FlexHook conditional evidence points, CRMOT lifecycle separation, and Mamba sequence-axis reasoning; no external source was copied |
| Proposed | Explicit predicted-box fine-grained evidence alignment; reliability-gated ID-keyed memory with causal read-before-write; optional identity/counterfactual objectives |
| Validated | Code shapes, masks, lifecycle invariants, strict state-dict round trips, no-GT path, real-sample forward, and nonzero TAM gradients |
| Not validated | Independent TAM gain, independent QIM gain, identity-temporal decoupling gain, cross-dataset generalization, and paper-level SOTA |

External provenance details are in RELATED_CODE_COMPARISON.md.

## D. Experiments and diagnostics

### D.1 Training-interface and integration evidence

The combined current-source random-initialization smoke completed on a real
Refer-KITTI sample:

    CURRENT_TAM_QIM_FORWARD_OK
    frames 1
    pred_logits_shape (1, 300, 1)
    pred_boxes_shape (1, 300, 4)
    pred_refers_shape (1, 300, 1)
    tam_enabled True
    qim_adapter ReliabilityAwareQueryInteraction
    total_parameters 178273518
    trainable_parameters 53996078
    tam_parameters 1118721
    qim_parameters 332033
    elapsed_seconds 1.8696
    peak_memory_mib 1871.44

TAM-only optimization smoke used the compatible Mamba checkpoint, froze the
base model, and ran two updates on a real four-frame training sample. Three
seeds produced the following nonzero-gradient checks:

| Seed | Loss step 0 | Loss step 1 | Grad step 0 | Grad step 1 | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 17 | 7.258664 | 6.590466 | 28.925404 | 27.274349 | optimization path live |
| 29 | 8.452438 | 7.572502 | 43.669483 | 36.235447 | optimization path live |
| 43 | 3.141252 | 2.719410 | 18.245129 | 16.683441 | optimization path live |

These are not validation scores, not full training, and not a new benchmark.
QIM-v2 has no trained metric.

### D.2 Dataset audit and difficulty diagnostics

The bounded staged audit found:

| Asset fact | Count/result |
| --- | ---: |
| Expression video directories | 41 |
| Expression JSON files | 323 |
| refer-kitti.train rows/videos | 14,750 / 22 |
| refer-kitti-full.train rows/videos | 26,650 / 41 |
| refer-kitti1.train rows/videos | 28,574 / 45 |
| Duplicate audited rows | 0 |
| Frame/image/box mismatches | 0 |
| Action/state lexical expressions | 67 |
| Color/appearance lexical expressions | 184 |
| Identity/number lexical expressions | 133 |
| Spatial lexical expressions | 58 |
| Group/entity lexical expressions | 323 |
| Exact sentences appearing in multiple videos | 51 |
| Collapsed source-name prefixes | 10 |
| Independent test manifest recovered | no |

The lexical categories are regex diagnostics, not human labels. Categories
overlap. In the 141-pair controlled manifest, the mean per-sequence online
diagnostics were:

| Lexical group | Pairs | Mamba mean IDF1 | Mamba mean Ref-F1 | TempRMOT mean IDF1 | TempRMOT mean Ref-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| action/state | 29 | 0.6601 | 0.7242 | 0.0125 | 0.0136 |
| color/appearance | 81 | 0.4782 | 0.6198 | 0.0050 | 0.0064 |
| identity/number | 59 | 0.4380 | 0.5381 | 0.0029 | 0.0031 |
| spatial relation | 23 | 0.7867 | 0.8812 | 0.0169 | 0.0184 |

These overlapping, model-specific diagnostics do not establish action truth,
hard-negative correctness, or a causal method gain. A valid identity/action
subset experiment still requires an authoritative grouped split and verified
negative intervals.

### D.3 Controls and missing experiments

Implemented TAM controls are capacity-matched pooled MLP, ordinary
cross-attention, fixed local evidence, context evidence, conditional sampling,
and lexical routing. They were not trained to convergence. Implemented QIM
temporal modes are EMA, GRU, attention, and optional Mamba over an explicit
time axis; none has a benchmark run. Identity and counterfactual loss
interfaces exist, but no verified annotations were found, so L1-L3 objectives
were not reported.

The following are deliberately not fabricated:

- no TAM-only HOTA/IDF1/Ref-F1 ablation;
- no QIM-only or TAM+QIM trained result;
- no final candidate checkpoint selected by a legal validation split;
- no multi-seed performance table (the three seeds above are smoke only);
- no hard-negative FPR or counterfactual accuracy;
- no Refer-Sports or Refer-KITTI-V2 generalization result.

## E. Reviewer-response record

| Point | Evidence-backed response | Final status |
| --- | --- | --- |
| R1-1 reproducibility/Mamba | Hashes, strict-load recipe, scan-axis audit, full controlled rerun, and explicit QIM-v2 time axis are recorded. | controlled rerun pass; paper split unverified |
| R1-2 split/statistics | 41 videos/323 expressions and manifest counts are audited; no independent test manifest or leakage-free grouping recovered. | unverified |
| R1-3 fair baseline/small gains | Old checkpoint hash is verified and both lineages use identical 141 keys; this is not a causal Mamba/TAM comparison. | controlled fairness pass; attribution unresolved |
| R1-4 action evidence | 67 action/state lexical expressions are diagnosed; interval truth outside labels is unknown. | insufficient for claim |
| R2-1 decoder equations/dimensions | Actual 256/512 dimensions, Triple Alignment, query-text attention, and cache overwrite order are documented. | audited |
| R2-2 dataset difficulty | Group/color/number/spatial/action diagnostics exist; no verified hard subset or human action labels. | partial |
| R2-3 recent strong baselines | DKGTrack, FlexHook, CRMOT, Mamba, and STORM code/provenance are recorded; none is silently counted as reproduced. | audit only |
| R2-4 QIM efficiency | Full historical runtime is measured; QIM-v2 parameters are 332,033, but trained speed/accuracy is not measured. | historical runtime pass; new QIM pending |
| R2-5 dataset scale | Bounded Refer-KITTI counts are reported; Refer-Sports and independent test scale remain unavailable. | partial |
| R3-2 source/match isolation | Ten source-name prefixes and repeated sentences are visible; no authoritative source/match split manifest. | unverified |
| R3-3 Mamba independent value | Mamba reproduces a pair and completes the controlled manifest; other source/checkpoint changes are confounded. | not proven |
| R3-4 TAM capacity/insertion | Pair head and pooled controls are implemented; predicted boxes sample pre-fusion projected features; metrics are absent. | code pass; gain pending |
| R3-5 annotation/LLM rewriting | Frame-indexed labels exist, but rewrite lineage and counterfactual correctness are not recovered. | unverified |
| R3-10 full runtime | Baseline/Mamba full-manifest FPS and memory are recorded; final TAM/QIM runtime is absent. | controlled runtime pass |
| R3-11 lifecycle | Physical ID ownership and referring-score separation are audited; ID-keyed state tests pass. | code/audit pass; online gain pending |

Remaining organization issues, including equation wording, split naming, and
benchmark protocol disclosure, must be fixed in any paper but cannot replace
the missing controlled training experiments.

## F. Final scientific judgment

### Proven or supported

- NAS asset lineage and checkpoint hashes are reproducible.
- The historical TempRMOT checkpoint can be strictly restored against its
  historical namespace; the Mamba checkpoint can be strictly restored against
  the current TempMam research namespace.
- Both lineages run through the same recovered 141-pair manifest with complete
  per-sequence artifacts and TrackEval summaries.
- TAM-v2 and QIM-v2 have real code paths, masks, lifecycle handling, tests,
  state-dict round trips, and a real-sample integration smoke.
- TAM-v2 receives gradients under three independent two-step smoke seeds.

### Not proven

- TAM-v2 independent language-discrimination gain.
- QIM-v2 independent identity or temporal gain.
- Identity-temporal decoupling as a measurable advantage.
- Cross-dataset generalization.
- Paper-reported Refer-Sports, Refer-KITTI, or Refer-KITTI-V2 numbers.
- Any CVPR/TMM SOTA claim.

### Negative results and stopping conditions

- The old baseline is extremely weak under the recovered controlled manifest;
  this is a diagnostic result, not permission to select a favorable protocol.
- The archived Mamba module scans active-track rows with reset state rather
  than persistent time; QIM-v2 fixes the interface but is not trained.
- The pre-Mamba checkpoint cannot be loaded into the current model by strict
  compatibility; permissive loading is not counted as reproduction.
- Refer-KITTI-V2 has a broken source symlink, and no independent test manifest
  or Refer-Sports protocol was recovered.
- Full TAM/QIM training should stop until a legal grouped validation/test
  protocol and verified identity/counterfactual annotations are available.

    FINAL GOAL STATUS
    BASELINE REPRODUCED: NO
    TAM INDEPENDENT GAIN: NOT PROVEN
    QIM INDEPENDENT GAIN: NOT PROVEN
    IDENTITY-TEMPORAL DECOUPLING: NOT PROVEN
    CROSS-DATASET GENERALIZATION: NOT PROVEN
    DATASET PROTOCOL VERIFIED: NO
    RECOMMENDED SUBMISSION: FURTHER RESEARCH
    FINAL GITHUB COMMIT: see final branch HEAD in the handoff

The honest next milestone is a protocol-recovered grouped validation/test
split, followed by B1/B2/B3 training with fixed seeds and a sealed evaluation.
Only after those gates should the submission recommendation be reconsidered.

