# FGRefMOT asset lineage audit

Audit scope: the three NAS directories named during the research task and the
checkpoint files found under their experiment trees. This is a code and file
provenance classification; directory ownership is not inferred from Unix
owner fields alone.

## Classification

| Directory | Evidence | Classification for this study |
| --- | --- | --- |
| `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT` | README is the released *Bootstrapping Referring Multi-Object Tracking* project. `models/qim_pro.py` is the short pre-Mamba QIM implementation, `models/transrmot_pro.py` has no `Triple_Alignment`, and `models/deformable_transformer_plus.py` is the older 27,330-byte implementation. It also contains `exps/default_rk/checkpoint0049.pth` and archived TrackEval outputs. | Upstream/original TempRMOT reference snapshot and baseline lineage. Use its checkpoint/results as the old baseline evidence; do not treat its later Mamba experiment folders as the clean upstream baseline without a separate source/config audit. |
| `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam` | Tracked source files are byte-identical to GitHub commit `d651fec18d4eda54bd141d90cd934b520343b789`. It adds Mamba-QIM, DeepFuse/Triple Alignment, generated visualization hooks, and the current FGRefMOT configs. | Primary FGRefMOT research source/archive. This is the code to branch from, but it is not the old upstream TempRMOT baseline. |
| `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam_frez` | `models/qim_pro.py`, `models/spatial_temporal_reason.py`, and `models/__init__.py` match `TempMam`; `main.py` explicitly freezes `deepfuse`, `mamba`, and `decoder`, and its configs point to `_frez` checkpoints and experiment paths. Its `transrmot_pro.py` removes the visualization/ablation hooks present in `TempMam`. | User/research freeze-and-training variant of the FGRefMOT tree. Keep as a separate experimental lineage; do not mix its checkpoints with either upstream TempRMOT or the current source without an audit. |

The NAS directories are not Git repositories, so their contents cannot be
assigned a commit beyond the byte-level comparison above. The timestamp and
configuration evidence supports the classifications, but does not prove
personal authorship of every generated file.

## Critical checkpoint finding

The file

`/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/official_checkpoint/checkpoint_0060.pth`

is stored under the `TempMam` archive but is architecturally an old
TempRMOT-lineage checkpoint. A strict historical restoration audit now
succeeds against the old `TempRMOT` source when the checkpoint's own RoBERTa
configuration and serialization details are reconstructed:

* against the current `TempMam`/GitHub source, its 969 keys produce 244 missing
  current keys, 3 unexpected keys, and 2 shared shape mismatches; the missing
  groups include all `deepfuse.*` and Mamba-QIM parameters;
* with the source's default text construction, the model removes the pooler
  and the installed Transformers version serializes `position_ids` as
  non-persistent, so the unadjusted comparison is short by three keys;
* using `max_position_embeddings=514`, `type_vocab_size=1`, restoring the
  historical `RobertaPooler`, and making `position_ids` persistent produces
  969 model keys, all 969 intersecting with 0 missing, 0 unexpected, and 0
  shape mismatches;
* strict `load_state_dict` then succeeds. The compatibility changes were
  process-local audit shims; no NAS source or checkpoint was modified;
* the stored metadata says `meta_arch='rmot_qim_v2'`, a label not exposed by
  the current `models/__init__.py`, which is further evidence that the file
  came from a historical fork/configuration rather than the current source.

This is a `HISTORICAL_CHECKPOINT_STRICT_LOAD` for the old TempRMOT lineage. It
is not a `CHECKPOINT_REPRODUCED` result for the current FGRefMOT source, which
still has the DeepFuse/Mamba/text-attention namespace mismatch. The checkpoint
SHA256 is recorded in `PHASE0_ASSET_AUDIT.md`.

## Archived experiment artifacts found after the initial bounded audit

The first scan missed deeper experiment directories. The current bounded audit
finds the following useful lineage anchors:

| Archive artifact | Evidence | Current classification |
| --- | --- | --- |
| `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT/exps/default_rk/checkpoint0049.pth` | 1,096,317,848 bytes; released TempRMOT source; matching `results_epoch49` contains 158 `predict.txt` and 158 `gt.txt` pairs. The archived summary reports HOTA 50.641, DetA 39.406, AssA 65.241. A local copy has SHA256 `55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`. | **UPSTREAM_BASELINE_ARCHIVE / ONE-PAIR REPRODUCED**. This is the strongest old-baseline candidate. Its output is still historical archive evidence except for the explicitly recorded single-pair rerun. |
| `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/official_checkpoint/checkpoint_0060.pth` | Same size and SHA256 as the TempRMOT baseline checkpoint; 969-key pre-Mamba QIM state dict; strict-restores against the old TempRMOT namespace with the RoBERTa compatibility shim described above. | **BYTE-IDENTICAL BASELINE COPY / HISTORICAL CHECKPOINT**. Do not pair it with current DeepFuse/Mamba source. |
| `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/default_debug_rk/checkpoint0089.pth` | 1,185,173,635 bytes; `data.pkl` names `track_embed.mamba_layers1/2`; corresponding `results_epoch89` has 141 prediction/GT pairs and archived summary HOTA 56.880, DetA 56.297, AssA 57.514. Local copy has SHA256 `adb2ad9591a4a14dcdd05718b271d6c7b391e97bd9cb20f4fd8e3bfe26e54038`; strict-load passes against TempMam and one complete 246-frame pair reproduces under local TrackEval. | **USER/RESEARCH MAMBA EXPERIMENT ARCHIVE**. One-pair checkpoint reproduction is established; dataset-wide reproduction is still pending. |
| `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT/exps/epoch120/checkpoint0119.pth` | 1,185,173,571 bytes; paired `results_epoch119` archived summary HOTA 48.777, DetA 37.150, AssA 64.158. | **RESEARCH MAMBA/VARIANT ARCHIVE**, not the clean upstream baseline. |

The following files were also found in `TempMam_frez`:

* `exps/official_checkpoint/checkpoint_0060.pth`, 1,096,317,848 bytes;
* `exps/official_checkpoint/v2/checkpoint_rk.pth`, 1,096,317,848 bytes;
* `exps/official_checkpoint/v2/checkpoint0002.pth`, 1,096,316,824 bytes;
* several `default_debug_rk` checkpoints around 845 MB.

The 845 MB `TempMam_frez/exps/default_debug_rk/checkpoint*.pth` series and the
`official_checkpoint/v2` files are **freeze/variant artifacts**, not the clean
upstream baseline. Their sizes and paths were recorded without treating them
as interchangeable. Two attempts to deserialize the variant files were
interrupted because the SSHFS read entered an uninterruptible remote-file
wait; no compatibility claim is made for those variants yet.

## Handling rule

For the FGRefMOT-v2 work, use the local checkout
`/data1/liuyeqiang/FGRefMOT_v2_research` on branch
`research/fgrefmot-v2-temporal-grounding` for source changes. Keep all NAS
checkpoints read-only. Treat `TempRMOT` as the upstream/reference baseline,
`TempMam` as the current research source, and `TempMam_frez` as an ablation or
freeze variant until a future checkpoint audit proves a different pairing.
