# Phase 0 asset audit

Audit date: 2026-10-19 (client time). The shell observations were made on
host `cieelab`.

## Access and storage

The NAS is mounted read-only at:

`/data1/liuyeqiang/nas/A100`

The mount source is `Valadmin@127.0.0.1:/home/A100` over SSHFS, reached through
the user-maintained reverse SSH forward on port `24557`. A broken forward can
leave a stale FUSE endpoint; remounting with the existing local helper restores
access. The NAS filesystem reports approximately 11 TB total, 7.1 TB used, and
3.8 TB available.

The current host is `cieelab`, user `liuyeqiang`, with local address
`10.130.255.5`. Local `/data1` has approximately 212 GB free. `/data2` is full
and is not a candidate for experiment outputs.

## Source candidates

The primary current-source archive is:

`/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam`

It contains the expected `models/transrmot_pro.py`,
`models/deformable_transformer_plus.py`, `models/qim_pro.py`,
`models/spatial_temporal_reason.py`, datasets, TrackEval, configs, logs, and
experiment directories. A second archive exists at:

`/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam_frez`

The latter contains a related Mamba/freeze variant and is retained as a
comparison source, not silently substituted for the primary current source.

The separate archive
`/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT` is an older
upstream TempRMOT snapshot. It has the released project README and the old
QIM/transformer implementation. Its deeper `exps/default_rk` directory
contains the upstream baseline checkpoint and archived TrackEval outputs. The
full provenance comparison is in `docs/FGREF_V2/ASSET_LINEAGE_AUDIT.md`.

The local checkout is:

`/data1/liuyeqiang/FGRefMOT_v2_research`

It is based on GitHub commit
`d651fec18d4eda54bd141d90cd934b520343b789` on branch
`research/fgrefmot-v2-temporal-grounding`. A tracked-file comparison found no
source differences between this checkout and the `TempMam` archive; the NAS
copy additionally contains data, logs, compiled extensions, and checkpoints.

## Checkpoint candidates

| Category | Absolute path | Size | Provenance status |
| --- | --- | ---: | --- |
| Upstream baseline | `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT/exps/default_rk/checkpoint0049.pth` | 1,096,317,848 bytes | Local copy hash matches the TempMam official candidate exactly; strict historical TempRMOT load and one complete-pair reproduction completed. |
| Historical official candidate | `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/official_checkpoint/checkpoint_0060.pth` | 1,096,317,848 bytes | Strictly restorable against the old `TempRMOT` source with the checkpoint's historical RoBERTa configuration/serialization; not compatible with current `TempMam`; SHA256 recorded below. |
| Freeze-variant official candidate | `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam_frez/exps/official_checkpoint/checkpoint_0060.pth` | 1,096,317,848 bytes | Separate freeze-variant file; not yet compatibility-certified because an SSHFS deserialization read stalled. |
| Training checkpoint | `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/default_debug_rk/checkpoint0059.pth` | 1,185,173,635 bytes | Refer-KITTI debug run; not interchangeable with the official checkpoint without a config and state-dict audit. |
| Training checkpoint | `/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/default_debug_rk/checkpoint0179.pth` | 1,185,173,763 bytes | Later debug run; provenance and evaluation protocol still pending. |

The primary official checkpoint SHA256 is:

`55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`

The local copy of the upstream baseline candidate
`TempRMOT/exps/default_rk/checkpoint0049.pth` has the same SHA256 and size.
This establishes byte-level identity between the old baseline checkpoint and
the archived TempMam official copy; it does not make the old checkpoint
compatible with the current DeepFuse/Mamba source.

Checkpoint loading is performed on CPU only. The primary candidate was loaded
and compared against a model instantiated from the current checkout. To avoid
network access during construction, the comparison used an offline tokenizer,
an offline RoBERTa constructor with the default RoBERTa configuration, and a
non-pretrained ResNet constructor; no checkpoint weights were modified or
written.

The primary checkpoint is not strictly compatible with the current checkout.
It is strictly restorable against the older `TempRMOT` source snapshot when
the checkpoint's historical text-encoder details are reconstructed:

* use `max_position_embeddings=514` and `type_vocab_size=1`, as shown by the
  checkpoint tensors;
* restore the historical `RobertaPooler` that the checked-in old source
  removes after construction;
* mark the RoBERTa `position_ids` buffer persistent, matching the checkpoint's
  serialization behavior in the installed Transformers version;
* the resulting old model has 969 state-dict keys: all 969 intersect the
  checkpoint, with 0 missing keys, 0 unexpected keys, and 0 shape mismatches;
* `model.load_state_dict(checkpoint['model'], strict=True)` succeeds in this
  explicit compatibility audit.

The audit is reproducible with
`tools/fgref_v2/audit_historical_checkpoint.py`; it uses process-local
compatibility shims and never writes to the NAS.

Against the current checkout:

* checkpoint metadata reports `meta_arch='rmot_qim_v2'`, while the current
  catalog exposes `temp_rmot` only;
* the current model has 1,210 state-dict keys and the checkpoint has 969;
* 966 keys intersect, with 964 exact shape matches;
* 244 current-model keys are missing from the checkpoint, 3 checkpoint keys
  are unexpected, and 2 shared keys have shape mismatches;
* the missing groups include all 120 current `deepfuse.*` keys, 88
  Mamba-QIM keys, and 36 decoder text-attention/normalization keys;
* the two shared shape mismatches are RoBERTa position embeddings
  `(512, 768)` versus `(514, 768)` and token-type embeddings `(2, 768)`
  versus `(1, 768)`.

This proves a `HISTORICAL_CHECKPOINT_STRICT_LOAD` for the old TempRMOT
lineage, not a `CHECKPOINT_REPRODUCED` restoration for the current FGRefMOT
source. The compatibility shim was used only in the audit process; the NAS
source and checkpoint were not modified. No `strict=False` load is accepted
as a reproduction result. Its top-level fields are
`model`, `optimizer`, `lr_scheduler`, `epoch`, and `args`; the stored epoch is
59 and the model tensor count is 175,411,974. The SHA256 above identifies the
exact file audited.

## Existing logs and configuration evidence

The primary archive contains historical training and evaluation logs, including
`train_sports.log`, `train_sports1.log`, `train_kittiv2.log`, `test.log`, and
`exps/default/train_log.txt`. The shell configs under `configs/` still contain
historical absolute paths such as `/data1/liuyeqiang/TempMam` and
`/data1/liuyeqiang/RMOT`; these paths must be mapped to verified local or NAS
locations before execution.

## Dataset status

The original Refer-KITTI data root is available at:

`/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti`

The source-side training split is
`KITTI/training/refer-kitti.train`. A real dataset-constructor smoke test
registered 22 videos and 14,750 image entries. It opened the first image
(`1280x720`, RGB), found its corresponding ID label, loaded six expression
JSON files for that video, and converted the first sample to 10 boxes/IDs with
one referring object. This certifies the original Refer-KITTI path for a
smoke run, subject to the full split audit before training.

The separate `refer-kitti-v2` directory is not certified: its `KITTI` symlink
points to `/data1/liuyeqiang/RMOT/data/Dataset/refer-kitti/KITTI`, which is
absent on this host. Its 5,171-line v2 split must not be paired with that
broken link.

## Blocking conditions and next actions

1. Keep the reverse SSH forward alive while reading the NAS; remount on a stale
   FUSE endpoint.
2. Run the strictly restored historical TempRMOT baseline on the verified
   Refer-KITTI path, with its original config and evaluation protocol.
3. Identify whether the `rmot_qim_v2` label denotes an additional historical
   wrapper beyond the strictly matched TempRMOT source.
4. Copy only the required source/config/test files into the local branch; keep
   checkpoints and private data on the NAS.
5. Complete the split-wide image/label/expression integrity audit and run a
   no-GPU import/config smoke test.
6. Reserve an actually free GPU only after checking that existing jobs have not
   released or changed ownership.
