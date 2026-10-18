# Storage plan — Refer-Sports RMOT

## Capacity snapshot

The local project filesystem had about 68 GB available at the Phase 0 sample and
was 93% full. The NAS filesystem had about 3.8 TB available and 7% inode use.
The local free-space value is a hard constraint: before a training process or a
checkpoint save, `scripts/storage_guard.py` must be run with a byte budget that
covers a complete optimizer-bearing resume checkpoint plus temporary writes.

The largest local trees identified without crossing the NAS mount were:

| path | size (approx.) | handling |
| --- | ---: | --- |
| `TAO-Amodal` | 357 GiB | existing project; do not delete or move automatically |
| `datasets` | 234 GiB | includes local SportsMOT and unrelated datasets |
| `IR-MFT_assets` | 176 GiB | existing project; do not delete or move automatically |
| `pandas_bytetrack_tao_val` | 33.5 GiB | existing project; do not delete or move automatically |
| `InterMOT_assets` | 27.5 GiB | existing project; do not delete or move automatically |
| `.venvs` | 20.1 GiB | shared environments; do not delete automatically |

No unrelated data is removed to make room for this study.

## NAS project layout

The writable NAS root is:

`/home/Valadmin/A100/FGREF_RMOT_SPORTS_20261019/`

It contains `assets/`, `experiments/`, `checkpoints/`, `metadata/`, `logs/` and
`reports/` subtrees. Raw SportsMOT frames are referenced in place; they are not
duplicated into this root.

## Local versus NAS policy

Local storage is reserved for source code, configurations, adapters, small
validation samples and current runtime files. NAS storage is used for complete
pretrained weights, optimizer-bearing checkpoints, tracker outputs, long logs
and reports. A checkpoint is not considered archived until its size and SHA256
are compared and a `.complete.json` marker is atomically published.

`checkpoint_archive.py` and `checkpoint_restore.py` implement local atomic
file-level archive/restore with hash verification. `nas_sync.sh` transfers to a
temporary NAS path through SCP/SFTP and publishes by rename; it never deletes
remote data. These scripts refuse path traversal and do not embed a password.

## Required run fields

Every run manifest must record:

`LOCAL_FREE_BEFORE`, `NAS_FREE_BEFORE`, `LOCAL_PEAK`, `NAS_USED_BY_PROJECT`,
`LARGEST_MODEL_SIZE`, `CHECKPOINT_RETENTION`, and `CURRENT_RESUME_PATH`.

Until a real training run supplies the last five values, they remain
`UNKNOWN`/`NOT_RUN`; no placeholder value is presented as a measurement.

## Phase status

`PHASE1_STORAGE_LAYOUT: COMPLETE` for directory creation and tooling.

`PHASE1_IO_BENCHMARK: NOT_RUN`; no benchmark or model transfer is started until
the baseline environment is fixed and a GPU is explicitly selected without
interfering with other jobs.
