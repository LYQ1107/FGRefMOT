# Phase 0 — Old QIM training pause audit

Date: 2026-10-20 UTC  
New worktree: `FGRefMOT_qim_fast6_rmot_fulltest_20261020`  
Source commit: `698019b2a099f7cd9ae20708a51d2aba61578dcd`  
Source branch: `research/qim-reviewer-ablation-20261019`

## Scope and termination

The seven explicitly identified old training processes were sent `SIGTERM`
by PID. No `kill -9`, `pkill`, process-group kill, or broad Python kill
was used. All seven training PIDs were absent after the termination check.
The five old wait/collection queues were also sent `SIGTERM`; PID 18338 had
already exited after its training children stopped. The queues were not
allowed to restart.

The launcher processes were started outside the current shell, so their exact
`wait(2)` exit codes are unavailable. The auditable termination status is
therefore `SIGTERM -> process absent after verification`, rather than an
invented numeric exit code. No traceback or OOM was present in the final log
tails.

## Training processes

| Variant | GPU | Python PID | Parent/launcher | Last observed log position | Approx. live global step | Persisted recovery point |
|---|---:|---:|---:|---|---:|---|
| Standard | 3 | 5694 | 5616/5632 | `Epoch [4] [13390/14684]` | 72,126 | 60,000 |
| LSTM | 4 | 5693 | 5618/5633 | `Epoch [4] [10960/14684]` | 69,696 | 60,000 |
| Temporal Attention | 5 | 5692 | 5617/5631 | `Epoch [4] [10820/14684]` | 69,556 | 60,000 |
| Original Mamba | 2 | 10129 | 10096/10101 | `Epoch [4] [8690/14684]` | 67,426 | 60,000 |
| Ordered Bi | 7 | 15453 | 15386/15387 | `Epoch [2] [13400/14684]` | 42,768 | 20,000 |
| Trackwise | 8 | 15452 | 15386/15388 | `Epoch [2] [13360/14684]` | 42,728 | 20,000 |
| Reliability Hybrid | 9 | 15450 | 15386/15389 | `Epoch [2] [13380/14684]` | 42,748 | 20,000 |

The live step is derived from the logged zero-based epoch and iteration using
`epoch * 14684 + iteration`, matching the checkpoint metadata convention.
It is not claimed to be recoverable: the last complete persisted checkpoint is
the recovery point in the final column.

## Recovery checkpoints

Every listed checkpoint was loaded with `torch.load(..., map_location='cpu')`
and contained all required fields: `model`, `optimizer`,
`lr_scheduler`, `global_step`, `step_in_epoch`, `epoch`,
`rng_state`, and `args`.

The original checkpoint files and all other historical checkpoints remain in
their original directories. The following seven last-valid points were copied
to the pause archive:

`/data1/liuyeqiang/fg_v2_runs/qim_fast6_pause_20261020/checkpoints/`

| Variant | Checkpoint | Bytes | SHA256 |
|---|---|---:|---|
| Standard | `checkpoint_step00060000.pth` | 730,562,673 | `4c37bcc7356501c6eaac6a1b80a234cadde5d2d0c572bb7aef10223347c3ef5a` |
| LSTM | `checkpoint_step00060000.pth` | 781,141,407 | `6e3832ab68bca800f98b69c044bc74dec33ac3a322408f940db110806186522e` |
| Temporal Attention | `checkpoint_step00060000.pth` | 755,937,137 | `6e6e24726f4c6fa923aabda16cf6a782addfab48a1e2c6b6e06b5e08cd93048c` |
| Original Mamba | `checkpoint_step00060000.pth` | 772,745,975 | `20aa697c81cf3d2768f80fb27d85a3ecedb7172a611399dd2e47ac92582ce15b` |
| Ordered Bi | `checkpoint_step00020000.pth` | 731,598,701 | `0061830c9e42844084279a4f6aeea0de79df5202dbd8905603835d1b9a1153d8` |
| Trackwise | `checkpoint_step00020000.pth` | 732,083,885 | `c51c9e1463a121844abfd7507864364ed699ea6bffd21ef20a503a6ca2118b81` |
| Reliability Hybrid | `checkpoint_step00020000.pth` | 732,632,299 | `04cbe584cea8254ff1bbf5f6d3b2eb6af2105d0d61eaf4a1d9b6efc3787a7f8b` |

Source and archived-copy SHA256 values matched byte-for-byte for all seven
files. The source training outputs were not deleted or overwritten.

## GPU and unrelated-process audit

At the initial audit, the seven old trainers occupied GPUs 2, 3, 4, 5, 7, 8,
and 9. After termination, those GPUs reported zero used memory. GPU 1 still
had an unrelated/stale 12,131 MiB allocation, and GPU 6 still had unrelated
processes owned by other projects (`sunming+` PID 4101 and `daiwenw+` PID
25388). Neither was touched. GPU 0 was idle.

The old experiment/queue process match was empty after termination, and the
remaining GPU processes were not part of this project.

## Data/protocol reference

The previously audited Refer-Sports manifest remains the source of record:

`/data1/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/metadata/datasets/refer_sports_manifest_global_ids_rev2.json`

Manifest SHA256:
`160245526fae5c7b5c043e5fce4ec5ddb90bed1bd0da4e4f2a8fcf8b2354da`

This pause audit does not claim that the new six-variant implementation or
full RMOT Test has completed; those are later phases.
