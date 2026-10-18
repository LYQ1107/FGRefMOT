# PHASE 0 — environment, NAS and data discovery

Observation time: `2026-10-19T02:33:50+08:00` on `user-SYS-4029GP-TRT2` as
`liuyeqiang`.

## Server and GPU

- Python available: `Python 3.12.3` (`python` itself is not on `PATH`).
- Git: `2.43.0`.
- `conda` is not on `PATH`; existing local virtual environments were recorded
  separately and are not modified by this phase.
- Ten `NVIDIA A100-SXM4-40GB` devices are present.
- GPU 1 and GPU 4 were idle apart from the driver reservation at the sample
  time. Other GPUs had Python processes owned by another user. Their PIDs and
  jobs were not stopped, reconfigured or otherwise touched.

## Storage and mounts

| filesystem/path | size | used | available | inode use | status |
| --- | ---: | ---: | ---: | ---: | --- |
| `/data3/liuyeqiang` (`/dev/loop23`) | 1007G | 889G | 68G | 7% | writable, near the 95% guard threshold |
| `/data3/liuyeqiang/nas/A100` (`Valadmin@127.0.0.1:/home/A100`) | 11T | 7.1T | 3.8T | 7% | mounted read-only by local SSHFS options |

The existing mount is `fuse.sshfs` with `ro,nosuid,nodev,default_permissions`.
The NAS account is reachable through the established reverse tunnel on
`127.0.0.1:24557`. A controlled write-and-cleanup probe through the existing
NAS SSH key succeeded, and the new project root was created remotely without
touching historical directories:

```
/home/Valadmin/A100/FGREF_RMOT_SPORTS_20261019/ (login-shell path; the SFTP
path used by the transfer script is `/home/A100/FGREF_RMOT_SPORTS_20261019/`.)
```

The local historical mount remains read-only by design. Large transfers should
use `scripts/nas_sync.sh` over the loopback tunnel or a separately validated
writable mount; no remount was performed in this phase. The NAS SFTP endpoint
rejects rsync remote-server mode, so the script uses SCP/SFTP with an atomic
temporary directory and a final rename, followed by an SSH SHA256 check.

## Refer-Sports discovery

The following sources were verified by bounded, path-specific inspection:

- Historical expression archive:
  `/data3/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti/sports_expression.zip`
- Historical training image list:
  `/data3/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti/KITTI/training/refer-kitti.train`
- Historical test expression list:
  `/data3/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/datasets/data_path/seqmap_sports.txt`
- Local SportsMOT frames and MOT labels:
  `/data3/liuyeqiang/datasets/SportsMOT/sportsmot_publish/dataset`

The archive contains 41 sequence directories and 323 JSON expression files.
The training list contains 22 distinct videos and 14,750 image paths. The
Sports test sequence map contains 19 distinct videos and 141 expression names.
All 41 expression videos were found under the local SportsMOT `train` frame
tree, while the Refer-Sports train/test assignment is the 22/19 assignment
from the historical lists. The JSON label format provides positive track IDs
per frame; labels outside the listed frames are retained as `UNKNOWN`, never
silently converted to negatives.

The full machine-readable result is generated at:

- `data/refer_sports_manifest.json`
- `data/refer_sports_manifest.csv`

The generator records source sizes, mtimes, SHA256 values, image counts,
dimensions, frame gaps, target IDs and contiguous temporal intervals. The
manifest warning explicitly records that the historical “approximately 492
expressions” figure does not match the verified archive count of 323.

## Phase status

`PHASE0_DISCOVERY: COMPLETE` for the checks above.

`PHASE0_TRAINING: NOT_RUN` and `PHASE0_OTHER_GPU_JOBS: UNTOUCHED`.
