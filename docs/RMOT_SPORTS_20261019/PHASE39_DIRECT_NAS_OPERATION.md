# Phase 39 — Direct NAS operation during YesTrack Y1

Status: `RUNNING_GPU1`.

The YesTrack Y1 run reads the base Qwen3-VL checkpoint from the NAS mount and
writes its relative `runs/` output through a project-local symlink whose target
is on the NAS.  No complete model or checkpoint is downloaded to local disk for
this run.

Verified paths:

- base model: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/yestrack/Qwen3-VL-2B-Instruct`
- run target: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/20261019_161037`
- project-facing link: `runs/gpu/yestrack_y1_sports_lora_seed42/runs`
- local pre-NAS safety backup: `runs/gpu/yestrack_y1_sports_lora_seed42/runs.local_before_nas.20261019162512`

The NAS SSH endpoint was checked through the restored reverse tunnel with the
read-only key and returned `NAS_OK`.  The training manifest records the storage
mode as `NAS_DIRECT_THROUGH_SSHFS_MOUNT`.  The active post-training watcher is
waiting for this process to finish, then waits for GPU1 to be free before
generating the fixed Refer-Sports SambaMOTR tracker outputs on NAS.

At report time the log had reached approximately 3,775 of 103,988 steps.  This
is a progress observation, not a completed metric or final result.
