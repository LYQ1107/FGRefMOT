# Phase 1 — storage I/O smoke benchmark

Status: `COMPLETE_SYNTHETIC_SMOKE; CHECKPOINT_TRANSFER_NOT_RUN`

On 2026-10-19, a self-cleaning 16 MiB deterministic temporary file was written,
fsynced, read and hashed on the local project filesystem and on the mounted NAS.
Thirty-two deterministic 4 KiB random reads were then issued on each file. The
temporary files were removed after each measurement; no project data or
checkpoint was copied.

| filesystem | sequential write | sequential read | 32 random reads |
| --- | ---: | ---: | ---: |
| `/data3/liuyeqiang` | 64.952 MiB/s | 221.140 MiB/s | 0.162 ms |
| `/data3/liuyeqiang/nas/A100` | 7.741 MiB/s | 8.040 MiB/s | 313.456 ms |

Both files produced the same SHA256 (`bb63a19be8c15da713b946c0a02cea5365825bc3b5b973b217e8f395339a0780`).
The result supports keeping long-lived checkpoints, tracker outputs and reports
on NAS while avoiding unnecessary random-frame hot-cache traffic there. A real
checkpoint write/restore transfer has not been claimed by this smoke benchmark;
that remains a separate archive/restore verification gate.

The command and machine-readable result are recorded in
`scripts/storage_io_benchmark.py` and `runs/storage_io_benchmark_20261019.json`.
