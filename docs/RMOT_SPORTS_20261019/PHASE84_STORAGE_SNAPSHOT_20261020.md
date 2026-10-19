# Phase 84 — Read-only local and NAS storage snapshot

Observed at `2026-10-20T06:49:52+08:00` with `df` and `du -x`. No run
目录 or artifact was changed.

The local `/data3/liuyeqiang` filesystem is 1,007G total, 896G used, and 61G
available (94%). The mounted NAS `/data3/liuyeqiang/nas/A100` is 11T total,
7.2T used, and 3.7T available (66%). The local research workspace occupies
6.5G, its project environment 386M, and the NAS project currently occupies
23G under `runs` plus 5.4G under `assets`. The local `runs/gpu` directory is
5.7G and is the largest project component.

This is an interim snapshot while the YesTrack and timed tracker jobs are live.
The final report must repeat the accounting after terminal runs so generated
outputs are included consistently.
