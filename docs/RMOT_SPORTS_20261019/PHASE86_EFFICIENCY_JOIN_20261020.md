# Phase 86 — Timed efficiency join

Status: `COMPLETE_FOR_FLEXHOOK_H8_HISTORICAL`.

The common SambaMOTR tracker completed all 19 registered historical-test
sequences, producing 11,900 frames in 1,580.706636 seconds. Its measured
physical throughput is 7.5283 FPS on GPU 0. The terminal tracker manifest is
stored at `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/common_tracker_refersports_timed_20261020/TRACKER_RUN_MANIFEST.json`.

The selected FlexHook H8 historical evaluation used 4,653.554735 seconds of
referring inference, 18.3013 expression-video-frame FPS, and 12,976 MiB peak
GPU memory. Joining the two terminal measurements gives

`11,900 / (1,580.706636 + 4,653.554735) = 1.9088 FPS`

for the complete tracker-plus-referring pipeline. The machine-readable join is
at `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/efficiency_flexhook_h8_historical_timed_20261020.json`.

This result uses the fixed historical-test tracker and the already selected H8
checkpoint. It does not use test data for training or selection, and it does
not provide YesTrack timing or a proposed-method improvement claim.
