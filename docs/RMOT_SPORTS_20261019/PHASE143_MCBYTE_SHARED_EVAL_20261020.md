# Phase 143 — Fast leakage-safe shared McByte evaluation (2026-10-20)

## Outcome

The speed-first detector run and both downstream evaluations completed successfully. One leakage-safe YOLOX-X detector was trained on the 26-video SportsMOT train remainder and its McByte outputs were reused unchanged by FlexHook and the reported `YesTrack+McByte` variant. No evaluation video or evaluation annotation entered detector training.

## Fast detector run

- Initialization: generic COCO YOLOX-X (`yolox_x_coco_init.pth`), not the official all-train SportsMOT detector.
- Clean split: 26 videos, 16,674 frames, 182,427 boxes; the 19 Refer-Sports evaluation videos were excluded by video ID.
- Speed settings: 8 A100 GPUs, global batch 64, fixed 512×896 input, FP16, 5 epochs, no random resize, no mixup, and no in-training validation.
- Wall time: about 28.3 minutes (11:00:24–11:28:40 +08:00).
- Checkpoint: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_sportsmot_loo19_fast5_20261020/mcbyte_yolox_x_loo19_latest_ckpt.pth.tar`.
- Checkpoint SHA-256: `640d475899f2c38e273bed8a0e8ceb4b223f78c653d84bac2791c89f02562944`.

## Shared tracker

`TRACKER_MANIFEST.json` records 19/19 output files, `used_ground_truth=false`, and `used_evaluation_frames_for_training=false`. The McByte settings were test size 512×896, detector confidence 0.1, NMS 0.7, track threshold 0.5, buffer 30, and match threshold 0.8. The exact outputs are consumed by both downstream methods.

## Final aggregate metrics

At TrackEval threshold 0.5:

| Method | HOTA(0) | LocA(0) | MOTA | MOTP | IDF1 | CLR_F1 |
|---|---:|---:|---:|---:|---:|---:|
| FlexHook + McByte | 0.595210 | 0.820196 | 0.380974 | 0.827922 | 0.527654 | 0.745966 |
| YesTrack + McByte | 0.668461 | 0.816588 | 0.719859 | 0.823856 | 0.611555 | 0.870964 |

YesTrack threshold sweep (same 142 expressions at every threshold):

| Threshold | HOTA(0) | MOTA | IDF1 | CLR_F1 | Prediction rows |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.662639 | 0.685285 | 0.603143 | 0.858385 | 407534 |
| 0.3 | 0.664748 | 0.696347 | 0.605950 | 0.862440 | 401944 |
| 0.4 | 0.666350 | 0.706637 | 0.608436 | 0.866137 | 396086 |
| 0.5 | 0.668461 | 0.719859 | 0.611555 | 0.870964 | 388486 |
| 0.6 | 0.671132 | 0.739325 | 0.615907 | 0.878052 | 376157 |
| 0.7 | 0.672464 | 0.752965 | 0.618650 | 0.882863 | 365587 |
| 0.8 | 0.672093 | 0.768362 | 0.619682 | 0.887470 | 347512 |

## YesTrack inference protocol

- Qwen3-VL-2B-Instruct with the existing one-epoch Y1 LoRA checkpoint; this is reported as `YesTrack+McByte`, not the untouched official TempRMOT* tracker.
- 32 expression shards ran in parallel. CPU thread caps (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) kept the GPUs saturated without host oversubscription.
- Every frame was inferred (`infer_every_n_frames=1`); re-refer lower/upper thresholds were 0.2/0.8. Stability testing was disabled as requested.
- 142/142 prediction files were merged before the single offline TrackEval pass. Ground truth was used only for metric computation.

## Artifacts

- Detector split manifest: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_leakage_safe_detector_loo19_20261020/SPLIT_MANIFEST.json`
- Shared tracker manifest: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_sportsmot_loo19_fast5_20261020/track_results/TRACKER_MANIFEST.json`
- FlexHook evaluation manifest: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_mcbyte_final_20261020/FLEXHOOK_EVAL_MANIFEST.json`
- YesTrack evaluation manifest: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/yestrack_y1_mcbyte_final_20261020/YESTRACK_EVAL_MANIFEST.json`
- YesTrack shard merge manifest: `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/yestrack_y1_mcbyte_final_20261020/YESTRACK_SHARD_MERGE_MANIFEST.json`

Previous SambaMOTR/DAB numbers are outside this leakage-safe shared-McByte protocol and are not included in the final comparison.
