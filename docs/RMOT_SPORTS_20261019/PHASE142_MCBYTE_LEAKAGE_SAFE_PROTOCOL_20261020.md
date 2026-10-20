# Phase 142 — Leakage-safe McByte protocol (2026-10-20)

## Decision

The original McByte route is retained for FlexHook and is deliberately shared
with a modified ``YesTrack+McByte`` variant.  The official YesTrack release
uses TempRMOT* pure-tracking candidates before referring, so the shared
McByte result must not be described as the untouched original YesTrack tracker.
The official SportsMOT detector checkpoint is not used for the Refer-Sports result, but the
same detector and the same per-video McByte files are used for both methods.
The 19 Refer-Sports evaluation videos are present in the public SportsMOT
`train` split.  A detector trained on the complete public train split can
therefore have seen the evaluation frames before tracking.

McByte itself is a training-free tracking-by-detection method.  The leakage
boundary is the dataset-specific YOLOX detector that supplies its detections,
not the McByte association code or its generic SAM/Cutie models.

## Clean detector split

`scripts/build_leakage_safe_sportsmot_detector_split.py` produced the split
manifest at:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_leakage_safe_detector_loo19_20261020/SPLIT_MANIFEST.json`

The source SportsMOT train annotation contains 45 videos.  The 19 evaluation
videos were removed by video ID, leaving 26 training videos, 16,674 training
frames, and 182,427 training boxes.  The evaluation sequence list and source
annotation hashes are recorded in the manifest.  The images remain available
through read-only symlinks; their evaluation annotations are absent from
`clean_train.json`.

The official SportsMOT validation split is reserved for detector checkpoint
selection, but this speed-first run deliberately skips validation.  The 19
Refer-Sports videos are not used for detector training, validation, checkpoint
selection, or tracker tuning.  Their GT is read only by the final TrackEval
stage.

## Detector training

YOLOX-X is initialized from the generic COCO checkpoint
`yolox_x.pth`, then trained with
`configs/mcbyte_yolox_x_loo19.py` on `clean_train.json`.  No SportsMOT
checkpoint is used as initialization.  The resulting detector checkpoint is
the only detector used to generate McByte outputs for both methods.

For the speed-first run, the detector uses fixed 512×896 inputs, global batch
64 over eight A100s, five epochs, no random multi-scale resize, no mixup, and
no in-training validation.  This is a reproducible fast detector variant, not
the original long YOLOX-X SportsMOT training schedule.  The final checkpoint
is written to
`runs/gpu/mcbyte_yolox_x_fast5_20261020/mcbyte_yolox_x_loo19/latest_ckpt.pth.tar`.

The detector/tracker stage is complete: all 19 McByte output files exist and
the manifest records `used_evaluation_frames=false` and
`used_evaluation_ground_truth=false` for the detector stage.

## Shared tracker outputs

The final checkpoint has SHA-256
`640d475899f2c38e273bed8a0e8ceb4b223f78c653d84bac2791c89f02562944`.
Eight independent GPU workers processed complete sequences, without
interleaving frames from one sequence across workers.  All 19 MOT text files
and the aggregate manifest are on NAS at:

`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_sportsmot_loo19_fast5_20261020/track_results/TRACKER_MANIFEST.json`

The manifest records `sequence_count=19`, `used_ground_truth=false`, and
`used_evaluation_frames_for_training=false`.  These exact McByte files are the
shared tracker input for FlexHook and `YesTrack+McByte`; no metric is claimed
until those downstream evaluations finish.

## Source references

- [McByte official README](https://github.com/tstanczyk95/McByte) describes
  McByte as a tracking-by-detection method that does not require training and
  recommends a SportsMOT-pretrained YOLOX detector.
- [McByte installation](https://github.com/tstanczyk95/McByte/blob/main/INSTALLATION.md)
  names `yolox_x_sports_mix.pth.tar` as the default SportsMOT detector.
- [MixSort official training instructions](https://github.com/MCG-NJU/MixSort#training)
  provide the clean alternative: initialize YOLOX-X from the generic COCO
  checkpoint and train it on the chosen MOT dataset.
- [YesTrack official README](https://github.com/ggbondrighthere/YesTrack) documents
  TempRMOT* as its released pure-tracking candidates; the shared McByte path is
  therefore recorded as `YesTrack+McByte`.

## Claim boundary

All earlier FlexHook and YesTrack numbers produced with SambaMOTR, and any run
using the official all-train SportsMOT detector, are superseded for this
protocol.  Downstream FlexHook and `YesTrack+McByte` metrics still need to be
run against these shared files before they are reported.
