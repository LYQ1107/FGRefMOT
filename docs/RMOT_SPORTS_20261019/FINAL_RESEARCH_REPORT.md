# Rebuilding and improving strong RMOT baselines on Refer-Sports

Status: `IN_PROGRESS — DO NOT TREAT AS FINAL RESULTS`

This report is the final-report entry point required by the research plan. It
is intentionally incomplete while the YesTrack historical evaluation,
multi-seed validation, postbaseline controls, and complete online timing are
running.

## Verified protocol

Training uses the registered Sports layouts. Checkpoint and threshold choices
use only the video-disjoint internal split. The 19-video historical split is
offline TrackEval only; its GT never enters model input, tracker input,
training, checkpoint selection, or threshold tuning.

## Current evidence

| field | value |
| --- | --- |
| FlexHook F1 historical HOTA | 68.385415 |
| YesTrack Y1 historical HOTA | NOT_RUN — formal manifest pending |
| FlexHook H8 temporal historical HOTA | 71.912813 (selected temporal experiment) |
| best baseline | PENDING YesTrack formal row |
| best proposed model | PENDING ablation gate |
| HOTA/AssA/IDF1 improvement | NOT_RUN |
| hard-subset improvement | NOT_RUN; labels insufficient for current candidates |
| generalization claim | NOT_SUPPORTED until multi-seed and final gates |
| FlexHook H8 complete-pipeline FPS | 1.908807 on the 19-video historical split; YesTrack comparison pending |
| total GPU hours | PENDING final accounting |
| local peak storage | PENDING final accounting |
| NAS checkpoint root | `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019` |
| data split verified | YES |
| final commit | PENDING final audit |
| final goal status | IN_PROGRESS |

The reviewer conclusion is therefore not yet a claim that the proposed method
beats the strongest baseline. The final answer must be filled only from
successful manifests and the completed gate audit.
