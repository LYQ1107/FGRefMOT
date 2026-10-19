# Refer-Sports asset manifest

Status: `AUDITED — LARGE ASSETS STAY ON NAS`

This index records the assets used by the isolated Sports route. Paths are
machine-local evidence paths; they are not public download links.

| asset | verified path | role | state |
| --- | --- | --- | --- |
| Refer-Sports expression archive | `/data3/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti/sports_expression.zip` | source expressions and frame-to-track labels | audited; 41 videos / 323 JSON files |
| historical training list | `/data3/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/data/Dataset/refer-kitti/KITTI/training/refer-kitti.train` | 22-video source split | audited |
| historical test list | `/data3/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/datasets/data_path/seqmap_sports.txt` | 19-video historical test split | audited |
| SportsMOT frames and labels | `/data3/liuyeqiang/datasets/SportsMOT/sportsmot_publish/dataset` | image links and `labels_with_ids` | reused read-only |
| fixed common tracker | `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_dab_refersports_train_20261019` | model-time candidate tracks | GPU run succeeded; `used_ground_truth=false` |
| FlexHook F1 checkpoint | `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_sports_trained_seed42/ckpt_epoch_best_0.pth` | Sports-trained baseline | SHA-256 recorded in Phase 29 |
| FlexHook H8 temporal checkpoint | `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_temporal_h8_internal_core_20261019/temporal_head_best.pt` | selected causal temporal head | SHA-256 `5c3d5c39115f...` |
| YesTrack Qwen base | `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/yestrack/Qwen3-VL-2B-Instruct` | referring-model initialization | pinned NAS asset |
| YesTrack seed42 LoRA | `/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/20261019_180459/train/lora_epoch_0` | full Sports training adapter | training succeeded; formal evaluation in progress |

The NAS project root is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019`. Raw videos,
weights, tracker outputs, logs, and credentials are excluded from GitHub.
Detailed source hashes and provenance are in Phases 0, 4, 14, 20–23, 29,
49, 60, 70, and 72.
