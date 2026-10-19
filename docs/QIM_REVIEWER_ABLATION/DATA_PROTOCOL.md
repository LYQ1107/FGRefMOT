# Refer-Sports data protocol

The archived audit manifest is `/data1/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/metadata/datasets/refer_sports_manifest_global_ids_rev2.json` with SHA256 `88e0ba5cfcf522887237af7fd4be075b7d0ad4cbac373ab56fbcd8fda21e25a`.

Verified archive facts:

- 41 videos; 22 historical Train videos and 19 historical Test videos.
- 323 expression JSON files (181 train, 142 test), not the approximately 492 cited in the attachment.
- 141 entries in the historical test seqmap; one archived test expression is retained as an audit warning.
- Expression IDs use the global `labels_with_ids` namespace.
- Positive annotations are `TRUE`; absent/outside-interval states are `UNKNOWN`, not explicit negatives.

The development protocol uses the fixed video split in `DEV_SPLIT_TRAIN.txt` (18 videos) and `DEV_SPLIT_VAL.txt` (4 videos), selected deterministically from the 22 manifest Train videos. The final historical protocol retrains the selected configuration on all 22 Train videos and evaluates once on the fixed 19-video historical Test split. Thresholds and checkpoint selection must be frozen before Test evaluation.

`datasets/refer_sports.py` converts MOT-style SportsMOT images and global `labels_with_ids` into the original FGRefMOT clip interface. It enforces same-video clips, deterministic expression selection per epoch/sample, and stable video-offset track IDs. `sports_ref_protocol=unknown_mask` marks only explicitly positive expression IDs as valid referring targets; absent IDs remain UNKNOWN and are excluded from the referring loss. `legacy_binary` is available only as a separately recorded compatibility control.

The test image tree is present, but the published test split has no local GT labels. Formal development metrics can therefore run on the fixed four-video development validation split; final historical Test tracking metrics remain blocked until the corresponding GT is supplied.

