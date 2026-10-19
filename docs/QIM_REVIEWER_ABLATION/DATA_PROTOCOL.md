# Refer-Sports data protocol

The archived audit manifest is `/data1/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/metadata/datasets/refer_sports_manifest_global_ids_rev2.json` with SHA256 `88e0ba5cfcf522887237af7fd4be075b7d0ad4cbac373ab56fbcd8fda21e25a`.

Verified archive facts:

- 41 videos; 22 historical Train videos and 19 historical Test videos.
- 323 expression JSON files (181 train, 142 test), not the approximately 492 cited in the attachment.
- 141 entries in the historical test seqmap; one archived test expression is retained as an audit warning.
- Expression IDs use the global `labels_with_ids` namespace.
- Positive annotations are `TRUE`; absent/outside-interval states are `UNKNOWN`, not explicit negatives.

The development protocol is intended to split the 22 Train videos by source video or match group. The final historical protocol retrains the selected configuration on all 22 Train videos and evaluates once on the fixed 19-video historical Test split. Thresholds and checkpoint selection must be frozen before Test evaluation.

This branch updates `datasets/refer_kitti.py` so legal clip starts are built per source video and cannot cross a video boundary. The original binary `is_ref` behavior is preserved for compatibility; the Sports manifest's `UNKNOWN` semantics still require a future Sports adapter/loss mask before formal training.

Current blocker: the manifest's image root and labels are not mounted locally, so neither the development nor final protocol can be executed yet.

