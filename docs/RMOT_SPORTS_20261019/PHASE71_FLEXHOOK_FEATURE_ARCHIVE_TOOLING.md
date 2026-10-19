# Phase 71 — FlexHook temporal feature archive tooling

Status: `SUCCEEDED_TRAIN_CORE_AND_INTERNAL_VAL_ARCHIVES`

`scripts/flexhook_feature_archive.py` now provides the reproducible bridge from
the frozen FlexHook checkpoint to the project-level causal temporal head. It
accepts only the registered `train_core` and `internal_val` layouts, hooks the
real `model.output_ca_qkv[-1]` representation and frozen text encoder, and
stores frozen visual features, pooled language features, logits, clip-frame
metadata, physical track IDs, terminal boxes, expressions and supervised labels
in hashable NAS shards.

The archive command never passes labels to FlexHook or to the common tracker;
labels are retained as temporal-head training targets. Historical test is not an
accepted archive split and remains an offline TrackEval-only evaluation set.
The real strict-core checkpoint produced a successful `train_core` archive with
4,360 samples across nine shards and an `internal_val` archive with 448 samples
in one shard. The manifest hashes and NAS paths are recorded in the JSON report.
The four focused helper tests remain green; no learned-method metric is claimed
until the temporal head is trained and selected.
