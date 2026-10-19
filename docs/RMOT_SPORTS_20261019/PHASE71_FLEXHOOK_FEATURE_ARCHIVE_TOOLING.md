# Phase 71 — FlexHook temporal feature archive tooling

Status: `IMPLEMENTED_NOT_RUN`

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
The implementation has four focused helper tests and has not yet produced an
archive or a learned-method metric. It will run after the strict FlexHook
core-only checkpoint is complete.
