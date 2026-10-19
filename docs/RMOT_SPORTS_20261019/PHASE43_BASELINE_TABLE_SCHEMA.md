# Phase 43 — Fair baseline-table schema

Status: `SCHEMA_READY_METRICS_PENDING`.

`scripts/build_baseline_table.py` creates one shared table for the historical
FGRefMOT reference, FlexHook F0/F1, YesTrack Y0/Y1/Y2/Y3, and the common
SambaMOTR control.  It records pretraining, Sports training, physical tracker,
referring model, checkpoint path, and the requested HOTA/DetA/AssA/IDF1/IDSW,
Ref-F1 and FPS fields.

The builder is conservative by design: it emits `NOT_RUN` until a completed
formal-evaluation manifest exists.  It never converts training loss, an
internal referring score, or the earlier FlexHook training F1 into a formal
tracking metric.  Historical paper values are tagged as a separate protocol
and are not mixed with the current common-tracker comparison.

The table is ready to be regenerated after the real YesTrack training,
SambaMOTR tracker run, and both formal evaluation adapters finish.
