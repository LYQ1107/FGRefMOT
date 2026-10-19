# Phase 43 — Fair baseline-table schema

Status: `SCHEMA_READY_VALIDATION_AND_FORMAL_METRICS_SEPARATED`.

`scripts/build_baseline_table.py` creates one shared table for the historical
FGRefMOT reference, FlexHook F0/F1, YesTrack Y0/Y1/Y2/Y3, and the common
SambaMOTR control.  It records pretraining, Sports training, physical tracker,
referring model, checkpoint path, and the requested HOTA/DetA/AssA/IDF1/IDSW,
Ref-F1 and FPS fields.

It records both `validation_split`/`validation_manifest` and
`evaluation_split`/`source_manifest`. Internal 5-video selection values are
written to `validation_*` columns only after a completed internal-validation
manifest. Historical 19-video comparison values are written to `formal_*`
columns only after a completed formal-test manifest, so the two splits cannot
be mistaken for one another.

The builder is conservative by design: it emits `NOT_RUN` until the relevant
completed manifest exists. It never converts training loss, an internal
referring score, or the earlier FlexHook training F1 into a tracking metric.
Historical paper values are tagged as a separate protocol and are not mixed
with the current common-tracker comparison.

The table can be regenerated after each internal or formal evaluator finishes;
the final common-tracker comparison still waits for the YesTrack training and
formal evaluation adapters.
