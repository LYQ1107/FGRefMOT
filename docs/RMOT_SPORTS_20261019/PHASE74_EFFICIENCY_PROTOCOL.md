# Phase 74 — reproducible efficiency and memory measurement

The efficiency gate now has a read-only measurement path.  `scripts/efficiency.py`
defines the units and joins the physical tracker and referring-module records;
`scripts/measure_efficiency.py` writes a JSON report without reading labels or
ground truth.  The complete-pipeline rate is only emitted when both stage
times are present.

The reported units are explicit:

- tracker FPS is the sum of the physical tracker output `max_frame` values
  divided by tracker wall time;
- video-frame FPS is unique video frames divided by referring inference time;
- referring-module FPS is expression-video-frame pairs divided by referring
  inference time;
- complete-pipeline FPS is unique video frames divided by tracker wall time
  plus referring inference time.

FlexHook baseline and learned temporal-head evaluation now record model-loading
time, inference time, peak GPU memory, and both referring rates in their
evaluation manifests.  YesTrack's upstream timing file records the same
values and the Refer-Sports adapter copies them into its evaluation manifest.
Future common-tracker runs record wall time and tracker FPS in their run
manifest.

The previously completed FlexHook F1 and common-tracker manifests were
created before this instrumentation and do not contain enough timing data.
They are therefore left unchanged and no efficiency number is claimed for
them.  The strict core runs are still active; their post-training watchers
will use the new fields.

Validation: 18 targeted tests passed, and all modified Python files compiled
in the project environment.
