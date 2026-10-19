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
evaluation manifests.  The completed five-video temporal internal evaluation
measured 1,719.179 seconds of inference, 334.918 seconds of model loading,
15.333 referring pairs/s, 1.901 unique-video-frames/s, and 12,966 MiB peak
GPU memory. The selected temporal checkpoint's 19-video historical run
measured 4,653.555 seconds of inference, 323.616 seconds of model loading,
18.301 referring pairs/s, 2.557 unique-video-frames/s, and 12,976 MiB peak GPU
memory. YesTrack's five-video internal inference completed in 9,605.726 seconds
with 570.194 seconds of model loading, 2.744 referring pairs/s, 0.340
unique-video-frames/s, and 13,516 MiB peak GPU memory; its timing file is
recorded alongside the successful reused-prediction evaluation manifest.
Future common-tracker runs record wall time and tracker FPS in their run
manifest.

The previously completed FlexHook F1 and common-tracker manifests were
created before this instrumentation and do not contain enough timing data.
They are therefore left unchanged. Complete-pipeline FPS still requires the
physical tracker wall-time join, and the full 19-video YesTrack timing/metric
manifest is still running.

Validation: 18 targeted tests passed, and all modified Python files compiled
in the project environment.
