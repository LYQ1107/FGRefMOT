# Speed and memory report

Status: `PARTIAL — COMPLETE-PIPELINE AND YES_TRACK_FORMAL TIMING PENDING`

Measurements use the registered efficiency definitions: referring-module
expression-video-frame pairs per second, unique video-frame FPS, model-load
time, and peak GPU memory. Complete-pipeline FPS additionally needs the common
tracker wall time.

| run | referring FPS | video FPS | model load (s) | peak GPU (MiB) |
| --- | ---: | ---: | ---: | ---: |
| FlexHook H8 internal | 15.333 | 1.901 | 334.918 | 12,966 |
| FlexHook H8 historical | 18.301 | 2.557 | 323.616 | 12,976 |
| YesTrack internal | 2.744 | 0.340 | 570.194 | 13,516 |
| YesTrack historical | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

The physical tracker manifest is complete, but its wall-time join and the full
YesTrack historical timing manifest remain pending. The measurement protocol
and read-only tooling are documented in Phase 74.
