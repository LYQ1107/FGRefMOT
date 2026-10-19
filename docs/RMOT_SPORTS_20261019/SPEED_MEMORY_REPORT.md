# Speed and memory report

Status: `PARTIAL — YESTRACK_FORMAL_TIMING_PENDING`

Measurements use the registered efficiency definitions: referring-module
expression-video-frame pairs per second, unique video-frame FPS, model-load
time, and peak GPU memory. Complete-pipeline FPS additionally needs the common
tracker wall time.

| run | referring FPS | video FPS | complete-pipeline FPS | model load (s) | peak GPU (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlexHook H8 internal | 15.333 | 1.901 | NOT_JOINED | 334.918 | 12,966 |
| FlexHook H8 historical | 18.301 | 2.557 | 1.909 | 323.616 | 12,976 |
| YesTrack internal | 2.744 | 0.340 | NOT_JOINED | 570.194 | 13,516 |
| YesTrack historical | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

The timed common tracker completed all 19 historical-test sequences (11,900
frames) in 1,580.706636 seconds, or 7.5283 physical tracker FPS. The H8
historical referring-evaluation timing was joined with that terminal tracker
manifest; the complete-pipeline result is 11,900 / (1,580.706636 +
4,653.554735) = 1.9088 FPS. The tracker manifest and machine-readable join are
recorded in Phase 86. YesTrack historical timing remains pending until its
formal evaluation writes a terminal manifest. The measurement protocol and
read-only tooling are documented in Phase 74.
