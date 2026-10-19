# FlexHook ablations

Status: `PREBASELINE_SCORE_CONTROLS_COMPLETE; POSTBASELINE_AND_LEARNED_COMPARISONS_PENDING`

The Phase 67 table is a frozen-score causal diagnostic using the same F1
checkpoint, common tracker, and threshold 0.5. It is not a learned temporal
head and is kept separate from the postbaseline gate.

| variant | HOTA | IDF1 | IDSW |
| --- | ---: | ---: | ---: |
| H0 | 68.385415 | 66.627791 | 409 |
| H1 | 68.385415 | 66.627791 | 409 |
| H2 | 68.512428 | 66.749697 | 382 |
| H3 | 68.063179 | 66.337556 | 426 |
| H4 | 68.604619 | 66.829675 | 375 |
| H5 | 68.491772 | 66.729259 | 390 |
| H6 | 68.570292 | 66.798313 | 378 |
| H7 | 68.391700 | 66.633793 | 404 |
| H8 | 68.632825 | 66.857910 | 370 |

The real H8 causal temporal head was trained on the 17-video core and selected
on the five-video internal split. It obtained internal HOTA `75.631646` and
historical HOTA `71.912813` (IDF1 `67.797011`, IDSW `386`). These values are
recorded in Phase 73. They do not establish an improvement until the native
YesTrack baseline and the registered capacity/EMA/GRU controls are complete.
The postbaseline H0–H8 watcher is waiting for the YesTrack formal manifest and
will use the frozen internal threshold 0.8.
