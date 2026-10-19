# Controlled baseline comparison

Status: `IN_PROGRESS — YES_TRACK_FORMAL_ROW_PENDING`

The table below keeps internal-selection evidence separate from historical-test
evidence. Both referring models use the fixed common SambaMOTR candidate
tracks.

| method | internal HOTA | internal IDF1 | historical HOTA | historical IDF1 | state |
| --- | ---: | ---: | ---: | ---: | --- |
| FlexHook F1 | 71.241265 | 71.282648 | 68.385415 | 66.627791 | complete |
| YesTrack Y1 | 74.652451 | 78.302335 | NOT_RUN | NOT_RUN | formal evaluation running |
| FlexHook H8 temporal | 75.631646 | 72.434749 | 71.912813 | 67.797011 | selected temporal result; proposed gate pending |

H8 is a selected learned temporal-head experiment, not the native FlexHook F1
baseline. The final “best baseline” and any improvement claim remain `PENDING`
until the YesTrack formal row, postbaseline controls, multi-seed results, and
complete online measurements are available. Historical paper numbers are kept
separate from this table.
