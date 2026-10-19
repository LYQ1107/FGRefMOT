# Refer-Sports diagnostic subsets

Status: `CANDIDATES_REGISTERED; METRICS_NOT_RUN`

The source labels support expression-level candidate registries, but missing
frame keys are `UNKNOWN`, not negative labels. Therefore no semantic subset
accuracy or event-delay metric is claimed yet.

| subset | train candidates | test candidates | state |
| --- | ---: | ---: | --- |
| S1 same-uniform identity | 104 | 81 | candidate; negative labels unknown |
| S2 temporally valid referring | 54 | 40 | candidate; events incomplete |
| S3 group/subset referring | 118 | 95 | candidate; framewise negatives unknown |
| S4 occlusion/identity persistence | 0 | 0 | `INSUFFICIENT_LABELS` |

The candidate registry is Phase 65. Historical candidates may be used only
after the final method and threshold are frozen.
