# YesTrack ablations

Status: `WAITING_FOR_Y1_FORMAL_BASELINE`

The registered causal comparison is YB0–YB6:

| variant | description | state |
| --- | --- | --- |
| YB0 | native Y1 | waiting for formal baseline |
| YB1 | native Y1 + official TCP/TRP | watcher registered |
| YB2 | YB1 + adaptive verification interval | watcher registered |
| YB3 | YB1 + causal EMA | watcher registered |
| YB4 | YB1 + evidence-validity control | watcher registered |
| YB5 | YB4 + identity memory | watcher registered |
| YB6 | evidence-supported combination | watcher registered |

All variants must reuse the fixed YesTrack seed42 adapter, common tracker,
historical sequence list, and internal-selected threshold 0.8. The current
watcher may collect the registered score grid for diagnostics, but the final
comparison is frozen at 0.8. No result is claimed until the corresponding
manifest succeeds; these score controls are not a substitute for retraining a
new Qwen model.
