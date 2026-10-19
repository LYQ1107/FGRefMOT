# Phase 79 — YesTrack multi-seed output isolation repair

Status: `V2_TRAINING_IN_PROGRESS`

An audit of the first multi-seed launch found that the unmodified YesTrack
launcher creates `runs/%Y%m%d_%H%M%S`. Seed43 and seed44 reached that code in
the same second and both resolved to
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/20261020_055735`.
The shared `config.json` identified seed43, so neither attempt was eligible
for a stability claim. Both processes were stopped at exit code 143 before
checkpoint save; their manifests and logs are retained as invalid-attempt
evidence.

The replacement v2 launchers use separate NAS roots:

- seed43: `.../runs/yestrack_y1_sports_lora_seed43_v2/`, GPU2;
- seed44: `.../runs/yestrack_y1_sports_lora_seed44_v2/`, GPU3.

The local `runs` symlink for each launcher resolves to its own NAS root, so a
same-second upstream timestamp cannot collide across seeds. Both v2 manifests
are `RUNNING`; the internal-evaluation watchers wait for their own manifests
and will reuse seed42's fixed threshold 0.8. GPU1's formal historical test
process was not changed.

Validation of the repair: launcher and watcher `bash -n` checks pass, the
v2 manifests record distinct roots and seeds, and the live GPU processes are
using GPU2/GPU3. No historical-test metric is claimed from either v1 or v2.
