# Phase 48 — YesTrack unknown-frame label protocol fix

Status: `CORRECTED_RUN_READY`

The first YesTrack Y1 attempt is retained as an invalidated audit artifact. It
was stopped at approximately 10,392 optimizer steps after the protocol audit
found that the upstream loader used `label_map.get(frame, [])`. That converts a
missing expression-frame key into a negative (`No`) sample, while the canonical
Refer-Sports manifest defines missing keys as `UNKNOWN`. The attempt produced no
valid checkpoint or baseline metric and must not be used in the comparison.

The adapter now constructs positive samples first and adds negatives only from
explicit expression-frame keys. An explicit empty list is an annotated frame
and can produce negatives; a missing key is skipped. The policy is recorded as
`EXPLICIT_FRAME_KEYS_ONLY; MISSING_KEYS_UNKNOWN` in the protocol config and in
the dataset object metadata.

The corrected Sports TRAIN_CORE loader smoke test produced:

- 181 train expressions across 22 videos;
- 723,877 supervised samples at `negative_downsample=0.5`;
- 303,818 explicit-key negative samples added;
- 214,833 unknown samples skipped;
- 90,485 optimizer steps for one epoch at batch size 8.

The interrupted attempt measured roughly 2.4–2.7 steps/second on the available
A100. The corrected one-epoch run is therefore expected to take about 9.5–10.5
hours of training, or 10–12 hours including startup and checkpoint finalization.
This is an estimate; the corrected run remains the source of truth.

Validation performed before launch:

- the layout dry-run reports 22 train videos / 14,750 train frames and 19 val
  videos / 11,900 val frames;
- the corrected dataset-construction smoke test reports the counts above;
- the project test suite passes (`24 passed`);
- the old invalid run was stopped without touching the common tracker on GPU 0
  or unrelated GPU jobs;
- the new run directory uses a direct symlink to the NAS `runs` directory.

Formal YesTrack and FlexHook metrics remain pending the corrected checkpoint,
the common tracker completion, and the frozen validation-only threshold scan.
