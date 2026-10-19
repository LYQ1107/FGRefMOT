# Phase 75 — FlexHook resume-state patch

The pinned FlexHook source currently commented out optimizer and AMP-scaler
serialization in its checkpoint writer. It also did not preserve Python,
NumPy, Torch, or CUDA random state, and current PyTorch defaults require an
explicit `weights_only=False` when restoring the serialized configuration.

The research launcher now applies the tracked
`patches/flexhook_resume_state.patch` to the ignored FlexHook checkout before
each future in-process launch. The patch is idempotent and fails closed if the
vendor file or marker does not match. A patched checkpoint contains model,
optimizer, scheduler, AMP scaler, epoch, and per-process random state. The
registered one-GPU path is verified; worker-local sampler state is represented
by the existing deterministic `set_epoch(seed)` policy and is not separately
serialized.

The currently running strict FlexHook core job started before this patch was
added. Its checkpoints remain valid evaluation artifacts, but this phase makes
no real resume claim for that job. A future patched run must produce a real
checkpoint and resume it before the recovery gate can close. This is recorded
as `PATCH_READY_REAL_RUN_PENDING` in the JSON companion.

Validation: the patch applies to an isolated copy, is idempotent, writes all
required state fields, and restores the Python/NumPy/Torch random streams;
both targeted tests passed. The full project suite remains the next combined
verification step after this change.
