# Phase 75 — FlexHook resume-state patch and real smoke-run verification

The pinned FlexHook source previously saved only model weights. The tracked
resume patch now stores the optimizer, scheduler, AMP scaler, epoch, and
Python/NumPy/Torch/CUDA random state, and restores them with an explicit
`weights_only=False`. The launcher applies the patch to the ignored vendor
checkout before each in-process launch and fails closed if the patch or marker
does not match. The runtime manifest records both tracked launch patches.

A real one-GPU recovery run is complete. The first two-epoch smoke run produced
the source checkpoint at
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_resume_smoke_seed42/ckpt_epoch_best_0.pth`.
The resumed run loaded that checkpoint with the log message `loaded
successfully ... (epoch 1)`, entered `Train: [2/3]`, completed epoch 2, measured
validation F1 68.721, saved `ckpt_epoch_best_0.pth`, and exited with code 0.
The NAS checkpoint hash was not computed because a direct read timed out; the
manifest records that limitation instead of inventing a digest.

The first resumed attempt is retained as an infrastructure record: it loaded
epoch 1 and then hit a legacy validation tuple mismatch. Another legacy
postprocess path reached a seqmap list/string type error. For the successful
smoke recovery only, `FGREFMOT_SKIP_FLEXHOOK_POSTPROCESS=1` bypassed that pinned
postprocess block so the training loop could exercise restore, continued
training, validation, and checkpoint save. This bypass is opt-in and is not a
claim that the legacy historical-evaluation postprocess is repaired.

The strict FlexHook core job started before the patch and remains a normal
evaluation artifact; it is not claimed as a patched resume run. Multi-GPU
recovery and worker-local state serialization remain unverified. The smoke F1
is a recovery-sanity observation, not a final baseline or improvement result.

Validation includes the three targeted resume tests, the full project suite
(75 passed, two unrelated deprecation warnings), Python compilation, and
whitespace checks. The machine-readable evidence and run manifests are in the
JSON companion.
