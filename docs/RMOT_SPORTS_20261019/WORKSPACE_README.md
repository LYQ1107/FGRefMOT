# Refer-Sports RMOT research route

This branch records the independent Refer-Sports execution route. Phase 0
verified the server/NAS/GPU state; Phase 1 created the NAS storage layout and
atomic checkpoint tooling; Phase 2 audited the existing 41-video expression
archive and wrote the generated manifest to the NAS metadata tree.

The historical source tree and raw frames remain external read-only inputs.
Training, evaluation and any claimed metric are marked `NOT_RUN` until a real
command produces a log, checkpoint hash and evaluation artifact.
