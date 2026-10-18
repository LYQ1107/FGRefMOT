# Refer-Sports RMOT research route

This branch records the independent Refer-Sports execution route. Phase 0
verified the server/NAS/GPU state; Phase 1 created the NAS storage layout and
atomic checkpoint tooling; Phase 2 audited the existing 41-video expression
archive and wrote the generated manifest to the NAS metadata tree; Phase 4
verified the pinned FlexHook and YesTrack source checkouts; Phase 5 prepared a
symlink-only FlexHook-compatible SportsMOT metadata layout.

The historical source tree and raw frames remain external read-only inputs.
Training, evaluation and any claimed metric are marked `NOT_RUN` until a real
command produces a log, checkpoint hash and evaluation artifact.  The Phase 5
layout contains expression metadata and normalized global-ID boxes for the
upstream training loader; it does not provide a detector or physical-tracker
output and must not be treated as one.
