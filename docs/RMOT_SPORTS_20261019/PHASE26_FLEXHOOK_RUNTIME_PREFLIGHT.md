# Phase 26 — FlexHook runtime dependency preflight

Audit date: 2026-10-19  
Status: **FLEXHOOK ENTRYPOINT IMPORT READY / DEPENDENCIES RECORDED / F1 GPU JOB NOT YET STARTED**

The F1 contract and data smoke were ready, but the first CPU-only import of the
pinned FlexHook entrypoint exposed two undeclared upstream runtime dependencies:
`tensorboardX` and `info_nce`. The dedicated environment was repaired without
editing `third_party/FlexHook`:

* `tensorboardX==2.6.2.2`;
* `protobuf==5.29.3` (the tensorboardX runtime dependency);
* `info-nce-pytorch==0.1.4` (provides the `info_nce` module imported by
  FlexHook's loss implementation).

The installed package files are inside
`/data3/liuyeqiang/.venvs/fgrefmot-rmot-sports-20261019`; `pip check` passes.
With `CUDA_VISIBLE_DEVICES=''`, the real upstream entrypoint was imported via
`scripts/launch_flexhook.py` and completed without a missing-module error. This
was an import/runtime preflight only; it did not initialize a training GPU or
write predictions.

The launch manifest
`runs/gpu/flexhook_f1_sports_trained_seed42/RUN_MANIFEST.json` records the
resolved environment versions, physical GPU selection, contract hash, complete
command, and output/log paths. Its SHA-256 is
`2028e6cb3bf7f5d22ca3ebe0932b14992d00402eeffe8c01a0c2337d9e00a2f9`.
The manifest was published to the NAS run directory before launch.

The vendored source remains unchanged. The pinned entrypoint hashes are:

* `third_party/FlexHook/main.py` —
  `9ab23f7cb3e253828d9afbd10697b5cec9bb68f46d70c0d9ed3d130d77c55089`;
* `third_party/FlexHook/loss.py` —
  `de196cb23842f66cc2fd95a45267bd5b75eb257d4478f3ced0fe09840192fa83`.

The next action is the already-authorized F1 single-GPU launch on the freshly
rechecked physical GPU 1. The other project processes remain out of scope.
