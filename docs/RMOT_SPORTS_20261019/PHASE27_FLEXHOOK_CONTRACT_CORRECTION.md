# Phase 27 — FlexHook F1 launch-gate correction

Audit date: 2026-10-19  
Status: **FIRST LAUNCH BLOCKED BY TYPE-CHECK ERROR / CONTRACT CORRECTED AND HASH-CHECKED / GPU RETRY READY**

The first authorized F1 launch reached the pinned FlexHook config parser but
did not initialize a model or start a training loop. The original contract
passed `DATA.TRAIN SportsMOT` and `DATA.TEST SportsMOT` as scalar strings while
the pinned `train-mix.yaml` defines the mixed dataset fields as lists. YACS
therefore stopped with:

```text
ValueError: Type mismatch (<class 'list'> vs. <class 'str'>) ... DATA.TRAIN
```

The failure is preserved in
`runs/gpu/flexhook_f1_sports_trained_seed42.log`. No checkpoint, prediction,
or result file was written. A post-failure device check found GPU 1 back at
0% utilization and 17 MiB used with no compute process.

The contract was corrected without editing `third_party/FlexHook`. The new
contract
`runs/contracts_20261019/flexhook_f1_sports_trained_seed42.json` has SHA-256
`c09a700ff83175c3af1c56c87c31f9705a47585d40a32b75cab6c720f09d7a91` and passes
`run_contract.py --check --require-ready`. Its type-correct overrides are:

```text
DATA.DATASET mix
DATA.TRAIN ['SportsMOT']
DATA.TEST ['SportsMOT']
sample_frame_stride [30]
val_sample_frame_stride [24]
val_set_scale [5]
SEED 42
```

The list overrides preserve the intended SportsMOT sampling protocol rather
than silently using the first entries of the original mixed-dataset lists.
The absolute layout, tracker, and RoBERTa paths are unchanged. The F1 run
manifest will be updated to this contract before retrying on the rechecked
single physical GPU 1. Other project processes remain untouched.
