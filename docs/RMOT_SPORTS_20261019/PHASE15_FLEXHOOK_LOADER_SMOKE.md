# Phase 15 — FlexHook SportsMOT loader smoke test

Audit date: 2026-10-19  
Status: **DATA LOADER SMOKE OK / EXTERNAL ASSETS PENDING / NO GPU JOB STARTED**

The new `scripts/flexhook_sportsmot_smoke.py` adapter exercised the pinned
FlexHook `RMOT_Dataset_mix` loader against the prepared SportsMOT layout.  It
constructed both splits and fetched one item from each without importing a
model or allocating CUDA.  Because the official RoBERTa directory is still
absent, this run used the script's explicit shape-only stub tokenizer; it is
not a model or a substitute asset.

| split | loader samples | first item tensor shapes |
|---|---:|---|
| train | 5,491 | images `[4,3,224,672]`; grid `[4,2,16,48]`; boxes `[4,4]`; labels `[8]`; text ids/mask `[8,25]` |
| val | 1,195 | images `[4,3,224,672]`; grid `[4,2,16,48]`; boxes `[4,4]`; labels `[8]`; text ids/mask `[8,25]` |

The complete machine-readable output is
`PHASE15_FLEXHOOK_LOADER_SMOKE.json`.  This verifies the dataset/layout
boundary and the configured four-frame/25-token tensor contract.  It does not
verify pretrained weights, official tokenization, model forward/backward,
tracker output, metrics, or a training run.

The external asset preflight remains pending.  After the official RoBERTa and
ROPE-Swin files plus the common tracker arrive, repeat this command with
`--tokenizer local --roberta-dir <verified-dir>`, pass the selected F0/F1/F2
contract, and only then start the model smoke test.

