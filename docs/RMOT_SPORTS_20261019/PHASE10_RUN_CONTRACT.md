# Phase 10 — reproducible baseline run contracts

Audit date: 2026-10-19  
Status: **PENDING EXTERNAL ASSETS / NO GPU JOB STARTED**

The workspace now has a read-only run contract gate in
[`scripts/run_contract.py`](../../scripts/run_contract.py).  A contract records
the exact preflight report, SportsMOT manifest, method-specific layout manifest,
finite asset snapshots and source revision together with the seed, split,
tracker policy, thresholds, hyperparameters and launch command.  The contract
builder never imports a model, allocates CUDA, launches a subprocess, or writes
tracker outputs.

The checked-in default contracts are deliberately pending:

| Method | Contract | Result |
|---|---|---|
| FlexHook | `FLEXHOOK_BASELINE_RUN_CONTRACT.json` | `PENDING` — the default contract includes RoBERTa, ROPE-Swin, task checkpoint and common tracker; protocol values are placeholders |
| YesTrack | `YESTRACK_BASELINE_RUN_CONTRACT.json` | `PENDING` — the default contract includes LoRA, Qwen base model and common tracker; protocol values are placeholders |

The contract builder is variant-aware.  The registered matrix is:

| Variant | Initialization | Task checkpoint required by the contract |
|---|---|---|
| FlexHook F0 `f0_zero_shot` | official public RMOT task checkpoint | yes |
| FlexHook F1 `f1_sports_trained` | RoBERTa/ROPE-Swin public pretraining, Sports-only training | no |
| FlexHook F2 `f2_sports_finetuned` | official task checkpoint followed by Sports training | yes, subject to overlap audit |
| YesTrack Y0 `y0_zero_shot` | public LoRA plus matching Qwen base | yes |
| YesTrack Y1 `y1_sports_lora` | fresh LoRA trained on Sports with matching Qwen base | no |
| YesTrack Y2 `y2_tcp_trp` | frozen Y1 adapter plus TCP/TRP inference | no public adapter; verified Y1 output is a prerequisite |
| YesTrack Y3 `y3_video_training` | Qwen base with video-training flags | no |

For example, after the relevant files arrive, a Sports-only FlexHook contract
is built with `--variant f1_sports_trained`; its aggregate preflight may still
list an absent optional task checkpoint without blocking that variant.  The
common tracker and all base components required by the selected variant remain
hard gates.

The gate rejects annotation/GT input as a tracker substitute.  It also rejects
placeholder strings such as `PENDING_*` and `TO_BE_FILLED`; after assets arrive,
the official run command and all thresholds/hyperparameters must be written
into a new contract.  A future launch must first pass both checks:

```bash
python3 scripts/run_contract.py --check docs/RMOT_SPORTS_20261019/<contract>.json --require-ready
python3 scripts/run_contract.py --write ... --variant f1_sports_trained --require-ready
```

The second command is expected to return status 2 until the selected variant's
required Phase 7 checks are `READY` and all protocol fields are concrete.  No
training, evaluation, metric, or improvement claim is made by this phase.  The
machine-readable summary is `PHASE10_RUN_CONTRACT.json`; the two checked-in
contract snapshots are the backwards-compatible default contracts.
