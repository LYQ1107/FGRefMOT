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

The current contracts are deliberately pending:

| Method | Contract | Result |
|---|---|---|
| FlexHook | `FLEXHOOK_BASELINE_RUN_CONTRACT.json` | `PENDING` — RoBERTa, ROPE-Swin, task checkpoint and common tracker are absent; protocol values are placeholders |
| YesTrack | `YESTRACK_BASELINE_RUN_CONTRACT.json` | `PENDING` — LoRA, Qwen base model and common tracker are absent; protocol values are placeholders |

The gate rejects annotation/GT input as a tracker substitute.  It also rejects
placeholder strings such as `PENDING_*` and `TO_BE_FILLED`; after assets arrive,
the official run command and all thresholds/hyperparameters must be written
into a new contract.  A future launch must first pass both checks:

```bash
python3 scripts/run_contract.py --check docs/RMOT_SPORTS_20261019/<contract>.json --require-ready
python3 scripts/run_contract.py --write ... --require-ready
```

The second command is expected to return status 2 until the Phase 7 preflight
is `READY`.  No training, evaluation, metric, or improvement claim is made by
this phase.  The machine-readable summary is `PHASE10_RUN_CONTRACT.json`; the
two checked-in contract snapshots are `FLEXHOOK_BASELINE_RUN_CONTRACT.json`
and `YESTRACK_BASELINE_RUN_CONTRACT.json`.
