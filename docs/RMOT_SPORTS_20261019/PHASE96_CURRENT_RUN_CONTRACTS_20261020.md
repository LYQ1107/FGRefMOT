# Phase 96 — current source-backed run contracts

Status: `CURRENT_PREFLIGHT_AND_F1_Y1_CONTRACTS_READY` at 07:39:47 CST.

The original method-level contracts were generated before the external assets
and common tracker were installed, so they intentionally described a pending
launch.  I reran the finite-file preflight against the current NAS assets and
generated source-backed contracts for the two variants that are actually used:

- `FLEXHOOK_F1_RUN_CONTRACT_20261020.json` is `READY` for the Sports-trained
  FlexHook route.
- `YESTRACK_Y1_RUN_CONTRACT_20261020.json` is `READY` for the fresh Sports-LoRA
  route.

The aggregate preflight still reports `PENDING` only because the optional
zero-shot/task-checkpoint assets are absent.  Those assets are not required by
F1 or Y1, and the variant-specific contract gate excludes them.  Both contracts
record the 17-video training core, the five-video internal selection split,
seed 42, the fixed common detector-output tracker, explicit thresholds, and the
actual launch command.  The preflight and contract files include SHA-256
provenance in the machine-readable companion JSON.

Validation commands:

```text
venv/bin/python scripts/preflight_baseline_assets.py --hash-files --output runs/baseline_asset_preflight_20261020/current_report.json
venv/bin/python scripts/run_contract.py --check docs/RMOT_SPORTS_20261019/FLEXHOOK_F1_RUN_CONTRACT_20261020.json --require-ready
venv/bin/python scripts/run_contract.py --check docs/RMOT_SPORTS_20261019/YESTRACK_Y1_RUN_CONTRACT_20261020.json --require-ready
```

This is reproducibility metadata only.  It does not promote partial YesTrack
predictions or running multi-seed jobs into metrics, and it does not use the
historical test ground truth for training or selection.
