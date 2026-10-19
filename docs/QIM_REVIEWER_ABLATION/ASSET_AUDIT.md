# FGRefMOT original-QIM asset audit

Audit date: 2026-10-19 (UTC). Study branch: `research/qim-reviewer-ablation-20261019`.

## Source and branch

- Baseline commit: `d651fec18d4eda54bd141d90cd934b520343b789`.
- The isolated worktree was created from that commit, so the FGRefMOT-v2 TAM/QIM changes are not in this study.
- Original NAS `TempMam/models/qim_pro.py` SHA256: `568c7d8c2af11e40d5ab540518c3a0f3ef9aa8bd8d4140d5ce16360b3a25132a`.
- `TempMam_frez/models/qim_pro.py` has the same SHA256. `TempRMOT/models/qim_pro.py` is a different historical source tree.
- A bounded search of `TempMam`, `TempMam_frez`, and `TempRMOT` found no separate `standard_qim`, `lstm_qim`, `temporal_qim`, or QIM-ablation implementation. The three alternatives in this branch are therefore labelled `REIMPLEMENTED_VARIANT`.

## Mamba checkpoint

`/data1/liuyeqiang/fgref_v2_runs/checkpoints/checkpoint0089.pth`

- SHA256: `adb2ad9591a4a14dcdd05718b271d6c7b391e97bd9cb20f4fd8e3bfe26e54038`.
- Strict loading into the original `QueryInteractionModule` passed with zero missing/unexpected QIM keys.
- Strict loading into the full original model passed in the fixed offline RoBERTa probe.
- This is a recovered checkpoint, not a verified Refer-Sports final 56.01-HOTA checkpoint. Its earlier 60.509 diagnostic must not be reported as Sports performance.

## NAS and data state

The active mount is `Valadmin@127.0.0.1:/home/A100` at `/data1/liuyeqiang/nas/A100`. The known archive contains the original TempMam trees and the separate Sports baseline audit, but not a local SportsMOT image/label tree.

The local `/data1/liuyeqiang/SportsMOT` contains 41 expression directories and 323 JSON expressions, but zero image files and zero `labels_with_ids` files. The canonical manifest points to `/data3/liuyeqiang/datasets/SportsMOT/sportsmot_publish/dataset`, which is absent on this server. The strong-baseline audit also marks target task checkpoints and a provenance-complete common SportsMOT tracker as missing.

Decision: no Refer-Sports tracking training or final evaluation was started, and no HOTA/DetA/AssA/IDF1/IDSW value is fabricated.

