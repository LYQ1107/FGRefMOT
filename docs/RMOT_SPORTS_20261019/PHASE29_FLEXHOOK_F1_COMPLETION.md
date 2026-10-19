# Phase 29 — FlexHook F1 SportsMOT baseline completion

Audit date: 2026-10-19  
Status: **SUCCEEDED / NAS ARTIFACTS PUBLISHED AND HASH-VERIFIED**

The corrected single-GPU F1 run completed all 25 training epochs and exited with
code 0 under `set -o pipefail`. It used the pinned FlexHook checkout, the
SportsMOT-only contract, the fixed SambaMOTR-DAB-DETR tracker outputs, and
physical GPU 1. The final log contains the complete training time (`2:24:32`)
and no traceback after the corrected retry began. Earlier type-mismatch and
Transformers tokenizer tracebacks are retained in the same append-only log and
are recorded as attempts 0 and 1 in the run manifest; they are not part of the
successful attempt.

The final aggregate validation lines are below. Values are the logged aggregate
values (three decimal places); the pinned run's top-five list is also emitted at
full float precision.

| Epoch | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0 | 57.839 | 92.251 | 71.100 |
| 1 | 58.497 | 92.528 | 71.678 |
| 2 | 60.910 | 92.274 | 73.381 |
| 3 | 60.699 | 93.245 | 73.532 |
| 4 | 60.196 | 93.824 | 73.339 |
| 5 | 62.201 | 93.222 | 74.616 |
| 6 | 61.224 | 93.500 | 73.995 |
| 7 | 61.425 | 94.148 | 74.345 |
| 8 | 61.603 | 93.338 | 74.221 |
| 9 | 61.932 | 93.292 | 74.444 |
| 10 | 61.940 | 93.777 | 74.604 |
| 11 | 63.189 | 91.765 | 74.842 |
| 12 | 61.337 | 93.176 | 73.976 |
| 13 | 61.780 | 94.078 | 74.583 |
| 14 | 62.128 | 93.731 | 74.726 |
| 15 | 61.137 | 94.032 | 74.098 |
| 16 | 61.274 | 94.101 | 74.220 |
| 17 | 62.506 | 93.361 | 74.879 |
| 18 | 61.995 | 93.430 | 74.534 |
| 19 | 61.330 | 94.101 | 74.261 |
| 20 | 61.231 | 93.639 | 74.044 |
| 21 | 62.131 | 93.361 | 74.609 |
| 22 | 61.647 | 93.361 | 74.259 |
| 23 | 61.522 | 93.685 | 74.271 |
| 24 | 61.764 | 93.454 | 74.374 |

The best checkpoint is epoch 17 with exact logged `F1=74.87940979003906`,
precision `62.50580596923828`, and recall `93.36109161376953`. FlexHook's
rotating top-five files are ordered by the final best list
`[17, 11, 14, 5, 21]`; the result file is
`runs/gpu/flexhook_f1_sports_trained_seed42/ckpt_epoch_best_0.pth`.

The run manifest is
`runs/gpu/flexhook_f1_sports_trained_seed42/RUN_MANIFEST.json` with SHA-256
`025d1621e00b0cf5ebaa97be27db44eed410bfd704bd641c6275ecc4673105a8`. It records
the exact command, contract SHA
`ea5384746c416514352946d60cf7250754de6fc97bbc0dfc4ce6099cdef99c93`, launcher
SHA `4b58ff965e597c19cf92289a514b09745c3d7bba961f6c36104bd3771aab1776`,
FlexHook revision `bd1acc38634b28525d54dc6e0fcb38335f0029f9`, environment, GPU
selection, attempt history, metrics, and artifact hashes.

Local artifacts were hashed after completion:

* aggregate log — `40598dd21bd7d69d805c841ac7df538cbd847c53bc645837b856e0cf98efd344`;
* rank-0 log — `729d048d36d0bf273df0e8159cbed5f41801ec771b8d7689da366d766b454643`;
* final config — `00500abd027a10cb598ecdc65127eeae6b45ba6a81a2608689ebb5e7f911032f`;
* result checkpoint (`best_0`) — `3fdc471d1fa17405ce8d57516bc6de74ba93b29f3760344bb1f0f30529818c7b`.

After the Windows reverse tunnel was restored, the complete run directory was
published to NAS at
`/home/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/flexhook_f1_sports_trained_seed42`.
The old placeholder target was preserved as
`.flexhook_f1_sports_trained_seed42.previous.20261019151924`; the refreshed
manifest was uploaded atomically and its local and remote SHA-256 both equal
`025d1621e00b0cf5ebaa97be27db44eed410bfd704bd641c6275ecc4673105a8`.
Every checkpoint, log, config, TensorBoard event file, and the manifest was
verified by remote `sha256sum`; no partial directory remains. The next
experiment is Y1 YesTrack, after its contract and GPU availability are rechecked.
