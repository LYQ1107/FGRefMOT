# Phase 28 — FlexHook tokenizer compatibility retry

Audit date: 2026-10-19  
Status: **SECOND F1 ATTEMPT REACHED MODEL AND INITIAL CHECKPOINT / DATA WORKER BLOCKED BY TRANSFORMERS API / COMPATIBILITY RETRY READY**

The corrected F1 command passed config parsing, built the 248,218,728-parameter
FlexHook model, and wrote `ckpt_epoch_-1.pth` before the first training batch.
It then failed in a DataLoader worker when the pinned loader called
`RobertaTokenizerFast.batch_encode_plus`, which Transformers 5.2.0 no longer
provides. The traceback is retained in
`runs/gpu/flexhook_f1_sports_trained_seed42.log`. No epoch, loss, validation
metric, or usable checkpoint was produced; the initial `ckpt_epoch_-1.pth` is a
startup artifact and is not a result.

The earlier smoke script already contained a compatibility shim, but the
training process did not import that script. The launcher is now corrected to
run the pinned FlexHook `main.py` in the same Python process after installing
the shim, so the monkey patch survives into DataLoader workers. A temporary
probe confirmed `RobertaTokenizerFast.batch_encode_plus=True` under the
launcher. The vendored checkout remains byte-for-byte unchanged.

The launcher hash is now
`4b58ff965e597c19cf92289a514b09745c3d7bba961f6c36104bd3771aab1776`. The F1
contract also records this launcher in its finite source-file hash set; the
updated contract hash is
`ea5384746c416514352946d60cf7250754de6fc97bbc0dfc4ce6099cdef99c93` and
`run_contract.py --check --require-ready` passes.

The first two shell launches used `tee` without `pipefail`, so their shell
return code could be zero even when the Python traceback was visible. The next
launch uses `set -o pipefail` and appends to the same log, preserving both
failure records and propagating the actual training exit code. GPU 1 is the
only selected device; after the second failure it returned to 17 MiB with no
compute process.
