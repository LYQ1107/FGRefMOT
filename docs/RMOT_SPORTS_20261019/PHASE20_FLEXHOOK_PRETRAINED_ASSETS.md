# Phase 20 — FlexHook pretrained assets and real tokenizer smoke

Audit date: 2026-10-19
Status: **FLEXHOOK PRETRAINED BASE ASSETS READY / FOUR EXTERNAL GATES PENDING / NO GPU JOB STARTED**

The NAS host itself can reach the approved HuggingFace mirror, so the two
FlexHook base assets were acquired directly into the project NAS tree and
checked against the sizes listed by the official FlexHook share:

* `roberta-base`: five required files, including the 501,200,538-byte
  `pytorch_model.bin`, are present under
  `assets/pretrained/flexhook/roberta-base`.
* `swin_rope_mixed_tiny_patch4_window7_224`: `config.json` and the
  114,039,533-byte `pytorch_model.bin` are present under
  `assets/pretrained/flexhook/swin_rope_mixed_tiny_patch4_window7_224`.
  The model manifest records the NAVER AI HuggingFace revision
  `3c078815db16ad2bdf3a83d9f8eae38aa5e2c546`.

The hash-based preflight now marks `flexhook_layout`, `yestrack_layout`,
`roberta_base`, and `rope_swin_tiny` as `READY`.  It remains `PENDING` only
for the FlexHook task checkpoint, YesTrack LoRA checkpoint, YesTrack base model,
and the provenance-complete common SportsMOT tracker.  Its exit code is `2`,
and no GPU launch is permitted.

With the actual NAS RoBERTa directory, the FlexHook train and validation loader
smoke passed: 5,491 train samples and 1,195 validation samples produced the
expected tensor shapes.  Transformers 5.2.0 no longer exposes the legacy
`batch_encode_plus` alias used by the pinned loader, so the research smoke
entry point installs a small compatibility shim that delegates to `__call__`
and preserves finite `max_length` truncation.  The vendored FlexHook checkout
was not edited.

The NAS project root was also made readable to the server's read-only SSHFS
mount (`0755` directories and `0644` files) without touching other NAS paths.
Machine-readable evidence and all file hashes are in
`PHASE20_FLEXHOOK_PRETRAINED_ASSETS.json`.
