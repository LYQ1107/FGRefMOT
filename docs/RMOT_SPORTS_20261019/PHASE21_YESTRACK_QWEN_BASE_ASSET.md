# Phase 21 — YesTrack Qwen3-VL base asset and NAS-local preflight

Audit date: 2026-10-19  
Status: **YESTRACK BASE MODEL READY / THREE EXTERNAL GATES PENDING / NO GPU JOB STARTED**

The NAS host downloaded the official YesTrack base model `Qwen/Qwen3-VL-2B-Instruct` from `hf-mirror.com` at revision `89644892e4d85e24eaac8bacfd4f463576704203`. The finite runtime file set was staged under a temporary NAS directory, hashed, and atomically published at:

`/home/Valadmin/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/yestrack/Qwen3-VL-2B-Instruct`

The published directory contains the 4,255,140,312-byte `model.safetensors` plus the processor, tokenizer, configuration, and chat-template files. The NAS `SOURCE_MANIFEST.json` records every source URL, size, and SHA-256; its SHA-256 is `62fb4ebfb3ecf58e3dee27326f7c4eccd2f1f93a5f6d4fea917cc5509373c16b`. No staging directory remains.

A hash-based preflight was executed on the NAS disk. It marks `flexhook_layout`, `yestrack_layout`, `roberta_base`, `rope_swin_tiny`, and `yestrack_base_model` as `READY`. It remains `PENDING` for the official FlexHook task checkpoint, the matching YesTrack LoRA checkpoint, and a provenance-complete SportsMOT detector-output tracker. The command exits `2` while any required gate is pending, so no GPU launch is allowed. The complete NAS-local report is copied to `runs/baseline_asset_preflight_20261019/phase21_nas_preflight.json`.

Using the server's read-only NAS mount, Transformers 5.2.0 loaded `Qwen3VLProcessor` with `local_files_only=True`; the tokenizer was `Qwen2Tokenizer`, the image processor was `Qwen2VLImageProcessorFast`, and `apply_chat_template(..., add_generation_prompt=True)` passed. This was a processor-only smoke: the 4.25 GB weight file was not loaded and no GPU job was started.

The official task checkpoints and detector-output tracker are still required before baseline training or evaluation. Ground-truth/annotation files remain excluded as tracker input, and the vendored FlexHook and YesTrack checkouts were not modified.

Machine-readable acquisition, preflight, and smoke evidence is in `PHASE21_YESTRACK_QWEN_BASE_ASSET.json`.
