# Phase 11 — external asset drop revalidation

Audit date: 2026-10-19  
Status: **PENDING EXTERNAL ASSETS / NO GPU JOB STARTED**

This is a bounded revalidation of the external dependency gate after the
Phase 10 run contracts were created.  It does not treat a pre-created NAS
directory as evidence that a model or tracker was uploaded.

The NAS SFTP view showed these target directories still empty:

```text
assets/pretrained/flexhook
assets/pretrained/yestrack
assets/pretrained/physical_tracker
assets/source_checkpoints
checkpoints/flexhook
checkpoints/yestrack
experiments/common_tracker
```

The local workspace, configured dataset/TempMam search, and the dedicated
Hugging Face cache contain no matching FlexHook, YesTrack, Qwen3-VL,
RoPE-Swin, or provenance-complete `tracker_manifest.json` asset.  The one
large cache blob is a Timm ViT checkpoint and is unrelated to either baseline;
it was not copied or substituted.

The current network split is also unchanged: GitHub responds through the
existing Codex proxy, while direct Hugging Face, hf-mirror, and ModelScope
probes fail at TLS before receiving model bytes.  The active GPU processes are
unrelated `habitat-llm` jobs; no RMOT process was started or modified.

The Phase 7 preflight and both Phase 10 run contracts therefore remain
`PENDING`.  The strict contract command returns status 2 until the official
assets, hashes, tracker provenance, and non-placeholder run parameters are
present.  The machine-readable evidence is in
`PHASE11_ASSET_REVALIDATION.json`.
