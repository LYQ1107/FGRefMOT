# Phase 13 — official external asset endpoint audit

Audit date: 2026-10-19  
Status: **PARTIAL METADATA / DOWNLOAD BLOCKED BY SIGNED-URL EXPIRY / NO GPU JOB STARTED**

This phase records a read-only acquisition attempt against the official public
shares referenced by the pinned upstream repositories.  It separates a
discoverable directory listing from an actually downloaded, hashable asset.
No model, checkpoint, tracker output, cookie, signed download URL, or secret was
written to the workspace or NAS.

## Official listings discovered

The FlexHook public share was reachable and its directory metadata exposed the
following finite entries:

| Official path | Listed size | Evidence status |
|---|---:|---|
| `FlexHook/pretrained/roberta-base/pytorch_model.bin` | 501,200,538 bytes | listing only |
| `FlexHook/pretrained/roberta-base/config.json` | 481 bytes | listing only |
| `FlexHook/pretrained/roberta-base/tokenizer.json` | 1,355,863 bytes | listing only |
| `FlexHook/pretrained/roberta-base/tokenizer_config.json` | 25 bytes | listing only |
| `FlexHook/pretrained/roberta-base/vocab.json` | 898,823 bytes | listing only |
| `FlexHook/pretrained/swin_rope_mixed_tiny_patch4_window7_224/pytorch_model.bin` | 114,039,533 bytes | listing only |
| `FlexHook/SOTA_ckpts/refer-kitti-best.pth` | 995,771,714 bytes | listing only |
| `FlexHook/SOTA_ckpts/refer-kitti-v2-best.pth` | 995,768,450 bytes | listing only |
| `FlexHook/tracker_outputs/McByte-SportsMOT/*.txt` | 45 files | listing only; no provenance manifest |

The YesTrack public share exposed `Yestrack/yestrack.zip` at 178,142,422
bytes.  The archive has not been downloaded or unpacked, so its LoRA contents
and hash are not accepted by the Phase 7 gate.

## Why the gate remains closed

The public-share listing and password verification succeeded.  A subsequent
download-ticket request returned a signed link whose expiry timestamp was about
3,600 seconds older than the API server time.  An immediate request to that
link returned HTTP 403 (`expire time out`, error code 31360).  Repeated ticket
requests also returned the provider's captcha/rate-limit response (`errno=-20`).
The signed link was not modified and no captcha was bypassed.  Therefore the
result is metadata discovery, not asset acquisition.

The tracker listing is also insufficient by itself: a folder of MOT text files
without a detector checkpoint hash, tracker configuration hash, source-manifest
hash, split, and explicit `used_ground_truth: false` cannot pass the common
tracker check.  The listed output is not substituted into that gate.

## Protocol consequence

The variant-aware contract now follows the study matrix.  FlexHook F1
(`f1_sports_trained`) and YesTrack Y1/Y2/Y3 do not require optional public task
adapters; FlexHook F0/F2 and YesTrack Y0 still do.  Every selected variant
continues to require its matching base components and a provenance-complete
common SportsMOT detector-output tracker.  Y2 additionally waits for the
verified Y1 adapter and run manifest; that is an experiment dependency rather
than an optional public download.  This distinction removes an
unrelated optional checkpoint from the F1/Y1 activation path without relaxing
the no-GT and no-unverified-substitution rules.

## Required next action

Use an authenticated browser or Baidu client on a machine that can complete the
official download, then place the resulting bytes in the NAS project asset
directories.  Preserve the upstream directory structure where possible:

```text
assets/pretrained/flexhook/roberta-base/
assets/pretrained/flexhook/swin_rope_mixed_tiny_patch4_window7_224/
assets/source_checkpoints/flexhook/
assets/source_checkpoints/yestrack/
assets/pretrained/yestrack/          # matching local Qwen base model
assets/pretrained/physical_tracker/  # outputs plus tracker_manifest.json
```

After transfer, record SHA256 values, rerun Phase 7 with `--hash-files`, and
build a contract for the selected matrix variant.  Until those checks pass, no
GPU job, evaluation, metric, or improvement claim is valid.
