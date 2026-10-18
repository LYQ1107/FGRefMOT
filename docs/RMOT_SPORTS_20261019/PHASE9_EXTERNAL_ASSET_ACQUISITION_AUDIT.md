# Phase 9 — external asset acquisition audit

Audit date: 2026-10-19  
Status: **PENDING EXTERNAL DOWNLOAD / NO ASSET BYTES CLAIMED**

This phase records bounded acquisition probes only.  No checkpoint or model
file was written to the workspace, and no third-party environment was changed.

## FlexHook Baidu share

The public metadata endpoint still exposes the official FlexHook directories.
A fresh request for the 481-byte
`/FlexHook/pretrained/roberta-base/config.json` file returned a signed link,
but the link's values were:

```text
server_time = 1792353448
dstime      = 1792349847
expires     = 1h
delta       = 3601 seconds
```

An immediate GET returned HTTP 403 with `error_code=31360` and
`expire time out error`.  The alternate PCS `locatedownload` route returned
`error_code=31045 user not exists` without a Baidu account session.  Replacing
the signed timestamp is invalid because it breaks the signature.  The
published FlexHook task weights and tracker outputs therefore remain metadata
only.

## Public RoPE-Swin route

The upstream RoPE-ViT model page
[`naver-ai/swin_rope_mixed_tiny_patch4_window7_224`](https://huggingface.co/naver-ai/swin_rope_mixed_tiny_patch4_window7_224)
is a possible source for the visual pretraining component, but the server's
direct TLS probes to both `huggingface.co` and `hf-mirror.com` failed with
`SSL_ERROR_SYSCALL` before any bytes were received.  It has not been treated
as a local asset or substituted into the FlexHook gate.

## Decision

The Phase 7 preflight remains **PENDING**.  The missing official RoBERTa and
ROPE-Swin directories, FlexHook task checkpoint, YesTrack LoRA/task
checkpoint, and provenance-complete SportsMOT tracker output must still be
downloaded through a working authenticated/public route, hashed, and passed
through the strict preflight before training or evaluation.
