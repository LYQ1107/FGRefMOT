# Phase 4 — baseline source and dependency audit

Audit date: 2026-10-19  
Status: **SOURCE AUDITED / TRAINING NOT_RUN**

## Fixed upstream revisions

The official repositories were inspected through their public GitHub trees at
the revisions in `third_party/SOURCES.md`:

* FlexHook `bd1acc38634b28525d54dc6e0fcb38335f0029f9` (`buptLwz/FlexHook`).
  The README specifies PyTorch 2.6.0/CUDA 12.4, RoBERTa, ROPE-Swin-T, CLIP,
  external dataset preparation, and externally hosted pretrained/best weights.
  Its checked-in training commands cover Refer-KITTI and Refer-Dance.  Its
  mixed inference script names an `mcbyte-sports` tracker input, but that is
  not evidence of a complete Refer-Sports expression training/evaluation route.
* YesTrack `ed42b8315b28c94dc595d83b0d49bf57002bb281`
  (`ggbondrighthere24/YesTrack`).  The README documents Refer-KITTI V1/V2
  released LoRA weights, pure-tracker candidate results, MOTChallenge-style GT,
  and a Refer-KITTI evaluation script.  It does not provide a verified
  Refer-Sports adapter or SportsMOT checkpoint at this revision.

The public README and source files are evidence of interfaces and requirements,
not evidence that external Baidu resources were downloaded.  Those resources
were not fetched or guessed.

## Local dependency check

The server's default `python3` is Python 3.12.3.  In that interpreter the
following core packages are absent: `torch`, `torchvision`, `transformers`,
`yacs`, `opencv-python`, and `numpy`.  PyYAML is present.  A read-only probe of
existing environments found `trackocd-a100` with Torch 2.6.0+cu118 but without
Transformers or YACS, and `agristl` with Torch 2.1.1+cu121 but without
Transformers or FairScale.  These environments belong to other projects and
were not modified or selected for training.  Neither one satisfies the full
FlexHook/YesTrack dependency contract, so an isolated, pinned environment still
has to be provisioned and import-checked.

## Status and guardrails

* `FLEXHOOK_SOURCE`: NOT_CLONED_NETWORK_BLOCKED.
* `YESTRACK_SOURCE`: NOT_CLONED_NETWORK_BLOCKED.
* `FLEXHOOK_WEIGHTS`: NOT_VERIFIED.
* `YESTRACK_WEIGHTS`: NOT_VERIFIED.
* `T_COMMON_SPORTS`: NOT_RUN (see Phase 3).
* FlexHook training/evaluation: NOT_RUN.
* YesTrack training/evaluation: NOT_RUN.

The server's repeated `git ls-remote` probe still fails at the configured HTTPS
proxy with `gnutls_handshake() failed: The TLS connection was non-properly
terminated`.  The fixed-commit fetch script remains the only approved source
acquisition path.  No baseline number is reported from the upstream README,
and no result is inferred from another dataset or from an unverified tracker.
