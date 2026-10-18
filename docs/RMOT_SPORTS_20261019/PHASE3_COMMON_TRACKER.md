# Phase 3 — common physical tracker and asset audit

Audit date: 2026-10-19  
Status: **INCOMPLETE / BLOCKED**

## Scope and rule

The proposed RMOT comparison requires one common physical tracker
`T_COMMON_SPORTS` for every method.  The tracker must consume detector outputs
and produce track IDs; Refer-Sports ground-truth boxes or target IDs must never
be supplied at inference time.  This phase therefore accepts only an asset
whose dataset provenance, split, detector source, tracker configuration, and
checkpoint hash can be reconstructed.

## Bounded findings

* The local SportsMOT publication at
  `/data3/liuyeqiang/datasets/SportsMOT/sportsmot_publish` contains the raw
  frames, labels/GT, COCO annotations, and train/val/test directories audited
  in Phase 2.  A bounded search through its first four directory levels found
  no `.pth`, `.pt`, or `.ckpt` detector/tracker weights.
* The local ByteTrack source at
  `/data3/liuyeqiang/CFP-FishMOT_references/ByteTrack` is a usable code
  candidate, but no SportsMOT-trained detector weights or tracker output
  provenance was found.  It is therefore **UNVERIFIED** and is not used.
* NAS candidates under
  `/data3/liuyeqiang/nas/A100/data1/LWR/vranlee/CELL/ByteTrack` and
  `Byte-soft-track` contain YOLOX code and output directories, but the bounded
  inspection did not establish SportsMOT training provenance, split identity,
  detector checkpoint hashes, or Refer-Sports compatibility.  They remain
  **UNVERIFIED** and are not used.
* The NAS `refer-kitti-v2` assets are Refer-KITTI materials, not a valid
  substitute for the Refer-Sports physical-tracker input.

## Decision

`T_COMMON_SPORTS`: **NOT_RUN**.  No eligible detector/physical-tracker output
set or checkpoint was identified, so no common tracker was fabricated from GT
and no evaluation was started.  FlexHook and YesTrack training/evaluation are
also **NOT_RUN** pending official source acquisition, dependency verification,
and an eligible detector/tracker asset.

The next executable step is to repair the server's HTTPS Git access and run
`scripts/fetch_official_repos.sh`, then audit the resulting detector/tracker
requirements and locate or train a provenance-complete SportsMOT detector.  Any
future tracker run must record the detector hash, tracker configuration, input
manifest, output hash, and the same output directory for all compared methods.
