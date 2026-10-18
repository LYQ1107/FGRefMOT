# Phase 3 — common physical tracker and asset audit

Audit date: 2026-10-19  
Status: **INCOMPLETE / NO ELIGIBLE TRACKER ASSET**

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
* A bounded audit also found four existing MeMOTR SportsMOT output folders
  under the local research references.  Each contains only a training
  `config.yaml` at the inspected depth and no MOT result files or checkpoint;
  the copies are not provenance-complete tracker assets and were not reused.
  The inspected roots were:
  `/data3/liuyeqiang/research_references/N72R20/MeMOTR/outputs/memotr_sportsmot`,
  `/data3/liuyeqiang/research_references/N72R20/MeMOTR/outputs/memotr_d_detr_sportsmot`,
  `/data3/liuyeqiang/research_refs/memory_tracking/MeMOTR/outputs/memotr_sportsmot`,
  and `/data3/liuyeqiang/CFP-FishMOT_references/phase7/MeMOTR/outputs/memotr_sportsmot`.
* The audited `labels_with_ids` files are annotation-derived global-ID boxes.
  They were used to construct evaluation-only GT for 142 test expressions
  (346,247 rows) and were not passed to a detector, tracker, or referring
  model.

## Decision

`T_COMMON_SPORTS`: **NOT_RUN**.  No eligible detector/physical-tracker output
set or checkpoint was identified, so no common tracker was fabricated from GT
and no evaluation was started.  FlexHook and YesTrack training/evaluation are
also **NOT_RUN** pending official source acquisition, dependency verification,
and an eligible detector/tracker asset.

The official source checkouts are now available and the FlexHook data layout is
prepared.  The next executable step is to provision an isolated compatible
environment and locate or train a provenance-complete SportsMOT detector and
tracker.  Any future tracker run must record the detector hash, tracker
configuration, input manifest, output hash, and the same output directory for
all compared methods.
