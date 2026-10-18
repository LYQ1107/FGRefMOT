# Phase 24 — Variant-ready baseline contracts

Audit date: 2026-10-19  
Status: **F1/Y1 CONTRACTS READY AND HASH-CHECKED / F0/F2/Y0 OFFICIAL CHECKPOINTS PENDING / NO GPU JOB STARTED**

This phase turns the independently audited assets into two explicit, launch-gated
SportsMOT variants. The aggregate preflight is
`runs/baseline_asset_preflight_20261019/phase24_variant_preflight.json` (SHA-256
`272860e6009615b2321be425f38ded0671812e87e68592cd9721f9097a33a33d`). It
combines the hash-bearing Phase 21 NAS report
(`runs/baseline_asset_preflight_20261019/phase21_nas_preflight.json`, SHA-256
`ac5fa7f0affc1ec16acf3464af92b0db1d943c84055db1ac57def51f50762b7e`) with
the Phase 23 tracker report
(`runs/baseline_asset_preflight_20261019/phase23_tracker_preflight.json`,
SHA-256 `00b2bfaae8b2b1a52b1bd962ad5c8fe6953774574f9fbb4d0fe79d5f25681c0a`).
The NAS login-shell prefix `/home/Valadmin/A100` is translated to the server's
read-only mount `/data3/liuyeqiang/nas/A100`; the two layout statuses are
revalidated against the current local layout manifests by each run contract.

The aggregate has six ready checks: `flexhook_layout`, `yestrack_layout`,
`roberta_base`, `rope_swin_tiny`, `yestrack_base_model`, and
`common_sports_tracker`. It remains **PENDING** only because the official
FlexHook task checkpoint and the official YesTrack LoRA checkpoint were not
acquired. Those missing gates are deliberately not substituted with a
checkpoint from another dataset or an unverified public file.

FlexHook F1 is recorded at
`runs/contracts_20261019/flexhook_f1_sports_trained_seed42.json` (SHA-256
`ea5384746c416514352946d60cf7250754de6fc97bbc0dfc4ce6099cdef99c93`). Its
`run_contract.py --check --require-ready` gate passed. It trains from the
public NAS RoBERTa and RoPE-Swin-Tiny assets with the SportsMOT training split,
the common fixed SambaMOTR-DAB-D-DETR tracker output root, and the exact
following protocol:

* seed `42`; batch/validation batch `10/40`; 25 epochs; base learning rate
  `1e-5`; weight decay `1e-4`;
* visual `rope-swin-tiny`, text `roberta`, pretrained mode `src`, text length
  `25`, expression sample count `8`, frame length/count/stride `8/4/30`;
* validation expression count `8`, validation frame stride `24`, validation
  scale `5`, `posw=0.7`, `gauss_sample=false`, and `noise1/noise2/noise3=true`;
* expression score thresholds `0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8`.

The upstream FlexHook checkout is not edited. Its relative pretrained lookup
is satisfied by the ignored runtime root
`runs/flexhook_runtime_20261019`, which contains only symlinks to the two NAS
directories. `RUNTIME_MANIFEST.json` has SHA-256
`0edb0bcd89280ee8a396525453aa1e6818372bdc58bbabcf0bf099a9546b2bd0`.
The runtime launcher and staging policy are implemented by
`scripts/launch_flexhook.py` (SHA-256
`4b58ff965e597c19cf92289a514b09745c3d7bba961f6c36104bd3771aab1776`) and
`scripts/prepare_flexhook_runtime.py` (SHA-256
`9cc08cc8d5179e7033580f5ad51163e4e4848061254c09fbd0d2629ebeb8a455`).

YesTrack Y1 is recorded at
`runs/contracts_20261019/yestrack_y1_sports_lora_seed42.json` (SHA-256
`cf60bbaded5e37c333c6c0f34f22aad8cb813985faace1cc8be7a1384afa523b`). Its
final NAS read-through completed with `contract: READY — hashes and policy
verified`. The supervised adapter trains against the SportsMOT train layout
from the local Qwen3-VL-2B-Instruct base model with:

* seed `42`; batch size `2`; one epoch; learning rate `5e-6`; image size `384`;
  margin ratio `0.2`; minimum crop side `8`;
* maximum text length `100`; negative downsample `0.5`; coordinate mode `xy`
  with three decimals;
* LoRA rank/alpha/dropout `8/16/0.0`; evaluation thresholds `0.2` through
  `0.8`; all sequences; evaluation every 5 frames; refinement interval
  `[0.2, 0.8]`.

Y1's adapter command is a supervised training command. Its later referring
evaluation must separately consume the same fixed tracker files in
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/sambamotr_sportsmot_dab/tracker_outputs`
using the pinned `third_party/YesTrack/refer_llm/llm_eval_from_mot.py` in
`--mot_input_type bytetrack` mode, with `--bytetrack_dir` set to that root,
`--image_frame_start_one 1`, `--mot_frame_start_one 1`, the seven evaluation
thresholds, `--re_refer_lower 0.2`, and `--re_refer_thresh 0.8`. The exact
LoRA output directory will be filled by the run manifest after training; no
evaluation has been claimed yet.

Both contracts use the same fixed SambaMOTR-DAB-D-DETR detector-plus-tracker
outputs. This is an integrated end-to-end tracker release, not an
external-detector-plus-ByteTrack run, and its 150 files were audited without
ground-truth boxes or IDs. The common tracker manifest is the only tracker
input allowed by this phase.

The official checkpoint gates remain separate. The FlexHook public inventory
contains task files for LaMOT, Refer-Dance, and Refer-KITTI but no
SportsMOT-named task checkpoint; the anonymous download route was recorded as
stale/403 in Phases 18–19. The YesTrack public release lists V1/V2 LoRA
artifacts, but its signed download returned Baidu error 31064 (HTTP 403), as
recorded in Phase 22. Neither route is treated as acquired, and no cross-task
checkpoint is substituted.

Before any GPU launch, the exact contract files must pass `run_contract.py
--check --require-ready` again, `nvidia-smi` must identify an idle device that
is not used by another project, and the selected device plus command must be
written to a run manifest. This phase intentionally starts no training or
evaluation process.

Machine-readable details are in `PHASE24_VARIANT_CONTRACTS.json`.
