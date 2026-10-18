# Phase 4 — baseline source and dependency audit

Audit date: 2026-10-19  
Status: **SOURCE CLONED AND AUDITED / TRAINING NOT_RUN**

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
were not fetched or guessed.  The two source checkouts now exist locally under
`third_party/` and their detached HEADs, repository URLs, and clean worktrees
were checked against `third_party/SOURCES.md`.

## Local dependency check

The server's default `python3` is Python 3.12.3.  In that interpreter the
following core packages are absent: `torch`, `torchvision`, `transformers`,
`yacs`, `opencv-python`, and `numpy`.  PyYAML is present.  A read-only probe of
existing environments found `trackocd-a100` with Torch 2.6.0+cu118,
`torchvision` 0.21.0, OpenCV, SciPy, timm and TrackEval dependencies, but
without Transformers, YACS, or wordninja.  `agristl` has Torch 2.1.1+cu121 and
YACS, but lacks the language-model stack.  These environments belong to other
projects and were not modified or selected for training.  Neither one satisfies
the full FlexHook/YesTrack dependency contract, so an isolated, pinned
environment still has to be provisioned and import-checked.

The pinned FlexHook checkout also contains two unused upstream helper modules
that fail Python parsing because their `try` blocks have no handler:
`deit/main.py` and `SwinTransformer/utils.py`.  The main RMOT path does not
import those helpers, but this remains an upstream source caveat; the checkout
was not edited.  A bounded compile audit found 127 other FlexHook Python files
and all 92 YesTrack Python files parse successfully; the two files above are
the only parse failures.  The isolated environment also lacks `skimage` in
addition to the language-model packages, so TrackEval imports have not been
claimed as ready.

An isolated environment now exists at
`/data3/liuyeqiang/.venvs/fgrefmot-rmot-sports-20261019`.  It uses a read-only
`.pth` entry for the existing Torch 2.6.0+cu118/vision stack and has its own
site-packages directory.  The canonical PyPI endpoint remains unreachable
through the current proxy (`SSLEOFError` while fetching
`https://pypi.org/simple/yacs/`).  The repaired GitHub route supplied pinned
source checkouts for the pure-Python packages; the binary wheels below were
installed from the reachable Tsinghua PyPI mirror with exact versions:

| package | source / version | verification |
|---|---|---|
| yacs | `0.1.8`, `rbgirshick/yacs` commit `32d5e4ac300e…` | import succeeds |
| wordninja | `keredson/wordninja` commit `0421d148cd3d…` | import and split smoke test succeed |
| transformers | tag `v5.2.0`, commit `7d9754a05193…` | package and model/tokenizer imports succeed after the pinned `tokenizers` wheel was installed |
| accelerate | tag `v1.10.1`, commit `623cc0ba588d…` | `PartialState` import succeeds |
| peft | tag `v0.18.1`, commit `e3398fc05c55…` | `LoraConfig`/`PeftModel` imports succeed |
| scikit-image | `0.25.2` copied into the dedicated environment from the existing agristl package | import succeeds |
| tokenizers | `0.22.1` manylinux wheel from the Tsinghua PyPI mirror | Rust extension import succeeds |
| typer-slim | `0.24.0` wheel from the Tsinghua PyPI mirror | `pip check` requirement satisfied |
| bitsandbytes | `0.47.0` wheel from the Tsinghua PyPI mirror | package import succeeds with the read-only Torch 2.6.0+cu118 stack |
| sentencepiece | `0.2.1` wheel from the Tsinghua PyPI mirror | package import succeeds |

The dedicated environment now passes `pip check`.  The following import
preflight succeeds: `RobertaTokenizerFast`, `RobertaModel`, FlexHook config,
FlexHook data/model modules, YesTrack `ReferKittiMOT`, Accelerate,
`LoraConfig`/`PeftModel`, bitsandbytes, and SentencePiece.  The optional
FlexHook fused-window and Tutel extensions remain absent and are reported by
the upstream code as optional warnings.  No package was installed into
another project's environment.

## Official external-asset inventory (metadata only)

The pinned FlexHook README's official Baidu share was queried without claiming
any downloaded bytes:

* `/FlexHook/pretrained/roberta-base` lists `config.json`,
  `pytorch_model.bin`, `tokenizer.json`, `tokenizer_config.json`, and
  `vocab.json`.
* `/FlexHook/pretrained/swin` lists `config.json` and `pytorch_model.bin`.
* `/FlexHook/SOTA_ckpts` lists the published Refer-KITTI, Refer-KITTI-v2,
  Refer-Dance, and LaMOT checkpoints.
* `/FlexHook/tracker_outputs/McByte-SportsMOT` lists tracker-result text
  files, but its detector/tracker provenance has not been independently
  established for this study.

The Baidu metadata endpoint required its challenge response.  Although a
challenge response returned a signed download URL, the URL was rejected by
the file endpoint with `403 error_code=31360 (expire time out error)` because
its `dstime` was already about one hour behind the current server time.  A
timestamp edit invalidated the signature.  Therefore these listings are an
inventory only; no local weight, checkpoint, or tracker result is marked
verified from them.

The YesTrack README's official share is recorded as a second acquisition
source for its Refer-KITTI V1/V2 LoRA weights and pure-tracker candidate
results.  Those materials are not a SportsMOT adapter, and no bytes from that
share are marked present in this workspace.

The executable guard and the point-in-time result are in
`PHASE7_BASELINE_ASSET_PREFLIGHT.md` and
`PHASE7_BASELINE_ASSET_PREFLIGHT.json`.  It requires the two generated layouts
plus verified local language/visual assets, task checkpoints, and a
provenance-complete common SportsMOT tracker before a GPU job can be started.

This resolves the Python dependency gate, but it does not make a baseline
training run reproducible by itself.  The required language/visual pretrained
directories, FlexHook/YesTrack task checkpoints, and a verified common SportsMOT
tracker output are still absent; the training and evaluation guards below
therefore remain active.

## Status and guardrails

* `FLEXHOOK_SOURCE`: CLONED_PINNED_VERIFIED.
* `YESTRACK_SOURCE`: CLONED_PINNED_VERIFIED.
* `FLEXHOOK_WEIGHTS`: NOT_VERIFIED.
* `YESTRACK_WEIGHTS`: NOT_VERIFIED.
* `T_COMMON_SPORTS`: NOT_RUN (see Phase 3).
* `RESEARCH_ENV_IMPORTS`: PASS (dedicated environment; `pip check` clean).
* FlexHook training/evaluation: NOT_RUN (weights and tracker input not verified).
* YesTrack training/evaluation: NOT_RUN (weights, tracker input, and SportsMOT wrapper not verified).

The HTTPS route used by the fixed-commit fetch script was repaired through the
existing Codex tunnel, with TLS verification left enabled.  Both repositories
were fetched with Git LFS smudge disabled and now match their pinned commits.
No baseline number is reported from the upstream README, and no result is
inferred from another dataset or from an unverified tracker.  The generated
FlexHook-compatible SportsMOT metadata layout and its limitations are recorded
in `PHASE5_FLEXHOOK_DATA_LAYOUT.md`.
