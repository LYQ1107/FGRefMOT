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
site-packages directory.  PyPI remains unreachable through the current proxy
(`SSLEOFError` while fetching `https://pypi.org/simple/yacs/`), so the pure
Python pieces that were independently obtainable through the repaired GitHub
route were installed from pinned local checkouts:

| package | source / version | verification |
|---|---|---|
| yacs | `0.1.8`, `rbgirshick/yacs` commit `32d5e4ac300e…` | import succeeds |
| wordninja | `keredson/wordninja` commit `0421d148cd3d…` | import and split smoke test succeed |
| transformers | tag `v5.2.0`, commit `7d9754a05193…` | package imports; model/tokenizer imports still require tokenizers |
| accelerate | tag `v1.10.1`, commit `623cc0ba588d…` | `PartialState` import succeeds |
| peft | tag `v0.18.1`, commit `e3398fc05c55…` | package installs; model helpers wait on Transformers/tokenizers |
| scikit-image | `0.25.2` copied into the dedicated environment from the existing agristl package | import succeeds |

The dedicated environment still lacks `tokenizers` and `typer-slim`, and the
YesTrack requirement set's `bitsandbytes` and SentencePiece are not installed.
`from transformers import RobertaTokenizerFast` and `RobertaModel`, the
FlexHook data/model imports, and PEFT model helpers therefore fail at the
missing `tokenizers` extension.  The YesTrack `ReferKittiMOT` data loader and
Accelerate import succeed when loaded without the umbrella package side
effects.  No package was installed into another project's environment, and
this environment remains **not ready for training**.

## Status and guardrails

* `FLEXHOOK_SOURCE`: CLONED_PINNED_VERIFIED.
* `YESTRACK_SOURCE`: CLONED_PINNED_VERIFIED.
* `FLEXHOOK_WEIGHTS`: NOT_VERIFIED.
* `YESTRACK_WEIGHTS`: NOT_VERIFIED.
* `T_COMMON_SPORTS`: NOT_RUN (see Phase 3).
* FlexHook training/evaluation: NOT_RUN.
* YesTrack training/evaluation: NOT_RUN.

The HTTPS route used by the fixed-commit fetch script was repaired through the
existing Codex tunnel, with TLS verification left enabled.  Both repositories
were fetched with Git LFS smudge disabled and now match their pinned commits.
No baseline number is reported from the upstream README, and no result is
inferred from another dataset or from an unverified tracker.  The generated
FlexHook-compatible SportsMOT metadata layout and its limitations are recorded
in `PHASE5_FLEXHOOK_DATA_LAYOUT.md`.
