# Official RMOT source manifest

This manifest pins the upstream code used by the Refer-Sports study.  It is a
source manifest only: it records the verified source checkouts below, but does
not claim environment installation, training, or evaluation on this server.

| Component | Repository | License observed | Pinned revision | Local status |
|---|---|---|---|---|
| FlexHook | <https://github.com/buptLwz/FlexHook> | MIT | `bd1acc38634b28525d54dc6e0fcb38335f0029f9` | CLONED_PINNED_VERIFIED |
| YesTrack | <https://github.com/ggbondrighthere24/YesTrack> | MIT | `ed42b8315b28c94dc595d83b0d49bf57002bb281` | CLONED_PINNED_VERIFIED |

The revisions and licenses were checked through the public GitHub repository
metadata and trees on 2026-10-19.  The FlexHook README at this revision
specifies a PyTorch 2.6.0/CUDA 12.4 environment and additional RoBERTa,
ROPE-Swin-T, CLIP, dataset, and pretrained-weight setup.  YesTrack exposes its
own requirements and Refer-KITTI evaluation entry points.  These are source
requirements, not evidence that the environments or weights are available
locally.

On 2026-10-19 the GitHub HTTPS route was repaired: GitHub domains now use the
same existing `codex-existing-tunnel-17899` proxy route as Codex, through the
local Mihomo listener at `127.0.0.1:7890`. TLS certificate verification remains
enabled. This route still depends on that existing forwarder; independent
server Internet access is not established by this repair.

`scripts/fetch_official_repos.sh` completed successfully for both repositories.
Their checked-out HEADs match the pinned revisions above and both worktrees
are clean. Git LFS smudge was disabled for this source fetch. No submodules,
pretrained weights, datasets, environment installation, training, or evaluation
were performed. The script refuses non-empty destinations, so it should not
be rerun against these populated directories.

The following items remain explicitly unavailable until independently verified:

* pretrained weights and their checksums;
* a SportsMOT-trained detector/physical-tracker output set;
* dependency lockfiles and successful import checks;
* FlexHook and YesTrack training or evaluation results.

Raw SportsMOT frames, GT, and Refer-Sports expression annotations are inputs
under the paths recorded in the Phase 2 audit.  They are not copied into this
source tree and are never used as inference detections.
