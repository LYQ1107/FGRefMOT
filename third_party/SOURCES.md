# Official RMOT source manifest

This manifest pins the upstream code used by the Refer-Sports study.  It is a
source manifest only: it does not claim that either repository has been cloned,
installed, trained, or evaluated on this server.

| Component | Repository | License observed | Pinned revision | Local status |
|---|---|---|---|---|
| FlexHook | <https://github.com/buptLwz/FlexHook> | MIT | `bd1acc38634b28525d54dc6e0fcb38335f0029f9` | NOT_CLONED_NETWORK_BLOCKED |
| YesTrack | <https://github.com/ggbondrighthere24/YesTrack> | MIT | `ed42b8315b28c94dc595d83b0d49bf57002bb281` | NOT_CLONED_NETWORK_BLOCKED |

The revisions and licenses were checked through the public GitHub repository
metadata and trees on 2026-10-19.  The FlexHook README at this revision
specifies a PyTorch 2.6.0/CUDA 12.4 environment and additional RoBERTa,
ROPE-Swin-T, CLIP, dataset, and pretrained-weight setup.  YesTrack exposes its
own requirements and Refer-KITTI evaluation entry points.  These are source
requirements, not evidence that the environments or weights are available
locally.

The first bounded clone attempt from this server failed during the HTTPS TLS
handshake (`gnutls_handshake() failed: The TLS connection was non-properly
terminated`; the configured proxy is `127.0.0.1:7890`).  No partial clone was
kept and no files in either upstream repository were modified.  Run
`scripts/fetch_official_repos.sh` after network access is repaired; it fetches
only the pinned commits and refuses to use a non-empty destination.

The following items remain explicitly unavailable until independently verified:

* official source trees in `third_party/`;
* pretrained weights and their checksums;
* a SportsMOT-trained detector/physical-tracker output set;
* dependency lockfiles and successful import checks;
* FlexHook and YesTrack training or evaluation results.

Raw SportsMOT frames, GT, and Refer-Sports expression annotations are inputs
under the paths recorded in the Phase 2 audit.  They are not copied into this
source tree and are never used as inference detections.
