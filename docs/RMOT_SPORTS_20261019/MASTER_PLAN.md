# Refer-Sports RMOT research execution index

Status: `IN_PROGRESS`

This branch is the isolated execution route for the Refer-Sports RMOT study.
The local workspace is `/data3/liuyeqiang/FGRefMOT_RMOT_Sports_20261019`, the
research branch is `research/sports-rmot-strong-baselines-20261019`, and the
NAS project root is
`/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019`.

The registered protocol uses the 22-video/181-expression Sports training set,
an internal 17-video/136-expression training core plus 5-video/45-expression
validation split for selection, and the historical 19-video/142-expression
test split only for final offline TrackEval. Model-time candidates come from
the fixed common SambaMOTR tracker outputs. Test GT is never passed to the
model, tracker, checkpoint selector or threshold tuner.

Completed evidence includes the environment/NAS/data audit, atomic NAS
storage tooling, the storage I/O and synthetic archive/restore smoke reports,
the common tracker manifests, the pinned FlexHook and YesTrack source audits,
and the FlexHook F1 formal test evaluation. FlexHook F1 currently reports
HOTA 68.385415, DetA 59.317842, AssA 78.842302, IDF1 66.627791 and IDSW 409
from its successful manifest; these are historical-test values and are not
internal validation values.

Active work is recorded in `EXPERIMENT_REGISTRY.csv`. The full and internal
YesTrack Sports-LoRA trainings have succeeded. The separate internal
YesTrack validation evaluation has also succeeded, selecting threshold 0.8
from internal mean-over-alpha HOTA (0.746525); its 19-video formal evaluation
remains live.
The strict FlexHook core-only training
run has completed on GPU3: it used 17 train videos/136 expressions, selected
the checkpoint on the reserved 5 videos/45 expressions, and wrote its
artifacts directly to NAS. Its separate internal TrackEval manifest is
complete; historical test remains blocked for that baseline. The earlier FlexHook F1 internal
validation and the frozen-checkpoint H0–H8 prebaseline diagnostic replay are
complete and recorded in Phases 66–67. The H0–H8 metrics are evidence only;
the proposed-method gate remains closed until the formal YesTrack baseline
exists. Watchers are registered for the YesTrack post-training evaluations and
score-level ablations. The learned temporal module package now has a
CPU-tested causal sequence adapter, Phase 69 verified the real F1 checkpoint
hook interface, and Phase 71 adds the reproducible train-core/internal-val
feature archive command with frozen visual and pooled language representations.
Phase 72 adds the causal temporal-head training command; the real H8 run
completed eight epochs and selected epoch 5 using only internal validation.
Phase 73 adds the inference adapter that applies the selected causal head before
the standard TrackEval tree is written. Its five-video internal TrackEval
completed successfully (HOTA 75.6316, IDF1 72.4347), followed by a successful
19-video historical evaluation (HOTA 71.9128, DetA 66.7403, AssA 77.4862,
MOTA 62.5810, IDF1 67.7970, IDSW 386). This is a selected temporal-head
result; it is recorded separately from the formal YesTrack baseline and does
not by itself establish a learned improvement.

The remaining gates are the YesTrack formal evaluation,
formal H0–H8/YB score and learned-method ablations after the baseline gate,
multi-seed confirmation, efficiency measurements, and the final reviewer
report. Phase 76 registers an idempotent postbaseline H0–H8 watcher that waits
for the full YesTrack manifest before rerunning the causal score controls.
Phase 65 has registered candidate
diagnostic subsets, but their metrics remain blocked by the manifest's UNKNOWN
negative/event labels. F0/Y0 remain `NOT_RUN/BLOCKED` because no eligible
official task checkpoint or YesTrack LoRA checkpoint is present on NAS.

Each completed gate must update the relevant phase report, preserve its real
manifest/log/checkpoint evidence, pass the scoped tests, create a local Git
commit, and verify the corresponding GitHub branch head through the GitHub
API. Raw videos, large weights, private NAS data and credentials stay off
GitHub.
