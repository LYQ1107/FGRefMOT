# Draft response to the TMM QIM reviewer

## What is independent of Mamba?

The controlled code path holds the original FGRefMOT Backbone, TAM/deepfuse, visual-language fusion, deformable decoder, STReasoner, losses, active-track selection, and RuntimeTrackerBase fixed. Only `track_embed` changes. The three missing historical alternatives are transparently marked reimplementations rather than claimed as recovered paper code.

## What does the original Mamba actually process?

The original module receives `[1, N_active, 256]` after the active tracks are selected. The second dimension is the ordered active-track token axis. The source creates no persistent Mamba inference state across frames; `inference_params` is reset to `None` for each pass. Thus the implementation should not be described as a true per-target long-term temporal memory without additional evidence.

## What can be said about efficiency now?

On the current V100 engineering probe, original Mamba QIM latency at 20 active tracks was 10.95 ms, compared with 2.26 ms for Standard, 4.62 ms for LSTM, and 5.74 ms for Temporal Attention. The full-model synthetic 360x640 probe produced 11.75 prepared-tensor FPS for Mamba, versus 11.23/12.02/11.94 for the three alternatives. These values are hardware- and probe-specific, not Refer-Sports paper results. Mamba's parameter count alone does not establish end-to-end FPS superiority.

## What cannot be claimed yet?

The recovered checkpoint is not verified as the paper's final Refer-Sports checkpoint, and the local server lacks the SportsMOT image/label root. Therefore there is currently no honest HOTA/AssA/IDF1/IDSW comparison, no reproducible 0.29-HOTA attribution, and no final paper conclusion. The branch contains the reproducible code, tests, manifests, and activation commands needed once those assets are supplied.

