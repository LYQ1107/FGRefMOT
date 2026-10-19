# Draft response to the TMM QIM reviewer

## What is independent of Mamba?

The controlled code path holds the original FGRefMOT Backbone, TAM/deepfuse, visual-language fusion, deformable decoder, STReasoner, losses, active-track selection, and RuntimeTrackerBase fixed. Only `track_embed` changes. The three missing historical alternatives are transparently marked reimplementations rather than claimed as recovered paper code.

## What does the original Mamba actually process?

The original module receives `[1, N_active, 256]` after the active tracks are selected. The second dimension is the ordered active-track token axis. The source creates no persistent Mamba inference state across frames; `inference_params` is reset to `None` for each pass. Thus the implementation should not be described as a true per-target long-term temporal memory without additional evidence.

## What can be said about efficiency now?

On the current V100 engineering probe, original Mamba QIM latency at 20 active tracks was 10.95 ms, compared with 2.26 ms for Standard, 4.62 ms for LSTM, and 5.74 ms for Temporal Attention. The full-model synthetic 360x640 probe produced 11.75 prepared-tensor FPS for Mamba, versus 11.23/12.02/11.94 for the three alternatives. These values are hardware- and probe-specific, not Refer-Sports paper results. Mamba's parameter count alone does not establish end-to-end FPS superiority.

The separate real-image step-200 benchmark used 60 native frames from
`v_-6Os86HzwCs_c001` (5 warmup, 55 timed, FP32, Tesla V100). Including decode
and validation transform, wall FPS was 5.64/5.55/5.94/5.77 for
Standard/LSTM/Temporal Attention/Original Mamba respectively; prepared-GPU FPS
was 7.54/7.26/7.93/7.63. Peak allocated memory was 2.244/1.985/1.919/1.925
GB. This is a single-clip engineering measurement, not a paper-wide speed
claim; the exact p95 values and protocol are in
`reports/qim_ablation/real_sportsmot_benchmark_step200.json`.

## What are the lightweight Mamba follow-ups?

Three explicitly labeled lightweight designs were implemented and trained in a
separate matched branch: M1 OrderedBi (geometry-ordered bidirectional Mamba),
M2 Trackwise (one causal sequence per active track), and M3
Reliability-Hybrid (M2 plus reliability-gated identity memory). They use the
same Refer-Sports loader, optimizer protocol, frozen four-video development
split, and exact step-200 resume. Their completed step-5000 development means
(HOTA/IDF1/AssA/IDSW sum) are M1 0.5978/0.5751/0.5349/157, M2
0.6205/0.6263/0.6368/80, and M3 0.5628/0.5476/0.5900/121. The parameter
counts are 1,992,000, 2,032,640, and 2,099,329 respectively. These are real
development measurements over four videos and 2,621 frames; the 20k runs and
their matched evaluations are still in progress, and no Test-set claim is made.
The detailed reports are in `docs/QIM_MAMBA_20261020/STEP5000_RESULTS.md` and
`reports/qim_mamba/*_step5000_dev_full.json` on the lightweight branch.

## What cannot be claimed yet?

The local SportsMOT image tree, `labels_with_ids`, expressions, and global-ID
manifest have now been audited and agree. The four step-200 QIM-only checkpoints
have been evaluated on the frozen four-video development split (2,621 frames,
one expression per video, all frames): Standard HOTA/IDF1/AssA =
0.0432/0.0078/0.1534, LSTM = 0.0284/0.0065/0.0108, Temporal Attention =
0.6376/0.6723/0.6707, and Original Mamba = 0.7328/0.7458/0.6254. The
corresponding IDSW sums are 1391, 4113, 44, and 74. These are explicitly
development results, not historical Test claims.

The exact-resume 5k milestone has now been evaluated on the same development
protocol. The four-way means (HOTA/IDF1/AssA/IDSW sum) are Standard
0.6720/0.7028/0.6506/65, LSTM 0.6519/0.6754/0.6462/62, Temporal Attention
0.5432/0.5148/0.5999/116, and Original Mamba 0.7222/0.7517/0.6378/75.
These are milestone development results over four videos and 2,621 frames;
they are not historical Test results. The exact 20k milestone is now also
available: Standard/LSTM/Temporal Attention/Original Mamba HOTA/IDF1/AssA/IDSW
sum are 0.6878/0.7172/0.6568/59, 0.7406/0.7721/0.6550/75,
0.5633/0.5955/0.6829/30, and 0.7236/0.7540/0.6556/76 respectively. The
60k/120k checkpoints and their evaluations are still running, so no final
long-run ranking or extrapolated training-cost claim is made yet.

The published Test images still have no matching local ground-truth labels;
therefore historical Test HOTA/AssA/IDF1/IDSW, a reproducible 0.29-HOTA
attribution, and a final paper conclusion must not be claimed until the Test
GT/protocol is supplied. The real-image efficiency benchmark is reported
separately from the synthetic engineering probe and is now complete for the
documented single clip; long-run training cost is being measured from actual
checkpoint logs. The measured checkpoint-cost records are available for steps
5000 and 20000; the observed GPU-hours from the step-200 checkpoint timestamp
are 1.713/1.772/1.760/1.845 at step 5000 and
6.971/7.212/7.174/7.454 at step 20000 for
Standard/LSTM/Temporal Attention/Original Mamba respectively. These are
checkpoint-timestamp observations, not power-meter measurements; 60k/120k
fields remain explicitly missing until their checkpoints are written.

The long-run logs also contain explicit no-gradient QIM batches: Standard at
steps 1535, 3316, and 13539, and Original Mamba at steps 3624 and 12472 in the
current snapshot. The training loop handles these by skipping only that
optimizer step when no trainable QIM path exists; this is recorded as a no-op,
not reported as a crash or converted into an accuracy claim.

