# Step-200 diagnostic record

The fixed Development-Val protocol is four videos, 2,621 frames, one
expression per video, `unknown_mask`, and `track_threshold=0`. The evaluator
report is the source of truth for every value below.

## M2 Trackwise Temporal Mamba

The step-200 checkpoint was evaluated successfully on all four videos. The
mean result is HOTA `0.0476121558`, DetA `0.0054709350`, AssA `0.4420127986`,
IDF1 `0.0073743303`, IDSW sum `673`, and referring F1 `0.0283685185`.
The per-video report shows `2,639,863`, `631,746`, `1,637,707`, and `747,828`
predicted detections respectively, versus `8,074`, `6,135`, `7,696`, and
`4,494` GT detections. This large false-positive count makes the result
diagnostic only; it is not used as a favorable comparison.

The result was produced with the current batched valid-history implementation
(`76e3805644b063079205f2e810cbdeebcf0667d3`) and the exact step-200
checkpoint. The ongoing exact-resume training is retained so the 5k/20k
milestones can distinguish early-training instability from a persistent
track-lifecycle or score-calibration defect.

## M3 Reliability Hybrid

The step-200 checkpoint was also evaluated successfully on all four videos.
The mean result is HOTA `0.0755403297`, DetA `0.0205235581`, AssA `0.2998716678`,
IDF1 `0.0193585475`, IDSW sum `765`, and referring F1 `0.0429394385`.
The per-video predicted-detection counts are `412,375`, `181,567`, `396,447`,
and `289,176`, again far above the corresponding GT counts. This is recorded
as a diagnostic result and is not used as a favorable accuracy comparison.

M3 used the exact step-200 checkpoint with the same fixed protocol and the
current source commit. The long-run M3 process remains active so later
milestones can show whether the reliability memory becomes calibrated after
training.
