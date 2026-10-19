# Phase 64 — YesTrack score-level ablation pipeline

Status: `REGISTERED_WAIT_FOR_Y1_FORMAL_EVAL`

`scripts/eval_yestrack_score_variant.py` now runs YB2–YB6 from an existing
Y1 `predict_with_conf.txt` archive. It first applies the causal controller to
each sequence/expression namespace, then invokes the existing Refer-Sports
TrackEval adapter at thresholds 0.2 through 0.8. No Qwen3-VL inference is
repeated, and test GT is used only by TrackEval after score transformation.
The registered comparison point is threshold `0.8`, selected on seed42's
internal validation split; the other grid values remain diagnostic traces and
are not used to tune the historical test.

YB1 remains a separate model-time control using the upstream TCP/TRP flags
(`infer_every_n_frames=4` plus the documented stability window). The score
pipeline is waiting for the full historical Y1 formal manifest before it can
produce metrics.
