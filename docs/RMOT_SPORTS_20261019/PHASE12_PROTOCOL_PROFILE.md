# Phase 12 — source-verified baseline protocol profile

Audit date: 2026-10-19  
Status: **TEMPLATE READY / ASSETS STILL MISSING / NO GPU JOB STARTED**

The source audit resolved the parameter ambiguity that would otherwise occur
after the external files arrive.  The machine-readable profile is
[`configs/rmot_sports_baseline_protocol.json`](../../configs/rmot_sports_baseline_protocol.json).

For FlexHook, the checked-in `train-mix.yaml` is routed through
`RMOT_Dataset_mix`; the profile explicitly overrides `DATA.DATASET`,
`DATA.TRAIN`, and `DATA.TEST` to `SportsMOT` and points both roots to the
prepared Phase 5 layout.  It keeps the upstream mixed-training values for
frame sampling, optimizer, epochs, and batch sizes, while fixing the study
seed at 42.  Its visual/text components are local RoPE-Swin-T and RoBERTa
directories; the code reads the Swin file directly from
`pretrained/swin_rope_mixed_tiny_patch4_window7_224/pytorch_model.bin`.

For YesTrack, the profile records the adapter command and the upstream parser
defaults that control LoRA training, sampling, multi-threshold validation,
and the two-stage refinement interval.  The `--model_name` value must resolve
to the verified local Qwen3-VL-2B-Instruct directory; an online model fallback
is disallowed.

Both profiles use the same 22/19 train/validation video partition, seed, seven
evaluation thresholds, and one common detector-output tracker.  The command
arrays contain `<...>` path slots and are templates only; neither command was
executed.  The Phase 7 preflight and Phase 10 contracts remain the final
activation gates, so this phase claims no training, evaluation, or metric.
