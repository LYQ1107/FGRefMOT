# Phase 30 — YesTrack Y1 post-F1 preflight

Audit date: 2026-10-19  
Status: **CPU IMPORT, SPORTSMOT ADAPTER, NAS ASSETS, AND Y1 CONTRACT PASS / GPU LAUNCH READY**

After the FlexHook F1 completion, the dedicated environment imported the pinned
YesTrack training and evaluation entry points on CPU:

* `torch 2.6.0+cu118`, `transformers 5.2.0`, `peft 0.18.1`, and
  `accelerate 1.10.1` imported;
* `refer_llm.llm_train` imported;
* `refer_llm.llm_eval_from_mot` imported;
* the YesTrack adapter dry-run passed with the generated SportsMOT layout.

The dry-run used `runs/yestrack_sports_layout_20261019_v1` and loaded supervised
labels only. It found 22 train sequences, 14,750 train frames, 160,489 train
boxes, and 181 train expressions; validation has 19 sequences, 11,900 frames,
130,152 boxes, and 142 expressions. No GPU process, model, tracker output, or
LLM result was started by this preflight.

The Y1 launch contract was refreshed after Phase 31 and is
`runs/contracts_20261019/yestrack_y1_sports_lora_seed42.json` with SHA-256
`81f478b1ac800786c7f62826dae4a0a2cfb8a8ca1712a5f652a833b3c85c47cf`. The contract sets `every_steps=0`, so no GT-based in-training evaluation is
run; the adapter owns `--eval_sequence` and the launch command does not pass it
to the upstream parser. The launch
gate now passes against the restored NAS-backed Qwen base and common tracker
manifest (`contract: READY — hashes and policy verified`). GPU 1 is free, so the
authorized Y1 launch can proceed with the same no-GT-as-tracker policy.
