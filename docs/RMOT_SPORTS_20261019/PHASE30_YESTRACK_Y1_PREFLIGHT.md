# Phase 30 — YesTrack Y1 post-F1 preflight

Audit date: 2026-10-19  
Status: **CPU IMPORT AND SPORTSMOT ADAPTER PASS / GPU LAUNCH WAITING FOR NAS TUNNEL**

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

The Y1 launch contract is
`runs/contracts_20261019/yestrack_y1_sports_lora_seed42.json` with SHA-256
`cf60bbaded5e37c333c6c0f34f22aad8cb813985faace1cc8be7a1384afa523b`. The launch
gate still requires the NAS-backed Qwen base and common tracker manifest. The
NAS reverse tunnel currently returns connection refused and the mounted NAS
path reports I/O errors, so the gate and GPU launch remain deferred. Once the
Windows reverse tunnel is restored, run the contract check, recheck GPU 1, and
launch Y1 with the same no-GT-as-tracker policy.
