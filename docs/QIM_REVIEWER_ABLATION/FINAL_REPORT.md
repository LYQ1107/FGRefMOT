# FGRefMOT original QIM ablation — current report

Status: **ENGINEERING COMPLETE / FORMAL REFER-SPORTS RESULTS BLOCKED BY MISSING DATA ASSETS**.

Completed in this snapshot:

- isolated original-source branch from `d651fec...`;
- historical QIM audit and checkpoint SHA256 record;
- unified Standard, LSTM, Temporal Attention, and untouched Original Mamba interfaces;
- same-video clip sampling fix;
- QIM gradient/lifecycle/checkpoint/permutation tests (8/8 passed);
- CUDA-event QIM latency and full-model fixed-resolution engineering probes;
- step-based initialization/freezing/resume flags and normal-loop memory-cleanup fix;
- reviewer-facing implementation and limitation records.

Not completed because the required local assets are absent:

- 200-step and long QIM-only FGRefMOT training;
- development validation and historical Test evaluation;
- Refer-Sports HOTA, DetA, AssA, IDF1, IDSW, referring F1;
- multi-seed/partial-unfreeze experiments;
- formal paper-level Full-model FPS on real SportsMOT clips.

The correct scientific conclusion at this point is that the QIM implementation and engineering measurements are reproducible, while Mamba's precision advantage and the reported 0.29 HOTA gain remain unverified in this environment. It would be incorrect to fill the empty tracking tables from the recovered diagnostic checkpoint or from the synthetic latency probe.

