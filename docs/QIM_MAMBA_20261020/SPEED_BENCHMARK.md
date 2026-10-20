# Speed benchmark

Initial V100 QIM-only benchmark (`CUDA_VISIBLE_DEVICES=6`, five warmups and
twenty timed repetitions per point) is recorded in
`reports/qim_mamba/speed_benchmark_v100.json`. It uses synthetic active-track
tensors and therefore is an engineering latency measurement, not full-model
image FPS.

| Variant | QIM params | N=5 mean ms | N=20 mean ms | N=50 mean ms |
|---|---:|---:|---:|---:|
| M1 Ordered Bi-Mamba | 1,992,000 | 8.59 | 7.45 | 5.48 |
| M2 Trackwise Mamba1 | 2,032,640 | 6.15 | 19.12 | 49.04 |
| M3 Reliability Hybrid | 2,099,329 | 10.08 | 24.72 | 47.18 |

The values include the shared QIM core and the variant mixer. They are not
full-model image FPS.

## Real-image step-200 benchmark

The matched step-200 checkpoints were also measured on the same 60-frame
SportsMOT clip (`v_-6Os86HzwCs_c001`), with five warmup frames, fp32 CUDA
events, and the validation image transform. The complete machine-readable
report is `reports/qim_mamba/real_sportsmot_benchmark_step200.json`, and the
compact table is `REAL_EFFICIENCY_RESULTS.csv`.

| Variant | Full-model params | Prepared mean ms / FPS | Wall mean ms / FPS | Peak allocated / reserved GB | Status |
|---|---:|---:|---:|---:|---|
| M1 Ordered Bi-Mamba | 178,814,764 | 248.215 / 4.029 | 321.712 / 3.108 | 1.914 / 3.238 | complete engineering |
| M2 Trackwise Mamba1 | 178,855,404 | 276.818 / 3.612 | 357.954 / 2.794 | 2.215 / 28.457 | diagnostic: tracks grew 10→566 |
| M3 Reliability Hybrid | 178,922,093 | 256.790 / 3.894 | 331.057 / 3.021 | 2.087 / 28.324 | diagnostic: tracks peaked at 329 |

The M2 and M3 rows are retained as diagnostic measurements rather than clean
steady-state efficiency claims: their step-200 track lifecycle produced a
rapidly increasing active-track count, which also explains the large reserved
memory values. Accuracy and final efficiency conclusions remain tied to the
fixed Development-Val protocol and the later long-run checkpoints.
