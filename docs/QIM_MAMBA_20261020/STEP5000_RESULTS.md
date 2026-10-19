# Matched step-5000 development results

All three variants were resumed exactly from their common step-200
checkpoint and evaluated with the same frozen Development-Val list:

- data source: `/data/DATASETS/TRACKING/JDE/SportsMOT/sportsmot_publish`;
- expressions: `/data1/liuyeqiang/SportsMOT/expression`;
- protocol: `unknown_mask`, all-object MOT metrics, UNKNOWN referring negatives
  ignored;
- evaluation set: 4 sequences and 2,621 frames;
- test-set metrics are not claimed because the published SportsMOT test split
  has no matching local ground-truth labels.

| Variant | HOTA | HOTA@0.5 | DetA | AssA | IDF1 | IDSW sum | IDSW mean/seq | Ref-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 OrderedBi | 0.597821 | 0.626957 | 0.691503 | 0.534909 | 0.575095 | 157 | 39.25 | 0.856879 |
| M2 Trackwise | 0.620469 | 0.649719 | 0.622179 | 0.636798 | 0.626328 | 80 | 20.00 | 0.855282 |
| M3 Reliability-Hybrid | 0.562850 | 0.587966 | 0.541054 | 0.589986 | 0.547645 | 121 | 30.25 | 0.666483 |

The corresponding reports are:

- `reports/qim_mamba/mamba_ordered_bi_step5000_dev_full.json`
- `reports/qim_mamba/mamba_trackwise_step5000_dev_full.json`
- `reports/qim_mamba/mamba_reliability_hybrid_step5000_dev_full.json`

The step-5000 checkpoint SHA256 values are recorded in `DEV_RESULTS.csv`.
Parameter counts and the controlled inference speed measurements remain in
`PARAMETER_BUDGET.md` and `SPEED_BENCHMARK.md`; no unmeasured latency, FPS,
memory, or training-cost values are inferred from these accuracy reports.
