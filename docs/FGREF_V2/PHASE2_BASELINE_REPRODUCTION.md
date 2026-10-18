# Phase 2 baseline reproduction

Status: `HISTORICAL_CHECKPOINT_STRICT_LOAD_PASS`, `ARCHIVED_RESULTS_FOUND`,
`ONE_PAIR_CHECKPOINT_REPRODUCED`, `CONTROLLED_141_MANIFEST_RERUN_PASS`,
`PAPER_SPLIT_REPRODUCTION_UNVERIFIED`.

This phase does not claim dataset-wide paper metrics. The values in the paper
and in the NAS README remain `PAPER_REPORTED` or archived historical values;
the one complete-pair numbers recorded below are a checkpoint/protocol
reproduction audit, not a new benchmark result:

| Dataset | Paper-reported HOTA | This audit |
| --- | ---: | --- |
| Refer-Sports | 56.01 | not reproduced |
| Refer-KITTI | 51.64 | not reproduced |
| Refer-KITTI-V2 | 35.31 | not reproduced |

## Checkpoint decision

The primary candidate is

`/data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/official_checkpoint/checkpoint_0060.pth`

with SHA256

`55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`.

It loads on CPU and contains `model`, `optimizer`, `lr_scheduler`, `epoch`,
and `args` (`epoch=59`). It is not a strict restoration of the current
FGRefMOT source: the current model has 1,210 state-dict keys while the file
has 969, with 244 missing current keys, 3 unexpected keys, and 2 shared shape
mismatches. The missing groups include the current DeepFuse and Mamba-QIM
parameters.

The same file is strictly restorable against the old `TempRMOT` source after
reconstructing its historical RoBERTa details: `514` position embeddings,
`1` token type, a retained pooler, and a persistent `position_ids` buffer.
The resulting state dict has `969` keys and matches the checkpoint exactly;
`strict=True` loading succeeds. This is recorded as
`HISTORICAL_CHECKPOINT_STRICT_LOAD`, not as a current FGRefMOT result.
The exact namespace audit can be rerun with
`tools/fgref_v2/audit_historical_checkpoint.py`.

## Archived baseline and research results found on NAS

The upstream archive does contain a baseline checkpoint that was missed by the
initial shallow scan:

```text
checkpoint: /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT/exps/default_rk/checkpoint0049.pth
size:      1096317848 bytes
output:    /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT/exps/default_rk/results_epoch49
pairs:     158 predict/GT expression directories
HOTA:      50.641  (archived TrackEval output)
DetA:      39.406  (archived TrackEval output)
AssA:      65.241  (archived TrackEval output)
SHA256:    55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8
```

The checkpoint was copied to
`/data1/liuyeqiang/fgref_v2_runs/checkpoints/tempRMOT_checkpoint0049.pth` and
the local SHA256 matches the TempMam official checkpoint byte-for-byte. The
archived 158-pair result remains `ARCHIVED_HISTORICAL_RESULT`; the separate
one-pair live rerun is documented below.

The research archive also contains a Mamba-QIM candidate:

```text
checkpoint: /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/default_debug_rk/checkpoint0089.pth
output:    /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam/exps/default_rk/results_epoch89
pairs:     141 predict/GT expression directories
HOTA:      56.880  (archived TrackEval output)
DetA:      56.297  (archived TrackEval output)
AssA:      57.514  (archived TrackEval output)
summary SHA256: b86df51601d7c50ef8ae62ffa5c5a86a9d4958fb1fd9dbf25883c6fda3d6049f
```

The `56.880` value is a historical research result, not evidence that the
current FGRefMOT-v2 code improves the baseline. The checkpoint/output pairing
is inferred from the epoch name; the checkpoint now strictly loads against the
current TempMam research source and has a one-complete-pair reproduction below.

## Controlled same-server reproduction (one complete pair)

The Mamba-QIM checkpoint was copied from NAS to the local read-only audit
staging area and passed ZIP integrity and strict namespace loading against the
TempMam source:

```text
source:    /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempMam
checkpoint: /data1/liuyeqiang/fgref_v2_runs/checkpoints/checkpoint0089.pth
checkpoint SHA256: adb2ad9591a4a14dcdd05718b271d6c7b391e97bd9cb20f4fd8e3bfe26e54038
sequence:  v_ApPxnw_Jffg_c015
expression: volleyball-players-who-are-serving
frames:    246 / 246
output:    /data1/liuyeqiang/fgref_v2_runs/historical_mamba_eval/checkpoint0089/
           v_ApPxnw_Jffg_c015/volleyball-players-who-are-serving
```

The independent historical evaluator produced the following local pair, and
the local TrackEval subset runner reported:

```text
HOTA:  33.38182122036139
DetA:  12.543201642959033
AssA:  88.9259970427912
IDF1:  23.97137745974955
IDSW:  0
FPS:   4.3965658757
peak memory: 2270.07 MiB
```

The corresponding NAS historical pair, copied without modifying NAS, gives
HOTA 33.336299050983236, DetA 12.517467721125644, AssA 88.85392156726101,
IDF1 23.822222222222223, and IDSW 0 under the same local TrackEval subset
runner. The files are not byte-identical: the archived writer uses global
track IDs and different coordinate precision, while the audit writer uses
local IDs and normalized formatting. The close metrics support source,
checkpoint, and protocol alignment for this pair, but do not establish
dataset-wide reproduction.

Classification: `CHECKPOINT_REPRODUCED` for this one complete sequence only;
Refer-Sports/Refer-KITTI/Refer-KITTI-V2 dataset-wide reproduction remains
pending.

## Upstream baseline one-pair rerun

The old TempRMOT source and the locally hash-verified baseline checkpoint were
run on the same complete 246-frame sequence used for the Mamba comparison:

```text
source:    /data1/liuyeqiang/nas/A100/codex_archive_20261014/TempRMOT
checkpoint: /data1/liuyeqiang/fgref_v2_runs/checkpoints/tempRMOT_checkpoint0049.pth
sequence:  v_ApPxnw_Jffg_c015 / volleyball-players-who-are-serving
frames:    246 / 246
FPS:       7.7910885168
peak memory: 1054.78 MiB
```

The local TrackEval subset runner reports HOTA 0.0, DetA 0.0, AssA 0.0,
IDF1 0.0, and IDSW 0. The online writer produced one referring prediction,
but it did not match a target at IoU 0.5. This is a valid one-pair
`CHECKPOINT_REPRODUCED` baseline diagnostic, not a full Refer-KITTI result;
the archived 50.641 HOTA is not replaced by this single-sequence outcome.

## Controlled 141-expression rerun

Both checkpoint lineages were then evaluated with the same 141 `(video,
expression)` keys, local staged Refer-KITTI images/labels, thresholds
(`det=0.6`, `refer=0.4`, `area=100`), and local TrackEval wrapper. The two
manifest key sets were identical: 141 unique pairs, no duplicates, no missing
pairs, and 84,640 frames per run. This is a controlled checkpoint rerun, not a
sealed paper test split: the recovered manifest is training-style and was
selected from the archived output namespace.

| Lineage | HOTA | DetA | AssA | IDF1 | IDSW | Weighted FPS | Peak MiB | Scope/class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TempRMOT `checkpoint0049.pth` | 1.5970 | 0.6261 | 4.1764 | 1.1816 | 141 | 8.2546 | 1054.81 | 141-pair `CHECKPOINT_REPRODUCED` |
| TempMam Mamba `checkpoint0089.pth` | 60.5086 | 64.4788 | 56.8343 | 63.5075 | 1276 | 4.0855 | 2277.49 | 141-pair `CHECKPOINT_REPRODUCED` |

The complete manifests and summaries are stored at:

```text
/data1/liuyeqiang/fgref_v2_runs/historical_baseline_manifest_full2/
  tempRMOT_checkpoint0049/trackeval_manifest_summary.json
/data1/liuyeqiang/fgref_v2_runs/historical_mamba_manifest_full2/
  checkpoint0089/trackeval_manifest_summary.json
```

The thresholded online referring diagnostic aggregated to
precision/recall/F1 of `0.1190/0.0080/0.0150` for the old baseline and
`0.9519/0.7269/0.8243` for the Mamba checkpoint. These are auxiliary
diagnostics; the main tracking comparison is the TrackEval table. The large
difference is evidence about the two recovered checkpoint lineages under this
manifest, not an isolated causal estimate of Mamba-QIM or of TAM/QIM-v2.

## Why this is not a paper-split reproduction claim

The checkpoint metadata reports `meta_arch='rmot_qim_v2'`, while the current
catalog exposes `temp_rmot`; the exact historical wrapper/configuration for
that label is not fully recovered even though the old source namespace loads
strictly. The separate `TempRMOT` archive does contain
`exps/default_rk/checkpoint0049.pth`; its source/result pairing and local
one-pair rerun are recorded above. In addition,
the available `refer-kitti-v2` directory has a broken `KITTI` symlink and is
not a valid evaluation root on this host.

The original Refer-KITTI root under `TempMam` passes a constructor smoke test.
A one-frame/8-frame live run was executed against the pre-Mamba
`official_checkpoint/checkpoint_0060.pth` using the strict historical TempRMOT
source. It did produce detections, but under the archived online threshold
(`det_score > 0.6`, `refer_score > 0.4`) it produced zero referring outputs;
the active-track snapshot had referring scores 0.0622, 0.1478, 0.0233, and
0.1104. This is a diagnostic compatibility clue, not a paper metric, and is
not used to replace the Mamba-QIM archive result.

The pre-Mamba official checkpoint still only provides the diagnostic result
described above: it produces detections but no referring outputs at the
archived threshold under the historical evaluator. The Mamba checkpoint is
the compatible research variant for the current TempMam namespace; its
one-pair result is now reproduced, but it must not be substituted for the
upstream TempRMOT baseline or generalized to a full split.

## Remaining required runs

1. Preserve the process-local historical RoBERTa compatibility shim and old
   TempRMOT source as an explicit evaluation recipe.
2. Recover an independent legal test split before calling the 141-pair run a
   dataset-wide paper reproduction.
3. Run the TAM/QIM controlled training matrix and report it separately from
   these checkpoint reruns.
4. Keep recording all requested metrics, memory, parameter count, and FPS with
   the result class (`CHECKPOINT_REPRODUCED` or `NEW_EXPERIMENT`).

Until those steps complete, no dataset-wide baseline or new-method metric is
inferred from the paper README, historical logs, or checkpoint filename. The
explicit single-pair diagnostics above remain limited to their stated source,
checkpoint, and sequence.

## Smoke results (not evaluation)

The saved tool
`tools/fgref_v2/historical_forward_smoke.py` was run with a one-frame sampler
on an available GPU. It strictly loaded the historical checkpoint and
processed one real Refer-KITTI sample:

```text
HISTORICAL_FORWARD_OK
frames 1
pred_logits_shape (1, 300, 1)
pred_boxes_shape (1, 300, 4)
pred_refers_shape (1, 300, 1)
elapsed_seconds 1.8514
peak_memory_mib 995.27
```

The sentence was `basketball players with blue uniforms`. This is an
end-to-end smoke only: it uses a training-split sample, one frame, no
TrackEval, no threshold selection, and no claim about paper metrics.
