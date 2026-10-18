# Phase 4 TAM-v2 implementation

Status: `IMPLEMENTED_AND_UNIT_TESTED`, `TRAINING_INTERFACE_SMOKE_PASS`,
`PERFORMANCE_NOT_ESTABLISHED`.

## Code path

The research-only package is under `models/fgref_v2/`:

- `text_evidence.py`: preserves all text tokens and emits identity, action,
  relation, unknown, negation, and validity masks. The first router is an
  auditable lexical diagnostic; it is not treated as ground-truth semantic
  annotation.
- `track_evidence_sampler.py`: fixed local/context grid sampling from the
  input-projected visual pyramid. Coordinates are normalized `cxcywh` boxes,
  and the returned mask rejects out-of-image and padded samples.
- `referring_pair_head.py`: per-track cross-attention plus an independent
  logit. There is no candidate-wise softmax, so zero/one/multiple referred
  targets remain representable. `ReferringMLPHead` is the pooled capacity
  control.
- `tam_v2.py`: composes text evidence, predicted-box sampling, optional
  history evidence, and the pair head.

`models/transrmot_pro.py` keeps the original decoder/detection path and adds
an opt-in post-decoder TAM branch. `main.py` exposes:

```text
--fgref_v2_tam
--fgref_v2_tam_local_points
--fgref_v2_tam_context_points
--fgref_v2_tam_conditional
--fgref_v2_tam_heads
```

The formal path uses `pred_boxes` emitted by the decoder. GT/diagnostic boxes
are only accepted through the explicit standalone diagnostic API and are not
used by the integrated path. The original decoder score is retained as
`pred_refers_baseline`; the new score is copied into the cache after legacy
cache loading and restored after STReasoner so `track_ref` cannot silently
replace it.

TAM is currently incompatible with the legacy activation-checkpoint tuple;
the build path fails explicitly when both `--fgref_v2_tam` and
`--use_checkpoint` are supplied. This is safer than returning a tuple whose
new named evidence fields are missing.

The three reproducible candidate manifests are:

- `configs/fgref_v2/tam_pair.yaml` — fixed local/context grid;
- `configs/fgref_v2/tam_local.yaml` — local-only capacity control;
- `configs/fgref_v2/tam_conditional.yaml` — query/text-conditioned offsets.

The YAML files are declarative launch manifests because the legacy repository
is CLI-driven; they are not silently parsed by `main.py`.

## Gradient and comparison boundaries

The retained visual pyramid is the projected feature before DeepFuse/fusion,
and no `detach()` is inserted in the formal TAM path. The optional debug
snapshot remains detached by design and is not a training input. All three
TAM variants use the same decoder, feature levels, query count, loss weights,
and seed in their manifests. Auxiliary decoder referring losses, when
`aux_loss` is enabled, remain the legacy auxiliary control because TAM is
computed only from the final decoded candidate set; this boundary must be
reported in any fair comparison.

## Verification

The focused CPU suite passes:

`33 passed, 1 warning`

It covers empty/multiple candidates, token masks, out-of-bounds sampling,
independent logits, prediction-box use, cache overwrite ordering, and
training/inference score consistency. No HOTA or referring metric is claimed
until a frozen baseline and valid evaluation split are available.

The real-sample integration smoke is reproducible with
`tools/fgref_v2/current_forward_smoke.py`; it uses random current-source
weights and must not be confused with a trained experiment. The TAM-v2 plus
QIM-v2 one-frame smoke completed with output shapes `(1,300,1)`, `(1,300,4)`,
and `(1,300,1)` for logits, boxes, and referring scores respectively; the
recorded GPU-2 runtime was 2.0063 seconds with a 1871.44 MiB peak allocation.
The separate restored-checkpoint/TAM-only gradient smoke is recorded in
`PHASE5_TAM_RESULTS.md`; neither smoke is a performance result.
