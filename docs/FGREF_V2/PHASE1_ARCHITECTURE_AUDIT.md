# Phase 1 architecture audit

Source audited: `models/` at
`d651fec18d4eda54bd141d90cd934b520343b789`, with the NAS `TempMam` archive
used to confirm historical generated files and logs. The repository entry point
is `models/__init__.py -> transrmot_pro.build`; the older `transrmot.py` is not
the active `temp_rmot` implementation.

## Tensor flow and dimensions

The default hidden dimension is 256 and the tracking query has dimension 512,
split into a 256 dimensional positional half and a 256 dimensional content
half. For a batch of one and `N` query slots:

| Stage | Actual shape / behavior |
| --- | --- |
| Text encoder | RoBERTa token states are projected to `[L, B, 256]`; the tokenizer attention mask becomes the PyTorch padding mask `[B, L]`. |
| Feature projection | Each visual level is projected from `[B, C_in, H_l, W_l]` to `[B, 256, H_l, W_l]`, then flattened to `[H_l W_l, B, 256]`. |
| Query split | `query_pos [N, B, 512]` is split into `query_embed_l [N, B, 256]` and `tgt_track [N, B, 256]`. |
| Triple alignment | Object, text, and visual sequences are concatenated as `[N + L + H_lW_l, B, 256]`; the output is split back into visual `[H_lW_l, B, 256]`, text `[L, B, 256]`, and object `[N, B, 256]`. |
| Decoder text attention | In the self-cross path, decoder queries `[N, B, 256]` query text keys/values `[L, B, 256]` through `self.text_self_attn` (the name is misleading), then use multi-scale deformable visual cross-attention. |
| Prediction heads | Each decoder level produces class logits `[B, N, 1]`, boxes `[B, N, 4]`, and referring logits `[B, N, 1]`. |

`Triple_Alignment_MHA` has three independent Q/K/V projections for object,
visual, and text streams and concatenates all three streams before a fully
dense multi-head attention. Its padding mask masks padded text and visual
locations but leaves all object slots valid. The default four-layer stack has
four copies of this module.

The current-source CPU profile with TAM-v2 and QIM-v2 enabled reports:

| Quantity | Parameters |
| --- | ---: |
| Full instantiated model | 178,273,518 |
| Parameters with `requires_grad=True` | 53,996,078 |
| TAM-v2 | 1,118,721 |
| QIM-v2 adapter | 332,033 |
| Base graph after removing those two modules | 176,822,764 |

The same profile is reproducible through the one-sample smoke tool; GPU
latency and peak memory are recorded separately because they depend on the
selected device and input path.

## TAM findings

`TransRMOT._forward_single_image()` retains no separate unmodified visual
pyramid for later track-centric sampling. It projects each level, applies
`Triple_Alignment` for levels other than zero, then applies the
vision-language fusion module before passing the resulting pyramid to the
deformable transformer. The existing code has an optional capture hook, but it
does not create a formal evidence interface.

The object stream in Triple Alignment is the current query set rather than a
track-specific local crop. The attention is global over object, text, and all
visual tokens at that level. This is a useful baseline for T0/T2 comparisons,
but it is not yet a fine-grained local identity evidence module.

## Decoder findings

The default decoder uses the self-cross branch. It first applies query self
attention, then a query-to-text attention, then deformable cross-attention over
the visual pyramid, followed by the FFN. `text_word_features` and
`text_word_mask` are passed through `combined_parameters`; no explicit token
groups or language-conditioned sampling points exist in the baseline.

The current code therefore does not implement a separate formula for a
different number of object and visual tokens: it performs attention over the
concatenated Triple Alignment sequence and separately performs decoder query to
text attention. Any paper equation claiming object and visual tokenwise sums
must be checked against these two concrete paths.

## STReasoner findings

`SpatialTemporalReasoner.forward()` receives `text_info`, but never reads it.
The actual sequence is per query slot, not an explicitly ID-keyed temporal
store:

1. `frame_shift()` shifts `hist_embeds` left and appends the current
   `output_embedding`, while marking the newest slot valid.
2. `forward_history_reasoning()` uses a three-layer temporal transformer to
   query `hist_embeds`, then a second three-layer object attention module.
3. The refined embedding is written to `output_embedding` and detached into
   the newest history slot.
4. `forward_history_refine()` writes refined class, referring, and box outputs
   into `cache_pred_logits`, `cache_pred_refers`, and `cache_pred_boxes`.

Thus the current frame is already inserted before history reasoning, and the
history refinement overwrites the decoder cache. This is a causalness and
attribution issue for the baseline audit: the cache no longer represents only
the raw decoder output after STReasoner runs. `frame_summarization()` then
copies the caches back to `pred_*`.

## Mamba-QIM findings

`QueryInteractionModule._update_track_embedding()` selects active rows with a
boolean mask and constructs `tgt` with shape `[N_active, 256]`. It then calls
Mamba with `tgt.unsqueeze(0)`, giving `[1, N_active, 256]`. Therefore the
Mamba sequence axis is the active-track row axis, not the historical time axis.
`inference_params` and `residual` are reset to `None` for every call, and no
per-track Mamba hidden state is stored in `Instances`.

The result depends on the order of active tracks. Boolean indexing preserves
the current row order, but there is no inverse permutation or track-ID keyed
state. The updated active rows are concatenated after a fresh initial query
set. A disappeared or new track therefore receives no explicit Mamba memory
reset because no temporal memory exists in this module; its behavior follows
the row and query fields passed through `Instances`.

## Track lifecycle findings

`RuntimeTrackerBase.update()` assigns new physical IDs to rows with
`obj_idxes == -1` and score above `score_thresh`, increments disappearance time
for existing rows below `filter_score_thresh`, and clears the ID after
`miss_tolerance`. It does not perform an explicit appearance or geometric
re-association.

Physical lifecycle uses `scores` derived from object class logits. Referring
scores do not keep a physical track alive. In `inference.py`, referring scores
are applied later as an output filter (`refers > 0.4`). The model inference path
does not read GT; GT is used in the training criterion and in the separate
evaluation-output writer.

## State and overwrite risks to test

- `load_detection_output_into_cache()` mutates the frame result by popping
  `query_feats` and copies raw predictions into cache fields.
- STReasoner replaces those cache fields with its refined predictions.
- `frame_summarization()` copies the refined cache back to `pred_*`, so an
  earlier referring head result can be silently replaced.
- `Instances.__getitem__()` and `Instances.cat()` operate on every field by
  row order. The current fields are tensors with a shared first dimension, but
  no field is keyed by physical object ID.
- QIM concatenates fresh initial queries with active rows. Any new memory state
  added to `Instances` must be initialized, sliced, concatenated, and reset at
  the same lifecycle boundaries.

The first Phase 1 tests cover zero/one/multiple rows, row permutation, cache
overwrite ordering, and the no-GT inference property. New memory-specific
tests will be added with QIM-v2 rather than weakening this baseline audit.

## Phase 1 verification

The current baseline invariants and FGRefMOT-v2 component tests pass the
focused test suite:

`36 passed, 1 warning in 10.37s` with `CUDA_VISIBLE_DEVICES=''`.

The no-GPU import smoke also passes (`NO_GPU_IMPORT_OK`, CUDA unavailable), and
standalone TAM-v2/QIM-v2 state dictionaries pass strict save/load round-trip
(`FGREF_V2_ROUNDTRIP_OK`, 21 and 18 state-dict entries respectively).

The warning is the existing timm deprecation warning. The tests run with the
`mamMOT` environment and do not require a dataset, GPU, checkpoint, or GT
files.
