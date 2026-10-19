# Phase 36 — Common tracker coverage correction and Refer-Sports generation

Audit date: 2026-10-19  
Status: **CORRECTED / GENERATION PREPARED / GPU INFERENCE PENDING**

The Phase 23 candidate is an official SambaMOTR-DAB-D-DETR release with 150
files for the SportsMOT `dataset/test` split. The Refer-Sports expression
manifest used by this study has 19 evaluation videos from the SportsMOT
`dataset/train` tree. A direct filename comparison reports zero overlap, so the
published test files cannot be supplied to the referring evaluator for these
19 videos. This corrects the earlier Phase 23 mapping statement before any
referring metrics are reported.

The official SambaMOTR checkout was cloned directly to NAS at
`/home/A100/FGREF_RMOT_SPORTS_20261019/assets/source/sambamotr`, pinned at
commit `f1c139a653c00a55c0873a64d7c59a67d7dbad44`. Its DAB-D-DETR checkpoint
and configuration are reused from the hash-verified Phase 23 asset. The
PyTorch 2.6 build requires a two-call API compatibility patch changing
`value.type()` to `value.scalar_type()` in the CUDA dispatch sites; the patch
SHA-256 is `3de2dd43a96f7fe106dac985efce1a88dc5c49f8372d2b8c5fd86fdcebf3905e`.
The compiled operator is
`MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so` with SHA-256
`9924ebe74085def9643cac54d684972bd2d5fb71532e28fe98d6b50fbbc19147`; a CUDA
forward smoke passed in the dedicated environment.

The reproducible launcher is
`scripts/run_sambamotr_refersports_tracker.py`. It builds an image-only
staging view containing exactly the 19 sequence names in
`configs/refer_sports_eval_sequences_20261019.txt`; each staged sequence has
only an `img1` symlink, with `gt`, `labels_with_ids`, and other annotations
omitted. The prepared NAS run root is
`/home/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/`
`sambamotr_dab_refersports_train_20261019`. Its preparation manifest is
`TRACKER_RUN_MANIFEST.json` with SHA-256
`d9cddc3fdf9f237edf4a33bc6af0cadc9440b1501bc2953fec0f9cbc3711d05a`; it
records the checkpoint, source, operator, runtime config, image-only policy,
exact sequence list, and future GPU command. No tracker inference has been
claimed yet, and no GT/label file is accepted as a fallback.

Once GPU 1 is free, the launcher will run the fixed checkpoint on this view and
write all 19 outputs directly to NAS. The resulting output directory must pass
row, frame, count, and SHA-256 checks before the common-tracker contracts are
updated and either FlexHook or YesTrack referring evaluation is started.
