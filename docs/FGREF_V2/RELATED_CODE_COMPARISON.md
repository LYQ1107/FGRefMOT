# Related code comparison

Audit date: 2026-10-18 UTC. External repositories were inspected at the
listed branch tip; the current FGRefMOT branch does not copy their source.
Claims below are mechanism/provenance notes, not reproduced benchmark results.

| Work | Code commit | Key code and mechanism | License / redistribution note | Use in FGRefMOT-v2 |
| --- | --- | --- | --- | --- |
| [DKGTrack](https://github.com/acyddl/DKGTrack) | `197f354443bd1e7b490d204456a7654b7d1e4ccd` (`main`) | `models/transrmot_pro.py`: `proceesing_text_decoupling`, `forward_text`, and `_forward_single_image`. The code maps tokenizer offsets to spaCy POS groups and forms subject/static/motion features; the public README reports Refer-KITTI HOTA 52.23. | README says research use only; no standalone LICENSE file was found in the repository root during this audit. Do not copy code without preserving applicable notices/permission. | Conceptual language-routing and text-conditioned temporal comparison only. Its static/motion decomposition is explicitly not claimed as our new contribution. |
| [FlexHook](https://github.com/buptLwz/FlexHook) | `bd1acc38634b28525d54dc6e0fcb38335f0029f` (`main`) | `models/mymodel.py`: `conditional_pos_decoder`, `conditional_pos_projector`, `conditional_f_proj`, `output_ca_qkv`, and masked pair scoring. Conditional points are generated for language/track matching in a two-stage referring-by-tracking model. | MIT; retain the copyright and license notice if code is reused. | Mechanism reference for conditional evidence sampling and pair heads. It is not a new physical MOT ID allocator and is not copied into the current implementation. |
| [CRMOT / CRTracker](https://github.com/chen-si-jia/CRMOT) | `50ffe32d1ce6938c0b80c2c5559c542b184249af` (`master`) | `CRTracker/src/lib/deep_sort/` contains explicit Deep SORT-style track lifecycle and assignment utilities; repository task is cross-view RMOT on CRTrack. | MIT; retain the copyright and license notice if code is reused. | Conceptual comparison for separating physical track lifecycle from language selection. No multi-camera or cross-view code is ported. |
| [Mamba](https://github.com/state-spaces/mamba) | `e9594ce1c732d97440f0332fdc43170a2294dbfa` (`main`) | `mamba_ssm/modules/mamba_simple.py` is a sequence state-space mixer. The scan axis must be established from tensor layout and state caching, not inferred from the name. | Apache-2.0; preserve LICENSE/NOTICE obligations for redistribution. | Correctness reference. The archived TempMam QIM call uses `[1, N_active, D]` with fresh `inference_params=None` per update, so its scan is over active-track rows rather than persistent time. QIM-v2 uses explicit `[N_tracks,T_history,D]` temporal inputs. |
| [STORM-Bench](https://github.com/amazon-science/storm-referring-multi-object-grounding) | `0d87c3ba52a024ffb0ea9c533ec278ae5361f4fa` (`main`) | Repository provides `storm-bench.json`; clips record source/split, frames, tracks, and referring expressions. It describes VidOR-derived coverage beyond pedestrian/vehicle RMOT. | No LICENSE file was found at the repository root during this audit; obtain dataset/code permission before redistribution or benchmark use. | Future external generalization candidate only. It is not merged into the current training/evaluation matrix because task, data access, and license compatibility are unverified. |

## Audit conclusion

The implementation boundary is intentional: DKGTrack is a language-routing
comparison, FlexHook is a conditional pair/evidence comparison, CRMOT is a
track-lifecycle comparison, and Mamba is a temporal-axis correctness
reference. None supplies evidence that the present TAM-v2/QIM-v2 combination
has an independent gain. No external implementation was counted as a
reproduced baseline in the current result tables.
