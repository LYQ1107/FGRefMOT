# Phase 17 — source fetch and external asset route audit

Audit date: 2026-10-19  
Status: **SOURCE FETCH VERIFIED / EXTERNAL ASSETS PENDING / NO GPU JOB STARTED**

The repaired HTTPS route was used by `scripts/fetch_official_repos.sh` to fetch
both pinned upstream sources.  The local checkouts are clean and their HEADs
match the study manifest:

| source | pinned/local commit | GitHub release assets | binary weights in source tree |
|---|---|---:|---|
| FlexHook | `bd1acc38634b28525d54dc6e0fcb38335f0029f9` | 0 | no; `SOTA_ckpts/CKPTS.md` and `tracker_outputs/TRACKERS.md` are instructions |
| YesTrack | `ed42b8315b28c94dc595d83b0d49bf57002bb281` | 0 | no model or LoRA binary |

This completes the source-code acquisition part of the study.  It does not
complete model acquisition: the fetch intentionally disables Git LFS smudge and
does not claim weights, datasets, or tracker outputs.

The official README routes remain the two Baidu shares recorded in Phase 13.
Their directory metadata was visible, but the server-side signed-download probe
returned HTTP 403 (`expire time out`, error 31360), followed by a rate-limit or
captcha response.  No model, checkpoint, or tracker bytes were written.  The
same network revalidation reaches GitHub through the existing Codex proxy, while
Hugging Face, hf-mirror, and ModelScope still fail during TLS before any model
bytes arrive.

The NAS project is reachable through the existing reverse SSH tunnel, and all
seven target asset directories are still empty.  The strict preflight therefore
remains `PENDING` for six named assets:

* FlexHook RoBERTa base and RoPE-Swin-Tiny base;
* FlexHook task checkpoint;
* YesTrack matching Qwen3-VL-2B-Instruct base and optional LoRA checkpoint;
* one provenance-complete SportsMOT physical-tracker output set shared by both
  methods.

The next required action is an authenticated download on a machine that can
complete the official share transfer, followed by `scripts/nas_sync.sh` to the
NAS paths, SHA-256 recording, and a strict preflight rerun.  A directory listing,
GT/labels, an unrelated checkpoint, or a tracker folder without detector
provenance will remain rejected.  No GPU job or metric claim is permitted until
the selected variant contract is `READY`.

Machine-readable evidence is in
`PHASE17_SOURCE_FETCH_AND_ASSET_ROUTE_AUDIT.json`.
