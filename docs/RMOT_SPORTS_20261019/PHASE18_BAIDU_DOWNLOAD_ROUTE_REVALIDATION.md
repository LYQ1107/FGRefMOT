# Phase 18 — Baidu official download route revalidation

Audit date: 2026-10-19 04:39 Asia/Shanghai
Status: **ROUTE REVALIDATED / EXTERNAL ASSETS PENDING / NO GPU JOB STARTED**

I re-ran the official FlexHook share route against the smallest listed file,
`/FlexHook/pretrained/roberta-base/config.json` (481 bytes), without changing
signed fields and without using credentials or cookies.  The listing and the
download-ticket endpoints respond, but the anonymous download ticket is stale:

* `wxlist` returned `errno=0`, with server time `1792355994`.
* `tplconfig` returned `errno=0`, with fresh timestamp `1792355995` and a sign.
* `api/sharedownload` with `clienttype=5` returned `errno=9019` (`need verify`).
* The `clienttype=0` request returned `errno=0` and a dlink, but its `dstime`
  was `1792352394`, 3,601 seconds older than the fresh ticket timestamp.
* A direct GET of that dlink returned HTTP 403 (`expire time out`, error 31360).
* The alternate `locatedownload` request returned HTTP 403 (`error 31045`,
  `user not exists`).

The bundled anonymous `BaiduPCS-Go` transfer path also failed while no logged-in
session was available (`获取分享项元数据错误`).  No model, checkpoint, or
tracker bytes were written; temporary partial files were removed.  The seven
NAS asset directories remain empty.  The strict preflight was rerun with the
dedicated environment and exited `2` with `baseline_asset_preflight: PENDING`
for the same six named assets.  No GPU job or metric claim is allowed yet.

The practical dependency is now explicit: complete the two official share
downloads on a machine/browser that can pass the provider's verification, then
transfer the resulting files to the declared NAS paths with
`scripts/nas_sync.sh`.  After that, record source paths and SHA-256 hashes and
rerun `scripts/preflight_baseline_assets.py --hash-files` before any training.

Machine-readable evidence is in
`PHASE18_BAIDU_DOWNLOAD_ROUTE_REVALIDATION.json`.
