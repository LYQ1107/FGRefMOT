# Phase 19 — Baidu egress comparison

Audit date: 2026-10-19
Status: **EGRESS COMPARISON COMPLETE / EXTERNAL ASSETS PENDING / NO GPU JOB STARTED**

I compared the official FlexHook share through the server's direct HTTPS
egress, the configured HTTPS proxy, and the official `yun.baidu.com` host.  The
probe requested only the 481-byte `config.json` entry and never changed a
signed field, supplied credentials/cookies, or bypassed verification.

The direct egress can list the share and obtain a fresh `tplconfig`, but
`api/sharedownload` returns `errno=-20` (`验证码输入错误，请重试`).  The configured
proxy returns `errno=0` and a dlink, yet its `dstime` is exactly 3,601 seconds
older than the fresh ticket timestamp; a direct GET then returns HTTP 403
(`expire time out`, error 31360).  The official `yun.baidu.com` host produces
the same one-hour-stale dlink and the same 403.  Switching official hosts or
using direct HTTPS therefore does not produce a usable anonymous ticket.

No bytes were written to the workspace or NAS, and no partial file remains.
The strict preflight was rerun and remains `PENDING` with six missing external
asset gates.  GPU training and metric generation remain disallowed until the
assets pass the hash-based preflight.

The remaining dependency is an authenticated/browser route that can complete
the provider's verification, followed by `scripts/nas_sync.sh` into the
declared NAS paths.  Machine-readable evidence is in
`PHASE19_BAIDU_EGRESS_COMPARISON.json`.
