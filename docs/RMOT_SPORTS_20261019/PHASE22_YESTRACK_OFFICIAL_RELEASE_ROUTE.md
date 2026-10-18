# Phase 22 — YesTrack official release route

Audit date: 2026-10-18T21:21:04.914419+00:00
  
Status: **BLOCKED_AT_SIGNED_DOWNLOAD**

The two official source checkouts are now verified at the requested pinned commits:

* FlexHook: `bd1acc38634b28525d54dc6e0fcb38335f0029f9`
* YesTrack: `ed42b8315b28c94dc595d83b0d49bf57002bb281`

This phase tested the official YesTrack public share from a process running on the NAS side of the existing reverse SSH tunnel. The public listing is reachable and identifies `/Yestrack/yestrack.zip` (`178,142,422` bytes, fs id `981448710771451`). The signed-link API also returns a `www.baidupcs.com` link from the NAS-side process.

A bounded `Range: bytes=0-1023` GET of each signed link returned HTTP `403` with Baidu error `31064` (`file is not authorized`). No ZIP magic bytes were observed, no partial archive was published, and no LoRA or tracker-result files were treated as acquired. The probe used `requests` with proxy inheritance disabled; cookie values and signed URLs are intentionally absent from this report.

The release route therefore remains **not acquired**. The next acceptable source is an authenticated browser/client download or another authorized endpoint, followed by the expected-size, SHA-256, and archive-member checks. No GPU job was started and no ground-truth boxes or IDs were used as tracker input.

Machine-readable evidence: `PHASE22_YESTRACK_OFFICIAL_RELEASE_ROUTE.json`.
