# Phase 78 — final report entrypoints and evidence ledger

Status: `IN_PROGRESS_ENTRYPOINTS_CREATED`

The plan-required report entrypoints now exist in this directory. They record
the audited assets, data/split policy, common tracker, FlexHook and YesTrack
baseline boundaries, registered ablations, diagnostic subsets, efficiency
measurements, reviewer evidence, and the final report fields. Every metric
that is not backed by a successful current manifest is explicitly `NOT_RUN`,
`PENDING`, or `IN_PROGRESS`.

Validation completed after the documentation change:

- project tests: **78 passed**, 2 deprecation warnings;
- experiment matrix: **VALID**, 18 registered variants;
- `git diff --check`: **PASS**.

The local source commit is `0eeb82c121972a83cdce9ae9710a78c8ce52e2b1`. The
same document tree was written through the GitHub data API to the research
branch at remote commit `472b8246820165e0120c25e5a7c086cea2ae57a6`; representative
Git blob hashes match the local files. The reports remain provisional until
the live YesTrack formal run, multi-seed internal evaluations, registered
postbaseline controls, complete online timing, and final gate audit finish.
