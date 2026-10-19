# Three lightweight designs

## M1 — Ordered Bi-Mamba

Active tracks are sorted by predicted `cx, cy, w, h`; known track IDs and a
deterministic current-feature fingerprint break ties. The projected sequence is
processed forward and backward with shared Mamba1 weights by default. The
backward output is flipped back to canonical order and the inverse permutation
restores caller row order before the existing FFN/query writer.

## M2 — Trackwise Temporal Mamba

Each active track supplies its own valid `hist_embeds` sequence. Padding is
removed before the Mamba call, and only the last valid state is projected back
to 256 dimensions and residually fused with the current query. The sequence is
causal and uses only the current/past history already produced by STReasoner.

## M3 — Reliability Hybrid

M3 reuses M2 and retains the existing STReasoner. A long memory is updated with
`m=(1-g)m_prev+g*phi(z)` only for active/born tracks. The gate sees detection
score, appearance consistency, motion consistency, memory validity, age, and
ambiguity; it never consumes referring score or target labels. Memory fields are
ordinary `Instances` tensors so lifecycle operations carry them through
concatenation, indexing, device transfer, and checkpoint reconstruction.

Optional `--qim_rmot_text_gate` affects only the short writer path. It is off by
default and cannot remove or overwrite the long identity memory.
