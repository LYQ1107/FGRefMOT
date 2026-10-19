# Memory and reliability test

The M3 tests cover:

- padding-safe per-track history and missing-observation births;
- ordinary `Instances` lifecycle propagation of all five `qim_*` fields;
- bounded reliability writes and explicit birth initialization;
- a real gradient path through the learned reliability gate;
- no output fields for a deleted/inactive row selected out of the active set.

Result at commit `63db294`: passed as part of the 18-test CUDA regression run.
The gate inputs are score, appearance consistency, motion consistency, memory
validity, age, and ambiguity only; referring scores/text are excluded.
