# Permutation test

The M1 test suite checks that canonical geometry ordering, deterministic
tie-breaking, and inverse permutation restore track-aligned outputs after a
random row permutation. It also tests equal predicted coordinates with unknown
(`obj_idxes=-1`) newborn tracks using a current-feature tie key.

Result at commit `63db294`: passed as part of the 18-test CUDA regression run.
The test is a structural invariance check; it does not use target IDs or GT
ordering to compute the result.
