# Phase 1 — checkpoint archive/restore smoke

Status: `SYNTHETIC_ARCHIVE_RESTORE_PASS; MODEL_RESUME_NOT_RUN`

The archive and restore tools were exercised with a 4 MiB deterministic
checkpoint-shaped file. `checkpoint_archive.py` wrote the file to a temporary
directory on the mounted NAS, fsynced it, published the complete marker, and
verified size and SHA256 before publication. `checkpoint_restore.py` verified
the marker and restored the file atomically to a local temporary directory.

The source and restored SHA256 were both
`f7ff53b035be35e2320b580670939ce2d2a206d227bae982f4e19d9bb6026527`, and the
NAS complete marker was present. Both temporary directories were removed after
the check. This validates the file-level archive protocol only; no real model
checkpoint or optimizer state has yet been resumed.
