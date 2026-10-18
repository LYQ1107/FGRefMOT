# Generated audit outputs

`refer_sports_manifest.json` and `refer_sports_manifest.csv` are generated
locally by `scripts/audit_refer_sports.py` from the read-only source paths in
`configs/paths.env.example`. They are intentionally kept out of the source
branch because they contain machine-local absolute paths and are regenerated
from the pinned source hashes. The current copies are archived under the NAS
project metadata tree; the checksum below is committed for provenance.

The expression IDs use the global SportsMOT `labels_with_ids` namespace. The
per-video IDs in `gt/gt.txt` are a different namespace and must not be used to
join Refer-Sports expressions. For evaluation GT, run
`scripts/build_refer_sports_mot_gt.py`; its output is evaluation-only and is
never a detector or tracker input.
