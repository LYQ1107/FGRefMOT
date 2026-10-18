#!/usr/bin/env python3
"""Prepare a symlink-only FlexHook runtime root.

FlexHook's pinned visual loader resolves ``pretrained/...`` relative to the
process working directory.  This staging directory supplies those paths as
symlinks to the hash-audited NAS assets without editing the vendored checkout
or copying multi-gigabyte weights into the repository.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _publish_link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"asset directory does not exist: {target}")
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target:
            raise RuntimeError(f"refusing to replace existing runtime path: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def prepare(runtime_root: Path, roberta_dir: Path, swin_dir: Path) -> dict:
    runtime_root = runtime_root.expanduser().resolve()
    _publish_link(runtime_root / "pretrained" / "roberta-base", roberta_dir)
    _publish_link(runtime_root / "pretrained" / "swin_rope_mixed_tiny_patch4_window7_224", swin_dir)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(runtime_root),
        "policy": "SYMLINK_ONLY; VENDORED_CHECKOUT_UNMODIFIED",
        "links": {
            "pretrained/roberta-base": {
                "target": str(roberta_dir.expanduser().resolve()),
                "path": str(runtime_root / "pretrained" / "roberta-base"),
            },
            "pretrained/swin_rope_mixed_tiny_patch4_window7_224": {
                "target": str(swin_dir.expanduser().resolve()),
                "path": str(runtime_root / "pretrained" / "swin_rope_mixed_tiny_patch4_window7_224"),
            },
        },
    }
    output = runtime_root / "RUNTIME_MANIFEST.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--roberta-dir", type=Path, required=True)
    parser.add_argument("--swin-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.runtime_root, args.roberta_dir, args.swin_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
