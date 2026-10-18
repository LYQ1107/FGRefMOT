#!/usr/bin/env python3
"""Atomically archive one checkpoint file to a writable filesystem.

The destination is first written as ``<name>.partial.<pid>``.  A JSON complete
marker is published only after fsync, size and SHA256 verification succeed.
Existing files are never overwritten silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination_dir", type=Path)
    parser.add_argument("--name", help="destination filename; defaults to source name")
    args = parser.parse_args()

    source = args.source.resolve()
    destination_dir = args.destination_dir.expanduser()
    if not source.is_file():
        raise SystemExit(f"checkpoint_archive: source is not a regular file: {source}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or source.name
    final = destination_dir / name
    marker = destination_dir / f"{name}.complete.json"
    if final.exists() or marker.exists():
        raise SystemExit(f"checkpoint_archive: refusing to overwrite existing archive: {final}")

    digest = sha256(source)
    temporary = destination_dir / f".{name}.partial.{os.getpid()}"
    try:
        shutil.copyfile(source, temporary)
        fsync_file(temporary)
        copied_size = temporary.stat().st_size
        copied_digest = sha256(temporary)
        if copied_size != source.stat().st_size or copied_digest != digest:
            raise RuntimeError("size or SHA256 mismatch before publish")
        os.replace(temporary, final)
        directory_fd = os.open(destination_dir, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "destination": str(final),
            "size_bytes": copied_size,
            "sha256": copied_digest,
        }
        marker_tmp = destination_dir / f".{marker.name}.partial.{os.getpid()}"
        with marker_tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(marker_tmp, marker)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f"checkpoint_archive: {exc}", file=sys.stderr)
        raise SystemExit(2)
