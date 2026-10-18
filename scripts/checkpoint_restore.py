#!/usr/bin/env python3
"""Restore a checkpoint only after verifying its complete marker and hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    marker = archive.with_name(f"{archive.name}.complete.json")
    if not archive.is_file() or not marker.is_file():
        raise SystemExit("checkpoint_restore: archive or complete marker is missing")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    expected_size = int(metadata["size_bytes"])
    expected_digest = str(metadata["sha256"])
    if archive.stat().st_size != expected_size or sha256(archive) != expected_digest:
        raise SystemExit("checkpoint_restore: archive verification failed")

    destination = args.destination.expanduser()
    if destination.exists():
        raise SystemExit(f"checkpoint_restore: refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    try:
        shutil.copyfile(archive, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if temporary.stat().st_size != expected_size or sha256(temporary) != expected_digest:
            raise SystemExit("checkpoint_restore: restored file verification failed")
        os.replace(temporary, destination)
        print(json.dumps({"status": "RESTORED", "destination": str(destination), "sha256": expected_digest}, indent=2))
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"checkpoint_restore: {exc}", file=sys.stderr)
        raise SystemExit(2)
