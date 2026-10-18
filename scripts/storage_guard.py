#!/usr/bin/env python3
"""Fail-fast free-space checks for a research run.

The guard only calls statvfs; it never deletes files or traverses a dataset.
It is intended to run before a job starts and before writing a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FilesystemCheck:
    path: str
    total_bytes: int
    available_bytes: int
    used_bytes: int
    use_percent: float
    required_bytes: int
    minimum_free_bytes: int
    ok: bool


def check_path(path: str, required: int, minimum_free: int, max_use: float) -> FilesystemCheck:
    target = Path(path).expanduser()
    st = os.statvfs(target)
    total = st.f_blocks * st.f_frsize
    available = st.f_bavail * st.f_frsize
    used = max(total - available, 0)
    percent = (100.0 * used / total) if total else 100.0
    ok = available >= required + minimum_free and percent <= max_use
    return FilesystemCheck(
        path=str(target),
        total_bytes=total,
        available_bytes=available,
        used_bytes=used,
        use_percent=round(percent, 3),
        required_bytes=required,
        minimum_free_bytes=minimum_free,
        ok=ok,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="directories on the target filesystems")
    parser.add_argument("--required-bytes", type=int, default=0)
    parser.add_argument("--minimum-free-bytes", type=int, default=0)
    parser.add_argument("--max-use-percent", type=float, default=95.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = []
    failed = False
    for raw in args.paths:
        try:
            result = check_path(
                raw, args.required_bytes, args.minimum_free_bytes, args.max_use_percent
            )
        except OSError as exc:
            print(f"storage_guard: cannot inspect {raw}: {exc}", file=sys.stderr)
            failed = True
            continue
        checks.append(result)
        failed |= not result.ok

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "required_bytes": args.required_bytes,
        "minimum_free_bytes": args.minimum_free_bytes,
        "max_use_percent": args.max_use_percent,
        "checks": [asdict(item) for item in checks],
        "ok": not failed and bool(checks),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in checks:
            print(
                f"{item.path}: available={item.available_bytes} bytes "
                f"use={item.use_percent:.2f}% required={item.required_bytes} "
                f"minimum_free={item.minimum_free_bytes} ok={item.ok}"
            )
        print(f"storage_guard: {'OK' if payload['ok'] else 'BLOCKED'}")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
