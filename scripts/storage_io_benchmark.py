#!/usr/bin/env python3
"""Run a small, self-cleaning storage I/O smoke benchmark.

The benchmark writes a temporary deterministic file on each requested
filesystem, fsyncs it, reads it sequentially while hashing the bytes, and
performs deterministic random reads.  It is intentionally small and does not
copy a dataset or a checkpoint.  The output is evidence about filesystem I/O,
not a claim about model checkpoint throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHUNK_BYTES = 1024 * 1024


def _payload(size: int) -> bytes:
    """Return a deterministic one-megabyte payload prefix."""

    return bytes(index % 251 for index in range(min(size, CHUNK_BYTES)))


def _mib_per_second(byte_count: int, elapsed: float) -> float:
    return round(byte_count / (1024 * 1024) / max(elapsed, 1e-9), 3)


def benchmark_path(path: str, size_mib: int, random_reads: int) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_dir():
        raise NotADirectoryError(target)
    size_bytes = size_mib * 1024 * 1024
    if size_bytes <= 0:
        raise ValueError("size_mib must be positive")
    payload = _payload(CHUNK_BYTES)
    digest = hashlib.sha256()
    temporary_name: str | None = None
    write_start = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".rmot-storage-", suffix=".bin", dir=target, delete=False
        ) as handle:
            temporary_name = handle.name
            remaining = size_bytes
            while remaining:
                block = payload[: min(len(payload), remaining)]
                handle.write(block)
                remaining -= len(block)
            handle.flush()
            os.fsync(handle.fileno())
        write_elapsed = time.perf_counter() - write_start

        read_start = time.perf_counter()
        with open(temporary_name, "rb", buffering=0) as handle:
            while True:
                block = handle.read(CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
        read_elapsed = time.perf_counter() - read_start

        random_start = time.perf_counter()
        random_bytes = 0
        if random_reads:
            with open(temporary_name, "rb", buffering=0) as handle:
                max_offset = max(size_bytes - 4096, 0)
                for index in range(random_reads):
                    offset = (index * 104729) % (max_offset + 1)
                    random_bytes += len(os.pread(handle.fileno(), 4096, offset))
        random_elapsed = time.perf_counter() - random_start

        return {
            "path": str(target),
            "temporary_file_bytes": size_bytes,
            "sequential_write_mib_s": _mib_per_second(size_bytes, write_elapsed),
            "sequential_read_mib_s": _mib_per_second(size_bytes, read_elapsed),
            "random_reads": random_reads,
            "random_bytes": random_bytes,
            "random_read_ms": round(random_elapsed * 1000, 3),
            "sha256": digest.hexdigest(),
        }
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="directories on the filesystems to test")
    parser.add_argument("--size-mib", type=int, default=32)
    parser.add_argument("--random-reads", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.random_reads < 0:
        parser.error("random-reads must be non-negative")

    results = []
    for path in args.paths:
        results.append(benchmark_path(path, args.size_mib, args.random_reads))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "synthetic_temporary_file",
        "size_mib": args.size_mib,
        "random_reads": args.random_reads,
        "results": results,
        "temporary_files_removed": True,
        "checkpoint_transfer_tested": False,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            print(
                f"{result['path']}: write={result['sequential_write_mib_s']} MiB/s "
                f"read={result['sequential_read_mib_s']} MiB/s "
                f"random={result['random_read_ms']} ms"
            )
        print("storage_io_benchmark: temporary files removed")


if __name__ == "__main__":
    main()
