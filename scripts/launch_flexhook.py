#!/usr/bin/env python3
"""Launch a FlexHook command from its isolated symlink runtime root.

The launcher is intentionally thin: it changes directory to the prepared
runtime root, then replaces itself with the requested upstream command.  It
does not allocate GPUs or modify the vendored checkout when imported or when
the help text is displayed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="upstream command after --")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("an upstream command is required after --")
    runtime_root = args.runtime_root.expanduser().resolve()
    if not (runtime_root / "RUNTIME_MANIFEST.json").is_file():
        raise FileNotFoundError(f"runtime manifest missing: {runtime_root}")
    os.chdir(runtime_root)
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
