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
import runpy
import sys
from pathlib import Path


def _run_flexhook_in_process(command: list[str], runtime_root: Path) -> int | None:
    """Run the pinned FlexHook script after installing the Transformers shim.

    The upstream loader calls ``batch_encode_plus``, which was removed by the
    installed Transformers 5 API.  Installing the alias in this launcher and
    then replacing the process with ``execvp`` would lose the monkey patch, so
    the FlexHook entrypoint is evaluated in this same interpreter.  Other
    commands keep the original ``execvp`` behavior.
    """

    if len(command) < 2:
        return None
    script = Path(command[1]).expanduser().resolve()
    if script.name != "main.py" or script.parent.name != "FlexHook":
        return None
    repo_root = Path(__file__).resolve().parents[1]
    upstream_root = script.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from scripts.transformers_compat import install_legacy_tokenizer_alias

    install_legacy_tokenizer_alias()
    os.chdir(runtime_root)
    sys.argv = [str(script), *command[2:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


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
    in_process_result = _run_flexhook_in_process(command, runtime_root)
    if in_process_result is not None:
        return in_process_result
    os.chdir(runtime_root)
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
