#!/usr/bin/env python3
"""Run deterministic subprocess fault cases against a command and record stdout/stderr/exit contracts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

FAULTS = ("exit_nonzero", "timeout", "malformed_output", "missing_binary")
TOOLS = ("launchctl", "br", "lsof", "uv", "stdin")


def fake_script(tool: str, fault: str) -> str:
    if fault == "exit_nonzero":
        return "#!/bin/sh\necho fault >&2\nexit 7\n"
    if fault == "timeout":
        return "#!/bin/sh\nsleep 30\n"
    if fault == "malformed_output":
        return "#!/bin/sh\nprintf 'not-json\\n'\n"
    return "#!/bin/sh\nexit 127\n"


def run_matrix(argv: list[str], timeout: float = 5.0) -> list[dict]:
    rows = []
    for tool in TOOLS:
        for fault in FAULTS:
            with tempfile.TemporaryDirectory(prefix="fault-matrix-") as tmp:
                root = Path(tmp)
                if fault != "missing_binary":
                    exe = root / tool
                    exe.write_text(fake_script(tool, fault))
                    exe.chmod(0o700)
                env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
                env["PATH"] = f"{root}{os.pathsep}{env.get('PATH', '')}"
                if tool == "stdin":
                    stdin = subprocess.DEVNULL
                else:
                    stdin = subprocess.PIPE
                try:
                    kwargs = {"capture_output": True, "timeout": timeout, "env": env}
                    if stdin is subprocess.DEVNULL:
                        kwargs["stdin"] = subprocess.DEVNULL
                    else:
                        kwargs["input"] = b""
                    command = [arg.replace("{tool}", tool) for arg in argv]
                    p = subprocess.run(["sh", "-c", 'exec "$@"', "fault-matrix", *command], **kwargs)
                    rows.append({"tool": tool, "fault": fault, "rc": p.returncode,
                                 "stdout": p.stdout.decode("utf-8", "replace")[:500],
                                 "stderr": p.stderr.decode("utf-8", "replace")[:500]})
                except subprocess.TimeoutExpired as exc:
                    rows.append({"tool": tool, "fault": fault, "rc": "timeout",
                                 "stdout": (exc.stdout or b"").decode("utf-8", "replace")[:500],
                                 "stderr": (exc.stderr or b"").decode("utf-8", "replace")[:500]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = ap.parse_args()
    if not args.command:
        ap.error("provide a command after --")
    if args.command[0] == "--":
        args.command = args.command[1:]
    rows = run_matrix(args.command)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"count": len(rows), "rows": rows}, indent=2) + "\n")
    print(f"FAULT-MATRIX {len(rows)} cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
