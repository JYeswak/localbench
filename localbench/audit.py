"""The mutation ledger: one JSONL row per real invocation of a state-changing verb (park, smol set, keep, doctor --fix,
...), whether it changed something, found nothing to do, was refused, or failed. `localbench audit` lists rows,
`localbench why <id>` prints one in full. A dry run writes nothing.

The ledger lives outside the clone (~/.localbench/audit.jsonl, next to the smol server's state) because the state it
describes does too: ollama names, omp profiles, LaunchAgents. The path is read from AUDIT_PATH at call time, so the
test package points it at a temporary file (tests/__init__.py)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import secrets
import socket
import time
from pathlib import Path

AUDIT_PATH = Path.home() / ".localbench" / "audit.jsonl"
OUTCOMES = ("done", "refused", "failed")


def path() -> Path:
    """The ledger file, resolved at call time from AUDIT_PATH."""
    return AUDIT_PATH


def _version() -> str | None:
    try:
        return importlib.metadata.version("localbench")
    except importlib.metadata.PackageNotFoundError:
        return None


def new_id(t: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t)) + "-" + secrets.token_hex(3)


def record(verb: str, argv: list[str], actions: list[str], outcome: str, detail: dict | None = None) -> str:
    """Append one row and return its id. `actions` are the steps executed (or attempted, for a failure; the planned
    ones, for a refusal); `outcome` is done, refused or failed."""
    if outcome not in OUTCOMES:
        raise ValueError(f"audit outcome {outcome!r}: one of {', '.join(OUTCOMES)}")
    t = time.time()
    row = {"id": new_id(t), "t": t, "verb": verb, "argv": list(argv), "actions": list(actions), "outcome": outcome,
           "detail": detail or {}, "localbench_version": _version(), "host": socket.gethostname(), "cwd": os.getcwd()}
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row["id"]


def rows(since: float | None = None) -> list[dict]:
    """Every row, oldest first; with `since` (epoch seconds), those at or after it. A torn last line (a writer killed
    mid-append) is skipped, not fatal."""
    p = path()
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and (since is None or r.get("t", 0) >= since):
            out.append(r)
    return out


def row(row_id: str) -> dict | None:
    return next((r for r in rows() if r.get("id") == row_id), None)
