"""omp's mnemopi memory store on this host (~/.omp/agent/memories/mnemopi/banks): which banks exist, whose working
directory each serves, how many rows and bytes, when it was last written — and pruning the banks localbench's own
children and probes created.

Since 2026-09-23 measured children keep their banks in localbench's own agent dir (workloads.AGENT_BANKS), so new
harness banks no longer land here; pruning clears the ones earlier runs left.

Why prune: omp scans banks/ at every session start (up to 64 directories) and recalls from the bank of the session's
cwd, so harness banks cost the user's sessions startup time and can put benchmark turns into recall. On 2026-09-23,
10 of 21 banks were leftovers from localbench probes.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import time
from pathlib import Path

BANKS = Path.home() / ".omp" / "agent" / "memories" / "mnemopi" / "banks"
# Working directories localbench children and probes run in. Every harness cwd now starts with /tmp/localbench-, so a
# bank's cwd (mnemopi names a bank `<basename(cwd)>-<hash>`) identifies it; lbmem-/lbside-/lbcap- are the
# 2026-09-23 probes that predate that rule.
HARNESS_CWD = re.compile(r"^(?:/private)?/tmp/(?:localbench-|lbmem-|lbside-|lbcap-)")
HARNESS_PREFIX = "localbench-"


def _cwds(db: Path) -> tuple[int, set[str]]:
    """(rows that carry metadata, distinct `$.cwd` values) over every table with a metadata_json column."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
    try:
        rows, cwds = 0, set()
        for (table,) in con.execute("select name from sqlite_master where type = 'table'").fetchall():
            if "metadata_json" not in {r[1] for r in con.execute(f'pragma table_info("{table}")')}:
                continue
            for cwd, n in con.execute(f"select json_extract(metadata_json, '$.cwd'), count(*) from \"{table}\" "
                                      "group by 1"):
                rows += n
                if cwd:
                    cwds.add(cwd)
        return rows, cwds
    finally:
        con.close()


def banks(root: Path = BANKS) -> list[dict]:
    """Every bank directory with its rows, cwds, size, last write, and whether localbench made it. A bank is
    harness-made when every cwd its rows record is a harness working directory, or when it records no cwd and its
    name carries the harness prefix (a harness cwd whose turn retained nothing)."""
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f for f in d.rglob("*") if f.is_file()]
        db = d / "mnemopi.db"
        error = None
        try:
            rows, cwds = _cwds(db) if db.exists() else (0, set())
        except sqlite3.Error as exc:
            rows, cwds, error = 0, set(), str(exc)
        harness = (bool(cwds) and all(HARNESS_CWD.match(c) for c in cwds)) or (
            not cwds and error is None and d.name.startswith(HARNESS_PREFIX))
        # -shm is the WAL index: readers (including this one) touch it, so it says nothing about the last write.
        modified = max((f.stat().st_mtime for f in files if not f.name.endswith("-shm")), default=d.stat().st_mtime)
        out.append({"bank": d.name, "rows": rows, "cwds": sorted(cwds),
                    "bytes": sum(f.stat().st_size for f in files),
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(modified)),
                    "harness": harness, **({"error": error} if error else {})})
    return out


def remove_banks(names: list[str], root: Path = BANKS) -> list[str]:
    """Delete the named bank directories (only direct children of `root`); returns the ones removed."""
    removed = []
    for name in names:
        path = root / name
        if path.parent == root and path.is_dir():
            shutil.rmtree(path)
            removed.append(name)
    return removed
