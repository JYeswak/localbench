#!/usr/bin/env python3
"""Mutation runner: does a planted defect fail the tests that claim to catch it?

Anti-ceremony (A12):
- Consumer: both agent panes grading each other's commits (docs/evidence/reviews/2026-09-23-cross-grades.md), and any
  author proving a new test before committing it.
- Gate: AGENTS.md "No self-grading without independent verification"; a cross-grade row records this runner's output.
- Defect class: a test that passes whatever the code does; a "caught" that was really another grader's plant; a
  plant that never ran because Python loaded stale bytecode; a restore that silently kept a plant.
- Delete when: the pre-commit hook runs each commit's declared mutations itself.

Each case in <cases.json> is {label, file, old, new, tests[]}:
1. The named tests pass on the known-good tree.
2. The edit is planted; `old` must occur exactly once in `file`.
3. The same tests fail on the planted tree.
4. The file is restored byte-for-byte, checked by sha256, but only when it still holds exactly the planted bytes.
   Bytes that changed during the case are someone else's edit (kit-mutate-clobber-67c: restores erased another
   agent's edits on 2026-10-01): they stay on disk, the original, planted and found versions are kept under
   runs/mutate-conflict-*/, the case is void and the run exits 2. A change before the plant voids it unplanted.

Two rules came from failures on 2026-09-23. Every run holds runs/.mutation.lock (atomic mkdir), because two graders
planting in one working tree voided each other's known-good run. Every test run gets a fresh PYTHONPYCACHEPREFIX: a
.pyc is validated by source mtime in whole seconds plus size, so a same-size plant written in the restore's second
ran the cached original and read "not caught".

    scripts/mutate.py cases.json        exit 0 only if every case is caught and restored; 2 on a concurrent edit
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "runs" / ".mutation.lock"
Runner = Callable[[list[str]], tuple[int, list[str]]]


@contextlib.contextmanager
def held(lock: Path = LOCK, wait_s: float = 900, poll_s: float = 5, files: list[str] | None = None) -> Iterator[None]:
    """Hold the mutation lock: one planting run per working tree. mkdir is atomic, so only one of two concurrent
    runners gets it. The owner file names who holds it and the files its cases plant in, for the one who waits and
    for editors checking before they write."""
    t0 = time.time()
    while True:
        try:
            lock.mkdir(parents=False)
            break
        except FileExistsError:
            if time.time() - t0 >= wait_s:
                owner = (lock / "owner").read_text().strip() if (lock / "owner").exists() else "?"
                raise TimeoutError(f"mutation lock held for {wait_s:.0f}s by {owner}") from None
            time.sleep(poll_s)
    try:
        (lock / "owner").write_text(f"pid {os.getpid()} since {time.ctime()}\n"
                                    + (f"files: {', '.join(files)}\n" if files else ""))
        yield
    finally:
        (lock / "owner").unlink(missing_ok=True)
        lock.rmdir()


def unittest_runner(root: Path = ROOT) -> Runner:
    """The named tests under `root`, each run in its own bytecode cache: stale .pyc cannot stand in for a plant."""
    def run(tests: list[str]) -> tuple[int, list[str]]:
        with tempfile.TemporaryDirectory() as cache:
            p = subprocess.run(["uv", "run", "--quiet", "python", "-m", "unittest", *tests], cwd=root, capture_output=True, text=True,
                               env={**os.environ, "PYTHONPYCACHEPREFIX": cache}, check=False)
        return p.returncode, [ln.split(" (")[0] for ln in p.stderr.splitlines() if ln.startswith(("FAIL:", "ERROR:"))]
    return run


def _keep_conflict(root: Path, case: dict, versions: dict[str, bytes]) -> Path:
    """Save every version of a file that changed under a case, for a human to reconcile; returns the directory."""
    out = root / "runs" / f"mutate-conflict-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    name = case["file"].replace("/", "__")
    for kind, data in versions.items():
        (out / f"{name}.{kind}").write_bytes(data)
    return out


def run_case(case: dict, root: Path, runner: Runner) -> dict:
    """One case -> {label, caught, restored, good_rc, bad_rc, fails}, {label, skipped} or {label, conflict, kept}.
    The plant is undone even when the runner raises, unless the file no longer holds the planted bytes: then another
    writer's bytes stay on disk (they may contain the plant, which the conflict message says) and the case is void."""
    path = root / case["file"]
    src = path.read_bytes()
    sha = hashlib.sha256(src).hexdigest()
    n = src.decode().count(case["old"])
    if n != 1:
        return {"label": case["label"], "skipped": f"anchor occurs {n} times, not once"}
    good_rc, _ = runner(case["tests"])
    now = path.read_bytes()
    if now != src:   # edited during the known-good run: planting over it would erase the edit
        kept = _keep_conflict(root, case, {"original": src, "found": now})
        return {"label": case["label"], "conflict": f"{case['file']} changed during the known-good run; nothing "
                "planted, the edit is untouched", "kept": str(kept)}
    planted = src.decode().replace(case["old"], case["new"]).encode()
    conflict = None
    try:
        path.write_bytes(planted)
        bad_rc, fails = runner(case["tests"])
    finally:
        now = path.read_bytes()
        if now == planted:
            path.write_bytes(src)
        else:
            kept = _keep_conflict(root, case, {"original": src, "planted": planted, "found": now})
            conflict = {"label": case["label"], "kept": str(kept),
                        "conflict": f"{case['file']} was written by someone else while the plant was in it; their "
                        "bytes were left on disk and may still contain the plant (diff against .planted)"}
    if conflict:
        return conflict
    restored = hashlib.sha256(path.read_bytes()).hexdigest() == sha
    return {"label": case["label"], "caught": good_rc == 0 and bad_rc != 0, "restored": restored,
            "good_rc": good_rc, "bad_rc": bad_rc, "fails": fails}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plant each case's defect and check its tests catch it.")
    ap.add_argument("cases", type=Path)
    a = ap.parse_args(argv)
    cases = json.loads(a.cases.read_text())
    if not cases:
        # Nothing planted proves nothing: an empty file must not read as every plant caught.
        print(f"no cases in {a.cases}: nothing was planted", file=sys.stderr)
        return 2
    ok, conflicts = True, []
    with held(LOCK, files=sorted({c["file"] for c in cases})):
        for case in cases:
            r = run_case(case, ROOT, unittest_runner())
            if "conflict" in r:
                conflicts.append(r)
            ok &= bool(r.get("caught") and r.get("restored"))
            print(json.dumps(r, ensure_ascii=False), flush=True)
    for r in conflicts:
        print(f"CONFLICT (case void): {r['conflict']}; versions kept in {r['kept']}", file=sys.stderr)
    return 2 if conflicts else 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
