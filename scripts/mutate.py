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
4. The file is restored byte-for-byte, checked by sha256.

Two rules came from failures on 2026-09-23. Every run holds runs/.mutation.lock (atomic mkdir), because two graders
planting in one working tree voided each other's known-good run. Every test run gets a fresh PYTHONPYCACHEPREFIX: a
.pyc is validated by source mtime in whole seconds plus size, so a same-size plant written in the restore's second
ran the cached original and read "not caught".

    scripts/mutate.py cases.json        exit 0 only if every case is caught and restored
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
def held(lock: Path = LOCK, wait_s: float = 900, poll_s: float = 5) -> Iterator[None]:
    """Hold the mutation lock: one planting run per working tree. mkdir is atomic, so only one of two concurrent
    runners gets it. The owner file names who holds it, for the one who waits."""
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
        (lock / "owner").write_text(f"pid {os.getpid()} since {time.ctime()}\n")
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


def run_case(case: dict, root: Path, runner: Runner) -> dict:
    """One case -> {label, caught, restored, good_rc, bad_rc, fails} or {label, skipped}. The file is restored even
    when the runner raises."""
    path = root / case["file"]
    src = path.read_bytes()
    sha = hashlib.sha256(src).hexdigest()
    n = src.decode().count(case["old"])
    if n != 1:
        return {"label": case["label"], "skipped": f"anchor occurs {n} times, not once"}
    good_rc, _ = runner(case["tests"])
    try:
        path.write_text(src.decode().replace(case["old"], case["new"]))
        bad_rc, fails = runner(case["tests"])
    finally:
        path.write_bytes(src)
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
    ok = True
    with held(LOCK):
        for case in cases:
            r = run_case(case, ROOT, unittest_runner())
            ok &= bool(r.get("caught") and r.get("restored"))
            print(json.dumps(r, ensure_ascii=False), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
