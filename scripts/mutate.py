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
import base64
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
JOURNAL = ROOT / "runs" / ".mutation-journal.json"
Runner = Callable[[list[str]], tuple[int, list[str]]]
def _journal_path(root: Path) -> Path:
    return root / "runs" / JOURNAL.name


def write_journal(root: Path, case: dict, original: bytes, planted: bytes) -> None:
    """Persist the restore before planting so SIGKILL recovery can undo the exact bytes."""
    payload = {"pid": os.getpid(), "file": case["file"], "label": case["label"],
               "original": base64.b64encode(original).decode(),
               "planted": base64.b64encode(planted).decode(),
               "original_sha": hashlib.sha256(original).hexdigest(),
               "planted_sha": hashlib.sha256(planted).hexdigest()}
    journal = _journal_path(root)
    journal.parent.mkdir(parents=True, exist_ok=True)
    tmp = journal.with_name(f".{journal.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(tmp, journal)
    finally:
        tmp.unlink(missing_ok=True)


def _replace_bytes_atomic(path: Path, data: bytes) -> None:
    """Replace one plant atomically so cancellation cannot leave a partial source file."""
    mode = path.stat().st_mode & 0o777
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def recover_journal(root: Path = ROOT) -> bool:
    """Restore a dead owner’s exact plant; refuse to overwrite a different writer’s bytes."""
    journal = _journal_path(root)
    if not journal.is_file():
        return False
    data = json.loads(journal.read_text())
    path = root / data["file"]
    original = base64.b64decode(data["original"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != data["planted_sha"]:
        raise RuntimeError(f"mutation journal conflict at {path}; refusing to overwrite another edit")
    _replace_bytes_atomic(path, original)
    if hashlib.sha256(path.read_bytes()).hexdigest() != data["original_sha"]:
        raise RuntimeError(f"mutation journal restore failed at {path}")
    journal.unlink(missing_ok=True)
    return True


def _owner_pid(lock: Path) -> int | None:
    try:
        text = (lock / "owner").read_text()
        return int(text.split()[1])
    except (OSError, ValueError, IndexError):
        return None


@contextlib.contextmanager
def held(lock: Path = LOCK, wait_s: float = 900, poll_s: float = 5, files: list[str] | None = None) -> Iterator[None]:
    """Hold the mutation lock: one planting run per working tree. mkdir is atomic, so only one of two concurrent
    runners gets it. The owner file names who holds it and the files its cases plant in, for the one who waits and
    for editors checking before they write."""
    t0 = time.time()
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            lock.mkdir(parents=False)
            break
        except FileExistsError:
            owner = _owner_pid(lock)
            if owner is not None:
                try:
                    os.kill(owner, 0)
                except ProcessLookupError:
                    recover_journal(lock.parent.parent)
                    (lock / "owner").unlink(missing_ok=True)
                    lock.rmdir()
                    continue
                except PermissionError:
                    pass
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




def _module_path(root: Path, name: str) -> Path | None:
    rel = Path(*name.split("."))
    for path in (root / (str(rel) + ".py"), root / rel / "__init__.py"):
        if path.is_file():
            return path
    return None


def dependency_closure(root: Path, tests: list[str]) -> list[Path]:
    """Resolve local Python imports from listed tests; unresolved third-party imports are outside the tree."""
    todo = list(tests)
    seen: set[Path] = set()
    while todo:
        name = todo.pop()
        path = _module_path(root, name)
        if path is None or path in seen:
            continue
        seen.add(path)
        try:
            tree = __import__("ast").parse(path.read_text(), str(path))
        except (OSError, SyntaxError):
            continue
        for node in __import__("ast").walk(tree):
            if isinstance(node, __import__("ast").Import):
                todo.extend(a.name for a in node.names if a.name.startswith(("localbench.", "tests.")))
            elif isinstance(node, __import__("ast").ImportFrom) and node.module:
                if node.module.startswith(("localbench", "tests")):
                    todo.append(node.module)
    return sorted(seen)


def case_key(case: dict, root: Path | None = None) -> str:
    """Content key for a plant: case JSON, target bytes, and local test/import closure bytes."""
    root = root or ROOT
    paths = {root / case["file"], *dependency_closure(root, case["tests"])}
    h = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode())
    for path in sorted(p for p in paths if p.is_file()):
        raw = path.read_bytes()
        rel = str(path.relative_to(root)).encode()
        h.update(len(rel).to_bytes(8, "big") + rel + len(raw).to_bytes(8, "big") + raw)
    return h.hexdigest()


def read_cache(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text())
        return doc if doc.get("schema") == 1 and isinstance(doc.get("entries"), dict) else {"schema": 1, "entries": {}}
    except (OSError, ValueError, TypeError):
        return {"schema": 1, "entries": {}}


def write_cache(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n")
    os.replace(tmp, path)

def unittest_runner(root: Path = ROOT) -> Runner:
    """Run the named tests under root with a fresh bytecode cache."""
    def run(tests: list[str]) -> tuple[int, list[str]]:
        from localbench import lifecycle

        with tempfile.TemporaryDirectory() as cache:
            child = lifecycle.spawn(
                ["uv", "run", "--quiet", "python", "-m", "unittest", *tests],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "PYTHONPYCACHEPREFIX": cache}, start_new_session=True,
            )
            try:
                _, stderr = child.communicate()
                returncode = child.returncode
            finally:
                lifecycle.unregister_child(child)
        failures = [line.split(" (")[0] for line in stderr.splitlines()
                    if line.startswith(("FAIL:", "ERROR:"))]
        return returncode, failures
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
        write_journal(root, case, src, planted)
        _replace_bytes_atomic(path, planted)
        bad_rc, fails = runner(case["tests"])
    finally:
        now = path.read_bytes()
        if now in (planted, src):
            if now == planted:
                _replace_bytes_atomic(path, src)
            journal = _journal_path(root)
            if journal.is_file():
                try:
                    data = json.loads(journal.read_text())
                except (OSError, ValueError):
                    data = {}
                if data.get("original_sha") == sha and data.get("planted_sha") == hashlib.sha256(planted).hexdigest():
                    journal.unlink(missing_ok=True)
        else:
            kept = _keep_conflict(root, case, {"original": src, "planted": planted, "found": now})
            conflict = {"label": case["label"], "kept": str(kept),
                        "conflict": f"{case['file']} was written by someone else while the plant was in it; their "
                        "bytes were left on disk and may still contain the plant (diff against .planted)"}
    if conflict:
        return conflict
    restored = hashlib.sha256(path.read_bytes()).hexdigest() == sha
    _journal_path(root).unlink(missing_ok=True)
    return {"label": case["label"], "caught": good_rc == 0 and bad_rc != 0, "restored": restored,
            "good_rc": good_rc, "bad_rc": bad_rc, "fails": fails}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plant each case's defect and check its tests catch it.")
    ap.add_argument("cases", type=Path)
    ap.add_argument("--wait-slot", type=float, default=0, metavar="SECONDS",
                    help="queue for the heavy-job slot up to SECONDS instead of refusing when held or busy")
    ap.add_argument("--force-load", action="store_true",
                    help="take the heavy-job slot past a busy admission")
    ap.add_argument("--cached", action="store_true",
                    help="reuse caught/restored verdicts when the case and local test/import closure are unchanged")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from localbench.lifecycle import install_cancel_handlers
    install_cancel_handlers()
    from localbench.heavyslot import SlotRefused, acquire
    try:
        slot = acquire("mutate", wait_s=max(0.0, a.wait_slot), force_load=a.force_load, needs_gpu=False)
    except SlotRefused as exc:
        print(f"heavy slot: {exc}", file=sys.stderr)
        return 1
    with slot:
        cases = json.loads(a.cases.read_text())
        if not cases:
            # Nothing planted proves nothing: an empty file must not read as every plant caught.
            print(f"no cases in {a.cases}: nothing was planted", file=sys.stderr)
            return 2
        ok, conflicts = True, []
        cache_path = ROOT / "runs" / ".mutation-cache.json"
        cache = read_cache(cache_path) if a.cached else {"schema": 1, "entries": {}}
        dirty_cache = False
        with held(LOCK, files=sorted({c["file"] for c in cases})):
            for case in cases:
                key = case_key(case) if a.cached else None
                if key and key in cache["entries"]:
                    r = {**cache["entries"][key], "label": case["label"], "reused": True}
                else:
                    r = run_case(case, ROOT, unittest_runner())
                    if key and r.get("caught") and r.get("restored"):
                        cache["entries"][key] = {k: r[k] for k in ("caught", "restored", "good_rc", "bad_rc", "fails")}
                        dirty_cache = True
                if "conflict" in r:
                    conflicts.append(r)
                ok &= bool(r.get("caught") and r.get("restored"))
                print(json.dumps(r, ensure_ascii=False), flush=True)
        if dirty_cache:
            write_cache(cache_path, cache)
        for r in conflicts:
            print(f"CONFLICT (case void): {r['conflict']}; versions kept in {r['kept']}", file=sys.stderr)
        return 2 if conflicts else 0 if ok else 1
    raise AssertionError("unreachable: the with-slot body always returns")


if __name__ == "__main__":
    raise SystemExit(main())
