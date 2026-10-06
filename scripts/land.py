#!/usr/bin/env python3
"""Self-serve landing: gate reserved files in an isolated worktree, then fast-forward main.

Each pane lands its own reserved files (reserve with Agent Mail file_reservation_paths first)
without a human bottleneck. Flow, mirroring land.sh: fresh DETACHED worktree at HEAD, copy the
named files (or only matching hunks), ruff + full suite + mutation gates selected by mutsel,
commit through the hook, fast-forward main, remove the worktree. Stops at the first red gate,
leaving the worktree for inspection.

Two safety properties beyond land.sh:
- Stale .git/index.lock: detected before committing (no lsof holder, older than 60 s) and
  REFUSED with the exact rm command for a human; land.py never removes a lock itself.
- Kill-safe: SIGINT/SIGTERM remove the disposable worktree and exit; all mutation happens in
  the worktree (the commit runs last among mutating steps), so an interrupted gate leaves no
  index.lock behind in either repository.

Stdlib only. Run from the repo root: python3 scripts/land.py NAME --message MSG files...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

STALE_LOCK_AGE_S = 60.0

_worktree: str | None = None


def clean_env() -> dict[str, str]:
    """Environment for git children: the caller's GIT_* variables are dropped.

    Under a git hook GIT_DIR / GIT_INDEX_FILE point at the caller's repository, and rev-parse
    would answer for it instead (backends.py, decision.py precedent; 2026-09-26 incident).
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def land_cmd(cmd: list[str], cwd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=clean_env(), **kw)


def _lock_holders(lock: str) -> list[str] | None:
    """PIDs holding the lock file via lsof; None when lsof is unavailable or fails."""
    try:
        out = subprocess.run(["lsof", "-t", lock], capture_output=True, text=True, timeout=15)
    except OSError:
        return None
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def lock_state(repo: str, now: float | None = None) -> str | None:
    """Describe .git/index.lock in repo: None when landing may proceed, else the refusal text.

    A lock with a live holder means another commit is in flight (wait, do not touch it). A lock
    with no holder older than STALE_LOCK_AGE_S is stale: refuse with the exact rm command for a
    human. land.py never removes a lock itself. A fresh holderless lock is racy: refuse briefly.
    """
    lock = os.path.join(repo, ".git", "index.lock")
    try:
        age = (time.time() if now is None else now) - os.path.getmtime(lock)
    except OSError:
        return None
    holders = _lock_holders(lock)
    if holders:
        return (f"refusing: {lock} is held by live process(es) {', '.join(holders)} "
                f"(another commit is in flight; wait and retry)")
    if holders is None:
        return (f"refusing: {lock} exists ({age:.0f}s old) but lsof is unavailable, so the holder "
                f"cannot be verified; have a human check it")
    if age < STALE_LOCK_AGE_S:
        return (f"refusing: {lock} exists ({age:.0f}s old) with no lsof holder; it may be fresh, "
                f"retry in a minute")
    return (f"refusing: stale {lock} ({age:.0f}s old, no holder). A human must remove it:\n"
            f"  rm {lock}\n"
            f"land.py never removes a lock itself.")


def refuse_if_locked(repo: str) -> None:
    state = lock_state(repo)
    if state is not None:
        print(state, file=sys.stderr)
        sys.exit(1)


def _cleanup_worktree() -> None:
    global _worktree
    wt, _worktree = _worktree, None
    if wt and os.path.isdir(wt):
        subprocess.run(["git", "worktree", "remove", "--force", wt],
                       capture_output=True, env=clean_env())


def _on_signal(signum: int, _frame) -> None:
    _cleanup_worktree()
    sys.exit(128 + signum)


def select_hunks(diff: str, old_text: str, pattern: str) -> str:
    """Rebuild a file as HEAD's content plus only hunks whose added/removed lines match.

    Hunks apply by OLD-side line numbers, bottom-up, so skipped hunks cannot shift the kept
    ones. Refuses when a kept hunk does not match HEAD (the tree moved underneath).
    """
    lines = old_text.splitlines(keepends=True)
    kept = []
    for h in re.split(r"(?m)^(?=@@ )", diff)[1:]:
        head, *body = h.splitlines(keepends=True)
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", head)
        if m is None:
            continue
        start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        removed = [ln[1:] for ln in body if ln.startswith("-")]
        added = [ln[1:] for ln in body if ln.startswith("+")]
        if any(pattern in ln for ln in removed + added):
            kept.append((start, count, removed, added))
    for start, count, removed, added in sorted(kept, reverse=True):
        # count == 0: pure insertion AFTER line `start`; otherwise replace start..start+count-1.
        at = start if count == 0 else start - 1
        if lines[at:at + count] != removed:
            sys.exit(f"hunk at old line {start} does not match HEAD; refusing")
        lines[at:at + count] = added
    return "".join(lines)


def split_spec(spec: str) -> tuple[str, str | None]:
    """A file argument is PATH (whole file) or PATH::TEXT (fixed-string matching hunks)."""
    if "::" in spec:
        path, text = spec.split("::", 1)
        return path, text
    return spec, None


def changed_hunks(diff: str) -> dict[str, list[tuple[int, int]]]:
    """New-side (start, end) line ranges per file from `git diff -U0 HEAD`."""
    hunks, cur = {}, None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m and cur:
            start, n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            hunks.setdefault(cur, []).append((start, start + max(n, 1) - 1))
    return hunks


def select_mutations(workdir: str, names: list[str]) -> list[str]:
    """Mutation gates to run: named ones plus every *-mutations.json with a case anchored in a
    changed hunk, planting into an untracked file, or naming a changed test module.

    Every case's anchor must be unique in its target file (static check, seconds); exits 2
    listing violations instead of running hours of mis-planted suites.
    """
    old = os.getcwd()
    os.chdir(workdir)
    try:
        diff = subprocess.run(["git", "diff", "-U0", "HEAD"], capture_output=True, text=True,
                              env=clean_env(), check=True).stdout
        untracked = set(subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                                       capture_output=True, text=True, env=clean_env(),
                                       check=True).stdout.split())
    finally:
        os.chdir(old)
    hunks = changed_hunks(diff)
    changed_tests = {f for f in [*hunks, *untracked] if f.startswith("tests/test_")}
    need, bad = set(names), []
    for path in sorted(glob.glob(os.path.join(workdir, "tests", "*-mutations.json"))):
        name = os.path.basename(path)[: -len("-mutations.json")]
        with open(path, encoding="utf-8") as f:
            cases = json.load(f)
        for c in cases:
            target = os.path.join(workdir, c["file"])
            try:
                with open(target) as f:
                    src = f.read()
            except OSError:
                bad.append(f"{name}: {c['label']}: {c['file']} missing")
                continue
            n = src.count(c["old"])
            if n != 1:
                bad.append(f"{name}: {c['label']}: anchor occurs {n} times")
                continue
            a = src[: src.index(c["old"])].count("\n") + 1
            b = a + c["old"].count("\n")
            if (c["file"] in untracked or path in untracked
                    or any(lo <= b and a <= hi for lo, hi in hunks.get(c["file"], []))
                    or any(f"tests/{t.split('.')[1]}.py" in changed_tests for t in c["tests"])):
                need.add(name)
    if bad:
        print("\n".join(bad), file=sys.stderr)
        sys.exit(2)
    return sorted(need)


def last_line(text: str) -> str:
    """Final output line, '(empty)' when there is none (no IndexError on empty gate output)."""
    lines = text.strip().splitlines()
    return lines[-1] if lines else "(empty)"


def run_gate(cmd: list[str], cwd: str, label: str) -> str:
    out: str = land_cmd(cmd, cwd).stdout
    print(f"== {label}: {last_line(out)}")
    return out


def mutation_problems(stdout: str) -> list[str]:
    """Uncaught, skipped, or unrestored mutation lines: the gate fails on these, not just rc."""
    return [ln for ln in stdout.splitlines()
            if '"caught": false' in ln or '"skipped"' in ln or '"restored": false' in ln]


def gate_failed(returncode: int, bad: list[str]) -> bool:
    """A mutation gate fails on a bad return code OR bad content (never content-blind)."""
    return returncode != 0 or bool(bad)

def parse_conflicts(text: str, agent: str | None) -> list[str]:
    """Holders other than agent from `am file_reservations conflicts` table output."""
    blockers = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 4 or parts[0] == "PATH" or parts[0].startswith("reservation_read_"):
            continue
        if agent is None or parts[1].lower() != agent.lower():
            blockers.append(f"{parts[0]} held by {parts[1]}")
    return blockers


def _reservation_problem(src: str, agent: str | None, paths: list[str]) -> list[str] | str | None:
    """None when clear, a blocker list when held, or an unreachable-reason string."""
    if shutil.which("am") is None:
        return "`am` not on PATH"
    try:
        r = subprocess.run(["am", "file_reservations", "conflicts", src, *paths],
                           capture_output=True, text=True, timeout=60, env=clean_env())
    except OSError as exc:
        return f"could not run `am`: {exc}"
    if r.returncode != 0:
        return f"`am` exited {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}"
    return parse_conflicts(r.stdout, agent) or None


def check_reservations(src: str, agent: str | None, paths: list[str],
                       owner_confirmed: bool) -> int:
    """Refuse when another pane holds the paths. Agent Mail unreachable: refuse unless
    --owner-confirmed, and record the bypass in the output."""
    problem = _reservation_problem(src, agent, paths)
    if problem is None:
        return 0
    if isinstance(problem, list):
        print("refusing: paths held by another pane:\n  " + "\n  ".join(problem),
              file=sys.stderr)
        return 1
    if not owner_confirmed:
        print(f"refusing: Agent Mail unreachable ({problem}); "
              f"retry or pass --owner-confirmed", file=sys.stderr)
        return 1
    print(f"OWNER-CONFIRMED reservation bypass recorded: Agent Mail unreachable ({problem}); "
          f"landing without a reservation check")
    return 0


def advance_main(src: str, sha: str, paths: list[str], runner=land_cmd) -> tuple[bool, str | None]:
    """Fast-forward main only if its ref remains at the revision we checked; never rewind a racing commit."""
    old_main = runner(["git", "rev-parse", "refs/heads/main"], src, check=True).stdout.strip()
    ancestor = runner(["git", "merge-base", "--is-ancestor", old_main, sha], src)
    if ancestor.returncode != 0:
        return False, f"main {old_main[:12]} is not an ancestor of {sha[:12]}"
    update = runner(["git", "update-ref", "refs/heads/main", sha, old_main], src)
    if update.returncode != 0:
        return False, f"main moved from {old_main[:12]} before compare-and-swap"
    reset = runner(["git", "reset", "-q", "--", *paths], src)
    if reset.returncode != 0:
        return False, f"main advanced to {sha[:12]} but index reset failed"
    return True, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="landing worktree suffix (wt-NAME)")
    ap.add_argument("--message", required=True, help="commit message")
    ap.add_argument("--mutations", default="",
                    help="extra mutation gate names (space-separated) beyond the anchored ones")
    ap.add_argument("--wait-slot", type=float, default=0.0, metavar="SECONDS",
                    help="queue for the heavyslot gate instead of refusing when busy")
    ap.add_argument("--src", default="~/Developer/localbench", help="canonical checkout")
    ap.add_argument("--agent", default=None,
                    help="Agent Mail identity landing (own holds do not block; without it every hold blocks)")
    ap.add_argument("--owner-confirmed", action="store_true",
                    help="record an OWNER-CONFIRMED bypass when Agent Mail is unreachable")
    ap.add_argument("files", nargs="+", help="PATH or PATH::REGEX (only matching hunks)")
    args = ap.parse_args(argv)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _on_signal)
    refuse_if_locked(args.src)

    sys.path.insert(0, args.src)
    from localbench import heavyslot
    slot = heavyslot.acquire("land", wait_s=max(0.0, args.wait_slot), needs_gpu=False)
    if slot is None:
        print("refusing: heavyslot gate busy (holder named above or in localbench status); "
              "retry with --wait-slot", file=sys.stderr)
        return 1
    if check_reservations(args.src, args.agent, [split_spec(f)[0] for f in args.files],
                          args.owner_confirmed) != 0:
        return 1

    global _worktree
    wt = os.path.join(os.path.expanduser("~/.localbench/tmp"), f"wt-{args.name}")
    if os.path.exists(wt):
        print(f"refusing: leftover worktree {wt} (a previous red gate); inspect or remove it",
              file=sys.stderr)
        return 1
    _worktree = wt
    try:
        r = land_cmd(["git", "worktree", "add", "--detach", wt, "HEAD"], args.src)
        if r.returncode != 0:
            print(f"worktree add failed: {r.stderr.strip()}", file=sys.stderr)
            _worktree = None
            return 1
        for spec in args.files:
            path, pattern = split_spec(spec)
            dest = os.path.join(wt, path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if pattern is None:
                shutil.copy2(os.path.join(args.src, path), dest)
            else:
                diff = land_cmd(["git", "diff", "-U0", "HEAD", "--", path], args.src,
                           check=True).stdout
                old = land_cmd(["git", "show", f"HEAD:{path}"], args.src, check=True).stdout
                with open(dest, "w") as f:
                    f.write(select_hunks(diff, old, pattern))
        out = run_gate(["uvx", "ruff", "check", "localbench"], wt, "ruff")
        if last_line(out) != "All checks passed!":
            return 1
        out = run_gate(["uv", "run", "--quiet", "python", "-m", "unittest", "discover",
                    "-s", "tests", "-t", "."], wt, "suite")
        if not any(ln == "OK" for ln in out.splitlines()):
            return 1
        names = select_mutations(wt, args.mutations.split())
        print(f"== mutation gates: {' '.join(names)}")
        for m in names:
            out = land_cmd(["uv", "run", "--quiet", "python", "scripts/mutate.py",
                       f"tests/{m}-mutations.json"], wt)
            bad = mutation_problems(out.stdout)
            print(f"== mut {m} rc={out.returncode} "
                  f"caught={out.stdout.count(chr(34) + 'caught' + chr(34) + ': true')}"
                  f"{' PROBLEMS: ' + '; '.join(bad) if bad else ''}")
            if gate_failed(out.returncode, bad):
                return 1
        # The commit runs last among mutating steps; a signal during it removes the whole
        # worktree (lock included). The source tree gets one atomic update-ref plus reset.
        r = land_cmd(["git", "add", "--", *[split_spec(f)[0] for f in args.files]], wt)
        if r.returncode != 0:
            print(f"worktree add failed: {r.stderr.strip()}", file=sys.stderr)
            return 1
        r = land_cmd(["git", "commit", "-m", args.message], wt)
        if r.returncode != 0:
            print(f"worktree commit failed: {(r.stdout + r.stderr).strip()[-2000:]}",
                  file=sys.stderr)
            return 1
        sha = land_cmd(["git", "rev-parse", "HEAD"], wt, check=True).stdout.strip()
        refuse_if_locked(args.src)
        paths = [split_spec(f)[0] for f in args.files]
        landed, problem = advance_main(args.src, sha, paths)
        if not landed:
            print(f"landing update failed: {problem}; worktree kept at {wt}", file=sys.stderr)
            _worktree = None
            return 1
    finally:
        _cleanup_worktree()
    print(f"LANDED {args.message[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
