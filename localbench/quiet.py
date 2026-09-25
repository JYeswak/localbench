"""Pause and resume the GPU users a measurement may silence: omp's own managed browser, nothing else.

Anti-ceremony (A12):
- Consumer: `localbench quiet` before a run whose preflight waits on a busy GPU, `--resume` after it; `localbench status`.
- Gate: the measurement law (one model on the machine; a run is CONTENDED when another process uses >25% GPU).
- Defect class: a run that waits out --wait-idle and refuses because omp's managed Chromium keeps rendering
  (2026-09-24: pid 37962 under ~/.omp/puppeteer at 17-46% GPU held the device near 25% for the whole preflight window).
- Delete when: omp freezes every idle managed tab on its own, or localbench stops needing a quiet GPU.

Pausing is SIGSTOP, resuming SIGCONT: reversible, nothing is killed. Only processes whose command runs from
~/.omp/puppeteer are candidates (the Chromium omp installs and freezes itself when a tab is idle); the user's own
applications, Terminal and WindowServer never are. Each pause is recorded with its command line, so a resume
never signals a pid that has since been reused by another program.
"""

from __future__ import annotations

import json
import os
import re
import signal

from . import sysstats
from .workloads import ROOT

STATE = ROOT / "runs" / "quiet.json"
PAUSABLE = re.compile(r"/\.omp/puppeteer/")


def candidates(procs: dict[int, str]) -> dict[int, str]:
    """pid -> command of every process that is omp's managed browser (pure; `procs` is pid -> command)."""
    return {pid: cmd for pid, cmd in procs.items() if PAUSABLE.search(cmd)}


def processes() -> dict[int, str]:
    out = {}
    for line in sysstats._run("ps", "-Ao", "pid=,command=").splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            out[int(pid)] = cmd.strip()
    return out


def paused() -> list[dict]:
    return json.loads(STATE.read_text()) if STATE.exists() else []


def _save(rows: list[dict]) -> None:
    if rows:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(rows, indent=2) + "\n")
    elif STATE.exists():
        STATE.unlink()


def plan_pause(procs: dict[int, str] | None = None) -> list[dict]:
    """The candidates pause() would stop now, changing nothing: every one not already paused, by pid."""
    procs = processes() if procs is None else procs
    have = {r["pid"] for r in paused()}
    return [{"pid": pid, "cmd": cmd} for pid, cmd in sorted(candidates(procs).items()) if pid not in have]


def pause(procs: dict[int, str] | None = None, kill=os.kill, plan: list[dict] | None = None) -> list[dict]:
    """SIGSTOP every row of `plan` (default plan_pause(procs)); record it. Returns the rows paused now."""
    plan = plan_pause(procs) if plan is None else plan
    now = []
    for r in plan:
        try:
            kill(r["pid"], signal.SIGSTOP)
        except ProcessLookupError:
            continue
        now.append(r)
    _save(paused() + now)
    return now


def plan_resume(procs: dict[int, str] | None = None) -> list[dict]:
    """The recorded pauses resume() would continue, changing nothing: those whose pid still runs the recorded
    command (a reused pid is left alone)."""
    procs = processes() if procs is None else procs
    return [r for r in paused() if procs.get(r["pid"]) == r["cmd"]]


def resume(procs: dict[int, str] | None = None, kill=os.kill, plan: list[dict] | None = None) -> list[dict]:
    """SIGCONT every row of `plan` (default plan_resume(procs)); forget every recorded pause. Returns resumed rows."""
    plan = plan_resume(procs) if plan is None else plan
    done = []
    for r in plan:
        try:
            kill(r["pid"], signal.SIGCONT)
        except ProcessLookupError:
            continue
        done.append(r)
    _save([])
    return done
