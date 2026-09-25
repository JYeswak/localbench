#!/usr/bin/env python3
"""Tick driver: keeps an agent pane working the AGENTS.md tick loop unattended, and never interrupts a measurement.

Anti-ceremony (A12):
- Consumer: the pane that owns the tick loop (runs/pair-plan-20260923.md), and whoever restarts it after a halt.
- Gate: AGENTS.md tick law step 5 ("three ticks in a row that move nothing: stop and report") and the measurement law
  (no message may start an agent turn, a redraw WindowServer composites, while a localbench run is alive).
- Defect class: an idle agent nobody wakes (the loop silently stops at the end of a turn); a wake that lands mid-run
  (2026-09-23: both dense A/A re-banks voided while agent panes were mid-turn); a driver that pesters a halted pane.
- Delete when: the agent harness can wake an idle session on a timer and applies the same run, idle and stale gates.

Replaces runs/loop-nudger.sh (untracked, untested). Every decision is `decide()`/`sent()`, pure functions tested in
tests/test_tick_driver.py; `main()` only reads facts, sends, and logs one JSON line per action.

    scripts/tick_driver.py --session localbench --pane %21 --first runs/handoff.md --tick runs/tick-nudge.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_PATTERN = "localbench (aa|run|ab|record)"
# omp's footer while a turn runs: a braille spinner then the elapsed time ("⠼ 5m >"), or "⎋ Working…".
SPINNER = re.compile(r"^ *[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏] |Working")


@dataclass(frozen=True)
class State:
    sent_first: bool = False
    last_send: float = 0.0
    head_at_send: str = ""
    stale: int = 0          # consecutive sends that found HEAD where the previous send left it
    idle_checks: int = 0    # consecutive checks with no run alive and the pane not mid-turn


def pane_working(capture: str) -> bool:
    """The pane is mid-turn when its footer (last three non-blank lines) shows omp's spinner or "Working"."""
    footer = [ln for ln in capture.splitlines() if ln.strip()][-3:]
    return any(SPINNER.search(ln) for ln in footer)


def decide(s: State, *, now: float, run_alive: bool, working: bool, stop: bool, head: str, idle_needed: int = 2,
           cooldown_s: float = 900, stale_limit: int = 3) -> tuple[str, State]:
    """One check -> (action, state). Actions: stop (stop file present), wait, send-first (the handoff, once),
    send-tick, exit-stale (`stale_limit` ticks in a row moved HEAD nowhere). A live run or a mid-turn pane resets the
    idle count, so a message never lands during a measurement or on top of a turn."""
    if stop:
        return "stop", s
    if run_alive or working:
        return "wait", replace(s, idle_checks=0)
    s = replace(s, idle_checks=s.idle_checks + 1)
    if s.idle_checks < idle_needed:
        return "wait", s
    if not s.sent_first:
        return "send-first", s
    if now - s.last_send < cooldown_s:
        return "wait", s
    if head == s.head_at_send and s.stale + 1 >= stale_limit:
        return "exit-stale", s
    return "send-tick", s


def sent(s: State, *, now: float, head: str) -> State:
    """State after a delivered send. A tick that finds HEAD where the last send left it moved nothing."""
    stale = (s.stale + 1 if head == s.head_at_send else 0) if s.sent_first else 0
    return replace(s, sent_first=True, last_send=now, head_at_send=head, stale=stale, idle_checks=0)


def _out(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False).stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--pane", required=True, help="tmux pane id, e.g. %%21")
    ap.add_argument("--first", type=Path, required=True, help="message sent once, at the first idle point")
    ap.add_argument("--tick", type=Path, required=True, help="message sent at every later idle point")
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--cooldown", type=float, default=900)
    ap.add_argument("--stop", type=Path, default=ROOT / "runs" / "LOOP-STOP")
    ap.add_argument("--log", type=Path, default=ROOT / "runs" / "tick-driver.jsonl")
    a = ap.parse_args(argv)
    state = State()

    def log(**row) -> None:
        with a.log.open("a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 1), **row}) + "\n")

    log(action="start", pane=a.pane, first=str(a.first), tick=str(a.tick))
    print("tick driver started", flush=True)
    while True:
        head = _out("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD").strip()
        run_alive = subprocess.run(["pgrep", "-f", RUN_PATTERN], capture_output=True, check=False).returncode == 0
        working = pane_working(_out("tmux", "capture-pane", "-p", "-t", a.pane))
        action, state = decide(state, now=time.time(), run_alive=run_alive, working=working,
                               stop=a.stop.exists(), head=head, cooldown_s=a.cooldown)
        if action in ("stop", "exit-stale"):
            log(action=action, head=head, stale=state.stale)
            return 0
        if action.startswith("send"):
            msg = a.first if action == "send-first" else a.tick
            op = msg.read_text().split()[0]
            rc = subprocess.run(["ntm", "send", a.session, f"--pane={a.pane}", f"--file={msg}", "--no-cass-check"],
                                capture_output=True, text=True, timeout=60, check=False).returncode
            time.sleep(10)
            received = op in _out("tmux", "capture-pane", "-p", "-J", "-t", a.pane, "-S", "-600")
            if rc == 0 and received:
                state = sent(state, now=time.time(), head=head)
            log(action=action, file=str(msg), op=op, rc=rc, received=received, head=head, stale=state.stale)
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
