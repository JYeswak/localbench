#!/usr/bin/env python3
"""Run the regression suite in the current directory under the heavy slot, as a CPU-only holder (background priority;
one heavy job at a time).

Anti-ceremony (A12):
- Consumer: scripts/suite_verdict.sh (the commit hook's suite run), which has no Python entry of its own.
- Gate: the AGENTS.md Pacing rule (one heavy job at a time), enforced by localbench/heavyslot.py.
- Defect class: two full suites running at once outside the slot (2026-10-03: a hook's suite_verdict and an
  orphaned one ran side by side while load climbed to 47).
- Delete when: suite_verdict.sh is replaced by a Python entry that acquires the slot itself.

    scripts/heavy_run.py <wait-seconds>     exit = the suite's, or 75 when the slot is refused
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localbench import lifecycle
from localbench.heavyslot import SlotRefused, acquire  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: heavy_run.py <wait-seconds>", file=sys.stderr)
        return 2
    rc = 2
    try:
        with acquire("suite", wait_s=float(argv[0]), needs_gpu=False):
            child = lifecycle.spawn(
                ["uv", "run", "--quiet", "python", "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                start_new_session=True,
            )
            try:
                rc = child.wait()
            finally:
                lifecycle.unregister_child(child)
    except SlotRefused as refused:
        print(f"heavy_run: {refused}", file=sys.stderr)
        return 75
    return rc


if __name__ == "__main__":
    from localbench.lifecycle import install_cancel_handlers
    install_cancel_handlers()
    raise SystemExit(main(sys.argv[1:]))
