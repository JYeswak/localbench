"""localbench's regression suite: pure logic only (no model, no GPU, no network). Run from the repo root:

    uv run python -m unittest discover -s tests -t .

Measurements stay in `localbench run|aa|ab`; these tests defend the code that judges them, and are the conformance
oracle for a future port (docs/port/)."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

# A git hook (the pre-commit hook runs this suite) exports GIT_DIR, GIT_INDEX_FILE and friends. Any test that runs
# git in a fixture repo then writes to THIS repository instead: on 2026-10-02 a fixture's `git init` and
# `git config user.*` inside the hook set core.bare=true and a fake user in localbench's own .git/config, which broke
# git status and commits for every pane until it was repaired by hand. Nothing in the suite may inherit them.
for _key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
             "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX", "GIT_COMMON_DIR"):
    os.environ.pop(_key, None)

# Every temp path the suite makes (tempfile.mkdtemp, TemporaryDirectory, NamedTemporaryFile, here and in any test
# module) lands under one scratch root that is removed when the interpreter exits. Before this, the three redirects
# below and module- or setUp-level mkdtemp calls were never removed; mutation proofs run as
# `env TMPDIR=<repo>/runs ... scripts/mutate.py` start the suite twice per case, so 2026-09-28..30 left 649
# localbench-test-*/tmp* dirs in runs/. The pid check keeps a forked child's exit from removing the parent's root.
SCRATCH = Path(tempfile.mkdtemp(prefix="localbench-test-"))
tempfile.tempdir = str(SCRATCH)
_OWNER = os.getpid()
atexit.register(lambda: os.getpid() == _OWNER and shutil.rmtree(SCRATCH, ignore_errors=True))

from localbench import smol

# No test may reach the live smol server (localbench smol): its state file names a real pid and port. On 2026-09-25
# the first suite run after `localbench smol set` sent the live server SIGTERM (a park test read the real state and
# parked it) and failed two preflight tests (they saw it up); the pre-commit hook runs this suite on every commit.
smol.STATE_DIR = Path(tempfile.mkdtemp(prefix="smol-"))
# Same for the LaunchAgent: a test that removed or rewrote it would unload the live server.
smol.LAUNCH_AGENTS = Path(tempfile.mkdtemp(prefix="launchagents-"))

from localbench import audit, heavyslot

# The mutation ledger (localbench audit / why): a test that parks, keeps or reverts through the CLI appends a row; it
# must land in a scratch file, never in the user's ~/.localbench/audit.jsonl.
audit.AUDIT_PATH = Path(tempfile.mkdtemp(prefix="audit-")) / "audit.jsonl"

# The heavy-job slot (AGENTS.md Pacing): decorated commands take it on every run. Tests use a scratch lock and
# calm sensors so they never touch ~/.localbench/heavy.lock or refuse on the live machine.
heavyslot.HOME = Path(tempfile.mkdtemp(prefix="heavy-slot-"))
heavyslot.LOAD_FN = lambda: (1.0, 1.0, 1.0)  # noqa: E731 - suite-wide diagnostic load stub
heavyslot.CPU_FN = lambda: 1.0  # noqa: E731 - suite-wide calm-CPU stub
heavyslot.MEMORY_FN = lambda: {"pressure_level": "normal"}  # noqa: E731 - suite-wide calm-memory stub
heavyslot.GPU_FN = lambda: {"device_pct": 1, "process_pct": 0.5, "coverage": None, "unattributed_pct": 0.0, "status": "IDLE"}  # noqa: E731 - suite-wide calm-GPU stub
