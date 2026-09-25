"""localbench's regression suite: pure logic only (no model, no GPU, no network). Run from the repo root:

    uv run python -m unittest discover -s tests -t .

Measurements stay in `localbench run|aa|ab`; these tests defend the code that judges them, and are the conformance
oracle for a future port (docs/port/)."""

import tempfile
from pathlib import Path

from localbench import smol

# No test may reach the live smol server (localbench smol): its state file names a real pid and port. On 2026-09-25
# the first suite run after `localbench smol set` sent the live server SIGTERM (a park test read the real state and
# parked it) and failed two preflight tests (they saw it up); the pre-commit hook runs this suite on every commit.
smol.STATE_DIR = Path(tempfile.mkdtemp(prefix="localbench-test-smol-"))
# Same for the LaunchAgent: a test that removed or rewrote it would unload the live server.
smol.LAUNCH_AGENTS = Path(tempfile.mkdtemp(prefix="localbench-test-launchagents-"))

from localbench import audit

# The mutation ledger (localbench audit / why): a test that parks, keeps or reverts through the CLI appends a row; it
# must land in a scratch file, never in the user's ~/.localbench/audit.jsonl.
audit.AUDIT_PATH = Path(tempfile.mkdtemp(prefix="localbench-test-audit-")) / "audit.jsonl"
