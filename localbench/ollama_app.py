"""Ollama.app, the supervisor of this Mac's `ollama serve`: what it runs, and a restart that leaves it aligned.

Why it exists (2026-09-27): the 0.34.4 install replaced /Applications/Ollama.app without relaunching the app, so the
supervisor kept running the moved-aside 0.34.2 image from /private/tmp (macOS cleans that directory) while the
server it spawns was 0.34.4. The app refuses AppleScript quit (-128) and ignores SIGTERM, and killing only `ollama
serve` makes the app respawn it. A restart therefore: asks it to quit, SIGKILLs the app if it is still there, stops
its server, reopens the bundle, and waits until the server lists models and the app runs from APP.

The server the harness pins (backend_sha) is APP/Contents/Resources/ollama either way; this module changes no pin.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import sysstats

APP = "/Applications/Ollama.app"
APP_EXE = "/Applications/Ollama.app/Contents/MacOS/Ollama"   # APP's executable, the image an aligned app runs
SERVE_MATCH = "Ollama.app/Contents/Resources/ollama serve"
APP_MATCH = "Ollama.app/Contents/MacOS/Ollama"


@dataclass
class State:
    app_pid: int | None
    app_image: str | None   # the executable the kernel mapped (lsof txt), which can lag a replaced bundle
    serve_pid: int | None

    @property
    def aligned(self) -> bool:
        return self.app_pid is not None and self.app_image == APP_EXE and self.serve_pid is not None


def _pids(pattern: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False).stdout.split()
    return [int(p) for p in out]


def _image(pid: int) -> str | None:
    """The first text-segment file lsof lists for pid: the image the process actually executes. lsof -F prints one
    field per line, a file's `f` field first (`ftxt`), then its other fields (`tREG`, `n<path>`)."""
    out = subprocess.run(["lsof", "-p", str(pid), "-Fftn"], capture_output=True, text=True, check=False).stdout.splitlines()
    in_txt = False
    for line in out:
        if line.startswith("f"):
            in_txt = line == "ftxt"
        elif in_txt and line.startswith("n"):
            return line[1:]
    return None


def state() -> State:
    apps, serves = _pids(APP_MATCH), _pids(SERVE_MATCH)
    app = apps[0] if apps else None
    return State(app_pid=app, app_image=_image(app) if app else None, serve_pid=serves[0] if serves else None)


def plan_restart(st: State) -> list[tuple[str, str]]:
    """(action, why) pairs, the same list the dry run prints and the restart executes."""
    steps = []
    if st.app_pid:
        steps.append((f"quit Ollama.app (pid {st.app_pid}); SIGKILL it if it refuses",
                      "the app refuses AppleScript quit and ignores SIGTERM; a stale image only changes on relaunch"))
    if st.serve_pid:
        steps.append((f"stop ollama serve (pid {st.serve_pid})", "the relaunched app spawns its own server"))
    steps.append((f"open {APP} and wait until the server answers and the app runs from {APP_EXE}",
                  "every omp session's local calls fail until the server is back"))
    return steps


def restart(up, kill=os.kill, sleep=time.sleep, quit_wait: float = 10, ready_timeout: float = 120
            ) -> State:
    """Execute plan_restart. `up()` says whether the server answers (the caller's /api/tags probe). Raises
    RuntimeError, naming what is wrong, when the app or its server does not come back aligned."""
    st = state()
    if st.app_pid:
        subprocess.run(["osascript", "-e", 'tell application "Ollama" to quit'], capture_output=True, text=True, check=False)
        for _ in range(int(quit_wait)):
            if not _pids(APP_MATCH):
                break
            sleep(1)
        for pid in _pids(APP_MATCH):
            kill(pid, signal.SIGKILL)
    for pid in _pids(SERVE_MATCH):
        kill(pid, signal.SIGTERM)
    for _ in range(int(quit_wait)):
        if not _pids(SERVE_MATCH):
            break
        sleep(1)
    for pid in _pids(SERVE_MATCH):
        kill(pid, signal.SIGKILL)
    subprocess.run(["open", "-a", APP], capture_output=True, text=True, check=False)
    for _ in range(int(ready_timeout / 2) + 1):   # counted polls, so an injected sleep bounds the wait in tests
        st = state()
        if st.aligned and up():
            return st
        sleep(2)
    raise RuntimeError(f"Ollama.app did not come back aligned within {ready_timeout:.0f} s: app pid {st.app_pid} "
                       f"image {st.app_image}, serve pid {st.serve_pid}; open {APP} by hand")


# Auto-update (owner decision 2026-09-30: manual Ollama upgrades; each release is A/B'd before adoption). The app's
# updater re-reads settings.auto_update_enabled before every download, but a bundle already staged under UPDATES
# installs at the next app start regardless of the setting (ollama app updater_darwin.go:395-412).
DB_ENV = "LOCALBENCH_OLLAMA_APP_DB"   # points the verb at another db.sqlite (tests use a fixture)
UPDATES = Path.home() / "Library/Caches/ollama/updates"
ROLLBACK = Path.home() / ".localbench" / "rollback"


def app_db() -> Path:
    return Path(os.environ.get(DB_ENV) or sysstats.OLLAMA_APP_DB).expanduser()


def staged_update(updates: Path | None = None) -> list[str]:
    """Entries under the updater's staging dir, sorted; empty when none (or no dir)."""
    d = updates or UPDATES
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def set_auto_update(enabled: bool, db: Path | None = None, rollback: Path | None = None) -> Path:
    """Back db up to <rollback>/ollama-db-<UTC>.sqlite, set settings.auto_update_enabled, read it back. Returns the
    backup. Raises RuntimeError when the settings row is missing or the read-back disagrees."""
    db, rollback = db or app_db(), rollback or ROLLBACK
    if sysstats.ollama_auto_update(db) is None:
        raise RuntimeError(f"{db}: no settings row with auto_update_enabled (no Ollama.app, or an app older than "
                           "that setting)")
    rollback.mkdir(parents=True, exist_ok=True)
    backup = rollback / f"ollama-db-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.sqlite"
    con = sqlite3.connect(db, timeout=10)   # busy timeout: the running app holds the db open
    try:
        dest = sqlite3.connect(backup)
        try:
            con.backup(dest)   # a consistent copy, WAL included, taken before any write
        finally:
            dest.close()
        with con:
            n = con.execute("UPDATE settings SET auto_update_enabled = ? WHERE id = 1", (int(enabled),)).rowcount
    finally:
        con.close()
    now = sysstats.ollama_auto_update(db)
    if n != 1 or now is not enabled:
        raise RuntimeError(f"{db}: wrote auto_update_enabled={int(enabled)} but read back {now}; backup at {backup}")
    return backup
