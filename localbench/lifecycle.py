"""Process-wide cancellation and child-process cleanup for localbench entry points."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time

_children: dict[subprocess.Popen, bool] = {}
_lock = threading.Lock()
_installed = False
_cleaned = False
_TERM_GRACE_SECONDS = 0.5


def register_child(process: subprocess.Popen, process_group: bool = False) -> subprocess.Popen:
    """Register a child for cancellation; ``process_group`` means its PID is a session/process-group leader."""
    with _lock:
        cleaned = _cleaned
        if not cleaned:
            _children[process] = process_group
    if cleaned:
        _stop_process(process, process_group)
    return process


def unregister_child(process: subprocess.Popen) -> None:
    with _lock:
        _children.pop(process, None)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _alive(process: subprocess.Popen, process_group: bool) -> bool:
    if process.poll() is None:
        return True
    return process_group and _group_exists(process.pid)


def _send(process: subprocess.Popen, process_group: bool, signum: int) -> None:
    try:
        if process_group:
            os.killpg(process.pid, signum)
        else:
            process.send_signal(signum)
    except OSError:
        pass


def _stop_process(process: subprocess.Popen, process_group: bool) -> None:
    if not _alive(process, process_group):
        return
    _send(process, process_group, signal.SIGTERM)
    deadline = time.monotonic() + _TERM_GRACE_SECONDS
    while _alive(process, process_group) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _alive(process, process_group):
        _send(process, process_group, signal.SIGKILL)
    try:
        process.wait(timeout=0.25)
    except (OSError, subprocess.TimeoutExpired):
        pass


def cleanup_children() -> None:
    """Terminate registered children and their groups, escalating after a bounded grace period."""
    global _cleaned
    with _lock:
        if _cleaned:
            return
        _cleaned = True
        children = tuple(_children.items())

    for process, process_group in children:
        if _alive(process, process_group):
            _send(process, process_group, signal.SIGTERM)

    deadline = time.monotonic() + _TERM_GRACE_SECONDS
    while time.monotonic() < deadline and any(_alive(process, group) for process, group in children):
        time.sleep(0.01)

    for process, process_group in children:
        if _alive(process, process_group):
            _send(process, process_group, signal.SIGKILL)
        try:
            process.wait(timeout=0.25)
        except (OSError, subprocess.TimeoutExpired):
            pass
        unregister_child(process)


def install_cancel_handlers() -> None:
    """Convert termination signals to KeyboardInterrupt so context managers and ``finally`` blocks unwind."""
    global _installed
    if _installed:
        return

    def cancel(_signum, _frame):
        cleanup_children()
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, cancel)
    _installed = True


def spawn(argv, /, **kwargs) -> subprocess.Popen:
    """Start and register a child; ``start_new_session=True`` enables process-group cleanup."""
    process_group = bool(kwargs.get("start_new_session", False))
    child = subprocess.Popen(argv, **kwargs)
    return register_child(child, process_group)
