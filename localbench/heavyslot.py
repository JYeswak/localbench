"""Heavy-job slot: one GPU/model-heavy job at a time (AGENTS.md Pacing rule).

The slot is a machine-wide flock on ~/.localbench/heavy.lock plus an owner file
(heavy.lock.owner: pid, verb, started_at, repo). The flock is the truth: it is held
only by a live process, so a dead holder needs no pid-liveness heuristics (its
owner file goes stale and is overwritten by the next take). Re-entrant in-process
via a refcount, so prove --due's per-spec runs share one slot.

Admission uses CPU busy and memory pressure on every fresh take, not on re-entry: refuse at
CPU_BUSY_LIMIT_PCT or above, or unless memory pressure is `normal`; unreadable signals fail closed.
GPU work also refuses at GPU_LIMIT_PCT busy (an unreadable GPU refuses too). `--force-load` skips
admission and is recorded in the caller's audit row.

Observability (read by `localbench slot`): a --wait-slot waiter keeps a ticket at heavy-queue/<pid>.json while it
polls. GPU tickets take precedence over CPU-only tickets; each class is FIFO. Tickets are removed on take or exit,
readers prune dead pids, and every last release appends one line to heavy-history.jsonl with the verb, class, times
and wait duration; per-verb medians give ETAs.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import statistics
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

CPU_BUSY_LIMIT_PCT = 80.0
GPU_LIMIT_PCT = 80
LOCK_NAME = "heavy.lock"
OWNER_NAME = "heavy.lock.owner"
QUEUE_DIR = "heavy-queue"
HISTORY_NAME = "heavy-history.jsonl"
HISTORY_TAIL = 2000   # history lines read back (the file is append-only; the read stays bounded)
ETA_SAMPLES = 20      # holds of one verb whose median wall_s is the ETA
POLL_S = 2.0
# Seams the suite redirects (tests/__init__.py): HOME moves the lock off ~/.localbench, and the sensor
# functions default quiet, so command tests never take the live slot or refuse on a busy machine.
HOME: Path | None = None
LOAD_FN = os.getloadavg


class SlotRefused(RuntimeError):
    """The slot is held (naming the holder) or the machine is too busy; --wait-slot queues instead."""


def lock_dir(home: Path | str | None = None) -> Path:
    """The directory holding the lock and owner files (tests point it at scratch)."""
    if home is None:
        home = HOME or Path.home()
    return Path(home) / ".localbench"


def lock_path(home: Path | None = None) -> Path:
    """The flocked slot file."""
    return lock_dir(home) / LOCK_NAME


def owner_path(home: Path | None = None) -> Path:
    """The owner file naming the holder (informational; the flock is the truth)."""
    return lock_dir(home) / OWNER_NAME


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _default_gpu() -> dict:
    from . import sysstats

    return sysstats.gpu_window(5.0)


GPU_FN = _default_gpu


def _default_cpu_busy():
    from . import sysstats

    return sysstats.cpu_busy_pct()


def _default_memory():
    from . import sysstats

    return sysstats.memory()


CPU_FN = _default_cpu_busy
MEMORY_FN = _default_memory


def readings(load_fn=None, cpu_fn=None, memory_fn=None, gpu_fn=None, *, check_gpu: bool = True) -> dict:
    """Admission evidence: load, CPU busy, memory pressure and, when checked, GPU window attribution."""
    try:
        load1 = float((load_fn or LOAD_FN)()[0])
        if not math.isfinite(load1) or load1 < 0:
            load1 = None
    except (OSError, ValueError, TypeError, IndexError, OverflowError):
        load1 = None
    try:
        cpu_busy = (cpu_fn or CPU_FN)()
        if (isinstance(cpu_busy, bool) or not isinstance(cpu_busy, (int, float))
                or not math.isfinite(cpu_busy) or not 0 <= cpu_busy <= 100):
            cpu_busy = None
        else:
            cpu_busy = float(cpu_busy)
    except Exception:  # noqa: BLE001 - unreadable CPU sensor refuses downstream
        cpu_busy = None
    try:
        memory = (memory_fn or MEMORY_FN)()
    except Exception:  # noqa: BLE001 - unreadable memory sensor refuses downstream
        memory = None
    mem_pressure = memory.get("pressure_level") if isinstance(memory, dict) else None
    if not isinstance(mem_pressure, str) or mem_pressure not in ("normal", "warn", "critical"):
        mem_pressure = None
    gpu = None
    if check_gpu:
        try:
            gpu = (gpu_fn or GPU_FN)()
        except Exception:  # noqa: BLE001 - any sensor failure reads as unreadable, which refuses
            gpu = None
    pct = gpu.get("device_pct") if isinstance(gpu, dict) else None
    if (isinstance(pct, bool) or not isinstance(pct, (int, float))
            or not math.isfinite(pct) or not 0 <= pct <= 100):
        pct = None
    process_pct = gpu.get("process_pct") if isinstance(gpu, dict) else None
    if (isinstance(process_pct, bool) or not isinstance(process_pct, (int, float))
            or not math.isfinite(process_pct) or process_pct < 0):
        process_pct = None
    return {"load1": load1, "cpu_busy": cpu_busy, "mem_pressure": mem_pressure,
            "gpu_pct": pct, "gpu_status": gpu.get("status") if check_gpu and isinstance(gpu, dict) else
            ("SKIPPED" if not check_gpu else None), "gpu_process_pct": process_pct,
            "gpu_coverage": gpu.get("coverage") if isinstance(gpu, dict) else None,
            "gpu_unattributed_pct": gpu.get("unattributed_pct") if isinstance(gpu, dict) else None,
            "gpu_source": gpu.get("source") if isinstance(gpu, dict) else None,
            "forced": False}


def background_priority() -> bool:
    """Lower this process to nice 10, so a CPU-only gate yields CPU to interactive work without stopping.
    Not macOS background QoS (`taskpolicy -b`): under load 100 that throttled gate processes to zero CPU time for
    40+ minutes, and no landing completed in 4 h (2026-10-03). Children inherit the nice value."""
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    except OSError:
        return False
    return True


def admission(load_fn=None, cpu_fn=None, memory_fn=None, gpu_fn=None,
              check_gpu: bool = True) -> tuple[dict, list[str]]:
    """(readings, problems). CPU busy must stay below the limit and memory pressure must be normal.
    GPU admission also requires a trustworthy device/process comparison; CPU-only holders skip that sensor."""
    readings_ = readings(load_fn, cpu_fn, memory_fn, gpu_fn, check_gpu=check_gpu)
    readings_["gpu_checked"] = bool(check_gpu)
    problems = []
    cpu_busy = readings_["cpu_busy"]
    if cpu_busy is None:
        problems.append("CPU busy percentage unreadable")
    elif cpu_busy >= CPU_BUSY_LIMIT_PCT:
        problems.append(f"CPU busy {cpu_busy:.1f}% >= {CPU_BUSY_LIMIT_PCT:g}%")
    pressure = readings_["mem_pressure"]
    if pressure is None:
        problems.append("memory pressure unreadable")
    elif pressure != "normal":
        problems.append(f"memory pressure {pressure} (normal required)")
    if check_gpu:
        pct = readings_["gpu_pct"]
        process_pct = readings_["gpu_process_pct"]
        status = readings_["gpu_status"]
        if pct is None or process_pct is None or status not in ("IDLE", "ATTRIBUTED", "UNATTRIBUTED", "UNALIGNED"):
            problems.append("GPU utilization unreadable")
        elif status == "IDLE":
            from . import sysstats
            idle_threshold_pct = sysstats.GPU_IDLE_THRESHOLD_PCT
            if pct >= idle_threshold_pct or process_pct >= idle_threshold_pct:
                problems.append(f"GPU idle accounting inconsistent (both device and process must be <{idle_threshold_pct:g}%)")
        elif status == "UNATTRIBUTED":
            coverage = readings_["gpu_coverage"]
            unattributed = readings_["gpu_unattributed_pct"]
            coverage_text = f"{coverage:.1f}%" if isinstance(coverage, (int, float)) else "unavailable"
            unattributed_text = f"{unattributed:.1f}%" if isinstance(unattributed, (int, float)) else "unavailable"
            problems.append(f"GPU attribution UNATTRIBUTED (coverage {coverage_text}; "
                            f"{unattributed_text} unattributed)")
        elif status == "UNALIGNED":
            problems.append(f"GPU attribution UNALIGNED (process {process_pct:.1f}% vs device {pct:.1f}%; coverage capped at 100%)")
        else:
            coverage = readings_["gpu_coverage"]
            if (isinstance(coverage, bool) or not isinstance(coverage, (int, float))
                    or not math.isfinite(coverage) or coverage < 90):
                problems.append(f"GPU attribution coverage {coverage!r} < 90% required")
            elif pct >= GPU_LIMIT_PCT:
                problems.append(f"GPU {pct}% >= {GPU_LIMIT_PCT}% busy")
    return readings_, problems


_HELD: dict[str, dict] = {}
_HELD_LOCK = threading.Lock()


class Slot:
    """A held heavy slot: .admission names the readings (and forced override) for the audit row."""

    def __init__(self, key: str, fd: int, owner: dict, admission: dict, home: Path | None = None):
        self._key, self._fd, self._owner = key, fd, owner
        self.admission = admission
        self._home = home
        self._acquired_mono = time.monotonic()
        self.waited_s = 0.0

    def release(self) -> None:
        """Release one hold; the last hold unlocks, closes and drops the owner file. Idempotent."""
        with _HELD_LOCK:
            rec = _HELD.get(self._key)
            if rec is None or rec["slot"] is not self:
                return
            rec["count"] -= 1
            if rec["count"] > 0:
                return
            del _HELD[self._key]
        try:
            Path(self._owner["path"]).unlink(missing_ok=True)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            _append_history(self._home, {
                "verb": self._owner.get("verb"), "repo": self._owner.get("repo"), "pid": self._owner.get("pid"),
                "needs_gpu": bool(self.admission.get("gpu_checked", True)),
                "acquired_at": self._owner.get("started_at"), "released_at": _utcnow(),
                "wall_s": round(time.monotonic() - self._acquired_mono, 3), "waited_s": round(self.waited_s, 3)})

    def __enter__(self) -> "Slot":
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False


def _take(home: Path | None, verb: str, repo: str | None) -> tuple[Slot | None, bool]:
    """Try one non-blocking take: (slot, fresh). A hold this process already owns is re-entered, never re-locked."""
    path = lock_path(home).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    with _HELD_LOCK:
        if key in _HELD:
            _HELD[key]["count"] += 1
            return _HELD[key]["slot"], False
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, False
    owner = {"pid": os.getpid(), "verb": verb, "started_at": _utcnow(),
             "repo": repo or str(Path.cwd()), "path": str(owner_path(home).expanduser())}
    owner_path(home).expanduser().write_text(json.dumps(owner, indent=1) + "\n")
    slot = Slot(key, fd, owner, {}, home)
    with _HELD_LOCK:
        _HELD[key] = {"fd": fd, "slot": slot, "count": 1}
    return slot, True


def holder(home: Path | None = None) -> dict | None:
    """The live holder's owner info, or None when the slot is free. A lock nobody holds is free even when a
    stale owner file remains; an unreadable owner file reports verb unknown rather than raising."""
    path = lock_path(home).expanduser()
    if not path.exists():
        return None
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                owner = json.loads(owner_path(home).expanduser().read_text())
            except (OSError, ValueError):
                owner = {}
            if not isinstance(owner, dict):
                owner = {}
            return {"pid": owner.get("pid"), "verb": owner.get("verb") or "unknown",
                    "started_at": owner.get("started_at"), "repo": owner.get("repo")}
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return None


def describe(holder_info: dict | None, problems: list[str]) -> str:
    """The refusal line: who holds the slot, and/or why the machine refuses."""
    bits = []
    if holder_info is not None:
        who = f"pid {holder_info['pid']}" if holder_info.get("pid") is not None else "an unknown process"
        bits.append(f"held by {who} ({holder_info.get('verb')}"
                    + (f" since {holder_info['started_at']}" if holder_info.get("started_at") else "")
                    + (f" in {holder_info['repo']}" if holder_info.get("repo") else "") + ")")
    bits.extend(problems)
    return "; ".join(bits)


def acquire(verb: str, *, wait_s: float = 0.0, force_load: bool = False, needs_gpu: bool = True,
            home: Path | None = None, load_fn=None, cpu_fn=None, memory_fn=None, gpu_fn=None,
            repo: str | None = None, poll_s: float = POLL_S) -> Slot:
    """Take the heavy slot for `verb`, or raise SlotRefused naming the holder and/or readings.
    Admission is checked on every fresh take and skipped on in-process re-entry. Waiting tickets are GPU-first,
    FIFO within each class; even a new zero-wait caller cannot bypass an existing ticket."""
    key = str(lock_path(home).expanduser())
    with _HELD_LOCK:
        owned = key in _HELD
    if owned:
        slot, _ = _take(home, verb, repo)
        assert slot is not None  # owned means _take re-enters, never None
        slot.admission = readings(load_fn, cpu_fn, memory_fn, gpu_fn, check_gpu=needs_gpu)
        slot.admission["forced"] = bool(force_load)
        slot.admission["gpu_checked"] = bool(needs_gpu)
        return slot
    start = time.monotonic()
    wait_s = max(0.0, float(wait_s))
    ticket = None
    queue = []
    try:
        while True:
            if force_load:
                readings_, problems = readings(load_fn, cpu_fn, memory_fn, gpu_fn, check_gpu=needs_gpu), []
                readings_["forced"] = True
                readings_["gpu_checked"] = bool(needs_gpu)
            else:
                readings_, problems = admission(
                    load_fn, cpu_fn, memory_fn, gpu_fn, check_gpu=needs_gpu)
            slot = None
            with _queue_guard(home):
                queue = _waiters_unlocked(home)
                is_turn = (not queue if ticket is None
                           else bool(queue and queue[0].get("pid") == os.getpid()))
                if is_turn and not problems:
                    slot, _ = _take(home, verb, repo)
                if ticket is None and time.monotonic() - start < wait_s:
                    orders = [t["queue_order"] for t in queue
                              if isinstance(t.get("queue_order"), int) and not isinstance(t.get("queue_order"), bool)]
                    ticket = _enqueue(home, verb, repo, needs_gpu, max(orders, default=0) + 1)
                if slot is not None and ticket is not None:
                    ticket.unlink(missing_ok=True)
                    ticket = None
            if slot is not None:
                if not needs_gpu:
                    readings_["background"] = background_priority()
                slot.admission = readings_
                slot.waited_s = time.monotonic() - start
                return slot
            elapsed = time.monotonic() - start
            if elapsed >= wait_s:
                msg = describe(holder(home), problems)
                if ticket is None and queue:
                    msg = "; ".join(part for part in (msg, f"{len(queue)} earlier queue ticket(s)") if part)
                elif ticket is not None:
                    msg += _queue_note(home)
                raise SlotRefused(msg)
            time.sleep(min(poll_s, max(0.0, wait_s - elapsed)))
    finally:
        if ticket is not None:
            with _queue_guard(home):
                ticket.unlink(missing_ok=True)


def queue_dir(home: Path | None = None) -> Path:
    """Where waiters' tickets live, one <pid>.json each."""
    return lock_dir(home).expanduser() / QUEUE_DIR


@contextmanager
def _queue_guard(home: Path | None):
    """Serialize queue snapshots, ticket creation/removal, and slot handoff."""
    directory = queue_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def history_path(home: Path | None = None) -> Path:
    """The append-only release log."""
    return lock_dir(home).expanduser() / HISTORY_NAME


def _enqueue(home: Path | None, verb: str, repo: str | None, needs_gpu: bool, queue_order: int) -> Path:
    """Write a ticket while the caller holds `_queue_guard`."""
    path = queue_dir(home) / f"{os.getpid()}.json"
    path.write_text(json.dumps({"pid": os.getpid(), "verb": verb, "repo": repo or str(Path.cwd()),
                                "needs_gpu": bool(needs_gpu), "enqueued_at": time.time(),
                                "queue_order": queue_order}) + "\n")
    return path


def _append_history(home: Path | None, row: dict) -> None:
    path = history_path(home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        pass   # observability only: a failed history write never fails the release


def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _waiters_unlocked(home: Path | None) -> list[dict]:
    """Queue snapshot for callers that already hold `_queue_guard`."""
    out = []
    d = queue_dir(home)
    if not d.is_dir():
        return out
    for path in d.glob("*.json"):
        try:
            t = json.loads(path.read_text())
        except (OSError, ValueError):
            t = None
        if not isinstance(t, dict) or not _alive(t.get("pid")):
            path.unlink(missing_ok=True)
            continue
        out.append(t)

    def order(t):
        queue_order = t.get("queue_order")
        if not isinstance(queue_order, int) or isinstance(queue_order, bool):
            queue_order = 0
        return (0 if t.get("needs_gpu", True) else 1, queue_order,
                float(t.get("enqueued_at") or 0), t.get("pid"))

    return sorted(out, key=order)


def waiters(home: Path | None = None) -> list[dict]:
    """Live tickets, GPU-first and FIFO within each class; dead or unreadable tickets are pruned."""
    d = queue_dir(home)
    if not d.is_dir():
        return []
    with _queue_guard(home):
        return _waiters_unlocked(home)


def history(home: Path | None = None, tail: int = HISTORY_TAIL) -> list[dict]:
    """The last `tail` release rows (oldest first); malformed lines are skipped."""
    path = history_path(home)
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block, data = 65536, b""
            while size > 0 and data.count(b"\n") <= tail:
                step = min(block, size)
                size -= step
                fh.seek(size)
                data = fh.read(step) + data
    except OSError:
        return []
    rows = []
    for line in data.decode(errors="replace").splitlines()[-tail:]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def median_wall(verb: str | None, rows: list[dict]) -> float | None:
    """Median wall_s of the last ETA_SAMPLES holds of `verb`; None without history."""
    walls = [float(r["wall_s"]) for r in rows if r.get("verb") == verb and isinstance(r.get("wall_s"), int | float)]
    walls = walls[-ETA_SAMPLES:]
    return statistics.median(walls) if walls else None


def _iso_age(stamp: str | None, now: float) -> float | None:
    try:
        return max(0.0, now - datetime.fromisoformat(stamp).timestamp()) if stamp else None
    except ValueError:
        return None


def report(home: Path | None = None, now: float | None = None) -> dict:
    """What `localbench slot` shows: holder and the GPU-first queue with FIFO positions and per-verb ETAs."""
    now = time.time() if now is None else now
    rows = history(home)
    h = holder(home)
    eta = 0.0
    if h is not None:
        held_s = _iso_age(h.get("started_at"), now)
        med = median_wall(h.get("verb"), rows)
        remaining = None if med is None or held_s is None else max(0.0, med - held_s)
        h = {**h, "held_s": held_s, "expected_remaining_s": remaining}
        eta = remaining
    queue = []
    for pos, t in enumerate(waiters(home), 1):
        queue.append({**t, "position": pos, "waited_s": max(0.0, now - float(t.get("enqueued_at") or now)),
                      "estimated_start_s": eta})
        med = median_wall(t.get("verb"), rows)
        eta = None if eta is None or med is None else eta + med
    return {"holder": h, "queue": queue}


def _queue_note(home: Path | None) -> str:
    """Suffix for a --wait-slot refusal: this process's queue position and ETA when known."""
    for t in report(home)["queue"]:
        if t.get("pid") == os.getpid():
            eta = t["estimated_start_s"]
            return (f"; queue position {t['position']}, estimated start in "
                    + (f"{eta:.0f}s" if eta is not None else "unknown (no history)"))
    return ""


def held(verb=None, *, wait_s: float = 0.0, force_load: bool = False, **kw):
    """Hold the heavy slot for one function call (Pacing rule: one heavy job at a time).
    Bare @held verbs from the function name; @held("verb", wait_s=...) names it and queues.
    A refused slot raises SlotRefused, never queues silently and never runs unheld."""
    import functools

    def decorate(fn):
        name = verb if isinstance(verb, str) else getattr(fn, "__name__", "unknown")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with acquire(name, wait_s=wait_s, force_load=force_load, **kw):
                return fn(*args, **kwargs)

        return wrapper

    if callable(verb):
        fn, verb = verb, None
        return decorate(fn)
    return decorate
