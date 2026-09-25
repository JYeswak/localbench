"""Machine state capture: a static fingerprint plus live contention signals.

Speed numbers are only comparable when the machine was in a comparable state,
so every run records: hardware identity, OS build, power/thermal state, memory
pressure, swap, GPU utilization, and whatever else was burning CPU.
Unprivileged probes always run; `powermetrics` and `purge` run only when
scripts/install-sudoers.sh granted them (sudo -n, never a password prompt).
"""

from __future__ import annotations

import json
import math
import os
import plistlib
import re
import sqlite3
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Self


def _run(*cmd: str, timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _sysctl(name: str) -> str:
    return _run("sysctl", "-n", name)


def gpu_utilization() -> dict:
    out = _run("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
    stats = {}
    for key, field in (("device_pct", "Device Utilization %"), ("renderer_pct", "Renderer Utilization %"),
                       ("tiler_pct", "Tiler Utilization %"), ("in_use_mem_bytes", "In use system memory")):
        m = re.search(rf'"{re.escape(field)}"=(\d+)', out)
        if m:
            stats[key] = int(m.group(1))
    return stats


_GPU_CREATOR = re.compile(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"')
_GPU_NS = re.compile(r'"accumulatedGPUTime"=(\d+)')


def gpu_time_by_pid() -> dict[int, dict]:
    """Cumulative GPU time (ns) per process, summed over its AGX driver clients' `AppUsage` records (what Activity
    Monitor's GPU column reads; no privileges). ioreg sorts keys, so each client block lists AppUsage BEFORE its
    IOUserClientCreator: blocks are parsed whole (pairing line by line credits the previous block's process).
    The counters only grow; the difference between two reads is which processes used the GPU in between."""
    procs: dict[int, dict] = {}
    for block in _run("ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXDeviceUserClient").split("+-o ")[1:]:
        m = _GPU_CREATOR.search(block)
        if m:
            proc = procs.setdefault(int(m.group(1)), {"name": m.group(2), "ns": 0})
            proc["ns"] += sum(map(int, _GPU_NS.findall(block)))
    return procs


def gpu_share(before: dict, after: dict, seconds: float, commands: dict[int, str] | None = None,
              min_pct: float = 0.5) -> list[dict]:
    """Per-process GPU busy % between two `gpu_time_by_pid()` reads, largest first (a process with several command
    queues can exceed 100%). Ollama runners carry the `model` they serve. `commands` caches pid -> command line."""
    commands = {} if commands is None else commands
    rows = [{"pid": pid, "name": a["name"],
             "pct": round(100 * (a["ns"] - before.get(pid, {}).get("ns", 0)) / (seconds * 1e9), 1)}
            for pid, a in after.items()]
    rows = [r for r in rows if r["pct"] >= min_pct]
    missing = [str(r["pid"]) for r in rows if r["pid"] not in commands]
    if missing:
        for line in _run("ps", "-o", "pid=,command=", "-p", ",".join(missing)).splitlines():
            pid, _, cmd = line.strip().partition(" ")
            commands[int(pid)] = cmd.strip()
    for r in rows:
        full = commands.get(r["pid"], "")
        r["cmd"] = full[:160]
        # From the full command: a GGUF blob path is ~130 characters, and the 160-character cut above left 58 of its 64
        # hex digits in every row observe.db recorded before this lookup, so they named no model.
        m = re.search(r"--model (\S+)", full)
        if m:
            r["model"] = m.group(1)
            names = ollama_blob_names(r["model"])
            if names:
                r["model"], r["model_names"] = ", ".join(names), names
    return sorted(rows, key=lambda r: -r["pct"])


OLLAMA_WEIGHTS = "application/vnd.ollama.image.model"


def ollama_blob_names(path: str) -> list[str]:
    """The names `ollama list` shows for a weights blob. Ollama.app runs a GGUF model as `llama-server --model
    <models>/blobs/sha256-<hex>`, so ps and ioreg show a hash where `ollama ps` shows a name; MLX models run under their
    name and need no lookup. Each `<models>/manifests/<registry>/<namespace>/<model>/<tag>` lists its layers, one of
    them the weights. Names follow ollama's display: `model:tag` for the default registry's library, `ns/model:tag` for
    another namespace there, the host kept for any other registry. A hex of 12+ digits matches as a prefix (rows cut
    short before this lookup existed). [] when the path is no ollama blob or no manifest lists it (deleted since)."""
    p = Path(path)
    hexpart = p.name.removeprefix("sha256-")
    if p.parent.name != "blobs" or hexpart == p.name or len(hexpart) < 12:
        return []
    root = p.parent.parent / "manifests"
    names = []
    for f in sorted(root.glob("*/*/*/*")):
        try:
            layers = json.loads(f.read_text()).get("layers") or []
        except (OSError, ValueError, AttributeError):
            continue
        if any(la.get("mediaType") == OLLAMA_WEIGHTS and str(la.get("digest", "")).startswith("sha256:" + hexpart)
               for la in layers):
            host, ns, model, tag = f.relative_to(root).parts
            base = model if ns == "library" else f"{ns}/{model}"
            names.append(f"{base}:{tag}" if host == "registry.ollama.ai" else f"{host}/{base}:{tag}")
    return names


def proc_label(r: dict) -> str:
    """`name pid N (model) P%` — one GPU row as refusals, status and contention text print it."""
    return f"{r['name']} pid {r['pid']}" + (f" ({r['model']})" if r.get("model") else "") + f" {r['pct']}%"


def gpu_is_ours(row: dict, backend: str, model: str) -> bool:
    """The backend's own GPU work: `ollama serve`, or an ollama runner serving `model` — `ollama runner` for MLX
    models, Ollama.app's `llama-server` for GGUF ones (named via `ollama_blob_names`); or the mlx-serve process."""
    if backend == "ollama":
        if row["name"] == "ollama" and row["cmd"].endswith(" serve"):
            return True
        return row["name"] in ("ollama", "llama-server") and model in (row.get("model_names") or [row.get("model")])
    return backend in row["cmd"]


# Local inference servers on this host: ollama, mlx-serve (localbench starts it), the localbench proxy, and Inco Splash
# (Homebrew `splash`, incoai/Qwen3.8-27B-Splash, started by hand; ~/.agents/skills/splash).
# 11235: the dedicated smol server (localbench smol, since 2026-09-25): omp's smol work, not the model under test.
SMOL_PORT = 11235
INFERENCE_PORTS = {11434: "ollama", 11234: "mlx-serve", SMOL_PORT: "mlx-smol", 11236: "omlx", 11299: "localbench-proxy",
                   8000: "splash"}


def omp_client_identity(cmd: str, env_cmd: str) -> dict:
    """Profile and agent dir of one omp process, from its argv and `ps eww` line.

    `--profile` / OMP_PROFILE / PI_PROFILE name a profile. PI_CODING_AGENT_DIR is the
    isolation pin: child_env() sets it to runs/omp-agent and strips the profile variables.
    A process that has only the agent dir is not the user's default profile.
    """
    out: dict = {}
    profile = (re.search(r"--profile[= ](\S+)", cmd)
               or re.search(r"\b(?:OMP_PROFILE|PI_PROFILE)=(\S+)", env_cmd))
    if profile:
        out["omp_profile"] = profile.group(1)
    agent = re.search(r"\bPI_CODING_AGENT_DIR=(\S+)", env_cmd)
    if agent:
        out["agent_dir"] = agent.group(1)
    elif re.search(r"\bomp\b", cmd) and "omp_profile" not in out:
        out["omp_profile"] = "default"
    return out


def inference_clients() -> list[dict]:
    """Processes holding a loopback TCP connection to a local inference server: who CAN send it GPU work right now
    (an idle keep-alive socket counts too; `gpu_share` says who DID use the GPU). Each carries its command line,
    working directory and, for omp, its profile (`--profile`, else OMP_PROFILE/PI_PROFILE) and PI_CODING_AGENT_DIR
    when the process env shows one. An omp with neither is recorded as profile default. An omp with only the agent
    dir is not."""
    clients: dict[int, dict] = {}
    pid = name = None
    for line in _run("lsof", "-nP", "-iTCP", "-sTCP:ESTABLISHED", "-Fpcn").splitlines():
        tag, val = line[:1], line[1:]
        if tag == "p":
            pid = int(val)
        elif tag == "c":
            name = val
        elif tag == "n" and "->" in val:
            local, remote = val.split("->", 1)
            port = int(remote.rsplit(":", 1)[1])
            if port in INFERENCE_PORTS and remote.startswith(("127.", "[::1]", "localhost")):
                c = clients.setdefault(pid, {"pid": pid, "name": name, "servers": {}, "conns": []})
                srv = INFERENCE_PORTS[port]
                c["servers"][srv] = c["servers"].get(srv, 0) + 1
                c["conns"].append([port, int(local.rsplit(":", 1)[1])])
    for c in clients.values():
        env_cmd = _run("ps", "eww", "-o", "command=", "-p", str(c["pid"]))
        c["cmd"] = _run("ps", "-o", "command=", "-p", str(c["pid"]))[:200]
        c["cwd"] = next((ln[1:] for ln in _run("lsof", "-a", "-p", str(c["pid"]), "-d", "cwd", "-Fn").splitlines()
                         if ln.startswith("n")), None)
        if re.search(r"\bomp\b", c["cmd"]):
            c.update(omp_client_identity(c["cmd"], env_cmd))
    return sorted(clients.values(), key=lambda c: c["pid"])


_OMP = re.compile(r"(?:^|\s)(?:\S*/)?omp(?:\s|$)")


def omp_processes() -> list[dict]:
    """Every running omp: pid, command, cwd. Unlike `inference_clients`, a session holding no connection at this
    instant is included: one that started while a smol model was parked keeps the model omp's fuzzy match gave it,
    whether or not it is mid-call when checked (2026-09-23: pid 2586 was idle for the one check that named 2779)."""
    procs = {}
    for line in _run("ps", "-Ao", "pid=,command=").splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit() and int(pid) != os.getpid() and _OMP.search(cmd):
            procs[int(pid)] = {"pid": int(pid), "cmd": cmd.strip()[:200], "cwd": None}
    if procs:
        pid = None
        for ln in _run("lsof", "-a", "-d", "cwd", "-p", ",".join(map(str, procs)), "-Fpn").splitlines():
            if ln.startswith("p"):
                pid = int(ln[1:])
            elif ln.startswith("n") and pid in procs:
                procs[pid]["cwd"] = ln[1:]
    return sorted(procs.values(), key=lambda p: p["pid"])


_LOOPBACK = ("127.", "::1", "[::1]")
_NETTOP_CONN = re.compile(r"^tcp[46] ([^\s<]+)[:.](\d+)<->([^\s,]+)[:.](\d+),(\d*),(\d*),")


def connection_bytes() -> dict[tuple[int, int], tuple[int, int]]:
    """Bytes carried so far by each open loopback connection to a local inference server, as the server counts them:
    (server port, client port) -> (request bytes received, response bytes sent). One `nettop` sample (~0.15 s, no
    privileges). Counters are per connection and cumulative; a connection that opens and closes between two samples
    is never seen (omp keeps its connections alive; a one-shot HTTP client may not be)."""
    out = {}
    for line in _run("nettop", "-m", "tcp", "-L", "1", "-n", "-J", "bytes_in,bytes_out", timeout=15).splitlines():
        m = _NETTOP_CONN.match(line)
        if m and int(m.group(2)) in INFERENCE_PORTS and m.group(1).startswith(_LOOPBACK) \
                and m.group(3).startswith(_LOOPBACK):
            out[(int(m.group(2)), int(m.group(4)))] = (int(m.group(5) or 0), int(m.group(6) or 0))
    return out


def traffic(before: dict, after: dict, conns: list) -> dict[str, dict[str, int]]:
    """What one client sent to (`up`, requests) and got back from (`down`, responses: generated tokens) each local
    server between two `connection_bytes()` samples, over its `conns` ([server port, client port]). A connection not
    in `before` opened inside the window and counts from zero, and so does one whose counters went backwards (a new
    connection on a reused port). An idle keep-alive connection yields zeros: connected is not used."""
    out: dict[str, dict[str, int]] = {}
    for sport, cport in conns:
        a = after.get((sport, cport))
        if a is None:
            continue
        b = before.get((sport, cport), (0, 0))
        if a[0] < b[0] or a[1] < b[1]:
            b = (0, 0)
        t = out.setdefault(INFERENCE_PORTS[sport], {"up": 0, "down": 0})
        t["up"] += a[0] - b[0]
        t["down"] += a[1] - b[1]
    return out


def cpu_busy_pct(interval_s: int = 2) -> float | None:
    """Whole-machine CPU busy % (100 - idle) over `interval_s`, from the second /usr/bin/top sample (the first
    covers time since boot). Load average is not a busy signal on this host: with ~15 idle omp/bun processes
    (~115 threads each) load1 sat at 17-24 while top reported 88-91% idle (ledger, 2026-09-22)."""
    out = _run("/usr/bin/top", "-l", "2", "-s", str(interval_s), "-n", "1", timeout=interval_s + 15)
    idle = re.findall(r"CPU usage:.*?([\d.]+)% idle", out)
    return round(100 - float(idle[-1]), 1) if len(idle) == 2 else None


def memory() -> dict:
    free = re.search(r"free percentage:\s*(\d+)%", _run("memory_pressure", "-Q"))
    swap = re.search(r"used = ([\d.]+)M", _sysctl("vm.swapusage"))
    level = _sysctl("kern.memorystatus_vm_pressure_level")
    return {
        "free_pct": int(free.group(1)) if free else None,
        "swap_used_mb": float(swap.group(1)) if swap else None,
        "pressure_level": {"1": "normal", "2": "warn", "4": "critical"}.get(level, level or None),
    }


def _json(url: str, timeout: float = 2.0) -> dict | None:
    """Parsed JSON, {} when nothing listens (connection refused), None when the server did not answer in time.
    An ollama scheduler that is loading a model can stall /api/ps for seconds; that is 'unknown', not 'empty'."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.URLError as exc:
        return {} if isinstance(exc.reason, ConnectionRefusedError) else None
    except (OSError, ValueError):
        return None


def resident_models() -> dict:
    """Models loaded on every local inference server we know of. Down server -> []; unresponsive -> None. Splash loads
    its model at startup, so a listed model is a resident one."""
    ps = _json("http://127.0.0.1:11434/api/ps")
    mx = _json("http://127.0.0.1:11234/v1/models")
    sp = _json("http://127.0.0.1:8000/v1/models")
    om = _json("http://127.0.0.1:11236/api/status")
    sm = _json("http://127.0.0.1:11235/v1/models")        # SMOL_PORT; literal like its neighbours (the port map reads it)
    return {
        "ollama": None if ps is None else sorted(m["name"] for m in ps.get("models", [])),
        "mlx-serve": None if mx is None else sorted(m["id"] for m in mx.get("data", [])),
        "splash": None if sp is None else sorted(m["id"] for m in sp.get("data", [])),
        # oMLX lists every model it could serve on /v1/models and loads on demand: residency is /api/status.
        "omlx": None if om is None else sorted(m if isinstance(m, str) else (m.get("id") or m.get("model_id"))
                                              for m in om.get("loaded_models", [])),
        # The smol server loads its one model at start, like Splash: listed means resident.
        "mlx-smol": None if sm is None else sorted(m["id"] for m in sm.get("data", [])),
    }


def keep_until(expires_at: str | None) -> str:
    """How long ollama keeps a loaded model, from /api/ps `expires_at`: `forever` for keep_alive -1 (ollama
    reports a date centuries out), local time for a timer, `unknown` when absent or unparseable."""
    if not expires_at:
        return "unknown"
    try:
        when = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", expires_at))
    except ValueError:
        return "unknown"
    return "forever" if when.year >= 2100 else when.astimezone().strftime("%m-%d %H:%M")


def ollama_residents(root: str = "http://127.0.0.1:11434", timeout: float = 2.0) -> list[tuple[str, str]] | None:
    """(model, keep_until) for every model ollama has loaded; [] when ollama is down or has nothing loaded; None when
    it did not answer in `timeout` s (its scheduler stalls /api/ps while loading a model): unknown, never 'none'."""
    ps = _json(root + "/api/ps", timeout=timeout)
    return None if ps is None else [(m["name"], keep_until(m.get("expires_at"))) for m in ps.get("models", [])]


OLLAMA_APP_DB = Path.home() / "Library/Application Support/Ollama/db.sqlite"


def _ollama_app_settings(db: Path = OLLAMA_APP_DB) -> dict | None:
    """Ollama.app's settings row, read-only; None when the database, the table or the row is absent."""
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=2)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute("SELECT * FROM settings WHERE id = 1").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return None if row is None else dict(row)


def ollama_auto_update(db: Path = OLLAMA_APP_DB) -> bool | None:
    """Ollama.app's own auto-update switch (settings.auto_update_enabled). Its updater re-reads it before every
    hourly download (app/updater/updater.go, v0.34.4), and an update it installs moves ollama under every golden.
    None when the app database or the column is absent (no app, or an app older than that setting)."""
    s = _ollama_app_settings(db)
    return None if s is None or "auto_update_enabled" not in s else bool(s["auto_update_enabled"])


def ollama_models_dir(db: Path = OLLAMA_APP_DB) -> Path:
    """Where ollama's server keeps models, resolved in Ollama.app's order (app/store Store.Settings, v0.34.4): the app's
    `models` setting, else OLLAMA_MODELS, else ~/.ollama/models. The app's setting overrides OLLAMA_MODELS for the
    server (the 0.34.4 upgrade, ledger 2026-09-25). `ollama create` from safetensors imports in the CLI's own process
    and resolves only OLLAMA_MODELS, so it must be given this path or it writes where the server never looks."""
    s = _ollama_app_settings(db) or {}
    return Path(s.get("models") or os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")


def splash_resident() -> bool:
    """True only when Splash answered /v1/models with at least one id. A refused connection, a timeout, or an
    empty list is not residency: those must not attach a Splash pin to an unrelated run."""
    listed = _json("http://127.0.0.1:8000/v1/models", timeout=0.4)
    return bool(listed and listed.get("data"))



def user_idle_s() -> float | None:
    """Seconds since the last keyboard or mouse input (IOHIDSystem HIDIdleTime, ns; no privileges). The person at the
    keyboard, measured directly: per-process GPU % cannot tell user activity apart, because a saturated GPU inflates
    every UI process's share (2026-09-24: 'foreign' load read 8.5-14.8% on dense-model legs, 2.8-3.6% on MoE legs)."""
    m = re.search(r'"HIDIdleTime" = (\d+)', _run("ioreg", "-c", "IOHIDSystem", "-d", "4", "-r"))
    return round(int(m.group(1)) / 1e9, 1) if m else None


def live() -> dict:
    """Cheap signals, safe to poll at 1 Hz."""
    gpu = {f"gpu_{k}": v for k, v in gpu_utilization().items()}
    return {"t": time.time(), "load1": round(os.getloadavg()[0], 2), **memory(), **gpu, "resident": resident_models(),
            "user_idle_s": user_idle_s()}


def foreign_models(resident: dict, backend: str, model: str) -> dict:
    """Everything resident that is not the model under test on its own backend (unknown servers are skipped)."""
    return {srv: [m for m in names if not (srv == backend and m == model)]
            for srv, names in resident.items()
            if names and any(not (srv == backend and m == model) for m in names)}


def top_processes(n: int = 8) -> list[dict]:
    rows = _run("ps", "-Ao", "pid=,pcpu=,rss=,comm=", "-r").splitlines()[:n]
    procs = []
    for row in rows:
        parts = row.split(None, 3)
        if len(parts) == 4:
            procs.append({"pid": int(parts[0]), "cpu_pct": float(parts[1]), "rss_mb": round(int(parts[2]) / 1024),
                          "comm": os.path.basename(parts[3])})
    return procs


def host() -> dict:
    """Static identity. Goldens are keyed by `host_id` so numbers never cross machines."""
    gpu_cores = re.search(r'"sppci_cores"\s*:\s*"?(\d+)', _run("system_profiler", "SPDisplaysDataType", "-json", timeout=20))
    chip = _sysctl("machdep.cpu.brand_string")
    mem_gb = int(_sysctl("hw.memsize") or 0) // 2**30
    info = {
        "hostname": _run("hostname", "-s"),
        "model": _sysctl("hw.model"),
        "chip": chip,
        "p_cores": int(_sysctl("hw.perflevel0.physicalcpu") or 0),
        "e_cores": int(_sysctl("hw.perflevel1.physicalcpu") or 0),
        "gpu_cores": int(gpu_cores.group(1)) if gpu_cores else None,
        "mem_gb": mem_gb,
        "macos": _run("sw_vers", "-productVersion"),
        "macos_build": _run("sw_vers", "-buildVersion"),
        "gpu_wired_limit_mb": int(_sysctl("iogpu.wired_limit_mb") or 0) or None,
    }
    info["host_id"] = re.sub(r"[^a-z0-9]+", "-", f"{info['hostname']}-{chip}-{mem_gb}gb".lower()).strip("-")
    return info


def power() -> dict:
    ps = _run("pmset", "-g", "ps")
    therm = _run("pmset", "-g", "therm")
    low_power = re.search(r"lowpowermode\s+(\d)", _run("pmset", "-g"))
    return {
        "source": "ac" if "AC Power" in ps else ("battery" if "Battery Power" in ps else None),
        "low_power_mode": low_power.group(1) == "1" if low_power else None,
        "thermal_warning": None if "No thermal warning level has been recorded" in therm or not therm
        else " ".join(therm.split()),
    }


def disk_free_gb(path: str = "/") -> float:
    st = os.statvfs(path)
    return round(st.f_bavail * st.f_frsize / 1e9, 1)


def snapshot() -> dict:
    return {"host": host(), "power": power(), "live": live(), "top": top_processes(), "disk_free_gb": disk_free_gb()}


INFERENCE_NAMES = ("ollama", "llama-server", "mlx-serve")
USER_ACTIVE_S = 10.0


def is_inference(row: dict) -> bool:
    """A GPU row that is a model server or runner (it carries the model it serves, or is one by name)."""
    return (bool(row.get("model")) or row.get("name") in INFERENCE_NAMES
            or bool(re.search(r"\b(omlx|splash) serve\b", row.get("cmd") or "")))


def classify(gpu_procs: list[dict], resident: dict, target: tuple[str, str], max_pct: float) -> tuple[dict, list, list]:
    """One sample -> (foreign resident models, other models' runners above max_pct, apps above max_pct).

    Only the first two void a run: another model competes for the GPU and `localbench park` controls it. Apps and the
    person at the keyboard are the condition the measurement is taken under (the owner, 2026-09-24: 'my machine is never
    going to be fully quiet - we need our testing to be while our system is working'); they are recorded as load."""
    foreign = {k: v for k, v in foreign_models(resident, *target).items() if v}
    busy = [r for r in gpu_procs if r["pct"] > max_pct and not gpu_is_ours(r, *target)]
    return foreign, [r for r in busy if is_inference(r)], [r for r in busy if not is_inference(r)]


def load_summary(series: list[dict], target: tuple[str, str] | None, max_pct: float) -> dict:
    """What else ran while a model was measured, per 1 Hz sample: GPU % of every non-model process summed (mean, p95),
    seconds in which one such process passed max_pct, and the share of samples with keyboard or mouse input within
    USER_ACTIVE_S. The app-GPU figures are inflated by the model's own saturation, so compare them only between legs
    that load the GPU alike; user_active_pct does not depend on the model."""
    app, spikes, active, seen = [], 0, 0, 0
    for s in series:
        rows = [r for r in s.get("gpu_procs") or [] if not is_inference(r) and not (target and gpu_is_ours(r, *target))]
        app.append(sum(r["pct"] for r in rows))
        spikes += any(r["pct"] > max_pct for r in rows)
        if s.get("user_idle_s") is not None:
            seen += 1
            active += s["user_idle_s"] < USER_ACTIVE_S
    app.sort()
    return {"samples": len(series),
            "app_gpu_mean_pct": round(statistics.fmean(app), 1) if app else None,
            "app_gpu_p95_pct": round(app[math.ceil(0.95 * len(app)) - 1], 1) if app else None,   # nearest rank
            "app_spike_seconds": spikes,
            "user_active_pct": round(100 * active / seen, 1) if seen else None}


class Sampler:
    """Polls `live()` on a background thread for the duration of a `with` block, plus which processes used the GPU
    in each interval (`gpu_procs`) and over the whole block (`summary()["gpu_by_process"]`).

    With `target=(backend, model)` every sample is checked for foreign resident models and, with
    `gpu_foreign_max_pct`, for any process other than the backend's own using more GPU than that in the interval
    (the device-wide % cannot see this: the model under test saturates it). The first sample of each contention
    episode goes to `on_contention` (live monitors see it).
    """

    def __init__(self, interval: float = 1.0, target: tuple[str, str] | None = None,
                 on_contention: Callable[[dict], None] | None = None, gpu_foreign_max_pct: float | None = None):
        self.interval = interval
        self.target = target
        self.on_contention = on_contention
        self.gpu_foreign_max_pct = gpu_foreign_max_pct
        self.series: list[dict] = []
        self.contention: list[dict] = []
        self.load_spikes: list[dict] = []
        self._commands: dict[int, str] = {}
        self._gpu_span: list[tuple[dict, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        in_episode = in_spike = False
        prev = (gpu_time_by_pid(), time.time())
        self._gpu_span = [prev, prev]
        # The first sample waits one interval, so it already carries a GPU window (an episode is reported by its first
        # sample); at least one sample is taken however short the block.
        self._stop.wait(self.interval)
        while True:
            s = live()
            now = (gpu_time_by_pid(), time.time())
            # Windows shorter than half an interval stay open: a 0.1 s window turns one frame into a large %.
            if now[1] - prev[1] >= self.interval / 2:
                s["gpu_procs"] = gpu_share(prev[0], now[0], now[1] - prev[1], self._commands)
                prev = now
            self._gpu_span[1] = now
            self.series.append(s)
            if self.target:
                limit = self.gpu_foreign_max_pct if self.gpu_foreign_max_pct is not None else float("inf")
                foreign, foreign_gpu, apps = classify(s.get("gpu_procs", []), s["resident"], self.target, limit)
                if (foreign or foreign_gpu) and not in_episode:
                    ev = {"t": round(s["t"], 3), "foreign": foreign, "foreign_gpu": foreign_gpu,
                          "resident": s["resident"], "gpu_device_pct": s.get("gpu_device_pct")}
                    self.contention.append(ev)
                    if self.on_contention:
                        self.on_contention(ev)
                in_episode = bool(foreign or foreign_gpu)
                if apps and not in_spike:
                    self.load_spikes.append({"t": round(s["t"], 3), "apps": apps, "user_idle_s": s.get("user_idle_s")})
                in_spike = bool(apps)
            if self._stop.wait(self.interval):
                break

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        out: dict = {"samples": len(self.series)}
        for key in ("load1", "gpu_device_pct", "free_pct", "swap_used_mb"):
            vals = [s[key] for s in self.series if s.get(key) is not None]
            if vals:
                out[key] = {"mean": round(statistics.fmean(vals), 1), "max": max(vals), "min": min(vals)}
        levels = {s.get("pressure_level") for s in self.series} - {None}
        out["pressure_levels"] = sorted(levels)
        out["resident_unknown_samples"] = sum(1 for s in self.series if None in s["resident"].values())
        (first, t0), (last, t1) = self._gpu_span or [({}, 0.0), ({}, 0.0)]
        if t1 > t0:
            out["gpu_by_process"] = gpu_share(first, last, t1 - t0, self._commands, min_pct=1.0)[:8]
        out["load"] = load_summary(self.series, self.target,
                                   self.gpu_foreign_max_pct if self.gpu_foreign_max_pct is not None else 25.0)
        return out


def powermetrics_available() -> bool:
    """True only when scripts/install-sudoers.sh granted passwordless powermetrics (checked, not run)."""
    return subprocess.run(["sudo", "-n", "-l", "/usr/bin/powermetrics"], capture_output=True,
                          check=False).returncode == 0


class PowerSampler:
    """Streams `powermetrics` plists (root-only): GPU/CPU/ANE power, GPU clock, thermal pressure.

    Without the sudoers grant this is a no-op and `summary()` reports
    `available: false` instead of inventing numbers.
    """

    SAMPLERS = "gpu_power,cpu_power,ane_power,thermal"

    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.samples: list[dict] = []
        self.available = powermetrics_available()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        assert self._proc and self._proc.stdout
        buf = b""
        for chunk in iter(lambda: self._proc.stdout.read(4096), b""):
            buf += chunk
            while b"\0" in buf:
                doc, buf = buf.split(b"\0", 1)
                if doc.strip():
                    self.samples.append(self._parse(plistlib.loads(doc.strip())))

    @staticmethod
    def _parse(p: dict) -> dict:
        proc = p.get("processor", {})
        gpu = p.get("gpu", {})
        return {
            "thermal_pressure": p.get("thermal_pressure"),
            "cpu_mw": proc.get("cpu_power"),
            "gpu_mw": proc.get("gpu_power"),
            "ane_mw": proc.get("ane_power"),
            "combined_mw": proc.get("combined_power"),
            "gpu_freq_mhz": round(gpu["freq_hz"] / 1e6) if gpu.get("freq_hz") else None,
            "gpu_active_pct": round((1 - gpu["idle_ratio"]) * 100, 1) if gpu.get("idle_ratio") is not None else None,
        }

    def __enter__(self) -> Self:
        if self.available:
            self._proc = subprocess.Popen(
                ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", self.SAMPLERS,
                 "-i", str(self.interval_ms), "-f", "plist"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=10)
            if self._thread:
                self._thread.join(timeout=5)

    def summary(self) -> dict:
        out: dict = {"available": self.available, "samples": len(self.samples)}
        for key in ("gpu_mw", "cpu_mw", "ane_mw", "combined_mw", "gpu_freq_mhz", "gpu_active_pct"):
            vals = [s[key] for s in self.samples if isinstance(s.get(key), (int, float))]
            if vals:
                out[key] = {"mean": round(statistics.fmean(vals), 1), "max": max(vals)}
        out["thermal_pressure"] = sorted({s["thermal_pressure"] for s in self.samples if s.get("thermal_pressure")})
        return out


class CpuSampler:
    """Streams /usr/bin/top (no privileges) for whole-machine CPU busy % during a run. Other agents share this
    host (2026-09-22: a `cp -r` plus Spotlight indexing held ~27% CPU mid-run and omp's own startup stretched), so
    CPU load is recorded next to every number. The first top sample covers time since boot and is dropped."""

    def __init__(self, interval_s: int = 2):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        assert self._proc and self._proc.stdout
        seen = 0
        for line in self._proc.stdout:
            m = re.search(r"CPU usage:.*?([\d.]+)% idle", line)
            if m:
                seen += 1
                if seen > 1:
                    self.samples.append({"t": round(time.time(), 3), "busy_pct": round(100 - float(m.group(1)), 1)})

    def __enter__(self) -> Self:
        self._proc = subprocess.Popen(["/usr/bin/top", "-l", "0", "-s", str(self.interval_s), "-n", "1"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        vals = [s["busy_pct"] for s in self.samples]
        out: dict = {"samples": len(vals)}
        if vals:
            out["busy_pct"] = {"mean": round(statistics.fmean(vals), 1), "max": max(vals), "min": min(vals)}
        return out



def purge_file_cache() -> bool:
    """Drop the unified buffer cache so the next model load reads from disk. Needs the sudoers grant."""
    return subprocess.run(["sudo", "-n", "/usr/sbin/purge"], capture_output=True, check=False).returncode == 0
