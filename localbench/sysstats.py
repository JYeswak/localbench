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

from .backends import owned_stop_events


def _run(*cmd: str, timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _sysctl(name: str) -> str:
    return _run("sysctl", "-n", name)


def gpu_utilization() -> dict:
    """Ancillary ioreg fields; device activity is measured by gpu_window()."""
    out = _run("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
    stats = {"device_pct": None, "device_source": "windowed macmon IOReport", "status": "WINDOW_REQUIRED"}
    for key, field in (("renderer_pct", "Renderer Utilization %"), ("tiler_pct", "Tiler Utilization %"),
                       ("in_use_mem_bytes", "In use system memory")):
        m = re.search(rf'"{re.escape(field)}"=(\d+)', out)
        if m:
            stats[key] = int(m.group(1))
    return stats


_GPU_CREATOR = re.compile(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"')
_GPU_NS = re.compile(r'"accumulatedGPUTime"=(\d+)')
GPU_IDLE_THRESHOLD_PCT = 5.0  # below this floor, a device/process ratio is measurement noise


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


def _parse_macmon_sample(line: str) -> dict | None:
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    usage = payload.get("gpu_usage")
    timestamp = payload.get("timestamp")
    if not isinstance(usage, list) or len(usage) < 2 or not isinstance(timestamp, str):
        return None
    try:
        sample_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        frequency_mhz, device_fraction = float(usage[0]), float(usage[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if (not math.isfinite(sample_time) or not math.isfinite(frequency_mhz) or frequency_mhz < 0
            or not math.isfinite(device_fraction) or not 0 <= device_fraction <= 1):
        return None
    return {"frequency_mhz": round(frequency_mhz, 1), "device_pct": round(device_fraction * 100, 1),
            "timestamp": timestamp, "sample_time": sample_time}


def _collect_macmon_window(seconds: float) -> tuple[list[dict], list[tuple[float, dict]]]:
    counter_points = []

    def sample_process_counters() -> None:
        started = time.time()
        counters = gpu_time_by_pid()
        ended = time.time()
        counter_points.append(((started + ended) / 2, counters))

    sample_process_counters()
    sample_count = max(2, math.ceil(seconds / 1.5))
    try:
        process = subprocess.Popen(["macmon", "pipe", "--samples", str(sample_count), "--interval", "1000"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except OSError:
        return [], counter_points
    deadline = time.monotonic() + max(30.0, seconds * 5)
    timed_out = False
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break
        try:
            process.wait(timeout=min(0.5, remaining))
        except subprocess.TimeoutExpired:
            pass
        sample_process_counters()
    stdout, _ = process.communicate()
    sample_process_counters()
    if timed_out or process.returncode != 0:
        return [], counter_points
    samples = []
    for line in stdout.splitlines():
        sample = _parse_macmon_sample(line)
        if sample is not None:
            samples.append(sample)
    return samples, counter_points


def _gpu_counters_at(points: list[tuple[float, dict]], when: float) -> dict[int, dict] | None:
    """Interpolate cumulative process counters at an IOReport sample boundary."""
    for (left_time, left), (right_time, right) in zip(points, points[1:]):
        if left_time <= when <= right_time and right_time > left_time:
            fraction = (when - left_time) / (right_time - left_time)
            counters = {}
            for pid in left.keys() | right.keys():
                before = left.get(pid, {})
                after = right.get(pid, {})
                before_ns = before.get("ns", 0)
                after_ns = max(before_ns, after.get("ns", before_ns))
                counters[pid] = {"name": after.get("name") or before.get("name") or "unknown",
                                 "ns": round(before_ns + (after_ns - before_ns) * fraction)}
            return counters
    return None


def gpu_coverage_summary(device_samples: list[float], process_pct: float | None) -> dict:
    values = []
    for value in device_samples:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 100):
            values = []
            break
        values.append(float(value))
    if (isinstance(process_pct, bool) or not isinstance(process_pct, (int, float))
            or not math.isfinite(process_pct) or process_pct < 0):
        return {"device_pct": None, "process_pct": None, "coverage": None, "unattributed_pct": None,
                "status": "UNAVAILABLE", "source": "macmon IOReport"}
    process_pct = float(process_pct)
    if not values:
        return {"device_pct": None, "process_pct": round(process_pct, 1), "coverage": None,
                "unattributed_pct": None, "status": "UNAVAILABLE", "source": "macmon IOReport"}
    device_pct = statistics.fmean(values)
    if device_pct < GPU_IDLE_THRESHOLD_PCT and process_pct < GPU_IDLE_THRESHOLD_PCT:
        status, coverage, unattributed_pct = "IDLE", None, 0.0
    elif device_pct < GPU_IDLE_THRESHOLD_PCT:
        status, coverage, unattributed_pct = "UNATTRIBUTED", None, None
    else:
        raw_coverage = 100 * process_pct / device_pct
        coverage = round(min(100.0, raw_coverage), 1)
        unattributed_pct = round(max(0.0, 100.0 - coverage), 1)
        if raw_coverage > 100.0:
            status = "UNALIGNED"
        else:
            status = "ATTRIBUTED" if coverage >= 90 else "UNATTRIBUTED"
    return {"device_pct": round(device_pct, 1), "process_pct": round(float(process_pct), 1),
            "coverage": coverage, "unattributed_pct": unattributed_pct, "status": status,
            "source": "macmon IOReport"}


def _device_window_mean(samples: list[dict]) -> float | None:
    if len(samples) < 2:
        return None
    raw_time = samples[0].get("sample_time")
    raw_pct = samples[0].get("device_pct")
    if (isinstance(raw_time, bool) or not isinstance(raw_time, (int, float))
            or isinstance(raw_pct, bool) or not isinstance(raw_pct, (int, float))):
        return None
    previous_time, previous_pct = float(raw_time), float(raw_pct)
    if (not math.isfinite(previous_time) or not math.isfinite(previous_pct)
            or not 0 <= previous_pct <= 100):
        return None
    first_time = previous_time
    area = 0.0
    for sample in samples[1:]:
        raw_time, raw_pct = sample.get("sample_time"), sample.get("device_pct")
        if (isinstance(raw_time, bool) or not isinstance(raw_time, (int, float))
                or isinstance(raw_pct, bool) or not isinstance(raw_pct, (int, float))):
            return None
        sample_time, sample_pct = float(raw_time), float(raw_pct)
        if (not math.isfinite(sample_time) or not math.isfinite(sample_pct)
                or sample_time <= previous_time or not 0 <= sample_pct <= 100):
            return None
        area += (previous_pct + sample_pct) * 0.5 * (sample_time - previous_time)
        previous_time, previous_pct = sample_time, sample_pct
    return area / (previous_time - first_time)


def gpu_window(seconds: float = 5.0) -> dict:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError):
        seconds = 0
    if not math.isfinite(seconds) or seconds <= 0:
        samples, counter_points = [], []
    else:
        samples, counter_points = _collect_macmon_window(seconds)
    raw_times = [sample.get("sample_time") for sample in samples]
    sample_times: list[float] = []
    for value in raw_times:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            sample_times = []
            break
        sample_times.append(float(value))
    valid_window = (len(samples) >= 2 and len(sample_times) == len(samples)
                    and all(right > left for left, right in zip(sample_times, sample_times[1:])))
    if valid_window:
        window_s = sample_times[-1] - sample_times[0]
        before = _gpu_counters_at(counter_points, sample_times[0])
        after = _gpu_counters_at(counter_points, sample_times[-1])
    else:
        window_s, before, after = 0.0, None, None
    if window_s < seconds * 0.8:
        before = after = None
    if before is not None and after is not None and window_s > 0:
        rows = gpu_share(before, after, window_s, min_pct=0.5)
        delta_ns = sum(max(0, process["ns"] - before.get(pid, {}).get("ns", 0))
                       for pid, process in after.items())
        process_pct = 100 * delta_ns / (window_s * 1e9)
        device_mean = _device_window_mean(samples)
        device_samples = [device_mean] if device_mean is not None else []
    else:
        rows, process_pct, device_samples = [], None, []
    report = gpu_coverage_summary(device_samples, process_pct)
    report.update({"window_s": round(window_s, 3), "gpu_by_process": rows,
                   "io_report_samples": len(samples),
                   "process_counter_samples": len(counter_points),
                   "io_report": [{"timestamp": sample.get("timestamp"),
                                  "device_pct": sample["device_pct"]} for sample in samples]})
    return report


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


# A harness server's process, by backend name, as its executable (the mlxfast backend runs Layr-Labs' `mlx-server`).
SERVER_EXES = {"mlxfast": "mlx-server"}


def gpu_is_ours(row: dict, backend: str, model: str, also: tuple[str, ...] = ()) -> bool:
    """The backend's own GPU work; `also` names additional models declared as part of the measured workload."""
    if backend == "ollama":
        if row["name"] == "ollama" and row["cmd"].endswith(" serve"):
            return True
        return row["name"] in ("ollama", "llama-server") and bool(
            {model, *also} & set(row.get("model_names") or [row.get("model")]))
    if backend == "omlx" and re.search(r"(?:^|/)omlx-server(?:\s|$)", row["cmd"]):
        return True
    exe = re.escape(SERVER_EXES.get(backend, backend))
    return bool(re.search(rf"(?:^|/){exe}(?:\s|$)", row["cmd"]))


# Local inference servers on this host: Ollama, the loopback residency gateway, MLX servers, and localbench proxy.
SMOL_PORT = 11235
OLLAMA_GATEWAY_PORT = 11300
INFERENCE_PORTS = {11434: "ollama", OLLAMA_GATEWAY_PORT: "ollama-gateway", 11234: "mlx-serve",
                   SMOL_PORT: "mlx-smol", 11236: "omlx", 11237: "mlxfast", 11299: "localbench-proxy", 8000: "splash"}


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


def _probe_error_class(exc: BaseException) -> str:
    return "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__


def _json(url: str, timeout: float = 2.0, *, probe: dict | None = None) -> dict | None:
    """Parsed JSON, {} when nothing listens, None when a responding server is unreadable or too slow.

    An ollama scheduler loading a model can stall /api/ps; that remains unknown, not empty.
    """
    started = time.monotonic() if probe is not None else None
    status = None
    error_class = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            status = getattr(r, "status", None)
            result = json.load(r)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error_class = "HTTPError"
        result = None
    except urllib.error.URLError as exc:
        error_class = (_probe_error_class(exc.reason) if isinstance(exc.reason, BaseException)
                       else type(exc).__name__)
        result = {} if isinstance(exc.reason, ConnectionRefusedError) else None
    except ValueError:
        error_class = "invalid_json"
        result = None
    except OSError as exc:
        error_class = _probe_error_class(exc)
        result = None
    finally:
        if probe is not None:
            probe.update({"probe_start": started, "probe_end": time.monotonic(), "error_class": error_class,
                          "http_status": status})
    return result


_RESIDENCY_SERVERS = (
    ("ollama", 11434, "http://127.0.0.1:11434/api/ps", "models", "name"),
    ("mlx-serve", 11234, "http://127.0.0.1:11234/v1/models", "data", "id"),
    # oMLX /v1/models inventories available models; /api/status reports the IDs actually resident.
    ("omlx", 11236, "http://127.0.0.1:11236/api/status", "loaded_models", None),
    ("mlx-smol", 11235, "http://127.0.0.1:11235/v1/models", "data", "id"),
    ("mlxfast", 11237, "http://127.0.0.1:11237/v1/models", "data", "id"),
)


def resident_models(probes: dict | None = None) -> dict:
    """Models loaded on every local inference server we know of. Down server -> []; unreadable or malformed -> None."""

    def fetch(server: str, port: int, url: str, field: str, item_field: str | None) -> list[str] | None:
        probe = {"server": server, "port": port}
        try:
            result = _json(url, probe=probe)
        except Exception as exc:
            probe.update({"probe_start": time.monotonic(), "probe_end": time.monotonic(),
                          "error_class": _probe_error_class(exc), "http_status": None})
            result = None
        if probes is not None:
            probes[server] = probe
        if result is None:
            return None
        # Connection refusal is the established down-server signal; unlike an HTTP 200 response, it has no schema.
        if result == {} and probe.get("http_status") is None and probe.get("error_class"):
            return []
        if not isinstance(result, dict) or not isinstance(result.get(field), list):
            probe["error_class"] = "schema"
            return None
        names = []
        for model in result[field]:
            if item_field is not None:
                name = model.get(item_field) if isinstance(model, dict) else None
            else:
                name = model if isinstance(model, str) else None
            if not isinstance(name, str) or not name:
                probe["error_class"] = "schema"
                return None
            names.append(name)
        return sorted(names)

    return {server: fetch(server, port, url, field, item_field)
            for server, port, url, field, item_field in _RESIDENCY_SERVERS}


def _unknown_residency_sample(exc: Exception) -> dict:
    error_class = _probe_error_class(exc)
    now = time.monotonic()
    probes = {
        server: {"server": server, "error_class": error_class, "probe_start": now, "probe_end": now,
                 "http_status": None}
        for server, *_ in _RESIDENCY_SERVERS
    }
    return {"t": time.time(), "resident": {server: None for server in probes}, "resident_probes": probes,
            "sampler_error_class": error_class}


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
    """Cheap signals, safe to poll at 1 Hz; residency probes carry per-server monotonic timing and failure class."""
    gpu = {f"gpu_{k}": v for k, v in gpu_utilization().items()}
    probes: dict = {}
    resident = resident_models(probes)
    return {"t": time.time(), "load1": round(os.getloadavg()[0], 2), **memory(), **gpu, "resident": resident,
            "resident_probes": probes, "user_idle_s": user_idle_s()}


def foreign_models(resident: dict, backend: str, model: str, also: tuple[str, ...] = ()) -> dict:
    """Everything resident outside the declared workload (unknown servers are skipped)."""
    ours = {model, *also}
    return {srv: [m for m in names if m not in ours]
            for srv, names in resident.items()
            if names and any(m not in ours for m in names)}


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


INFERENCE_NAMES = ("ollama", "llama-server", "mlx-serve", "mlx-server", "omlx-server")
USER_ACTIVE_S = 10.0


def is_inference(row: dict) -> bool:
    """A GPU row that is a model server or runner (it carries the model it serves, or is one by name)."""
    return (bool(row.get("model")) or row.get("name") in INFERENCE_NAMES
            or bool(re.search(r"\b(omlx|splash) serve\b|(?:^|/)omlx-server(?:\s|$)", row.get("cmd") or "")))


def classify(gpu_procs: list[dict], resident: dict, target: tuple[str, ...], max_pct: float) -> tuple[dict, list, list]:
    """One sample -> (foreign resident models, other models' runners above max_pct, apps above max_pct).

    Only the first two void a run: another model competes for the GPU and `localbench park` controls it. Apps and the
    person at the keyboard are the condition the measurement is taken under; declared auxiliary models also belong to
    the workload rather than counting as contention."""
    backend, model, *also = target
    also = tuple(also)
    foreign = {k: v for k, v in foreign_models(resident, backend, model, also).items() if v}
    busy = [r for r in gpu_procs if r["pct"] > max_pct and not gpu_is_ours(r, backend, model, also)]
    return foreign, [r for r in busy if is_inference(r)], [r for r in busy if not is_inference(r)]


def load_summary(series: list[dict], target: tuple[str, ...] | None, max_pct: float) -> dict:
    """What else ran while a model was measured, per 1 Hz sample: GPU % of every non-model process summed (mean, p95),
    seconds in which one such process passed max_pct, and the share of samples with keyboard or mouse input within
    USER_ACTIVE_S. The app-GPU figures are inflated by the model's own saturation, so compare them only between legs
    that load the GPU alike; user_active_pct does not depend on the model."""
    app, spikes, active, seen = [], 0, 0, 0
    for s in series:
        rows = [r for r in s.get("gpu_procs") or [] if not is_inference(r) and not (
            target and gpu_is_ours(r, target[0], target[1], tuple(target[2:])))]
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


def _owned_stop_for_sample(sample: dict, target: tuple[str, ...] | None, stops: list[dict]) -> dict | None:
    unknown = [server for server, models in sample["resident"].items() if models is None]
    if not unknown or target is None or any(server != target[0] for server in unknown):
        return None
    probes = sample.get("resident_probes")
    if not isinstance(probes, dict):
        return None
    evidence = None
    for server in unknown:
        probe = probes.get(server)
        if not isinstance(probe, dict) or probe.get("server") != server or not probe.get("error_class"):
            return None
        probe_start, probe_end, port = probe.get("probe_start"), probe.get("probe_end"), probe.get("port")
        if (not isinstance(probe_start, int | float) or not isinstance(probe_end, int | float)
                or probe_start > probe_end):
            return None
        match = None
        for stop in stops:
            if not isinstance(stop, dict) or stop.get("server") != server or stop.get("port") != port:
                continue
            start, end = stop.get("t_term"), stop.get("t_exit")
            if not isinstance(start, int | float) or not isinstance(end, int | float):
                continue
            if not start <= probe_start <= probe_end <= end:
                continue
            if stop.get("pre") != sorted({target[1], *target[2:]}):
                continue
            if stop.get("rc") is None or stop.get("killed"):
                continue
            match = stop
            break
        if match is None:
            return None
        evidence = match
    return evidence


class Sampler:
    """Polls `live()` on a background thread for the duration of a `with` block, plus which processes used the GPU
    in each interval (`gpu_procs`) and over the whole block (`summary()["gpu_by_process"]`).

    With `target=(backend, model[, auxiliary_model, ...])` every sample is checked for foreign resident models and, with
    `gpu_foreign_max_pct`, for any process other than the backend's own using more GPU than that in the interval
    (the device-wide % cannot see this: the model under test saturates it). The first sample of each contention
    episode goes to `on_contention` (live monitors see it).
    `on_sample` runs on the sampler thread for each completed sample before it is stored; callback exceptions are recorded
    on that sample as `sample_callback_error`.
    """

    def __init__(self, interval: float = 1.0, target: tuple[str, ...] | None = None,
                 on_contention: Callable[[dict], None] | None = None, gpu_foreign_max_pct: float | None = None,
                 on_sample: Callable[[dict], None] | None = None):
        self.interval = interval
        self.target = target
        self.on_contention = on_contention
        self.gpu_foreign_max_pct = gpu_foreign_max_pct
        self.on_sample = on_sample
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
            try:
                s = live()
                now = (gpu_time_by_pid(), time.time())
                # Windows shorter than half an interval stay open: a 0.1 s window turns one frame into a large %.
                if now[1] - prev[1] >= self.interval / 2:
                    s["gpu_procs"] = gpu_share(prev[0], now[0], now[1] - prev[1], self._commands)
                    prev = now
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
                        self.load_spikes.append(
                            {"t": round(s["t"], 3), "apps": apps, "user_idle_s": s.get("user_idle_s")})
                    in_spike = bool(apps)
            except Exception as exc:
                s = _unknown_residency_sample(exc)
                try:
                    now = (gpu_time_by_pid(), time.time())
                    prev = now
                except Exception:
                    now = (prev[0], time.time())
                in_episode = in_spike = False
            self._gpu_span[1] = now
            if self.on_sample:
                try:
                    self.on_sample(s)
                except Exception as exc:
                    s["sample_callback_error"] = f"{type(exc).__name__}: {exc}"
            self.series.append(s)
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
        stops = owned_stop_events()
        unknown = owned = 0
        for sample in self.series:
            sample.pop("resident_owned_stop", None)
            if None not in sample["resident"].values():
                continue
            evidence = _owned_stop_for_sample(sample, self.target, stops)
            if evidence is None:
                unknown += 1
            else:
                owned += 1
                sample["resident_owned_stop"] = evidence
        out["resident_unknown_samples"] = unknown
        out["resident_owned_stop_samples"] = owned
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
