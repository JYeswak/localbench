from __future__ import annotations

import os
import resource
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DEFAULT_SECONDS = 15
MAX_SECONDS = 32
IO_SAMPLE_SECONDS = 1
PARENT_DEPTH = 64
TOP_PROCESS_LIMIT = 2000


class LoadError(RuntimeError):
    """A required read-only system sample could not be collected."""


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    age_s: float
    cpu_time_s: float
    cpu_pct: float
    rss_kib: int
    comm: str
    args: str


@dataclass(frozen=True)
class Activity:
    pid: int
    cpu_pct: float
    csw_s: float
    sysbsd_s: float
    sysmach_s: float
    command: str
    ppid: int = 0


@dataclass(frozen=True)
class _TopProcess:
    pid: int
    ppid: int
    cpu_pct: float
    csw: float
    sysbsd: float
    sysmach: float
    command: str


@dataclass
class _ProbeLog:
    invocations: list[str]
    pids: list[int]


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _seconds(value: str) -> float:
    """Parse ps elapsed/CPU clocks (MM:SS, HH:MM:SS, or DD-HH:MM:SS), without regex."""
    value = value.strip()
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return days * 86400 + int(minutes) * 60 + float(seconds)
    return days * 86400 + float(value)


def parse_ps(text: str) -> dict[int, Process]:
    """Parse one ps snapshot; args is the only unbounded-width field."""
    out: dict[int, Process] = {}
    for line in text.splitlines():
        fields = line.split(None, 7)
        if len(fields) < 7:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
            age_s, cpu_time_s = _seconds(fields[2]), _seconds(fields[3])
            cpu_pct = float(fields[4].rstrip("%"))
            rss_kib = int(fields[5])
        except ValueError:
            continue
        comm = fields[6]
        args = fields[7] if len(fields) == 8 else comm
        out[pid] = Process(pid, ppid, age_s, cpu_time_s, cpu_pct, rss_kib, comm, args)
    return out


def _number(value: str) -> float:
    value = value.strip().rstrip("+%").replace(",", "")
    if not value or value in {"-", "?"}:
        return 0.0
    return float(value)


@dataclass
class _ActivityTotal:
    cpu_area: float = 0.0
    csw: float = 0.0
    sysbsd: float = 0.0
    sysmach: float = 0.0
    ppid: int = 0
    command: str = ""


def _parse_top_tables(text: str) -> list[dict[int, _TopProcess]]:
    tables: list[dict[int, _TopProcess]] = []
    columns: list[str] | None = None
    current: dict[int, _TopProcess] = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "PID" and "%CPU" in fields:
            if columns is not None:
                tables.append(current)
            columns = fields
            current = {}
            continue
        if columns is None or not fields or not fields[0].isdigit():
            continue
        row = line.split(None, len(columns) - 1)
        if len(row) < len(columns):
            continue
        try:
            pid = int(row[columns.index("PID")])
            ppid = int(row[columns.index("PPID")]) if "PPID" in columns else 0
            cpu = _number(row[columns.index("%CPU")])
            csw = _number(row[columns.index("CSW")])
            sysbsd = _number(row[columns.index("SYSBSD")])
            sysmach = _number(row[columns.index("SYSMACH")])
        except (ValueError, IndexError):
            continue
        command_index = columns.index("COMMAND") if "COMMAND" in columns else len(row) - 1
        current[pid] = _TopProcess(pid, ppid, cpu, csw, sysbsd, sysmach, row[command_index])
    if columns is not None:
        tables.append(current)
    return tables


def _aggregate_top_tables(tables: list[dict[int, _TopProcess]], seconds: int) -> dict[int, Activity]:
    delta_tables = tables[1:] if len(tables) > 1 else []
    if seconds <= 0 or not delta_tables:
        return {}
    interval_s = seconds / len(delta_tables)
    totals: dict[int, _ActivityTotal] = {}
    for table in delta_tables:
        for pid, sample in table.items():
            total = totals.setdefault(pid, _ActivityTotal())
            total.cpu_area += sample.cpu_pct * interval_s
            total.csw += sample.csw
            total.sysbsd += sample.sysbsd
            total.sysmach += sample.sysmach
            total.ppid = sample.ppid
            total.command = sample.command
    return {pid: Activity(pid, total.cpu_area / seconds, total.csw / seconds,
                          total.sysbsd / seconds, total.sysmach / seconds,
                          total.command, total.ppid)
            for pid, total in totals.items()}


def _top_spawn_events(tables: list[dict[int, _TopProcess]]) -> list[_TopProcess]:
    if len(tables) < 2:
        return []
    seen = set(tables[0])
    events: list[_TopProcess] = []
    for table in tables[1:]:
        for pid, process in table.items():
            if pid not in seen:
                events.append(process)
        seen = set(table)
    return events


def parse_top(text: str, seconds: int) -> dict[int, Activity]:
    """Aggregate /usr/bin/top delta tables over the requested interval."""
    return _aggregate_top_tables(_parse_top_tables(text), seconds)

def parse_iostat(text: str, window_seconds: int = 1) -> dict:
    """Parse the final per-device interval row from macOS iostat -d."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    for index, devices in enumerate(lines):
        if not devices or not all(device.startswith("disk") and device[4:].isdigit() for device in devices):
            continue
        expected = [field for _ in devices for field in ("KB/t", "tps", "MB/s")]
        header_index = next((i for i in range(index + 1, len(lines)) if lines[i] == expected), None)
        if header_index is None:
            continue
        samples = []
        for row in lines[header_index + 1:]:
            if len(row) != len(expected):
                continue
            try:
                samples.append([float(value) for value in row])
            except ValueError:
                continue
        if not samples:
            return {}
        latest = samples[-1]
        return {"window_seconds": window_seconds,
                "devices": [{"device": device, "kb_per_transfer": latest[i * 3],
                             "transfers_per_s": latest[i * 3 + 1], "mb_per_s": latest[i * 3 + 2]}
                            for i, device in enumerate(devices)]}
    return {}


def _vm_stat_number(value: str) -> int | None:
    value = value.rstrip(".,")
    multiplier = 1
    if value and value[-1] in "KMG":
        multiplier = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}[value[-1]]
        value = value[:-1]
    try:
        return int(float(value) * multiplier)
    except ValueError:
        return None


def parse_vm_stat(text: str, window_seconds: int = 1) -> dict:
    """Parse vm_stat -c output: last-row event deltas and current page-queue gauges."""
    page_size = None
    columns = None
    rows = []
    for line in text.splitlines():
        if "page size of " in line:
            value = line.partition("page size of ")[2].split()
            if value:
                try:
                    page_size = int(value[0].rstrip(")."))
                except ValueError:
                    continue
        fields = line.split()
        if "free" in fields and "pageins" in fields and "swapouts" in fields:
            columns = fields
            continue
        if columns is None or len(fields) != len(columns):
            continue
        values = [_vm_stat_number(value) for value in fields]
        if all(value is not None for value in values):
            rows.append(dict(zip(columns, values)))
    if page_size is None or window_seconds < 1 or len(rows) < 2:
        return {}
    current = rows[-1]
    current_pages = {key: current[source] for key, source in (
        ("free", "free"), ("active", "active"), ("speculative", "specul"),
        ("inactive", "inactive"), ("throttled", "throttle"),
        ("wired", "wired"), ("purgeable", "prgable"))}
    pages_per_s = {key: current[source] / window_seconds for key, source in (
        ("pageins", "pageins"), ("pageouts", "pageout"),
        ("swapins", "swapins"), ("swapouts", "swapouts"))}
    return {"window_seconds": window_seconds, "page_size_bytes": page_size,
            "current_pages": current_pages, "pages_per_s": pages_per_s}


def parse_memory_pressure(text: str) -> dict:
    """Parse available pages and the system-wide free-memory percentage from memory_pressure -Q."""
    available_bytes = available_pages = page_size = free_pct = None
    for line in text.splitlines():
        if "pages with a page size of" in line:
            before, _, after = line.partition("(")
            parts = after.split()
            try:
                available_bytes = int(before.split()[-1])
                available_pages = int(parts[0])
                page_size = int(parts[-1].rstrip(")."))
            except (IndexError, ValueError):
                continue
        if "free percentage:" in line:
            value = line.partition(":")[2].strip().rstrip("%")
            try:
                free_pct = int(value)
            except ValueError:
                continue
    if (available_bytes is None or available_pages is None or page_size is None
            or free_pct is None or not 0 <= free_pct <= 100):
        return {}
    return {"available_bytes": available_bytes, "available_pages": available_pages,
            "page_size_bytes": page_size, "free_pct": free_pct}


def _load_probe(argv: list[str], timeout_s: float, runner: Runner | None, commands: _ProbeLog,
               warnings: list[str]) -> str:
    commands.invocations.append(argv[0])
    try:
        if runner is None:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            commands.pids.append(process.pid)
            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.communicate()
                raise
            result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        else:
            result = runner(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
            pid = getattr(result, "pid", None)
            if isinstance(pid, int) and pid > 0:
                commands.pids.append(pid)
    except (OSError, subprocess.TimeoutExpired) as exc:
        warnings.append(f"{argv[0]}: {type(exc).__name__}: {exc}")
        return ""
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        warnings.append(f"{argv[0]} exited {result.returncode}" + (f": {detail[-1][:160]}" if detail else ""))
    return result.stdout or ""


def _load_and_cpu(top: str) -> tuple[list[float] | None, dict[str, float]]:
    load: list[float] | None = None
    cpu: dict[str, float] = {}
    for line in top.splitlines():
        if line.startswith("Load Avg:"):
            values = line.partition(":")[2].split(",")
            try:
                load = [float(value.strip()) for value in values[:3]]
            except ValueError:
                load = None
        elif line.startswith("CPU usage:"):
            words = line.partition(":")[2].replace(",", "").split()
            for number, label in zip(words[::2], words[1::2]):
                if number.endswith("%"):
                    try:
                        cpu[label.lower()] = float(number[:-1])
                    except ValueError:
                        continue
    return load, cpu


def _tmux_owners(text: str) -> dict[int, str]:
    out = {}
    for line in text.splitlines():
        fields = line.split("|", 2)
        if len(fields) == 3:
            try:
                out[int(fields[0])] = f"tmux:{fields[1]}:{fields[2]}"
            except ValueError:
                continue
    return out


def _launchd_owners(text: str) -> dict[int, str]:
    out = {}
    for line in text.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[0].isdigit():
            out[int(fields[0])] = f"launchd:{fields[2]}"
    return out


def _omp_roots(processes: dict[int, Process]) -> dict[int, str]:
    roots = {}
    for proc in processes.values():
        tokens = proc.args.split()
        if proc.comm == "omp" or any(Path(token.strip("\"'" )).name == "omp" for token in tokens):
            roots[proc.pid] = f"omp:{proc.pid}"
    return roots


def _cwd_by_pid(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    pid: int | None = None
    for line in text.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
        elif line.startswith("n") and pid is not None:
            out[pid] = line[1:]
    return out


def _process_name(proc: Process) -> str:
    """Use argv[0] for display: macOS ps comm is fixed-width and truncates long paths."""
    argv0 = proc.args.split(None, 1)[0] if proc.args else proc.comm
    return Path(argv0.strip("\"'")).name or Path(proc.comm).name or proc.comm

def _owner_for(pid: int, processes: dict[int, Process], panes: dict[int, str],
               omp_roots: dict[int, str], launchd: dict[int, str]) -> str:
    seen = set()
    current = pid
    for _ in range(PARENT_DEPTH):
        if current in omp_roots:
            return omp_roots[current]
        if current in panes:
            return panes[current]
        if current in launchd:
            return launchd[current]
        if current <= 1 or current in seen:
            break
        seen.add(current)
        parent = processes.get(current)
        if parent is None:
            break
        current = parent.ppid
    proc = processes.get(pid)
    return f"unowned:{_process_name(proc) if proc else 'exited'}"


def _script_for(pid: int, processes: dict[int, Process]) -> str:
    current = processes.get(pid)
    if current is None:
        return "(parent exited)"
    current_pid = current.ppid
    seen = set()
    for _ in range(PARENT_DEPTH):
        if current_pid <= 1 or current_pid in seen:
            break
        seen.add(current_pid)
        parent = processes.get(current_pid)
        if parent is None:
            break
        for token in parent.args.split():
            path = token.strip("\"'")
            if Path(path).suffix.lower() in {".sh", ".py", ".js", ".mjs", ".cjs", ".ts"}:
                return path
        current_pid = parent.ppid
    return "(no script ancestor)"


def _probe_cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _ranked(rows: list[dict], key: str, limit: int = 15) -> list[dict]:
    return sorted(rows, key=lambda row: row.get(key, 0), reverse=True)[:limit]


def _add_owner_totals(by_owner: dict[str, dict], row: dict) -> None:
    owner = row["owner"]
    totals = by_owner.get(owner)
    if totals is None:
        totals = {"owner": owner, "cpu_pct": 0.0, "csw_s": 0.0, "sysbsd_s": 0.0,
                  "sysmach_s": 0.0, "rss_mib": 0.0, "processes": 0}
        by_owner[owner] = totals
    for key in ("cpu_pct", "csw_s", "sysbsd_s", "sysmach_s", "rss_mib"):
        totals[key] += row[key]
    totals["processes"] += 1


def _process_rows(processes: dict[int, Process], activity: dict[int, Activity],
                  owners: dict[int, str]) -> tuple[list[dict], dict[str, dict]]:
    by_owner: dict[str, dict] = {}
    rows = []
    for pid, proc in processes.items():
        stats = activity.get(pid)
        row = {"pid": pid, "command": _process_name(proc), "owner": owners[pid],
               "cpu_pct": stats.cpu_pct if stats else 0.0,
               "csw_s": stats.csw_s if stats else 0.0,
               "sysbsd_s": stats.sysbsd_s if stats else 0.0,
               "sysmach_s": stats.sysmach_s if stats else 0.0,
               "rss_mib": round(proc.rss_kib / 1024, 1)}
        rows.append(row)
        _add_owner_totals(by_owner, row)
    return rows, by_owner


def _omp_session_rows(omp_labels: dict[int, str], cwd_by_pid: dict[int, str],
                      process_rows: list[dict]) -> list[dict]:
    rows = []
    for pid, owner in sorted(omp_labels.items()):
        members = [row for row in process_rows if row["owner"] == owner]
        rows.append({"pid": pid, "cwd": cwd_by_pid.get(pid),
                     "cpu_pct": round(sum(row["cpu_pct"] for row in members), 1),
                     "rss_mib": round(sum(row["rss_mib"] for row in members), 1),
                     "processes": len(members)})
    return rows


def _spawner_rows(events: list[_TopProcess], processes: dict[int, Process],
                  owners: dict[int, str]) -> tuple[int, list[dict]]:
    counts: dict[tuple[str, str], int] = {}
    for event in events:
        if event.pid == os.getpid():
            continue
        if event.pid not in processes:
            continue
        key = (_script_for(event.pid, processes), owners.get(event.pid, "unowned"))
        counts[key] = counts.get(key, 0) + 1
    rows = [{"script": script, "owner": owner, "count": count}
            for (script, owner), count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
    return len(events), rows


def _top_rows(activity: dict[int, Activity], process_rows: list[dict],
              by_owner: dict[str, dict], owners: dict[int, str]) -> list[dict]:
    row_by_pid = {row["pid"]: row for row in process_rows}
    rows = []
    for pid, stats in activity.items():
        row = row_by_pid.get(pid)
        if row is None:
            command = stats.command.split(None, 1)[0] if stats.command else "(unknown)"
            row = {"pid": pid, "command": Path(command).name,
                   "owner": owners.get(pid, "exited during sample"),
                   "cpu_pct": stats.cpu_pct, "csw_s": stats.csw_s, "sysbsd_s": stats.sysbsd_s,
                   "sysmach_s": stats.sysmach_s, "rss_mib": 0.0}
            _add_owner_totals(by_owner, row)
        rows.append(row)
    return rows

def collect(seconds: int = DEFAULT_SECONDS, *, runner: Runner | None = None) -> dict:
    """Measure one process snapshot and one /usr/bin/top delta sample; all other probes are single-shot."""
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= MAX_SECONDS:
        raise LoadError(f"--seconds must be an integer from 1 to {MAX_SECONDS}")
    start_wall = time.monotonic()
    start_cpu = _probe_cpu_seconds()
    commands = _ProbeLog([], [])
    warnings: list[str] = []
    top_text = _load_probe(["/usr/bin/top", "-l", str(seconds + 1), "-s", "1", "-d", "-n",
                            str(TOP_PROCESS_LIMIT), "-stats",
                            "pid,ppid,cpu,csw,sysbsd,sysmach,th,command"], seconds * 3 + 10,
                           runner, commands, warnings)
    top_tables = _parse_top_tables(top_text)
    if len(top_tables) != seconds + 1:
        raise LoadError(f"/usr/bin/top returned {max(0, len(top_tables) - 1)} process intervals; expected {seconds}")
    activity = _aggregate_top_tables(top_tables, seconds)
    if not activity:
        raise LoadError("/usr/bin/top returned no process delta table")
    spawn_events = _top_spawn_events(top_tables)
    ps_text = _load_probe(["ps", "-Ao", "pid=,ppid=,etime=,time=,%cpu=,rss=,comm=,args="], 3,
                          runner, commands, warnings)
    if not ps_text:
        raise LoadError("ps returned no process snapshot")
    processes = parse_ps(ps_text)
    iostat_text = _load_probe(["iostat", "-d", "-w", str(IO_SAMPLE_SECONDS), "-c", "2"], 3,
                       runner, commands, warnings)
    vm_stat_text = _load_probe(["vm_stat", "-c", "2", str(IO_SAMPLE_SECONDS)], 3,
                        runner, commands, warnings)
    pressure_text = _load_probe(["memory_pressure", "-Q"], 2, runner, commands, warnings)
    io = parse_iostat(iostat_text, IO_SAMPLE_SECONDS)
    paging = parse_vm_stat(vm_stat_text, IO_SAMPLE_SECONDS)
    pressure = parse_memory_pressure(pressure_text)
    if not io:
        warnings.append("iostat: no disk interval sample")
    if not paging:
        warnings.append("vm_stat: no two-sample page data")
    if not pressure:
        warnings.append("memory_pressure: no pressure data")
    panes_text = _load_probe(["tmux", "list-panes", "-a", "-F", "#{pane_pid}|#{session_name}|#{pane_id}"], 2,
                      runner, commands, warnings)
    launchd_text = _load_probe(["launchctl", "list"], 2, runner, commands, warnings)
    panes = _tmux_owners(panes_text)
    launchd = _launchd_owners(launchd_text)
    omp_roots = _omp_roots(processes)
    cwd_by_pid = {}
    if omp_roots:
        pids = ",".join(str(pid) for pid in sorted(omp_roots))
        lsof_text = _load_probe(["lsof", "-a", "-d", "cwd", "-p", pids, "-Fpn"], 2,
                         runner, commands, warnings)
        cwd_by_pid = _cwd_by_pid(lsof_text)
    omp_labels = {pid: f"omp:{cwd_by_pid.get(pid) or 'pid ' + str(pid)} (pid {pid})" for pid in omp_roots}
    attribution_processes = dict(processes)
    for event in spawn_events:
        attribution_processes.setdefault(
            event.pid, Process(event.pid, event.ppid, 0.0, 0.0, event.cpu_pct, 0, event.command, event.command))
    owners = {pid: _owner_for(pid, attribution_processes, panes, omp_labels, launchd)
              for pid in attribution_processes}
    process_rows, by_owner = _process_rows(processes, activity, owners)
    session_rows = _omp_session_rows(omp_labels, cwd_by_pid, process_rows)
    observed_spawns, spawners = _spawner_rows(spawn_events, attribution_processes, owners)
    top_rows = _top_rows(activity, process_rows, by_owner, owners)
    load_avg, cpu = _load_and_cpu(top_text)
    elapsed = max(time.monotonic() - start_wall, 0.001)
    cpu_s = max(_probe_cpu_seconds() - start_cpu, 0.0)
    cpu_pct = 100 * cpu_s / elapsed
    return {"schema": "localbench.load/v1", "sample_seconds": seconds,
            "sampled_at": datetime.now(timezone.utc).isoformat(), "load_avg": load_avg,
            "cpu": cpu, "process_count": len(processes),
            "cpu_basis": {
                "system_percent": "machine-wide (last /usr/bin/top table)",
                "process_percent": "one logical core = 100%",
                "process_window_seconds": seconds,
                "process_interval_seconds": 1,
            },
            "top_cpu": _ranked(top_rows, "cpu_pct"),
            "top_csw_per_s": _ranked(top_rows, "csw_s"),
            "top_sysbsd_per_s": _ranked(top_rows, "sysbsd_s"),
            "top_sysmach_per_s": _ranked(top_rows, "sysmach_s"),
            "io": io,
            "paging": paging,
            "pressure": pressure,
            "spawns": {"observed_in_window": observed_spawns,
                       "method": ("new PIDs first seen across /usr/bin/top -d tables sampled every 1s; "
                                  "retained after exit; sub-second starts may be missed"),
                       "by_parent_script": spawners},
            "owners": _ranked(list(by_owner.values()), "cpu_pct"),
            "omp_sessions": session_rows,
            "probe_cost": {"wall_s": round(elapsed, 3), "cpu_s": round(cpu_s, 3),
                           "cpu_pct_one_core": round(cpu_pct, 3),
                           "cpu_pct_machine": round(cpu_pct / max(os.cpu_count() or 1, 1), 4),
                           "budget_cpu_basis": "machine", "spawns": len(commands.pids),
                           "probe_invocations": len(commands.invocations),
                           "spawn_method": "PIDs returned by successful probe process launches",
                           "budget_ok": cpu_pct / max(os.cpu_count() or 1, 1) < 2 and len(commands.pids) < 50},
            "warnings": warnings}


def render(report: dict) -> str:
    lines = [f"load ({report['sample_seconds']}s): loadavg={report['load_avg']} cpu={report['cpu']}",
             f"processes={report['process_count']} observed_spawns={report['spawns']['observed_in_window']} "
             "(new PIDs first seen in top's 1s tables; exits retained; sub-second starts may be missed)"]
    lines.append(
        f"CPU basis: system=machine-wide (last /usr/bin/top table); "
        f"process=one logical core = 100%; process intervals=1s over {report['sample_seconds']}s"
    )
    cost = report["probe_cost"]
    lines.append(f"probe cost: {cost['cpu_s']} CPU s / {cost['wall_s']} wall s, "
                 f"{cost['cpu_pct_one_core']}% one core, {cost['cpu_pct_machine']}% machine, "
                 f"{cost['spawns']} child starts/{cost['probe_invocations']} probe invocations, "
                 f"budget_ok={cost['budget_ok']} (<2% machine CPU and <50 child starts)")
    for title, key, field in (
        ("top CPU", "top_cpu", "cpu_pct"),
        ("top context switches per second", "top_csw_per_s", "csw_s"),
        ("top BSD syscalls per second", "top_sysbsd_per_s", "sysbsd_s"),
        ("top Mach syscalls per second", "top_sysmach_per_s", "sysmach_s"),
    ):
        lines.append(f"\n{title} (value, pid, process, owner)")
        for row in report[key][:10]:
            lines.append(f"{row[field]:8.2f}  {row['pid']:>6}  {row['command'][:24]:<24}  {row['owner']}")
    io = report["io"]
    lines.append(f"\ndisk I/O ({io.get('window_seconds', 'unknown')}s; device, MB per second, transfers per second, KB/transfer)")
    for row in io.get("devices", [])[:10]:
        lines.append(f"{row['device']:>6}  {row['mb_per_s']:8.2f}  {row['transfers_per_s']:10.1f}  {row['kb_per_transfer']:9.2f}")
    if not io:
        lines.append("  unavailable")
    paging = report["paging"]
    if paging:
        events = paging["pages_per_s"]
        pages = paging["current_pages"]
        lines.append(f"\npaging ({paging['window_seconds']}s): pageins={events['pageins']:.1f} per second "
                     f"pageouts={events['pageouts']:.1f} per second swapins={events['swapins']:.1f} per second "
                     f"swapouts={events['swapouts']:.1f} per second")
        lines.append(f"  current pages: free={pages['free']} active={pages['active']} "
                     f"inactive={pages['inactive']} wired={pages['wired']}")
    else:
        lines.append("\npaging: unavailable")
    pressure = report["pressure"]
    if pressure:
        lines.append(f"\nmemory pressure: free={pressure['free_pct']}% "
                     f"available={pressure['available_bytes']} B ({pressure['available_pages']} pages)")
    else:
        lines.append("\nmemory pressure: unavailable")
    lines.append("\nprocess starts first seen in top interval tables (count, parent script, owner; sub-interval starts may be missed)")
    for row in report["spawns"]["by_parent_script"][:10]:
        lines.append(f"{row['count']:6}  {row['script']}  {row['owner']}")
    lines.append("\nOMP session trees (CPU%, RSS MiB, processes, pid, cwd)")
    for row in report["omp_sessions"][:25]:
        lines.append(f"{row['cpu_pct']:7.1f}  {row['rss_mib']:9.1f}  {row['processes']:5}  {row['pid']:>6}  {row['cwd'] or '(cwd unavailable)'}")
    if report["warnings"]:
        lines.append("\nwarnings: " + "; ".join(report["warnings"]))
    return "\n".join(lines)
