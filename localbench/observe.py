"""Local-model use over time: a once-a-minute record of GPU time by process and model, models loaded on each local
server, and the processes connected to them — and a report over any window of it.

`localbench gpu` answers "what is using the GPU now"; this answers "what used it today, for how long, and who was
connected". The watcher reads ioreg, the servers' model lists, and lsof — no model is touched, so it may run during
measurements (the measurement law forbids local inference by monitors, not observation).

Store: runs/observe.db (stdlib sqlite3, WAL; one writer — see the fsqlite SURVEY row for why not fsqlite).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import sysstats
from .workloads import ROOT

DB = ROOT / "runs" / "observe.db"
SCHEMA = """
create table if not exists samples (t real primary key, window_s real, device_pct int, free_pct int, swap_mb real);
create table if not exists gpu (t real, pid int, name text, model text, pct real);
create table if not exists resident (t real, server text, model text);
create table if not exists clients (t real, pid int, name text, omp_profile text, cwd text, servers text);
create table if not exists traffic (t real, pid int, server text, up int, down int, resident text);
create index if not exists gpu_t on gpu(t);
create index if not exists clients_t on clients(t);
create index if not exists traffic_t on traffic(t);
"""


def connect(path: Path = DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.execute("pragma journal_mode=wal")
    con.executescript(SCHEMA)
    return con


def record(con: sqlite3.Connection, before: dict, t0: float, commands: dict[int, str],
           conn_before: dict | None = None) -> tuple[dict, float, dict]:
    """Write one sample covering (t0, now]; returns the GPU counters, the connection byte counters, and the time the
    next window starts from. Traffic needs the previous sample's counters, so a watch's first sample writes none.
    A traffic row keeps what its server had loaded then: bytes down while one model was resident are that model's."""
    after, conn_after, t1 = sysstats.gpu_time_by_pid(), sysstats.connection_bytes(), time.time()
    window = t1 - t0
    live = sysstats.live()
    clients = sysstats.inference_clients()
    traffic = [] if conn_before is None else [
        (t1, c["pid"], srv, t["up"], t["down"], json.dumps(live["resident"].get(srv)))
        for c in clients for srv, t in sysstats.traffic(conn_before, conn_after, c["conns"]).items()
        if t["up"] or t["down"]]
    with con:
        con.execute("insert into samples values (?, ?, ?, ?, ?)",
                    (t1, window, live.get("gpu_device_pct"), live.get("free_pct"), live.get("swap_used_mb")))
        con.executemany("insert into gpu values (?, ?, ?, ?, ?)",
                        [(t1, r["pid"], r["name"], r.get("model"), r["pct"])
                         for r in sysstats.gpu_share(before, after, window, commands)])
        con.executemany("insert into resident values (?, ?, ?)",
                        [(t1, srv, m) for srv, models in live["resident"].items() for m in (models or [])])
        con.executemany("insert into clients values (?, ?, ?, ?, ?, ?)",
                        [(t1, c["pid"], c["name"], c.get("omp_profile"), c["cwd"], json.dumps(c["servers"]))
                         for c in clients])
        con.executemany("insert into traffic values (?, ?, ?, ?, ?, ?)", traffic)
    return after, t1, conn_after


def watch(interval: float = 60.0, samples: int | None = None, path: Path = DB) -> None:
    """Record every `interval` seconds until interrupted (or `samples` times)."""
    con = connect(path)
    commands: dict[int, str] = {}
    before, conn, t0 = sysstats.gpu_time_by_pid(), sysstats.connection_bytes(), time.time()
    n = 0
    while samples is None or n < samples:
        time.sleep(max(0.0, t0 + interval - time.time()))
        before, t0, conn = record(con, before, t0, commands, conn)
        n += 1


def report(since_s: float, path: Path = DB) -> dict:
    """GPU-seconds by process/model, model residency, clients seen, and the bytes each client moved to and from each
    server over the last `since_s` seconds."""
    con = connect(path)
    t_min = time.time() - since_s
    covered = con.execute("select count(*), coalesce(sum(window_s), 0), min(t), max(t) from samples where t >= ?",
                          (t_min,)).fetchone()
    gpu_s: dict[tuple[str, str], float] = {}
    for name, model, s in con.execute(
            "select g.name, coalesce(g.model, ''), sum(g.pct / 100.0 * s.window_s) from gpu g join samples s "
            "on s.t = g.t where g.t >= ? group by 1, 2", (t_min,)):
        # Rows written before sysstats named ollama runners carry a (cut) blob path; name them now so old and new
        # samples of one model add up. A blob no manifest lists any more keeps its path.
        key = (name, ", ".join(sysstats.ollama_blob_names(model)) or model)
        gpu_s[key] = gpu_s.get(key, 0.0) + s
    gpu = sorted(((n, m, round(s, 1)) for (n, m), s in gpu_s.items()), key=lambda g: -g[2])
    resident = con.execute(
        "select r.server, r.model, round(sum(s.window_s), 0) from resident r join samples s on s.t = r.t "
        "where r.t >= ? group by 1, 2 order by 3 desc", (t_min,)).fetchall()
    clients = con.execute(
        "select coalesce(omp_profile, name), cwd, servers, count(*), min(t), max(t) from clients where t >= ? "
        "group by 1, 2, 3 order by 4 desc", (t_min,)).fetchall()
    used: dict[tuple, list[int]] = {}
    for who, cwd, server, res, up, down in con.execute(
            "select coalesce(c.omp_profile, c.name), c.cwd, tr.server, tr.resident, tr.up, tr.down from traffic tr "
            "left join clients c on c.t = tr.t and c.pid = tr.pid where tr.t >= ?", (t_min,)):
        models = json.loads(res) if res else None
        model = (models[0] if len(models) == 1 else "several: " + ", ".join(models)) if models else "none resident"
        v = used.setdefault((who, cwd, server, model), [0, 0])
        v[0] += up
        v[1] += down
    return {"samples": covered[0], "covered_s": round(covered[1]), "first": covered[2], "last": covered[3],
            "gpu": [{"process": n, "model": m, "gpu_s": s} for n, m, s in gpu],
            "resident": [{"server": srv, "model": m, "seconds": s} for srv, m, s in resident],
            "clients": [{"who": w, "cwd": c, "servers": json.loads(sv), "samples": k, "first": a, "last": b}
                        for w, c, sv, k, a, b in clients],
            "traffic": [{"who": w, "cwd": c, "server": srv, "while_resident": m, "up": u, "down": d}
                        for (w, c, srv, m), (u, d) in sorted(used.items(), key=lambda kv: -kv[1][1])]}
