"""localbench: is the local model fast on THIS machine, and did it get slower?

  localbench stats
  localbench run  ollama:qwen3.6:35b-mlx                      # measure + compare to its golden
  localbench aa   ollama:qwen3.6:35b-mlx --write-golden       # A/A pair -> banked receipt + golden
  localbench ab   ollama:qwen3.8:27b-mlx ollama:qwen3.6:35b-mlx --bank ab-incumbent-vs-moe
  localbench record --label lean ollama:qwen3.6:35b-mlx -- --no-skills ...

A backend spec is `ollama:<model>`, `mlx-serve:<model dir>` or `omlx:<model dir>` (the process is pinned to that model).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, NoReturn

from . import audit, backends, golden, memory, models, observe, park, quiet, render, smol, sysstats
from .backends import OMLX, MlxServe, Ollama, _first_line, _post, sha16, splash_pin
from .workloads import (
    AGENT_CONFIG,
    AGENT_DIR,
    CHILD_CONFIG,
    FIXTURES,
    MEM_CONFIG,
    MEM_ROUNDS,
    ROOT,
    TIERS,
    Ctx,
    Result,
    child_env,
    ensure_localbench_model,
    fixtures_sha,
    omp_bin,
)

RUNS = ROOT / "runs"
RECEIPTS = ROOT / "docs" / "evidence" / "receipts"
# Verbs that read nothing under ROOT: they run from any install. Every other verb needs a clone's data root. doctor
# reports a missing root itself, as one FAIL row.
ROOTLESS = frozenset({"stats", "memory", "keep", "pull", "create", "audit", "why", "validate", "doctor"})

EPILOG = """\
exit status:
  0    ok; every verdict a claim could rest on is sound
  1    unsound, regressed or refused: a MUST FAIL (not listed in DISCREPANCIES.md), a CONTENDED run, a golden row
       that REGRESSED / FAILed / is MISSING / GENERATION-MISMATCH / TOL-UNPROVEN, or a refusal (a run is alive,
       preflight, a download that does not fit); a claim must not rest on it
  2    usage error: a bad flag or argument value, or no data root (see LOCALBENCH_HOME)
  141  stdout was closed early (`localbench memory | head`)
Failures print to stderr; stdout carries only results (with --json, one JSON document). No color output.

environment:
  LOCALBENCH_HOME     the clone holding fixtures/, goldens/, runs/ (default: the checkout this install runs from)
  LOCALBENCH_HF_DIR   `pull hf:` download root (default ~/.cache/localbench/hf)
  LOCALBENCH_OMP, LOCALBENCH_MLX_SERVE   the omp / mlx-serve executables (default: PATH)
"""


def _fail(msg: str) -> int:
    """A failure or refusal: the message on stderr, exit 1."""
    print(msg, file=sys.stderr)
    return 1


def _usage(msg: str) -> NoReturn:
    """A bad argument value found after parsing: the message on stderr, exit 2 like argparse's own usage errors."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


class Step(NamedTuple):
    """One planned change of a state-changing verb: `action` is the line a dry run prints, `why` what --explain adds
    (what the action does and why it is needed)."""
    action: str
    why: str


class Mutation:
    """One invocation of a state-changing verb (park, smol revert, keep, ...). The handler computes its plan, the same
    steps for a dry run and a real run, and hands it to gate(): a dry run prints it (or, with --json, {verb, actions,
    would_refuse, noop}) and ends; a refusal or a no-op is written to the audit ledger and ends the command; anything
    else proceeds, and main() writes the outcome (done, or failed on a nonzero exit or an exception) when the handler
    returns. A dry run writes nothing, to the ledger or anywhere else."""

    def __init__(self, verb: str, argv: list[str], *, dry_run: bool = False, explain: bool = False,
                 as_json: bool = False, audited: bool = True):
        self.verb, self.argv, self.dry_run, self.explain, self.as_json = verb, argv, dry_run, explain, as_json
        self.audited = audited
        self.actions: list[str] = []
        self.detail: dict = {}
        self.outcome: str | None = None   # set by a handler for an outcome its exit code does not tell (refused)
        self.proceeded = False
        self.row_id: str | None = None

    def _print(self, steps: list[Step]) -> None:
        for s in steps:
            print(s.action)
            if self.explain:
                print(f"    why: {s.why}")

    def gate(self, steps: list[Step], refuse: str | None = None, noop: str | None = None) -> int | None:
        """None: go ahead and change things. Otherwise the exit code to return now: a dry run 0 (1 when the real
        command would refuse); a refusal 1; a no-op 0 (nothing to do is success, and says so)."""
        actions = [s.action for s in steps]
        if self.dry_run:
            if self.as_json:
                print(json.dumps({"verb": self.verb, "actions": actions, "would_refuse": refuse, "noop": noop},
                                 indent=2))
            else:
                self._print(steps)
                if noop and not refuse:
                    print(noop)
            if refuse:
                print(refuse, file=sys.stderr)
                return 1
            return 0
        if self.explain:
            self._print(steps)
        if refuse:
            self.record(actions, "refused", {"reason": refuse})
            return _fail(refuse)
        if noop:
            self.record([], "done", {"noop": noop})
            print(noop, file=sys.stderr if self.as_json else sys.stdout)
            return 0
        self.actions, self.proceeded = actions, True
        return None

    def record(self, actions: list[str], outcome: str, detail: dict) -> None:
        if self.audited and self.row_id is None:
            self.row_id = audit.record(self.verb, self.argv, actions, outcome, detail)

    def finish(self, rc: int | None, exc: BaseException | None = None) -> None:
        """The ledger row for a run that got past gate(), or that raised before reaching it. A usage error (exit 2)
        changed nothing and is not a mutation; neither did a dry run, even one whose planning raised."""
        if self.dry_run:
            return
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            rc, exc = 0, None
        if isinstance(exc, SystemExit) and exc.code == 2 and not self.proceeded:
            return
        if not self.proceeded and exc is None:
            return
        detail = {"rc": rc, **self.detail}
        if exc is not None:
            detail["error"] = str(exc.code) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
        self.record(self.actions, self.outcome or ("done" if rc == 0 and exc is None else "failed"), detail)


def _mut(args) -> Mutation:
    """The invocation's Mutation (main() attaches it); a handler called directly gets one that proceeds unrecorded."""
    return getattr(args, "mutation", None) or Mutation("direct", [], audited=False)


def _version() -> str:
    try:
        return importlib.metadata.version("localbench")
    except importlib.metadata.PackageNotFoundError:
        return "(not installed: `uv tool install -e .` in the clone)"


def _check_root(cmd: str) -> None:
    """A wheel or git+ install resolves ROOT to site-packages: no goldens, `parked: none`, every answer silently empty.
    Refuse instead, naming the two ways to point localbench at its data."""
    if cmd in ROOTLESS or (FIXTURES / "omp").is_dir():
        return
    _usage(f"localbench {cmd}: no data root at {ROOT} (no fixtures/omp there). localbench runs from a clone: "
           "`git clone <repo> && cd localbench && uv tool install -e .`, or set LOCALBENCH_HOME=<clone>")


def _rev() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         check=False)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "localbench"],
                           capture_output=True, text=True, check=False).stdout.strip()
    return (out.stdout.strip() or "uncommitted") + ("-dirty" if dirty else "")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def omp_pins(mem_config: Path = MEM_CONFIG) -> dict:
    binary = omp_bin()
    return {"omp_version": _first_line(binary, "--version").removeprefix("omp/").strip() or None,
            "omp_sha": sha16(binary), "omp_path": binary, "omp_child_config": sha16(str(CHILD_CONFIG)),
            "omp_mem_config": sha16(str(mem_config)), "omp_agent_config": sha16(str(AGENT_CONFIG))}


def run_pins(backend, model: str, host: dict, mem_config: Path = MEM_CONFIG) -> dict:
    pins = {**backend.pins(model), **omp_pins(mem_config), "fixtures_sha": fixtures_sha(),
            "macos_build": host["macos_build"], "host_id": host["host_id"]}
    return golden.attach_splash(pins, backend=backend.name, resident=sysstats.splash_resident(), splash=splash_pin())


def _overlay(path: str) -> Path:
    """argparse type for an omp config overlay: an existing file, absolute, so the default compares equal."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"no such overlay file: {path}")
    return p


def _exe_path(path: str) -> Path:
    """argparse type for an executable (omp, mlx-serve): an existing executable file, absolute."""
    p = Path(path).expanduser().absolute()
    if not (p.is_file() and os.access(p, os.X_OK)):
        raise argparse.ArgumentTypeError(f"not an executable file: {path}")
    return p


@contextlib.contextmanager
def _binaries_for_leg(overrides: dict[str, Path | None]):
    """Run one leg under other binaries, e.g. {"LOCALBENCH_OMP": <omp>, "LOCALBENCH_MLX_SERVE": <mlx-serve>}: omp_bin()
    and mlx_serve_bin() read these at every call, so the leg's launches and its start/end pins all name them. Each is
    restored afterwards, so the next A leg is back on the default. None leaves a variable alone."""
    old = {k: os.environ.get(k) for k, v in overrides.items() if v is not None}
    os.environ.update({k: str(v) for k, v in overrides.items() if v is not None})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _mem_rounds(text: str) -> int:
    """argparse type: a mem-tier round count of at least 1. Zero rounds would record a vacuous pass."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"mem rounds must be an integer, got {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"mem rounds must be at least 1, got {n}")
    return n


@contextlib.contextmanager
def open_backend(spec: str, server_args: tuple[str, ...] = ()):
    """Yield (backend, model) for `ollama:<model>`, `mlx-serve:<dir>` or `omlx:<dir>`; mlx-serve and oMLX are started
    and stopped here. One model at a time: a run refuses while another harness server is serving."""
    kind, _, rest = spec.partition(":")
    servers = {"mlx-serve": MlxServe("."), "omlx": OMLX(".")}
    busy = [f"{n} on :{s.port}" for n, s in servers.items() if n != kind and s.up()]
    if busy and rest:
        sys.exit(f"{', '.join(busy)} is serving during a {kind} run; stop it first (one model at a time)")
    if kind == "ollama" and rest:
        yield Ollama(), rest
    elif kind in ("mlx-serve", "omlx") and rest:
        unload_ollama()
        with (MlxServe if kind == "mlx-serve" else OMLX)(rest, server_args) as srv:
            yield srv, srv.model_id()
    else:
        _usage(f"bad backend spec {spec!r}: use ollama:<model>, mlx-serve:<model dir> or omlx:<model dir>")


def unload_ollama() -> list[str]:
    freed = []
    with contextlib.suppress(OSError):
        ol = Ollama()
        for m in ol.loaded():
            _post(ol.root + "/api/generate", {"model": m["name"], "keep_alive": 0})
            freed.append(m["name"])
    return freed


GPU_BUSY_MAX_PCT = 25.0


def _busy_check() -> tuple[dict, float | None, list[str]]:
    with sysstats.Sampler(0.5) as s:
        cpu = sysstats.cpu_busy_pct()
    summ = s.summary()
    problems = []
    if "gpu_device_pct" not in summ:
        problems.append("GPU signal unavailable (ioreg IOAccelerator utilization keys missing)")
    else:
        # Only another model running refuses a start: apps and the person at the keyboard are the condition the
        # measurement is taken under (the owner, 2026-09-24), recorded per leg as system.during.load, not a veto.
        busy = [r for r in summ.get("gpu_by_process", []) if r["pct"] > GPU_BUSY_MAX_PCT and sysstats.is_inference(r)]
        if busy:
            problems.append("another model is running: " + ", ".join(sysstats.proc_label(r) for r in busy)
                            + " (a run would be CONTENDED)")
    if cpu is None:
        problems.append("CPU signal unavailable (/usr/bin/top printed no CPU usage line)")
    if summ.get("swap_used_mb", {}).get("max", 0) > 1024:
        problems.append("swapping")
    with contextlib.suppress(OSError):
        loadable_smol = park.reachable_smol()
        if loadable_smol:
            problems.append(f"omp smol model(s) {loadable_smol} are loadable: every omp session's titles/memory will "
                            "contend mid-run; run `localbench park` first")
    live_smol = smol.load_state()
    if live_smol and smol.server_up(live_smol["port"]):
        problems.append(f"the smol server ({smol.PROVIDER}/{live_smol['model_id']} on :{live_smol['port']}) is up: every "
                        "omp session's smol work lands on it mid-run; run `localbench park` first")
    stuck_sessions = park.stuck_sessions(sysstats.omp_processes())
    if stuck_sessions:
        try:
            loadable = park.fallbacks()
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            loadable = [f"unknown ({exc})"[:160]]
        stuck = _stuck_problem(stuck_sessions, loadable)
        if stuck:
            problems.append(stuck)
    return summ, cpu, problems


def _stuck_problem(stuck: list[dict], loadable: list[str]) -> str | None:
    """A preflight problem naming the omp sessions that started while a smol model was parked, while a model omp
    could have handed their smol role (`park.fallbacks`) is still loadable. Idle at preflight, such a session still
    calls it mid-run (2026-09-23: qwen3.8-uncensored voided an A/B and blocked a re-bank for 30 min). `localbench
    park` parks the fallbacks, after which the sessions' calls fail instead; restarting them is the user's call."""
    if not stuck or not loadable:
        return None
    who = ", ".join(f"pid {s['pid']} ({s['cwd']})" for s in stuck)
    return (f"omp sessions that started while a smol model was parked may load {', '.join(loadable)} mid-run: "
            f"{who}; `localbench park` parks it too, or restart them (a run would be CONTENDED)")


def preflight(allow_busy: bool, wait_idle_s: float = 0, emit=None) -> dict:
    """Refuse to measure while another model can run (a loadable smol model, a stuck session's fallback, another
    model's runner busy), while swapping, or when the GPU/CPU signal cannot be read. A busy machine is not refused:
    apps and user activity are the measured condition (system.during.load). load1 is recorded, not judged.
    With wait_idle_s, re-check every 30 s until the machine is idle or the wait runs out; the verdict is the
    same gate, applied to the state the run actually starts in (other agents share this host)."""
    t0 = time.time()
    while True:
        summ, cpu, problems = _busy_check()
        if not problems or allow_busy or time.time() - t0 >= wait_idle_s:
            break
        if emit:
            emit({"event": "preflight_wait", "problems": problems})
        time.sleep(30)
    if problems and not allow_busy:
        sys.exit("preflight refused: " + "; ".join(problems) + "  (rerun with --allow-busy to measure anyway; "
                 "such a run is non-proof)")
    return {"idle_check": summ, "cpu_busy_pct": cpu, "problems": problems, "allow_busy": allow_busy,
            "waited_s": round(time.time() - t0, 1)}


def _emitter(progress: Path):
    """Events go to progress.jsonl whole. The terminal, and any agent reading it, gets one `event k=v` line each
    (render.event_line: every field, rounded, free text clipped with a marker). Until 2026-09-23 this echoed JSON cut
    at 300 characters: invalid JSON for long events, fields past the cut silently lost, 16-digit floats."""
    print(f"  events: {progress.relative_to(ROOT)} (full precision; the lines below round and clip)", flush=True)

    def emit(ev: dict) -> None:
        ev = {"t": round(time.time(), 3), **ev}
        with progress.open("a") as fh:
            fh.write(json.dumps(ev, default=str) + "\n")
        print("  " + render.event_line(ev), flush=True)
    return emit


def execute(backend, model: str, *, tiers: list[str], repeats: int, allow_busy: bool, purge: bool,
            label: str = "run", wait_idle_s: float = 0, mem_config: Path = MEM_CONFIG,
            mem_rounds: int = MEM_ROUNDS) -> dict:
    """One measurement of one model: preflight, isolate, tiers under the samplers, summary on disk."""
    stamp = _stamp()
    run_dir = RUNS / f"{stamp}__{label}__{backend.name}__{golden.slug(model)}"
    run_dir.mkdir(parents=True)
    latest = RUNS / "LATEST"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    emit = _emitter(run_dir / "progress.jsonl")
    emit({"event": "start", "backend": backend.name, "model": model, "tiers": tiers, "label": label})

    before = sysstats.snapshot()
    pre = preflight(allow_busy, wait_idle_s, emit)
    freed = unload_ollama() if backend.name != "ollama" else []
    purged = sysstats.purge_file_cache() if purge else None
    evicted = backend.isolate(model)
    fp = backend.fingerprint(model)
    pins = run_pins(backend, model, before["host"], mem_config)
    emit({"event": "isolated", "evicted": evicted, "freed": freed, "purged": purged, "fingerprint": fp})

    ctx = Ctx(backend=backend, model=model, repeats=repeats, run_dir=run_dir, emit=emit, pins=pins,
              loaded_context=fp.get("loaded_context"), mem_config=mem_config, mem_rounds=mem_rounds)
    results: list[Result] = []
    def on_contention(ev):
        emit({"event": "contention", **ev})
    with sysstats.Sampler(1.0, target=(backend.name, model), on_contention=on_contention,
                          gpu_foreign_max_pct=GPU_BUSY_MAX_PCT) as smp, \
            sysstats.PowerSampler(1000) as pwr, sysstats.CpuSampler(2) as cpu:
        for tier in tiers:
            emit({"event": "tier", "tier": tier})
            results += TIERS[tier](ctx)
    after = sysstats.snapshot()
    # A pin that moved during the run (omp was upgraded in place mid-campaign on 2026-09-23) means part of the
    # run measured another generation; the start pins would be a false label for it.
    pins_changed = golden.pin_diff(pins, run_pins(backend, model, after["host"], mem_config))

    metrics, conformance = golden.flatten(results)
    listed = golden.listed_discrepancies(backend.name, model)
    must_fail = sorted(c for c, e in conformance.items()
                       if e["level"] == "MUST" and e["verdict"] == "FAIL" and c not in listed)
    summary = {
        "provenance": {"pins": pins, "fingerprint": fp, "localbench_rev": _rev(), "created": stamp,
                       "tiers": tiers, "repeats": repeats, "label": label},
        "verdicts": {"contended": bool(smp.contention), "must_fail": must_fail,
                     "preflight_problems": pre["problems"], "allow_busy": allow_busy, "pins_changed": pins_changed},
        "metrics": metrics, "conformance": conformance,
        "results": [r.__dict__ for r in results],
        "system": {"before": before, "after": after, "preflight": pre, "during": smp.summary(),
                   "contention": smp.contention, "power": pwr.summary(), "cpu": cpu.summary()},
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in ctx.samples))
    # 1 Hz machine series (GPU, memory, load, resident models on every local server): the contention evidence.
    (run_dir / "sampler.jsonl").write_text("".join(json.dumps(s) + "\n" for s in smp.series))
    emit({"event": "done", "contended": bool(smp.contention), "must_fail": must_fail})
    return summary


def unsound(summary: dict) -> list[str]:
    v = summary["verdicts"]
    reasons = []
    if v["contended"]:
        reasons.append("CONTENDED: another model was resident or running during the run (system.contention)")
    if v["must_fail"]:
        reasons.append(f"MUST FAIL: {', '.join(v['must_fail'])}")
    if v["preflight_problems"]:
        reasons.append(f"preflight: {'; '.join(v['preflight_problems'])}")
    if v.get("pins_changed"):
        reasons.append(f"PINS CHANGED mid-run (another generation measured part of it): {v['pins_changed']}")
    return reasons


def _receipt_path(name: str) -> Path:
    return RECEIPTS / f"{golden.slug(name)}.json"


def _receipt_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str) + "\n"


def _bank(name: str, payload: dict) -> Path:
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    path = _receipt_path(name)
    path.write_text(_receipt_text(payload))
    return path


# Tiers whose per-case details stay in runs/ only. Every other tier's details are banked, so a new tier is
# auditable from its receipt by default (think was dropped for a day because the kept set was a hand list).
DETAILS_NOT_BANKED = frozenset({"micro", "conf", "replay"})


def _receipt_view(summary: dict) -> dict:
    """What a banked receipt keeps: everything except raw per-sample rows (those stay in runs/), plus the details of
    every tier outside DETAILS_NOT_BANKED (e2e/rel call splits and wrong answers verbatim, mem recall attempts, sess
    per-turn rows and memory-LLM overlap, think answers and cut-offs) that a reader needs to audit a rate, accuracy
    or wall-time row. mem/sess details were dropped until 2026-09-23, think until 2026-09-24."""
    details = {r["case"]: r["detail"] for r in summary.get("results", []) if r["tier"] not in DETAILS_NOT_BANKED}
    return {k: summary[k] for k in ("provenance", "verdicts", "metrics", "conformance", "run_dir")} | {
        "details": details,
        "system": {k: summary["system"].get(k) for k in ("preflight", "during", "contention", "power", "cpu")} | {
            "before_live": summary["system"]["before"]["live"], "host": summary["system"]["before"]["host"]}}


def cmd_stats(_args) -> int:
    snap = sysstats.snapshot()
    snap["powermetrics"] = sysstats.powermetrics_available()
    print(json.dumps(snap, indent=2))
    return 0


def golden_states(host: dict) -> list[dict]:
    """Per golden on this host: {golden, unavailable: why} when its current pins cannot be read or its model is not
    present, else {golden, tiers: [{state: CURRENT | GENERATION-MISMATCH, tiers, moved}]} grouped by the pins that
    moved."""
    goldens = []
    for path in sorted((golden.GOLDENS / host["host_id"]).glob("*.json")):
        g = golden.load(path)
        gp = g["pins"]
        row: dict = {"golden": path.name}
        goldens.append(row)
        try:
            if gp["backend"] == "ollama":
                backend = Ollama()
            else:
                receipt = json.loads((ROOT / g["aa_receipt"]).read_text())
                backend = MlxServe(receipt["runs"][0]["provenance"]["fingerprint"]["model_dir"])
            pins = run_pins(backend, gp["model"], host)
        except (OSError, KeyError, ValueError) as exc:
            row["unavailable"] = f"cannot read current pins: {exc}"[:200]
            continue
        if pins.get("model_digest") is None:
            hint = " (a parked name: exists only during `localbench park`)" if gp["model"].startswith(park.PREFIX) else ""
            row["unavailable"] = f"model {gp['model']} not present on {gp['backend']}{hint}"
            continue
        by_diff: dict[str, list[str]] = {}
        for tier in golden.tiers_of(g):
            by_diff.setdefault(json.dumps(golden.tier_diff(g, tier, pins)), []).append(tier)
        row["tiers"] = [{"state": "GENERATION-MISMATCH" if diff != "{}" else "CURRENT", "tiers": tiers,
                         "moved": json.loads(diff)} for diff, tiers in by_diff.items()]
    return goldens


def status_report() -> dict:
    """The facts `localbench status` shows, as data: per golden either why it is UNAVAILABLE or its tiers grouped by
    the pins that moved (CURRENT when none did), replay fixtures against the running omp, park and quiet state,
    ollama residency and auto-update, the smol server, sessions started while parked, and GPU users over 3 s."""
    host = sysstats.host()
    goldens = golden_states(host)
    running = omp_pins()
    running_gen = (running["omp_version"], running["omp_sha"], running["omp_child_config"])
    fixtures = []
    for meta_path in sorted((FIXTURES / "omp").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        gen = (meta.get("omp_version"), meta.get("omp_sha"), (meta.get("child_config") or {}).get("sha16"))
        fixtures.append({"fixture": meta_path.name.removesuffix(".meta.json"), "omp_version": gen[0], "omp_sha": gen[1],
                         "prompt_tokens": meta.get("prompt_tokens"), "matches_running_omp": gen == running_gen})
    residents = sysstats.ollama_residents(timeout=10.0)
    live = smol.load_state()
    smol_view = None
    if live:
        up = smol.server_up(live["port"])
        smol_view = {"selector": live["selector"], "port": live["port"],
                     "state": "up" if up else "parked" if _smol_parked() else "down"}
    before, t0 = sysstats.gpu_time_by_pid(), time.time()
    time.sleep(3)
    return {"host_id": host["host_id"], "goldens": goldens,
            "running_omp": {"omp_version": running_gen[0], "omp_sha": running_gen[1]}, "fixtures": fixtures,
            "parked": json.loads(park.STATE.read_text()) if park.STATE.exists() else [],
            "ollama_loaded": None if residents is None else [{"model": m, "until": u} for m, u in residents],
            "ollama_auto_update": sysstats.ollama_auto_update(), "smol": smol_view,
            "quiet_paused": quiet.paused(), "started_while_parked": park.stuck_sessions(sysstats.omp_processes()),
            "gpu_last_3s": sysstats.gpu_share(before, sysstats.gpu_time_by_pid(), time.time() - t0, min_pct=5.0)}


def cmd_status(args) -> int:
    """Which configs on this host have a live regression gate, per tier: each golden tier against the pins that tier
    depends on now (CURRENT, or GENERATION-MISMATCH naming the moved pins; UNAVAILABLE when the model is not present,
    e.g. a parked name while unparked), whether the replay fixtures still match the running omp, the park state, and
    who used the GPU in the last 3 s. Exit 1 while any tier is GENERATION-MISMATCH: re-bank just those tiers with
    `localbench aa <spec> --tiers <tiers> --write-golden`."""
    st = status_report()
    rc = 1 if any(t["state"] == "GENERATION-MISMATCH" for g in st["goldens"] for t in g.get("tiers", [])) else 0
    if args.json:
        print(json.dumps(st, indent=2, default=str))
        return rc
    print(f"goldens ({st['host_id']}):")
    for g in st["goldens"]:
        print(f"  {g['golden']}")
        if "unavailable" in g:
            print(f"    UNAVAILABLE         {g['unavailable']}")
        for t in g.get("tiers", []):
            print(f"    {t['state']:<19} {', '.join(t['tiers'])}" + (f"  {json.dumps(t['moved'])}" if t["moved"] else ""))
    run = st["running_omp"]
    for f in st["fixtures"]:
        fresh = "matches the running omp" if f["matches_running_omp"] else (
            f"running omp is {run['omp_version']} ({run['omp_sha']}): replay still measures the recorded request; "
            "`localbench record` re-records it (a new replay generation)")
        print(f"fixture {f['fixture']}: recorded under omp {f['omp_version']} ({f['omp_sha']}), "
              f"{f['prompt_tokens']} prompt tokens — {fresh}")
    print("parked: " + (", ".join(f"{p['name']} as {p['parked_as']}" for p in st["parked"]) or "none"))
    loaded = st["ollama_loaded"]
    print("loaded on ollama: " + (
        "unknown: /api/ps did not answer within 10 s (ollama stalls it while loading a model); rerun `localbench status`"
        if loaded is None else ", ".join(f"{m['model']} (until {m['until']})" for m in loaded) or "none"))
    auto = st["ollama_auto_update"]
    print("ollama app auto-update: " + ("unknown (no Ollama.app settings database)" if auto is None else
                                        "ON: the app can install a new ollama under every golden" if auto else "off"))
    if st["smol"]:
        s = st["smol"]
        print(f"smol: {s['selector']} on :{s['port']} " + {
            "up": "up", "parked": "parked (stopped by `localbench park`)",
            "down": "DOWN: every omp session's smol/memory calls fail; `localbench smol start`"}[s["state"]])
    held = st["quiet_paused"]
    if held:
        print(f"paused by `localbench quiet`: {len(held)} process(es) of omp's managed browser "
              f"(pids {', '.join(str(r['pid']) for r in held)}); resume with `localbench quiet --resume`")
    for line in _stuck_lines(st["started_while_parked"]):
        print(line)
    print("GPU, last 3 s: " + (", ".join(sysstats.proc_label(r) for r in st["gpu_last_3s"]) or "no process above 5%"))
    return rc


def cmd_gpu(args) -> int:
    """Who is using the GPU, and who can send it inference work: per-process GPU share over a window, models
    resident on each local server, processes connected to those servers, and, for each omp client, the features
    its profile routes to a local model (from omp's own resolved settings)."""
    before, conn0, t0 = sysstats.gpu_time_by_pid(), sysstats.connection_bytes(), time.time()
    time.sleep(args.seconds)
    after, conn1 = sysstats.gpu_time_by_pid(), sysstats.connection_bytes()
    report = {"window_s": args.seconds,
              "gpu_by_process": sysstats.gpu_share(before, after, time.time() - t0),
              "device": sysstats.gpu_utilization(), "resident": sysstats.resident_models(),
              "clients": sysstats.inference_clients()}
    for c in report["clients"]:
        c["traffic"] = sysstats.traffic(conn0, conn1, c["conns"])
    for s in park.stuck_sessions(report["clients"]):
        next(c for c in report["clients"] if c["pid"] == s["pid"])["started_while_parked"] = s["window"]
    routes: dict[str, dict] = {}
    for c in report["clients"]:
        if "omp_profile" in c:
            prof = c["omp_profile"]
            routes.setdefault(prof, park.local_routes(prof))
            c["local_routes"] = routes[prof]
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"GPU by process over {args.seconds:g} s (% of wall time the process kept the GPU busy; a runner with several "
          f"queues can pass 100%; device {report['device'].get('device_pct')}% busy now):")
    for r in report["gpu_by_process"] or [{"pct": 0, "pid": "-", "name": "(nothing above 0.5%)", "cmd": ""}]:
        what = f"model {r['model']}" if r.get("model") else r["cmd"][:90]
        print(f"  {r['pct']:6.1f}%  pid {r['pid']:<6} {r['name']:<18} {what}")
    print("Resident models: " + "; ".join(f"{srv}: {', '.join(m) if m else '(none)' if m is not None else '(no answer)'}"
                                          for srv, m in report["resident"].items()))
    print(f"Clients of local inference servers (open connections; bytes sent up / received down in the {args.seconds:g} s):")
    for c in report["clients"]:
        if c.get("agent_dir"):
            who = f"omp agent_dir={c['agent_dir']}"
            if "omp_profile" in c:
                who += f" profile={c['omp_profile']}"
        elif "omp_profile" in c:
            who = f"omp profile={c['omp_profile']}"
        else:
            who = c["cmd"][:70]
        used = "; ".join(f"{srv} {_size(t['up'])} up / {_size(t['down'])} down"
                         for srv, t in c["traffic"].items() if t["up"] or t["down"])
        stuck = "  STARTED WHILE PARKED: restart it" if c.get("started_while_parked") else ""
        print(f"  pid {c['pid']:<6} {who}  cwd={c['cwd']}  -> {c['servers']}  {used or 'idle'}{stuck}")
        for feature, target in c.get("local_routes", {}).items():
            print(f"      {feature}: {target}")
    return 0


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB" if n >= 1e3 else f"{n} B"


def cmd_watch(args) -> int:
    """Record local-model use once per `--interval` seconds into runs/observe.db until interrupted."""
    print(f"recording every {args.interval:g} s into {observe.DB} (Ctrl-C to stop)", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        observe.watch(args.interval, args.samples)
    return 0


def _since(text: str) -> float:
    """argparse type for `report --since`: seconds, or a number with m, h or d."""
    unit = {"m": 60, "h": 3600, "d": 86400}.get(text[-1:])
    try:
        seconds = float(text[:-1] if unit else text) * (unit or 1)
    except ValueError:
        raise argparse.ArgumentTypeError(f"window {text!r}: use 90m, 24h, 7d (or plain seconds)") from None
    if not seconds > 0:
        raise argparse.ArgumentTypeError(f"window {text!r}: must be longer than zero, e.g. 90m, 24h, 7d")
    return seconds


def cmd_report(args) -> int:
    """What used local models over `--since` (e.g. 90m, 24h, 7d): GPU-seconds by process/model, residency, clients.
    A window with no samples is an empty answer, not an error: exit 0, and the same shape under --json."""
    r = observe.report(args.since)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    if not r["samples"]:
        print(f"no samples in the last {args.since / 3600:g} h; `localbench watch` records them", file=sys.stderr)
        return 0
    span = time.strftime("%m-%d %H:%M", time.localtime(r["first"])) + " – " + \
        time.strftime("%m-%d %H:%M", time.localtime(r["last"]))
    print(f"{r['samples']} samples, {r['covered_s'] / 3600:.2f} h covered ({span})")
    print("GPU time by process (GPU-seconds; share of covered time):")
    for g in r["gpu"][:12]:
        who = f"{g['process']} ({g['model']})" if g["model"] else g["process"]
        print(f"  {g['gpu_s']:>9.1f}  {100 * g['gpu_s'] / max(r['covered_s'], 1):5.1f}%  {who}")
    print("Models loaded (seconds resident):")
    for m in r["resident"]:
        print(f"  {m['seconds']:>9.0f}  {m['server']}: {m['model']}")
    print("Connected to a local inference server (samples seen):")
    for c in r["clients"][:15]:
        print(f"  {c['samples']:>5}  {c['who']}  cwd={c['cwd']}  {c['servers']}")
    print("Traffic by client (down = responses, i.e. generated tokens; up = requests; model = what that server had "
          "loaded then):")
    for t in r["traffic"][:15]:
        print(f"  {_size(t['down']):>9} down {_size(t['up']):>9} up  {t['who']}  cwd={t['cwd']}  "
              f"-> {t['server']} ({t['while_resident']})")
    if not r["traffic"]:
        print("  none recorded (traffic is sampled since this watcher version; restart `localbench watch`)")
    return 0


def cmd_models(args) -> int:
    """Local models: installed on ollama, mlx-serve and Splash, plus the ones omp runs on the CPU itself (tiny models,
    mnemopi's embedding model); whether each is the newest build of its source, which omp profiles/features route to
    it, and what appeared upstream in the last `--days` days."""
    installed = models.ollama_models() + models.mlx_models()
    cpu = models.omp_cpu_models()
    names = models.profiles()
    uses = models.routes_by_model(names)
    for m in installed + cpu:
        m["routes"] = uses.get(m.get("source", m["name"]), {})
    new = models.releases(args.days, installed)
    if args.json:
        # Profiles once; per model {feature: [profiles]} (was a flat "profile: feature" string per pair, which repeated
        # every feature name once per profile: 3.5k tokens, audit 2026-09-23).
        print(json.dumps({"profiles": names, "installed": installed, "omp_cpu": cpu, "releases": new},
                         separators=(",", ":")))
        return 0
    print(f"omp profiles ({len(names)}): {', '.join(names)}")
    for m in installed + cpu:
        label = m["name"] + (f"  [parked: {m['source']}]" if m.get("parked") else "")
        print(f"{m['server']:<9} {label:<48} {m.get('digest', m.get('upstream_sha', '')):<12} {m['gb']:>6.1f} GB  "
              f"{m['freshness']}")
        for feature, who in m["routes"].items():
            print(f"{'':12}{feature}: {render.members(who, names, 'profiles')}")
    print(f"\nNew or updated upstream in the last {args.days} days (publishers and families in use):")
    for r in new:
        print(f"  {r['modified']}  {r['id']}  ({r['task'] or '-'})")
    return 0


def cmd_memory(args) -> int:
    """omp's memory store: every mnemopi bank with the cwd(s) it serves, rows, size, last write, and whether
    localbench made it. `--prune` deletes the harness-made banks (benchmark and probe turns); with nothing to prune it
    says so and lists the banks as usual."""
    rows = memory.banks()
    if args.prune:
        doomed = [b for b in rows if b["harness"]]
        steps = [Step(f"delete bank {b['bank']} ({b['rows']} rows, {b['bytes'] / 1e6:.1f} MB) in {memory.BANKS}",
                      "every cwd it serves is a localbench child's or probe's (or it has none and a harness name); no "
                      "user session reads it, and it keeps growing with each run") for b in doomed]
        rc = _mut(args).gate(steps, noop=None if doomed else "no harness bank to prune")
        if rc is not None and (args.dry_run or rc):
            return rc
        if doomed:
            removed = memory.remove_banks([b["bank"] for b in doomed])
            print(f"removed {len(removed)} harness bank(s): {', '.join(removed)}",
                  file=sys.stderr if args.json else sys.stdout)
            rows = memory.banks()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{len(rows)} bank(s) in {memory.BANKS}:")
    for b in rows:
        where = ", ".join(b["cwds"]) or b.get("error") or "(no cwd recorded)"
        tag = "  [localbench]" if b["harness"] else ""
        print(f"  {b['bank']:<40} {b['rows']:>6} rows {b['bytes'] / 1e6:>8.1f} MB  {b['modified']}  {where}{tag}")
    return 0


def _park_step(entry: dict) -> Step:
    if entry.get("kind") == park.SMOL_SERVER:
        return Step(f"stop the smol server {entry['name']}",
                    "omp sessions' smol and memory calls go to this server's model on the GPU; stopped, they fail at "
                    "request time instead of competing with the run. `localbench unpark` restarts it")
    why = ("an omp profile's smol role (and mnemopi memory through it) names this model, so any omp session would load "
           "it on the GPU mid-run" if entry["role"] == "smol" else
           "omp's resolver hands a parked smol role to this installed model, and a session started meanwhile keeps it "
           "after unpark")
    return Step(f"park ollama {entry['name']} as {entry['parked_as']} ({entry['role']}, digest "
                f"{entry['digest'].removeprefix('sha256:')[:12]})",
                why + "; the name is copied to the parked name (shared blobs, same digest), unloaded and deleted, so "
                "those calls fail instead. `localbench unpark` restores the name")


def cmd_park(args) -> int:
    """Park every omp smol model (and what omp would hand the role instead), and stop the dedicated smol server, for
    a test window. Parking what is parked already is a no-op."""
    if args.json and not (args.status or args.dry_run):
        _usage("park --json: `localbench park --status` prints park state as JSON, `park --dry-run --json` the plan; "
               "plain `park` parks")
    if args.status:
        print(json.dumps({"smol_targets": park.smol_targets(), "loadable": park.reachable_smol(),
                          "fallbacks": park.fallbacks(), "parked": park.parked_now()}, indent=2))
        return 0
    plan = park.plan_park()
    already = [p["name"] for p in park.parked_now()]
    noop = None if plan else "nothing to park: " + (
        f"already parked: {', '.join(already)}; restore with: localbench unpark" if already else
        "no smol target or fallback is installed under its own name, and no smol server is up")
    if (rc := _mut(args).gate([_park_step(e) for e in plan], noop=noop)) is not None:
        return rc
    park.park(plan=plan)
    for p in plan:
        if p.get("kind") == park.SMOL_SERVER:
            print(f"stopped the smol server ({p['name']}); smol/memory calls fail until `localbench unpark`")
            continue
        why = "omp's fallback for a parked smol model" if p["role"] == "fallback" else "smol/memory calls to it now fail"
        print(f"parked {p['name']} as {p['parked_as']} (digest {p['digest'][:12]}); {why}")
    print("restore with: localbench unpark")
    return 0


def cmd_unpark(args) -> int:
    """Restore parked models, then refresh every omp profile's ollama catalog and read it back: exit 1 when a
    profile still does not list a restored tag (its smol calls would fail with 'Model not found'). Nothing parked is
    a no-op."""
    parked = park.parked_now()
    steps = [Step(f"restart the smol server {p['name']}", "park stopped it; smol and memory calls fail until it serves")
             if p.get("kind") == park.SMOL_SERVER else
             Step(f"restore ollama {p['name']} from {p['parked_as']} (digest {p['digest'].removeprefix('sha256:')[:12]})",
                  "copies the parked name back (same digest, checked), then deletes the parked copy; omp's smol role "
                  "resolves to it again") for p in parked]
    ollama_names = [p["name"] for p in parked if p.get("kind") != park.SMOL_SERVER]
    if ollama_names:
        steps.append(Step("refresh every omp profile's ollama catalog and read it back",
                          "omp caches its ollama model list per profile for 24 h; a list taken during the park lacks "
                          "the restored tags, and their smol calls would fail with 'Model not found'"))
    if (rc := _mut(args).gate(steps, noop=None if parked else "nothing parked; nothing to restore")) is not None:
        return rc
    restored = park.unpark()
    for p in restored:
        print(f"restarted the smol server ({p['name']})" if p.get("kind") == park.SMOL_SERVER
              else f"restored {p['name']} (digest {p['digest'][:12]})")
    rc = 0
    ollama_names = [p["name"] for p in restored if p.get("kind") != park.SMOL_SERVER]
    if ollama_names:
        for profile, gone in park.refresh_catalogs(ollama_names, models.profiles()).items():
            print(f"omp catalog {profile}: " + (f"still missing {', '.join(gone)}" if gone else "lists every restored tag"))
            rc = rc or (1 if gone else 0)
    for line in _stuck_lines(park.stuck_sessions(sysstats.omp_processes())):
        print(line)
    return rc


def _stuck_lines(stuck: list[dict]) -> list[str]:
    """One line per omp session that started while a smol model was parked. Its smol role resolved, by omp's fuzzy
    match, to another local model it keeps after unpark (ledger rows on park; 2026-09-23: two proj-c sessions
    kept qwen3.8-uncensored resident for hours). Restarting the session is the user's call."""
    def hm(t: float) -> str:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(t))
    return [f"started while parked: pid {s['pid']} cwd={s['cwd']} at {hm(s['started'])} (park {hm(s['window'][0])} – "
            f"{hm(s['window'][1])}); its smol role may still point at another local model: restart that session"
            for s in stuck]


RUN_PATTERN = "localbench (aa|run|ab|record)"
KEEP_ALIASES = {"forever": -1, "unload": 0, "0": 0}


def _run_alive() -> bool:
    """A measurement is in progress (loading a model or downloading one now would perturb it)."""
    return subprocess.run(["pgrep", "-f", RUN_PATTERN], capture_output=True, check=False).returncode == 0


def _ollama_model(spec: str) -> str:
    backend, _, model = spec.partition(":")
    if backend != "ollama" or not model:
        _usage(f"{spec}: this manages ollama models; give ollama:<name>")
    return model


def cmd_keep(args) -> int:
    """Set how long ollama keeps a model loaded: `forever`, a duration (`30m`, `2h`), or `0`/`unload`; loads it if
    needed. A run's isolate reloads the model under test with ollama's default keep-alive, so a model the user keeps
    resident needs this afterwards. Refused while a run is alive: loading a model mid-run is contention. Keeping
    forever what is kept forever, or unloading what is not loaded, is a no-op."""
    model = _ollama_model(args.spec)
    keep = KEEP_ALIASES.get(args.duration, args.duration)
    refuse = "a localbench run is alive; set keep-alive after it ends" if _run_alive() else None
    noop = None
    residents = None if refuse else sysstats.ollama_residents(Ollama().root, timeout=10.0)
    if residents is not None:
        until = dict(residents).get(model)
        if keep == 0 and until is None:
            noop = f"{model}: not loaded; nothing to unload"
        elif keep == -1 and until == "forever":   # sysstats.keep_until's reading of keep_alive -1
            noop = f"{model}: already loaded until forever"
    step = (Step(f"unload ollama {model} (keep_alive 0)", "frees its memory and GPU now; the next request reloads it")
            if keep == 0 else
            Step(f"set ollama {model} keep_alive {args.duration} (loads it if it is not loaded)",
                 "a run's isolate reloads the model under test with ollama's default keep-alive (5 min), so a model "
                 "kept resident needs this afterwards"))
    if (rc := _mut(args).gate([step], refuse=refuse, noop=noop)) is not None:
        return rc
    body = {"model": model, "keep_alive": keep}
    try:
        backends._post(Ollama().root + "/api/generate", body, timeout=900)
    except urllib.error.HTTPError as exc:
        return _fail(f"ollama refused {model}: HTTP {exc.code} {exc.read().decode(errors='replace')[:200]}")
    residents = sysstats.ollama_residents(Ollama().root, timeout=120.0)
    if residents is None:
        return _fail(f"{model}: requested, but ollama did not answer /api/ps within 120 s; loaded state unknown")
    loaded = dict(residents)
    if keep == 0:
        if model in loaded:
            return _fail(f"{model}: still loaded")
        print(f"{model}: unloaded")
        return 0
    if model not in loaded:
        return _fail(f"{model}: not loaded after the request")
    print(f"{model}: loaded until {loaded[model]}")
    return 0


def cmd_quiet(args) -> int:
    """Pause omp's managed browser (SIGSTOP) so a run's preflight finds the GPU idle; `--resume` continues it
    (SIGCONT). Nothing else is touched and nothing is killed. Resume is refused while a run is alive: it would put
    the paused GPU load back in the middle of a measurement. Pausing what is paused, or resuming with nothing
    paused, is a no-op."""
    m = _mut(args)
    held = quiet.paused()
    if args.resume:
        refuse = "a localbench run is alive; resume after it ends" if _run_alive() else None
        plan = quiet.plan_resume()
        steps = [Step(f"SIGCONT pid {r['pid']}  {r['cmd'][:120]}", "continues omp's managed browser where "
                      "`localbench quiet` stopped it") for r in plan]
        if len(held) > len(plan):
            steps.append(Step(f"forget {len(held) - len(plan)} recorded pause(s) whose pid is gone or runs another "
                              "command", "a reused pid belongs to another program, which must not get SIGCONT"))
        if (rc := m.gate(steps, refuse=refuse, noop=None if held else "nothing paused; nothing to resume")) is not None:
            return rc
        rows = quiet.resume(plan=plan)
        print(f"resumed {len(rows)} process(es) of omp's managed browser")
        return 0
    plan = quiet.plan_pause()
    steps = [Step(f"SIGSTOP pid {r['pid']}  {r['cmd'][:120]}", "omp's managed browser draws on the GPU; stopped (not "
                  "killed), a run's preflight finds the GPU idle. `localbench quiet --resume` continues it") for r in plan]
    if args.display:
        steps.append(Step("put the display to sleep (pmset displaysleepnow)",
                          "the screen is a GPU client: other panes' output drew WindowServer past 25% for a whole leg "
                          "(2026-09-24); a sleeping display composites nothing, and any input wakes it"))
    noop = None if steps else "nothing to pause: " + (
        f"{len(held)} already paused; resume with `localbench quiet --resume`" if held else
        "omp's managed browser is not running")
    if (rc := m.gate(steps, noop=noop)) is not None:
        return rc
    for r in quiet.pause(plan=plan):
        print(f"paused pid {r['pid']}  {r['cmd'][:120]}")
    held = quiet.paused()
    print(f"{len(held)} paused; resume with `localbench quiet --resume`" if held
          else "nothing to pause: omp's managed browser is not running")
    if args.display:
        # The screen is a GPU client: other panes' output drew WindowServer past 25% for a whole leg (2026-09-24).
        # A sleeping display composites nothing; any key or mouse movement wakes it.
        rc = subprocess.run(["pmset", "displaysleepnow"], capture_output=True, check=False).returncode
        if rc:
            return _fail(f"pmset displaysleepnow failed (rc {rc}); the display is still on")
        print("display asleep; any input wakes it, and a woken screen with busy panes can void the run")
    return 0


def _smol_parked() -> bool:
    return any(p.get("kind") == park.SMOL_SERVER for p in (json.loads(park.STATE.read_text()) if park.STATE.exists() else []))


def _smol_verify(st: dict, expect: dict[str, str]) -> bool:
    """Per profile: the smol role omp itself reports, and (when it is the dedicated server's) whether omp's registry
    lists that model. True when every profile says what `expect` says."""
    ok = True
    for profile, want in expect.items():
        got = smol.omp_smol(profile)
        listed = smol.omp_lists(profile, st["model_id"]) if want == st.get("selector") else None
        good = got == want and listed is not False
        ok &= good
        print(f"  {profile:<10} smol={got}" + ("" if listed is None else f", registry lists it: {'yes' if listed else 'NO'}")
              + ("" if good else f"   <- expected {want}"))
    return ok


def _smol_plan(args, st: dict | None, new: dict | None) -> tuple[list[Step], str | None, str | None]:
    """(steps, refusal, no-op) for a changing `smol` action; reads the server, launchd and the profiles, changes
    nothing. `new` is the state `set` would write."""
    sel = st["selector"] if st else None
    up = bool(st) and smol.server_up(st["port"])
    stop = Step(f"stop the smol server {sel} on :{st['port']}" if st else "",
                "SIGTERM to the process on the port (SIGKILL after 60 s), waiting until it has exited; smol calls "
                "fail until it serves again")
    if args.action == "start":
        if not st:
            return [], "smol: nothing to start (no `localbench smol set` yet)", None
        if _smol_parked():
            return [], "smol server is parked for measurement; `localbench unpark` restarts it", None
        how = "under launchd" if smol.autostart_on() else "in its own session"
        noop = f"smol server already up on :{st['port']}, serving {', '.join(smol.served_ids(st['port']))}" if up else None
        return [Step(f"start the smol server {sel} on :{st['port']} {how}",
                     "omp profiles point their smol role at it; it serves once it answers /v1/models")], None, noop
    if args.action == "stop":
        if not st:
            return [], None, "nothing to stop: smol is not managed by localbench (no `localbench smol set`)"
        return [stop], None, None if up else f"smol server already stopped ({sel} on :{st['port']})"
    if args.action == "autostart":
        if not st:
            return [], "smol autostart: nothing to start at login (no `localbench smol set` yet)", None
        plist = smol.plist_path()
        if args.spec == "off":
            return ([Step(f"unload the LaunchAgent {plist} and delete it (launchd stops its server)",
                          "the server no longer comes back at login; `localbench smol start` runs one in this session")],
                    None, None if smol.autostart_on() else "autostart already off: no LaunchAgent")
        if _smol_parked():
            return [], "smol server is parked for measurement; unpark first", None
        if smol.autostart_on() and plist.read_bytes() == smol.launchd_plist(st) and up:
            return [], None, f"autostart already on ({plist}), server up"
        return ([stop] if up else []) + [
            Step(f"write the LaunchAgent {plist} and load it", "launchd starts the server now (RunAtLoad) and after "
                 "every login; no KeepAlive, so park's stop sticks. Its binary needs Full Disk Access to read an "
                 "external volume"),
            Step("wait until the server answers under launchd", "a job that cannot read its model exits; the log "
                 f"{smol.STATE_DIR / 'server.log'} names why")], None, None
    if args.action == "revert":
        if not st:
            return [], None, "smol: nothing to revert (not managed by localbench; profiles use their own smol lines)"
        steps = [Step(f"{name}: smol line back to {prev}, {smol.PROVIDER} block removed from models.yml" if prev else
                      f"{name}: smol line changed by hand since set, left as it is; {smol.PROVIDER} block removed",
                      "revert undoes exactly set's edits, so other edits made since survive")
                 for name, prev in smol.plan_revert(st["profiles"], sel, smol.profile_dirs())]
        if up:
            steps.append(stop)
        if smol.autostart_on():
            steps.append(Step(f"unload the LaunchAgent {smol.plist_path()} and delete it",
                              "nothing brings the server back at login"))
        steps.append(Step(f"move {smol.state_path()} to state.reverted-<stamp>.json",
                          f"smol is no longer managed; set's whole-file backups stay in {st.get('backup')}"))
        return steps, None, None
    # set
    steps = []
    if st and any(st.get(k) != new[k] for k in ("binary", "model_dir", "server_args")) and up:
        steps.append(Step(stop.action + " (binary, model or args differ from the new set)", stop.why))
    steps += [Step(f"start {new['binary']} --model {new['model_dir']} on :{smol.PORT} {' '.join(new['server_args'])}".rstrip(),
                   "the dedicated server omp's smol role will point at"),
              Step(f"check it serves only {new['model_id']} and answers a 16-token chat",
                   "a server that serves another model or cannot answer leaves the profiles unchanged"),
              Step(f"back up and point {len(smol.profile_dirs())} omp profile(s) at {new['selector']} "
                   f"(backups under {smol.BACKUPS})",
                   "every profile's smol line and a marked provider block in models.yml; `localbench smol revert` "
                   "undoes exactly these edits"),
              Step(f"write {smol.state_path()}", "what revert, start, park and status need")]
    return steps, None, None


def cmd_smol(args) -> int:
    """omp's smol role on a dedicated local server (localbench/smol.py): `set mlx-serve:<dir>` starts the server and
    points every profile at it (backups, verified through omp itself); `status`; `start`/`stop` the server only;
    `autostart on|off` runs it as a LaunchAgent so it comes back after a reboot (needs Full Disk Access for the
    mlx-serve binary, see smol.py); `revert` puts every profile back, stops it and removes the LaunchAgent. Changes
    are refused while a run is alive, and `start` while the server is parked (unpark restarts it). Starting what is
    up, stopping what is down, autostart on when on (off when off), and reverting nothing are no-ops."""
    st = smol.load_state()
    if args.json and args.action != "status" and not args.dry_run:
        _usage(f"smol {args.action} --json: `smol status --json` reads state, `smol {args.action} --dry-run --json` "
               "prints the plan")
    if args.action == "status":
        if not st:
            if args.json:
                print(json.dumps({"managed": False}))
            else:
                print("smol: not managed by localbench (profiles use their own smol lines)")
            return 0
        up = smol.server_up(st["port"])
        profiles = {}
        for name, agent in smol.profile_dirs().items():
            m = smol.SMOL_LINE.search((agent / "config.yml").read_text())
            profiles[name] = m.group("value") if m else None
        rc = 0 if up or _smol_parked() else 1
        if args.json:
            print(json.dumps({"managed": True, "selector": st["selector"], "port": st["port"], "up": up,
                              "parked": _smol_parked(), "serving": smol.served_ids(st["port"]) if up else [],
                              "binary": st["binary"], "server_args": st.get("server_args") or [],
                              "set_at": st.get("set_at"), "autostart": smol.autostart_on(),
                              "plist": str(smol.plist_path()), "profiles": profiles}, indent=2))
            return rc
        print(f"smol: {st['selector']} on :{st['port']} ({'up, serving ' + ', '.join(smol.served_ids(st['port'])) if up else 'DOWN'})"
              f"; binary {st['binary']}; args {' '.join(st.get('server_args') or []) or '-'}; set {st.get('set_at')}"
              f"; autostart {'on (' + str(smol.plist_path()) + ')' if smol.autostart_on() else 'off'}")
        for name, value in profiles.items():
            print(f"  {name:<10} {value or '(no smol line)'}")
        return rc
    m = _mut(args)
    if _run_alive():
        return m.gate([], refuse="a localbench run is alive; change smol after it ends")
    if args.action == "autostart" and args.spec not in ("on", "off"):
        _usage(f"smol autostart {args.spec or ''}: give on or off")
    new = None
    if args.action == "set":
        backend, _, model_dir = (args.spec or "").partition(":")
        if backend != "mlx-serve" or not model_dir:
            _usage(f"smol set {args.spec or ''}: give mlx-serve:<model dir>")
        model_dir = str(Path(model_dir).expanduser().absolute())
        new = {"binary": str(args.mlx_serve or backends.mlx_serve_bin()), "model_dir": model_dir, "port": smol.PORT,
               "server_args": list(args.server_arg or []), "model_id": Path(model_dir).name,
               "selector": f"{smol.PROVIDER}/{Path(model_dir).name}"}
    steps, refuse, noop = _smol_plan(args, st, new)
    if (rc := m.gate(steps, refuse=refuse, noop=noop)) is not None:
        return rc
    if args.action == "start":
        print(f"smol server up, pid {smol.start_server(st)}, serving {', '.join(smol.served_ids(st['port']))}")
        return 0
    if args.action == "autostart":
        if args.spec == "off":
            smol.remove_autostart()
            print("autostart off: LaunchAgent removed (its server stopped); `localbench smol start` runs one in this "
                  "session")
            return 0
        smol.stop_server(st)
        smol.install_autostart(st)
        print(f"LaunchAgent {smol.plist_path()} loaded; waiting for it to serve", flush=True)
        print(f"smol server up under launchd, pid {smol.start_server(st)}, serving "
              f"{', '.join(smol.served_ids(st['port']))}")
        return 0
    if args.action == "stop":
        smol.stop_server(st)
        print("smol server stopped; smol calls fail until `localbench smol start`")
        return 0
    if args.action == "revert":
        dirs = smol.profile_dirs()
        untouched = smol.revert_profiles(st["profiles"], st["selector"], dirs)
        smol.stop_server(st)
        smol.remove_autostart()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        smol.state_path().rename(smol.STATE_DIR / f"state.reverted-{stamp}.json")
        for name in untouched:
            print(f"  {name}: smol line was changed by hand since `set`; left as it is")
        print(f"reverted; server stopped, LaunchAgent removed; backups of the pre-set files stay in {st.get('backup')}")
        m.detail["hand_edited"] = untouched
        return 0 if _smol_verify(st, {p: r["previous"] for p, r in st["profiles"].items()
                                      if r.get("previous") and p not in untouched}) else 1
    # set
    if st and any(st.get(k) != new[k] for k in ("binary", "model_dir", "server_args")) and smol.server_up(st["port"]):
        smol.stop_server(st)
    print(f"starting {new['binary']} --model {new['model_dir']} on :{smol.PORT} {' '.join(new['server_args'])}", flush=True)
    new["pid"] = smol.start_server(new)
    served = smol.served_ids(smol.PORT)
    if served != [new["model_id"]]:
        return _fail(f"smol server serves {served}, expected [{new['model_id']}]; profiles left unchanged")
    reply = _post(f"http://{smol.HOST}:{smol.PORT}/v1/chat/completions",
                  {"model": new["model_id"], "max_tokens": 16, "messages": [{"role": "user", "content": "Reply with exactly: OK"}]},
                  timeout=300)
    print(f"server answers: {(reply['choices'][0]['message'].get('content') or '').strip()[:40]!r}")
    ctx = smol.context_length(new["model_id"]) or 262144
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = smol.BACKUPS / f"smol-{stamp}"
    new["profiles"] = smol.set_profiles(new["selector"], smol.provider_block(new["model_id"], ctx), smol.profile_dirs(),
                                        backup, prior=(st or {}).get("profiles"))
    new.update(context_window=ctx, backup=str(backup), set_at=stamp)
    smol._save_state(new)
    print(f"profiles pointed at {new['selector']} (backups in {backup}); omp's own view:")
    ok = _smol_verify(new, {p: new["selector"] for p, r in new["profiles"].items() if r.get("previous")})
    print("running omp sessions keep the smol they started with; new sessions use this one. "
          "Undo everything: `localbench smol revert`")
    return 0 if ok else 1


def cmd_pull(args) -> int:
    """Download an ollama model (a library tag, or hf.co/<org>/<repo>:<quant>) through ollama's API, show progress,
    and confirm the tag is installed with its digest; or, for `hf:<org>/<repo>`, a Hugging Face repo's files (see
    _pull_hf). Refused while a run is alive: the download's disk and CPU load would perturb it."""
    hf = args.spec.startswith("hf:")
    model = None if hf else _ollama_model(args.spec)
    repo = args.spec.removeprefix("hf:")
    if hf and repo.count("/") != 1:
        _usage(f"hf:{repo}: give hf:<org>/<repo>")
    m = _mut(args)
    if _run_alive():
        return m.gate([], refuse="a localbench run is alive; pull after it ends")
    if hf:
        plan = _plan_pull_hf(repo, args.to)
        steps = [Step(f"download {repo} @ {plan['sha'][:12]} ({plan['stored'] / 1e9:.1f} GB stored, "
                      f"{plan['need'] / 1e9:.1f} GB to fetch) into {plan['dest']} with `hf download`",
                      "the revision is pinned to the sha the API reports now; resumable: a rerun continues"),
                 Step(f"write {plan['dest'] / '.localbench-source.json'}", "records the repo and revision the files are")]
        if (rc := m.gate(steps, refuse=plan["refuse"])) is not None:
            return rc
        return _pull_hf(repo, plan)
    step = Step(f"download ollama {model} through ollama's /api/pull",
                "an installed tag is checked against its registry and updated; the tag and digest are confirmed after")
    if (rc := m.gate([step])) is not None:
        return rc
    root = Ollama().root
    req = urllib.request.Request(root + "/api/pull", json.dumps({"model": model, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    shown = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            ev = json.loads(raw)
            if ev.get("error"):
                return _fail(f"pull failed: {ev['error']}")
            total, done = ev.get("total"), ev.get("completed")
            step = f"{ev.get('status', '')}" + (f" {100 * done // total}%" if total and done is not None else "")
            decile = (ev.get("status"), 10 * done // total if total and done is not None else None)
            if decile != shown:
                print(step, flush=True)
                shown = decile
    name = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
    digest = {m["name"]: m["digest"] for m in backends._get(root + "/api/tags").get("models", [])}.get(name)
    if not digest:
        return _fail(f"{name}: not in ollama's model list after the pull")
    print(f"installed {name} (digest {digest.removeprefix('sha256:')[:12]})")
    return 0


# Default under the user's cache; LOCALBENCH_HF_DIR moves it to a bigger volume (MLX/safetensors repos are 20-56 GB).
# On macOS a Time Machine volume's root refuses new entries (backupd's `everyone deny add_file,add_subdirectory` ACL),
# so point it at a subdirectory there.
HF_DIR = Path(os.environ.get("LOCALBENCH_HF_DIR") or Path.home() / ".cache" / "localbench" / "hf").expanduser()
HF_HEADROOM_GB = 20


def _plan_pull_hf(repo: str, to: Path | None) -> dict:
    """What `pull hf:<repo>` would fetch and where, from the Hugging Face API (a read): {dest, sha, stored, need,
    gated, refuse}. `refuse` is set when the volume lacks the repo's stored size (the API's usedStorage, an upper
    bound) less what is already there, plus HF_HEADROOM_GB."""
    dest = (to or HF_DIR / repo).expanduser().absolute()
    info = backends._get(f"https://huggingface.co/api/models/{repo}?expand[]=usedStorage&expand[]=sha&expand[]=gated",
                         timeout=30)
    sha, stored = info["sha"], info.get("usedStorage") or 0
    have = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) if dest.exists() else 0
    need = max(stored - have, 0)
    anchor = next(p for p in (dest, *dest.parents) if p.exists())
    free = shutil.disk_usage(anchor).free
    refuse = None
    if free < need + HF_HEADROOM_GB * 1e9:
        refuse = (f"{repo}: needs {need / 1e9:.1f} GB plus {HF_HEADROOM_GB} GB headroom on {anchor}; "
                  f"{free / 1e9:.1f} GB free (--to or LOCALBENCH_HF_DIR picks another volume)")
    return {"dest": dest, "sha": sha, "stored": stored, "need": need, "gated": bool(info.get("gated")), "refuse": refuse}


def _pull_hf(repo: str, plan: dict) -> int:
    """Download a Hugging Face repo with the `hf` CLI (resumable: rerun to continue) as `plan` (_plan_pull_hf) says:
    into its dest, default HF_DIR/<org>/<repo> (LOCALBENCH_HF_DIR, else ~/.cache/localbench/hf), the revision pinned
    to the sha the API reported and recorded in .localbench-source.json. A gated repo needs a token in the caller's
    environment, e.g. `HF_TOKEN=<token> localbench pull hf:<org>/<repo>`: `hf` reads it, localbench never reads or
    prints it."""
    dest, sha = plan["dest"], plan["sha"]
    print(f"{repo} @ {sha[:12]}: {plan['stored'] / 1e9:.1f} GB stored{' (gated)' if plan['gated'] else ''} -> {dest}",
          flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["hf", "download", repo, "--revision", sha, "--local-dir", str(dest)],
                        env={**os.environ, "HF_HUB_DISABLE_UPDATE_CHECK": "1"}, check=False).returncode
    if rc:
        return _fail(f"hf download failed (rc {rc}); a gated repo needs HF_TOKEN from an account that was granted access")
    files = [f for f in dest.rglob("*") if f.is_file() and ".cache" not in f.relative_to(dest).parts]
    (dest / ".localbench-source.json").write_text(json.dumps(
        {"repo": repo, "revision": sha, "downloaded": datetime.now(UTC).isoformat(timespec="seconds")}) + "\n")
    print(f"downloaded {repo} @ {sha[:12]}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB "
          f"in {dest}")
    return 0


def cmd_create(args) -> int:
    """Build an ollama model from a local safetensors directory with `ollama create`, quantizing with --quantize (e.g.
    nvfp4, the incumbent smol model's format). ollama imports safetensors in the CLI's own process, on the GPU through
    MLX, and writes blobs where OLLAMA_MODELS points, so the server's models dir is passed explicitly: without it the
    blobs land in ~/.ollama/models, which the server here does not read (ollama v0.34.4 cmd.go createSafetensorsModel).
    The renderer and parser decide how omp's thinking and tool calls are formatted and parsed; --like copies them from
    the model the new one would replace. Refused while a run is alive, and when the name already exists."""
    name = _ollama_model(args.spec)
    m = _mut(args)
    if _run_alive():
        return m.gate([], refuse="a localbench run is alive; create after it ends")
    src = args.src.expanduser().absolute()
    if not (src / "config.json").is_file():
        _usage(f"create --from {src}: no config.json there; give a safetensors model directory")
    root = Ollama().root
    exists = any(t["name"] == name for t in backends._get(root + "/api/tags").get("models", []))
    renderer, parser = args.renderer, args.parser
    if args.like and not exists:
        ref = backends._post(root + "/api/show", {"model": _ollama_model(args.like)})
        renderer, parser = renderer or ref.get("renderer"), parser or ref.get("parser")
    modelfile = src / "Modelfile.localbench"
    store = sysstats.ollama_models_dir()
    steps = [Step(f"write {modelfile}: FROM {src}, renderer {renderer or '-'}, parser {parser or '-'}",
                  "ollama create reads the source, renderer and parser from it; the renderer and parser decide how "
                  "omp's thinking and tool calls are formatted and parsed"),
             Step(f"ollama create {name} (quantize {args.quantize or 'none'}) into {store}",
                  "ollama imports safetensors in the CLI's own process (GPU, MLX) and writes blobs where OLLAMA_MODELS "
                  "points; the server's models dir, or the server never sees them")]
    refuse = f"{name} already exists; pick another name (ollama would replace it)" if exists else None
    if (rc := m.gate(steps, refuse=refuse)) is not None:
        return rc
    modelfile.write_text(f"FROM {src}\n" + (f"RENDERER {renderer}\n" if renderer else "")
                         + (f"PARSER {parser}\n" if parser else ""))
    print(f"creating {name} from {src} (quantize {args.quantize or 'none'}, renderer {renderer or '-'}, parser "
          f"{parser or '-'}) into {store}", flush=True)
    rc = subprocess.run(["ollama", "create", name, "-f", str(modelfile),
                         *(["--quantize", args.quantize] if args.quantize else [])],
                        env={**os.environ, "OLLAMA_MODELS": str(store)}, check=False).returncode
    if rc:
        return _fail(f"ollama create failed (rc {rc})")
    show = backends._post(root + "/api/show", {"model": name})
    digest = {t["name"]: t["digest"] for t in backends._get(root + "/api/tags").get("models", [])}.get(name)
    if not digest:
        return _fail(f"{name}: created, but the server does not list it (blobs written outside {store}?)")
    print(f"created {name} (digest {digest.removeprefix('sha256:')[:12]}; renderer {show.get('renderer') or '-'}, "
          f"parser {show.get('parser') or '-'}, capabilities {', '.join(show.get('capabilities') or [])})")
    return 0


def judge(s: dict, as_json: bool = False) -> int:
    """Compare a summary against its golden, write report.md, print it (or, as_json, the golden path, the compared
    rows and the unsound reasons); 1 if anything a claim rests on is unsound."""
    pins = s["provenance"]["pins"]
    gpath = golden.golden_path(pins["host_id"], pins["backend"], pins["model"])
    base = golden.load(gpath)
    rows = golden.compare(s["metrics"], s["conformance"], base, pins, s["provenance"]["tiers"]) if base else []
    gstate = f"compared to {gpath.relative_to(ROOT)}" if base else f"NO GOLDEN at {gpath.relative_to(ROOT)}"
    bad = unsound(s) + [f"{r['key']}: {r['status']}" for r in rows if r["status"] in golden.FAILING]
    report = render.run_report(s, rows, gstate, bad)
    (ROOT / s["run_dir"] / "report.md").write_text(report)
    if as_json:
        print(json.dumps({"run_dir": s["run_dir"], "golden": str(gpath.relative_to(ROOT)) if base else None,
                          "rows": rows, "unsound": bad}, indent=2, default=str))
    else:
        print("\n" + report)
    for b in bad:
        print(f"UNSOUND  {b}", file=sys.stderr)
    return 1 if bad else 0


def cmd_run(args) -> int:
    with open_backend(args.backend, tuple(args.server_arg or ())) as (backend, model):
        s = execute(backend, model, tiers=args.tiers.split(","), repeats=args.repeats, allow_busy=args.allow_busy,
                    purge=args.purge, wait_idle_s=args.wait_idle, mem_config=args.mem_config,
                    mem_rounds=args.mem_rounds)
    return judge(s)


def _run_summary(verb: str, run_dir: str) -> dict:
    """A run dir's summary.json, or a usage error naming what `verb` expects."""
    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        _usage(f"{verb}: no {path}; give a run dir under runs/ (runs/<stamp>__<label>__<backend>__<model>, or "
               f"runs/LATEST), e.g. `localbench {verb} runs/LATEST`")
    return _load_json(verb, path)


def _load_json(verb: str, path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _usage(f"{verb}: {path} is not JSON ({exc}); {verb} reads a receipt .json (docs/evidence/receipts/), a "
               "golden .json (goldens/<host>/) or a run dir (runs/<dir>, its summary.json)")


def cmd_compare(args) -> int:
    """Re-judge an existing run against the golden on disk now (no measurement)."""
    return judge(_run_summary("compare", args.run_dir), as_json=args.json)


def cmd_bank(args) -> int:
    """Bank an existing run's receipt view under docs/evidence/receipts/<name>.json (summary.json is immutable, so
    banking after the fact records the same evidence). Refuses to bank without printing why the run is unsound.
    Banking the same run under the same name again is a no-op."""
    s = _run_summary("bank", args.run_dir)
    problems = unsound(s)
    payload = {"kind": "run", "problems": problems, "run": _receipt_view(s)}
    path = _receipt_path(args.name)
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    same = path.is_file() and path.read_text() == _receipt_text(payload)
    step = Step(f"{'replace' if path.exists() else 'write'} {rel} (receipt of {s['run_dir']}"
                f"{f', UNSOUND: {len(problems)} problem(s)' if problems else ''})",
                "a banked receipt is the evidence a claim cites; it keeps everything but the raw samples")
    m = _mut(args)
    m.detail["unsound"] = problems
    rc = m.gate([] if same else [step], noop=f"already banked: {rel} holds this receipt" if same else None)
    if rc is not None and (args.dry_run or rc):
        return rc
    if not same:
        print(f"banked {_bank(args.name, payload).relative_to(ROOT)}")
    for p in problems:
        print(f"UNSOUND  {p}", file=sys.stderr)
    m.outcome = "done"   # an unsound run is banked all the same, with its problems in the receipt; exit 1 says so
    return 1 if problems else 0


def cmd_show(args) -> int:
    """A receipt, golden or run dir as a compact reading view (render.show); one subtree of it unrounded with
    --path <RFC 6901 pointer>; with --diff REV, what a re-bank changed in a golden since git revision REV."""
    target = Path(args.target)
    path = (target / "summary.json") if target.is_dir() else target
    if not path.is_file():
        _usage(f"show: no such file {path} (give a receipt .json, a golden .json, or a runs/<dir>)")
    label = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)
    doc = _load_json("show", path)
    if args.path is not None:
        try:
            print(render.subtree(doc, args.path, label, cursor=args.cursor, limit=args.limit), end="")
        except KeyError as exc:
            _usage(f"show: {label} --path {args.path}: {exc.args[0]}")
        return 0
    if args.diff:
        if render.kind_of(doc) != "golden":
            _usage(f"show: --diff compares goldens; {label} is a {render.kind_of(doc)}")
        old = subprocess.run(["git", "-C", str(ROOT), "show", f"{args.diff}:{label}"], capture_output=True, text=True,
                             check=False)
        if old.returncode:
            sys.exit(f"show: `git show {args.diff}:{label}` failed: {old.stderr.strip()}")
        print(render.golden_diff(json.loads(old.stdout), doc, label, args.diff), end="")
        return 0
    print(render.show(doc, label, cursor=args.cursor, limit=args.limit), end="")
    return 0


def cmd_aa(args) -> int:
    """Run one config twice; bank the pair as a receipt and (with --write-golden) derive the golden from it."""
    tiers = args.tiers.split(",")
    m = _mut(args)
    refuse = None
    if args.write_golden and (args.server_arg or args.mem_config != MEM_CONFIG or args.mem_rounds != MEM_ROUNDS):
        refuse = ("refusing --write-golden with --server-arg, --mem-config, or --mem-rounds: the golden for a spec "
                  "is its default launch; measure a variant as the B leg of `localbench ab`")
    steps = [Step(f"measure {args.backend} twice (aa1, aa2): tiers {args.tiers}, {args.repeats} repeat(s), each after "
                  "preflight", "the A/A pair is the null: its spread sets the band a later run is judged against"),
             Step("bank the pair as docs/evidence/receipts/aa__<backend>__<model>__<created>.json",
                  "the receipt is the evidence the golden cites")]
    if args.write_golden:
        steps.append(Step(f"write the golden for {args.backend} under {golden.GOLDENS.relative_to(ROOT)}/<host_id>/ "
                          f"(tiers {args.tiers} re-banked, other tiers kept) unless the pair is unsound",
                          "goldens are written only here (the packet's UPDATE_GOLDENS); review with `localbench show "
                          "<golden> --diff HEAD` before committing"))
    if (rc := m.gate(steps, refuse=refuse)) is not None:
        return rc
    with open_backend(args.backend, tuple(args.server_arg or ())) as (backend, model):
        s1 = execute(backend, model, tiers=tiers, repeats=args.repeats, allow_busy=args.allow_busy,
                     purge=args.purge, label="aa1", wait_idle_s=args.wait_idle, mem_config=args.mem_config,
                     mem_rounds=args.mem_rounds)
        s2 = execute(backend, model, tiers=tiers, repeats=args.repeats, allow_busy=args.allow_busy,
                     purge=args.purge, label="aa2", wait_idle_s=args.wait_idle, mem_config=args.mem_config,
                     mem_rounds=args.mem_rounds)
    pins = s1["provenance"]["pins"]
    problems = unsound(s1) + unsound(s2)
    if golden.pin_diff(pins, s2["provenance"]["pins"]):
        problems.append(f"pins changed between A/A runs: {golden.pin_diff(pins, s2['provenance']['pins'])}")
    if args.allow_busy:
        problems.append("--allow-busy: an A/A pair from a busy machine cannot derive a band")
    name = f"aa__{pins['backend']}__{pins['model']}__{s1['provenance']['created']}"
    receipt = _bank(name, {"kind": "aa", "runs": [_receipt_view(s1), _receipt_view(s2)], "problems": problems})
    rel = str(receipt.relative_to(ROOT))
    m.detail["receipt"] = rel
    print(f"\nbanked A/A receipt {rel}\n")
    print(render.show(json.loads(receipt.read_text()), rel))
    if problems:
        for p in problems:
            print(f"UNSOUND  {p}", file=sys.stderr)
        m.outcome, m.detail["unsound"] = "refused", problems
        return 1
    g, refusals = golden.from_aa(s1["results"], s2["results"], pins, rel, tiers)
    if refusals:
        for r in refusals:
            print(f"REFUSED  {r}", file=sys.stderr)
        m.outcome, m.detail["refusals"] = "refused", refusals
        return 1
    if args.write_golden:
        gpath = golden.golden_path(pins["host_id"], pins["backend"], pins["model"])
        # Only the tiers this pair ran are re-banked: after an omp update `--tiers e2e,rel` refreshes the omp-bound
        # rows and keeps conf/micro/replay rows, whose pins did not move.
        g, dropped = golden.merge(golden.load(gpath), g, tiers)
        golden.write(gpath, g)
        m.detail["golden"] = str(gpath.relative_to(ROOT))
        if dropped:
            print(f"dropped tiers {dropped}: the backend or model changed, so their old rows are another generation")
        print(f"wrote {gpath.relative_to(ROOT)} — review with `localbench show {gpath.relative_to(ROOT)} --diff HEAD` before committing")
    return 0


def ab_order(pairs: int) -> list[tuple[str, str]]:
    """(arm, label) per leg: A,B repeated `pairs` times, then A. One pair keeps the original labels ab_a1, ab_b, ab_a2."""
    if pairs == 1:
        return [("A", "ab_a1"), ("B", "ab_b"), ("A", "ab_a2")]
    return [leg for i in range(1, pairs + 1) for leg in (("A", f"ab_a{i}"), ("B", f"ab_b{i}"))] + [("A", f"ab_a{pairs + 1}")]


def cmd_ab(args) -> int:
    """Same invocation, interleaved: A,B,...,A (`--pairs`). A's legs are the A/A null; each arm is the median of its
    legs. On a machine in use, more pairs spread bursts of activity across both arms."""
    tiers = args.tiers.split(",")
    order = ab_order(args.pairs)
    legs = []
    for arm, label in order:
        spec = args.a if arm == "A" else args.b
        extra = () if arm == "A" else tuple(args.b_server_arg or ())
        mem_config = args.mem_config if arm == "A" else (args.b_mem_config or args.mem_config)
        binaries = {"LOCALBENCH_OMP": args.b_omp, "LOCALBENCH_MLX_SERVE": args.b_mlx_serve} if arm == "B" else {}
        with _binaries_for_leg(binaries), open_backend(spec, tuple(args.server_arg or ()) + extra) as (backend, model):
            legs.append(execute(backend, model, tiers=tiers, repeats=args.repeats, allow_busy=args.allow_busy,
                                purge=args.purge, label=label, wait_idle_s=args.wait_idle, mem_config=mem_config,
                                mem_rounds=args.mem_rounds))
    a_legs = [leg for leg, (arm, _) in zip(legs, order) if arm == "A"]
    b_legs = [leg for leg, (arm, _) in zip(legs, order) if arm == "B"]
    problems = [f"{leg['provenance']['label']}: {p}" for leg in legs for p in unsound(leg)]
    drift = golden.arm_pin_drift([leg["provenance"]["pins"] for leg in a_legs],
                                 [leg["provenance"]["pins"] for leg in b_legs], tiers)
    def busy(leg):
        return (((leg.get("system") or {}).get("cpu") or {}).get("busy_pct") or {}).get("mean")
    balance = golden.load_balance([busy(leg) for leg in a_legs], [busy(leg) for leg in b_legs])
    table = golden.ab_table([leg["metrics"] for leg in a_legs], [leg["metrics"] for leg in b_legs], void_tiers=drift,
                            load_favours=balance["favours"])
    a1, b = a_legs[0], b_legs[0]
    receipt = {"kind": "ab", "a": args.a, "b": args.b, "order": [arm for arm, _ in order], "problems": problems,
               "pin_drift": drift, "load_balance": balance, "table": table,
               "legs": [_receipt_view(x) for x in legs]}
    name = args.bank or f"ab__{a1['provenance']['pins']['model']}__vs__{b['provenance']['pins']['model']}__{a1['provenance']['created']}"
    path = _bank(name, receipt)
    rel = str(path.relative_to(ROOT))
    print(f"\nbanked A/B receipt {rel}\n")
    print(render.show(json.loads(path.read_text()), rel))
    for p in problems:
        print(f"UNSOUND  {p}", file=sys.stderr)
    return 1 if problems else 0


def cmd_record(args) -> int:
    """Capture the first chat request omp sends for a flag set as fixtures/omp/<label>.json plus its
    generation sidecar <label>.meta.json (omp pins, prompt tokens, profile, provider path, backend pins)."""
    from .proxy import Proxy

    save = FIXTURES / "omp" / ".capture"
    dest = FIXTURES / "omp" / f"{args.label}.json"
    omp_flags = args.omp_flags[1:] if args.omp_flags[:1] == ["--"] else args.omp_flags
    rel = dest.relative_to(ROOT)
    steps = [Step(f"clear {save.relative_to(ROOT)}", "the capture dir holds only this recording's bodies"),
             Step(f"start {args.backend}, run omp once (flags: {' '.join(omp_flags) or '-'}) through the timing proxy",
                  "the fixture is the first chat request omp itself sends for this flag set, not a hand-built body"),
             Step(f"{'replace' if dest.exists() else 'write'} {rel} and {dest.with_suffix('.meta.json').name}",
                  "replay binds to the body's hash; the sidecar pins the omp, profile and backend it was recorded "
                  "under, so a changed fixture is a new generation for the replay tier")]
    if (rc := _mut(args).gate(steps)) is not None:
        return rc
    save.mkdir(parents=True, exist_ok=True)
    for stale in save.glob("*.json"):
        stale.unlink()
    calls = save / "calls.jsonl"
    calls.unlink(missing_ok=True)
    with open_backend(args.backend) as (backend, model):
        backend.isolate(model)
        fp = backend.fingerprint(model)
        if not fp.get("loaded_context"):
            sys.exit(f"{backend.name} did not report a loaded context for {model}; refusing to guess one for omp")
        with Proxy(backend.base_url, calls, save_dir=save, label=args.label):
            ensure_localbench_model(model, fp["loaded_context"])
            proc = subprocess.run([omp_bin(), "-p", "Reply with exactly: OK", "--model", f"localbench/{model}",
                                   "--smol", f"localbench/{model}", "--config", str(CHILD_CONFIG),
                                   "--mode", "json", "--no-session", *omp_flags],
                                  cwd="/tmp", capture_output=True, text=True, timeout=1800, env=child_env(),
                                  stdin=subprocess.DEVNULL, check=False)
        pins = backend.pins(model)
    (save / f"{args.label}.omp-stdout.jsonl").write_text(proc.stdout)
    # Main-agent requests carry tools; smol (memory) calls routed through the proxy do not.
    captured = [c for c in sorted(save.glob(f"{args.label}-*.json")) if json.loads(c.read_bytes()).get("tools")]
    rows = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    rows = [r for r in rows if r.get("tools")]
    # omp sometimes sends the identical first request twice (2026-09-22: the first stream carried no usage);
    # the fixture is the first body, and its token count comes from any call that sent that same body.
    first = captured[0].read_bytes() if captured else b""
    tokens = next((r["prompt_tokens"] for r, c in zip(rows, captured, strict=False)
                   if c.read_bytes() == first and r.get("prompt_tokens")), None)
    if proc.returncode or not captured or not tokens:
        sys.exit(f"record failed rc={proc.returncode} calls={len(rows)} tokens={tokens}: {proc.stderr[-500:]}")
    body = json.loads(first)
    dest.write_text(json.dumps(body, indent=1) + "\n")
    meta = {
        **omp_pins(), "prompt_tokens": tokens, "recorded_at": _stamp(), "omp_calls": len(rows),
        "profile": f"isolated agent dir ({AGENT_DIR.relative_to(ROOT)}, {AGENT_CONFIG.relative_to(ROOT)})",
        "omp_flags": omp_flags,
        "child_config": {"path": str(CHILD_CONFIG.relative_to(ROOT)), "sha16": sha16(str(CHILD_CONFIG))},
        "provider_path": "localbench provider (openai-completions) -> localbench proxy -> backend",
        "messages": len(body.get("messages", [])), "tools": len(body.get("tools") or []),
        "backend_pins": pins,
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"recorded {dest.relative_to(ROOT)} + {dest.with_suffix('.meta.json').name}: "
          f"{meta['prompt_tokens']} prompt tokens, {meta['tools']} tools, omp {meta['omp_version']}")
    return 0


def _when(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def cmd_audit(args) -> int:
    """The mutation ledger (localbench/audit.py): one row per real invocation of a state-changing verb, oldest first.
    `localbench why <id>` prints one row in full."""
    rows = audit.rows(None if args.since is None else time.time() - args.since)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(f"no audit rows{'' if args.since is None else ' in the window'} ({audit.path()})", file=sys.stderr)
        return 0
    print("\n".join(render.table(["id", "time", "verb", "outcome", "actions", "first action / reason"],
                       [[r.get("id", "?"), _when(r.get("t", 0)), r.get("verb", "?"), r.get("outcome", "?"),
                         str(len(r.get("actions") or [])),
                         render.clip((r.get("actions") or [None])[0] or (r.get("detail") or {}).get("reason")
                                     or (r.get("detail") or {}).get("noop") or "-", 80)] for r in rows])))
    return 0


def cmd_why(args) -> int:
    """One audit row in full: what ran (argv, cwd, host, localbench version), when, every action and the outcome."""
    r = audit.row(args.id)
    if r is None:
        recent = [x.get("id") for x in audit.rows()[-3:]]
        _usage(f"why: no audit row {args.id!r} in {audit.path()}; `localbench audit` lists the ids"
               + (f" (latest: {', '.join(recent)})" if recent else " (the ledger is empty)"))
    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    print(f"{r['id']}  {r.get('verb')}  {r.get('outcome')}  {_when(r.get('t', 0))}")
    print(f"argv: localbench {' '.join(r.get('argv') or [])}")
    print(f"cwd: {r.get('cwd')}  host: {r.get('host')}  localbench {r.get('localbench_version')}")
    print("actions:" if r.get("actions") else "actions: none")
    for a in r.get("actions") or []:
        print(f"  {a}")
    for k, v in (r.get("detail") or {}).items():
        print(f"{k}: {v if isinstance(v, str) else json.dumps(v, default=str)}")
    return 0


# Pins every leg of every receipt and golden on disk carries (2026-09-25: 139 of 139). A leg without them cannot be
# placed in a generation, so `status` and a golden comparison could not judge it.
REQUIRED_PINS = ("backend", "backend_version", "backend_sha", "model", "model_digest", "macos_build", "host_id")


def validate_doc(doc) -> list[str]:
    """Why a receipt, golden or run summary is not one `show` and a golden comparison can use; [] when it is."""
    if not isinstance(doc, dict):
        return [f"top level is a {type(doc).__name__}, not an object"]
    kind = render.kind_of(doc)
    if kind == "json":
        return [("not a receipt (kind aa|ab|run), a golden (pins, metrics, aa_receipt) or a run summary (provenance, "
                 "verdicts, metrics)")]
    reasons = []
    if kind == "golden":
        sets = [("pins", doc.get("pins"))]
        if not isinstance(doc.get("metrics"), dict) or not doc["metrics"]:
            reasons.append("metrics: missing or empty")
    else:
        legs = render._legs(doc, kind)
        if not legs or not all(isinstance(leg, dict) for leg in legs):
            return [f"{kind}: no legs ({ {'ab': 'legs', 'aa': 'runs', 'run': 'run', 'summary': 'itself'}[kind]})"]
        if kind == "aa" and len(legs) != 2:
            reasons.append(f"aa: {len(legs)} runs, an A/A pair has 2")
        if kind == "ab":
            reasons += [f"ab: no {k}" for k in ("a", "b", "order", "table") if not doc.get(k)]
        sets = []
        for i, leg in enumerate(legs):
            where = f"{render._leg_ptr(kind, i) or '/'}"
            prov = leg.get("provenance")
            if not isinstance(prov, dict):
                reasons.append(f"{where}: no provenance")
                continue
            reasons += [f"{where} provenance: no {k}" for k in ("tiers", "created") if not prov.get(k)]
            reasons += [f"{where}: no {k}" for k in ("metrics", "verdicts") if not isinstance(leg.get(k), dict)]
            sets.append((f"{where} provenance.pins", prov.get("pins")))
    for where, pins in sets:
        if not isinstance(pins, dict):
            reasons.append(f"{where}: missing")
            continue
        missing = [k for k in REQUIRED_PINS if k not in pins]
        if missing:
            reasons.append(f"{where}: lacks {', '.join(missing)}")
    if not reasons:
        try:
            render.show(doc, "validate")
        except Exception as exc:   # noqa: BLE001 - any crash of the reading view is the finding
            reasons.append(f"show cannot render it: {type(exc).__name__}: {exc}")
    return reasons


def cmd_validate(args) -> int:
    """Pure read: does a receipt, golden or run dir parse, carry the fields `show` needs and the pins a generation
    check needs? Exit 0 valid, 1 invalid with each reason on stderr."""
    target = Path(args.file)
    path = (target / "summary.json") if target.is_dir() else target
    if not path.is_file():
        _usage(f"validate: no such file {path} (give a receipt .json, a golden .json, or a runs/<dir>)")
    try:
        doc = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _fail(f"INVALID {path}: not JSON ({exc})")
    reasons = validate_doc(doc)
    if reasons:
        print(f"INVALID {path} ({render.kind_of(doc) if isinstance(doc, dict) else 'json'})", file=sys.stderr)
        for r in reasons:
            print(f"  {r}", file=sys.stderr)
        return 1
    print(f"valid {render.kind_of(doc)}: {path}")
    return 0


def cmd_doctor(args) -> int:
    """One row per subsystem localbench depends on (localbench/doctor.py): PASS, WARN or FAIL, what was found, and the
    command that fixes it. `--fix` performs only the safe, reversible, idempotent repairs, each recorded in the audit
    ledger; the rest keep their command. Exit 1 when any row is FAIL."""
    from . import doctor

    rows = doctor.checks(fix=args.fix)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("\n".join(render.table(["subsystem", "status", "detail", "fix"],
                           [[r["subsystem"], r["status"] + (" (fixed)" if r.get("fixed") else ""),
                             render.clip(r["detail"], 120), r.get("fix") or "-"] for r in rows])))
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


def main(argv: list[str] | None = None) -> int:
    """Parse and dispatch. Python ignores SIGPIPE, so a closed pipe is a BrokenPipeError where it happened: the proxy
    marks a stream omp abandoned as `aborted`, the rpc driver reads a dead omp as EOF. A process-wide SIG_DFL (as
    here until 2026-09-23) turned the first of those into a silent kill (exit 141): the sess tier died when omp
    aborted a classifier stream queued behind a 10 s memory extraction. Only stdout gets the Unix-filter treatment."""
    ap = argparse.ArgumentParser(prog="localbench", description=__doc__, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stats", help="print a full system snapshot (always JSON)")
    st.add_argument("--json", action="store_true", help="accepted; stats always prints JSON")
    sta = sub.add_parser("status", help="which configs have a live golden (generation check), park state, GPU users")
    sta.add_argument("--json", action="store_true", help="one JSON document instead of the text view")
    gp = sub.add_parser("gpu", help="who is using the GPU now, and which sessions/features can send it work")
    gp.add_argument("--seconds", type=float, default=10.0, help="attribution window")
    gp.add_argument("--json", action="store_true")
    mem = sub.add_parser("memory", help="omp memory banks: whose, how big, how fresh; --prune removes localbench's")
    mem.add_argument("--prune", action="store_true", help="delete banks localbench children/probes created")
    mem.add_argument("--json", action="store_true")
    mdl = sub.add_parser("models", help="installed local models: freshness vs upstream, who uses them, new releases")
    mdl.add_argument("--days", type=int, default=14, help="release window")
    mdl.add_argument("--json", action="store_true")
    w = sub.add_parser("watch", help="record local-model use (GPU by process, loaded models, clients) every minute")
    w.add_argument("--interval", type=float, default=60.0)
    w.add_argument("--samples", type=int, help="stop after N samples")
    rp = sub.add_parser("report", help="what used local models over a window (from `localbench watch`)")
    rp.add_argument("--since", type=_since, default="24h", help="window: 90m, 24h, 7d (or seconds)")
    rp.add_argument("--json", action="store_true")

    def measured(p, *specs):
        for s in specs:
            p.add_argument(s, help="backend spec: ollama:<model> | mlx-serve:<model dir> | omlx:<model dir>")
        p.add_argument("--tiers", default="conf,micro,replay,e2e")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--allow-busy", action="store_true", help="measure anyway; the run is non-proof")
        p.add_argument("--purge", action="store_true", help="drop the file cache first (needs sudoers grant)")
        p.add_argument("--server-arg", action="append", help="extra mlx-serve flag (repeatable)")
        p.add_argument("--wait-idle", type=float, default=0, metavar="SECONDS",
                       help="re-check a busy machine every 30 s for up to SECONDS before refusing")
        p.add_argument("--mem-config", type=_overlay, default=MEM_CONFIG, metavar="OVERLAY",
                       help="omp config overlay for the mem tier's children (default fixtures/omp/child-config-mem.yml)")
        p.add_argument("--mem-rounds", type=_mem_rounds, default=MEM_ROUNDS, metavar="N",
                       help=f"mem-tier rounds (default {MEM_ROUNDS}); each round plants three fresh facts")
        return p

    measured(sub.add_parser("run", help="measure one model and compare to its golden"), "backend")
    aa = measured(sub.add_parser("aa", help="A/A pair: banked receipt, optionally the golden"), "backend")
    aa.add_argument("--write-golden", action="store_true")
    ab = measured(sub.add_parser("ab", help="same-invocation interleaved A,B,...,A comparison"), "a", "b")
    ab.add_argument("--pairs", type=int, default=1,
                    help="A,B pairs before the closing A (default 1: A,B,A); more spread a busy machine's bursts")
    ab.add_argument("--bank", help="receipt name under docs/evidence/receipts/")
    ab.add_argument("--b-server-arg", action="append",
                    help="extra mlx-serve flag for the B leg only (repeatable), e.g. --b-server-arg=--mtp")
    ab.add_argument("--b-mem-config", type=_overlay, metavar="OVERLAY",
                    help="mem-tier omp overlay for the B leg only, e.g. fixtures/omp/child-config-mem-fts.yml")
    ab.add_argument("--b-omp", type=_exe_path, metavar="PATH",
                    help="omp executable for the B legs only (A uses LOCALBENCH_OMP or PATH), e.g. an isolated "
                         "install of an older release to A/B omp itself")
    ab.add_argument("--b-mlx-serve", type=_exe_path, metavar="PATH",
                    help="mlx-serve executable for the B legs only (A uses LOCALBENCH_MLX_SERVE or PATH), e.g. a "
                         "release extracted beside Homebrew's to A/B mlx-serve itself")
    cmp = sub.add_parser("compare", help="re-judge an existing run dir against its golden (no measurement)")
    cmp.add_argument("run_dir", help="a runs/<dir> holding summary.json, e.g. runs/LATEST")
    cmp.add_argument("--json", action="store_true", help="golden path, compared rows and unsound reasons as JSON")
    bk = sub.add_parser("bank", help="bank an existing run dir as docs/evidence/receipts/<name>.json")
    bk.add_argument("run_dir")
    bk.add_argument("name")
    sh = sub.add_parser("show", help="read a receipt, golden or run dir compactly; --path prints one subtree raw")
    sh.add_argument("target", help="a receipt or golden .json, or a runs/<dir>")
    sh.add_argument("--path", metavar="POINTER", help="RFC 6901 pointer, printed unrounded, e.g. /legs/1/system/during")
    sh.add_argument("--diff", metavar="REV", help="golden only: rows, pins and tiers changed since git revision REV")
    sh.add_argument("--cursor", type=int, default=0, help="first row or entry (a paged view names the next cursor)")
    sh.add_argument("--limit", type=int, default=render.PAGE, help="rows or entries per page")
    rec = sub.add_parser("record", help="record omp's request body + sidecar into fixtures/omp/")
    rec.add_argument("backend", help="backend spec: ollama:<model> | mlx-serve:<model dir> | omlx:<model dir>")
    rec.add_argument("--label", required=True)
    rec.add_argument("omp_flags", nargs=argparse.REMAINDER)
    pk = sub.add_parser("park", help="move omp smol models out of reach for a test window (memory may fail)")
    pk.add_argument("--status", action="store_true", help="print park state as JSON; parks nothing")
    pk.add_argument("--json", action="store_true", help="with --status: accepted (--status always prints JSON)")
    unpk = sub.add_parser("unpark", help="restore models parked by `localbench park`")
    kp = sub.add_parser("keep", help="set how long ollama keeps a model loaded (forever, 30m, 0 to unload)")
    kp.add_argument("spec", help="ollama:<model>")
    kp.add_argument("duration", nargs="?", default="forever", help="forever | 30m | 2h | 0 (default forever)")
    pl = sub.add_parser("pull", help="download an ollama model (library tag or hf.co/<org>/<repo>:<quant>), or a Hugging "
                                     "Face repo's files (hf:<org>/<repo>; token from HF_TOKEN in the environment)")
    pl.add_argument("spec", help="ollama:<model> | hf:<org>/<repo>")
    pl.add_argument("--to", type=Path, help="hf: download directory (default $LOCALBENCH_HF_DIR/<org>/<repo>, "
                                            "LOCALBENCH_HF_DIR defaulting to ~/.cache/localbench/hf)")
    cr = sub.add_parser("create", help="build an ollama model from a safetensors dir (import + quantize, e.g. nvfp4)")
    cr.add_argument("spec", help="ollama:<new name>")
    cr.add_argument("--from", dest="src", type=Path, required=True, help="safetensors model directory (config.json)")
    cr.add_argument("--quantize", help="e.g. nvfp4 (the incumbent smol model's format)")
    cr.add_argument("--like", help="ollama:<model> whose renderer and parser to copy (the model it would replace)")
    cr.add_argument("--renderer", help="override the renderer")
    cr.add_argument("--parser", help="override the parser")
    sm = sub.add_parser("smol", help="omp's smol role on a dedicated local server: set, status, start, stop, autostart, "
                                     "revert")
    sm.add_argument("action", choices=["set", "status", "start", "stop", "autostart", "revert"])
    sm.add_argument("spec", nargs="?", help="set: mlx-serve:<model dir>; autostart: on|off")
    sm.add_argument("--mlx-serve", type=_exe_path, help="set: mlx-serve binary (default LOCALBENCH_MLX_SERVE or PATH)")
    sm.add_argument("--server-arg", action="append", help="set: server flag, one argv element each (e.g. --server-arg=--mtp)")
    sm.add_argument("--json", action="store_true", help="status: one JSON document instead of the text view")
    qt = sub.add_parser("quiet", help="pause omp's managed browser so the GPU is idle for a run (--resume after)")
    qt.add_argument("--resume", action="store_true", help="continue what `localbench quiet` paused")
    qt.add_argument("--display", action="store_true", help="also put the display to sleep (the screen is a GPU client)")
    au = sub.add_parser("audit", help="the mutation ledger: one row per park, unpark, smol change, keep, pull, ...")
    au.add_argument("--since", type=_since, help="window: 90m, 24h, 7d (or seconds); default every row")
    au.add_argument("--json", action="store_true", help="the rows as one JSON array")
    wh = sub.add_parser("why", help="one audit row in full: argv, cwd, host, version, actions, outcome")
    wh.add_argument("id", help="a row id from `localbench audit`")
    wh.add_argument("--json", action="store_true")
    va = sub.add_parser("validate", help="check a receipt, golden or run dir: parses, has what `show` needs, pins present")
    va.add_argument("file", help="a receipt or golden .json, or a runs/<dir>")
    dr = sub.add_parser("doctor", help="PASS/WARN/FAIL per subsystem with the command that fixes each")
    dr.add_argument("--fix", action="store_true", help="perform the safe, reversible repairs (each is audited)")
    dr.add_argument("--json", action="store_true", help="the rows as one JSON array")
    for p, has_json in ((pk, True), (unpk, False), (kp, False), (pl, False), (cr, False), (sm, True), (qt, False),
                        (mem, True), (bk, False), (aa, False), (rec, False)):
        p.add_argument("--dry-run", action="store_true",
                       help="print the planned actions, one per line, and change nothing (exit 1 if it would refuse)")
        p.add_argument("--explain", action="store_true", help="print what each action does and why, then proceed")
        if not has_json:
            p.add_argument("--json", action="store_true", help="with --dry-run: the plan as one JSON document")
            p.set_defaults(plan_json_only=True)
    args = ap.parse_args(argv)
    _check_root(args.cmd)
    verb = _mutation_verb(args)
    if getattr(args, "dry_run", False) and verb is None:
        _usage(f"{args.cmd} --dry-run: this invocation only reads (park --status, smol status, memory without --prune)")
    if getattr(args, "plan_json_only", False) and args.json and not args.dry_run:
        _usage(f"{args.cmd} --json: prints the plan with --dry-run; the command itself prints text")
    mut = args.mutation = None if verb is None else Mutation(
        verb, list(sys.argv[1:] if argv is None else argv), dry_run=args.dry_run, explain=args.explain,
        as_json=args.json, audited=not (args.cmd == "aa" and not args.write_golden))
    handler = {"stats": cmd_stats, "status": cmd_status, "gpu": cmd_gpu, "run": cmd_run, "aa": cmd_aa, "ab": cmd_ab,
               "record": cmd_record, "memory": cmd_memory, "models": cmd_models, "watch": cmd_watch,
               "report": cmd_report, "compare": cmd_compare, "bank": cmd_bank, "park": cmd_park, "unpark": cmd_unpark,
               "show": cmd_show, "keep": cmd_keep, "pull": cmd_pull, "create": cmd_create, "smol": cmd_smol,
               "quiet": cmd_quiet, "audit": cmd_audit, "why": cmd_why, "validate": cmd_validate,
               "doctor": cmd_doctor}[args.cmd]
    try:
        rc = handler(args)
        sys.stdout.flush()
    except BrokenPipeError as exc:
        if mut:
            mut.finish(None, exc)
        # `localbench memory | head`: the reader left; end quietly like a Unix filter. Point stdout at /dev/null so
        # the interpreter's own exit-time flush does not raise again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 141
    except BaseException as exc:
        if mut:
            mut.finish(None, exc)
        raise
    if mut:
        mut.finish(rc)
    return rc


def _mutation_verb(args) -> str | None:
    """The audit verb of an invocation that changes state (`smol revert`, `memory --prune`, ...); None for a read."""
    c = args.cmd
    if c == "park":
        return None if args.status else c
    if c == "smol":
        return None if args.action == "status" else f"smol {args.action}"
    if c == "memory":
        return "memory --prune" if args.prune else None
    if c == "quiet":
        return "quiet --resume" if args.resume else c
    if c == "aa":
        return "aa --write-golden" if args.write_golden else c
    return c if c in ("unpark", "keep", "pull", "create", "bank", "record") else None


if __name__ == "__main__":
    sys.exit(main())
