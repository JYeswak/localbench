"""localbench: is the local model fast on THIS machine, and did it get slower?

  localbench stats
  localbench run  ollama:qwen3.6:35b-mlx                      # measure + compare to its golden
  localbench aa   ollama:qwen3.6:35b-mlx --write-golden       # A/A pair -> banked receipt + golden
  localbench ab   ollama:qwen3.8:27b-mlx ollama:qwen3.6:35b-mlx --bank ab-incumbent-vs-moe
  localbench record --label lean ollama:qwen3.6:35b-mlx -- --no-skills ...

A backend spec is `ollama:<model>`, `mlx-serve:<model dir>`, `omlx:<model dir>` or `mlxfast:<model dir>` (the process is
pinned to that model).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import plistlib
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, NoReturn

from . import (
    audit,
    backends,
    gateway,
    golden,
    heavyslot,
    memory,
    models,
    observe,
    ollama_app,
    park,
    quiet,
    render,
    smol,
    sysstats,
    workloads,
)
from .backends import OMLX, MlxFast, MlxServe, Ollama, _first_line, _post, sha16, splash_pin
from .client import RequestCancellation, RunAborted
from .watchdog import RunWatchdog
from .workloads import (
    AGENT_CONFIG,
    AGENT_DIR,
    CHILD_CONFIG,
    FIXTURES,
    MEM_CONFIG,
    MEM_ROUNDS,
    MEM_TOOLS,
    ROOT,
    TIERS,
    Ctx,
    Result,
    child_env,
    embedding_pins,
    ensure_localbench_model,
    fixtures_sha,
    omp_bin,
    smol_pins,
)

RUNS = ROOT / "runs"
RECEIPTS = ROOT / "docs" / "evidence" / "receipts"
LOAD_COMMAND = "load"

# Verbs that read nothing under ROOT: they run from any install. Every other verb needs a clone's data root. doctor
# reports a missing root itself, as one FAIL row.
ROOTLESS = frozenset({"stats", "memory", "keep", "pull", "create", "audit", "why", "validate", "doctor",
                      "ollama-app", "gateway", "slot", LOAD_COMMAND})

EPILOG = """\
exit status:
  0    ok; every verdict a claim could rest on is sound
  1    unsound, regressed or refused: an unlisted MUST FAIL, a MUST VOID without proof, a CONTENDED run, a golden
       row that REGRESSED / FAILed / is MUST-VOID / MISSING / GENERATION-MISMATCH / TOL-UNPROVEN, or a refusal
       (a run is alive, preflight, a download that does not fit); a claim must not rest on it
  2    usage error: a bad flag or argument value, or no data root (see LOCALBENCH_HOME)
  141  stdout was closed early (`localbench memory | head`)
Failures print to stderr; stdout carries only results (with --json, one JSON document). No color output.

environment:
  LOCALBENCH_HOME     the clone holding fixtures/, goldens/, runs/ (default: the checkout this install runs from)
  LOCALBENCH_HF_DIR   `pull hf:` download root (default ~/.cache/localbench/hf)
  LOCALBENCH_OMP, LOCALBENCH_MLX_SERVE, LOCALBENCH_MLXFAST   the omp / mlx-serve / mlx-server (mlxfast) executables
                      (default: PATH)
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


def _held(verb, *, run_only: bool = False, inference: bool = True):
    """Serialize heavy verbs on the machine slot (AGENTS.md Pacing): refuse naming the holder and the
    readings, or --wait-slot queue instead; the admission lands in the audit row (--force-load included).
    Dry runs plan without taking anything. run_only limits decision to its run action. inference=False
    checks the load average only, for CPU-only verbs beside a permanently busy smol GPU."""
    def wrap(fn):
        def inner(args):
            if run_only and getattr(args, "decision_action", "run") not in ("run", None):
                return fn(args)
            m = _mut(args)
            if m.dry_run:
                return fn(args)
            try:
                slot = heavyslot.acquire(verb, wait_s=max(0.0, float(getattr(args, "wait_slot", 0) or 0)),
                                         force_load=bool(getattr(args, "force_load", False)),
                                         needs_gpu=inference)
            except heavyslot.SlotRefused as exc:
                print(f"heavy slot: {exc}", file=sys.stderr)
                m.record([], "refused", {"reason": f"heavy slot: {exc}"})
                return 1
            m.detail["heavy_slot"] = slot.admission
            try:
                return fn(args)
            finally:
                slot.release()
        return inner
    return wrap


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
    """omp's identity and the overlays its children run, plus the embedding model the mem overlay's children load
    (embedder / embedder_digest; both None when that overlay runs no embeddings)."""
    binary = omp_bin()
    return {"omp_version": _first_line(binary, "--version").removeprefix("omp/").strip() or None,
            "omp_sha": sha16(binary), "omp_path": binary, "omp_child_config": sha16(str(CHILD_CONFIG)),
            "omp_mem_config": sha16(str(mem_config)), "omp_agent_config": sha16(str(AGENT_CONFIG)),
            "omp_mem_tools": ",".join(sorted(MEM_TOOLS)), **embedding_pins(mem_config)}


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
    """Run one leg under other binaries, e.g. {"LOCALBENCH_OMP": <omp>, "LOCALBENCH_MLX_SERVE": <mlx-serve>}: omp_bin(),
    mlx_serve_bin() and mlxfast_bin() read these at every call, so the leg's launches and its start/end pins all name
    them. Each is restored afterwards, so the next A leg is back on the default. None leaves a variable alone."""
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


def _unique_tiers(text: str) -> str:
    """Reject duplicate case IDs before a repeated tier could overwrite earlier receipt diagnostics."""
    tiers = text.split(",")
    if len(tiers) != len(set(tiers)):
        raise argparse.ArgumentTypeError("duplicate tier in --tiers")
    return text


# The servers the harness starts and stops, by backend name (= spec prefix = pins["backend"]).
SERVERS = {"mlx-serve": MlxServe, "omlx": OMLX, "mlxfast": MlxFast}


@contextlib.contextmanager
def open_backend(spec: str, server_args: tuple[str, ...] = ()):
    """Yield (backend, model) for `ollama:<model>` or `<server>:<model dir>` (mlx-serve, omlx, mlxfast); servers are
    started and stopped here. One model at a time: a run refuses while another harness server is serving."""
    kind, _, rest = spec.partition(":")
    busy = [f"{n} on :{s.port}" for n, cls in SERVERS.items() if n != kind and (s := cls(".")).up()]
    if busy and rest:
        sys.exit(f"{', '.join(busy)} is serving during a {kind} run; stop it first (one model at a time)")
    if kind == "ollama" and rest:
        yield Ollama(), rest
    elif kind in SERVERS and rest:
        unload_ollama()
        with SERVERS[kind](rest, server_args) as srv:
            yield srv, srv.model_id()
    else:
        _usage(f"bad backend spec {spec!r}: use ollama:<model>, mlx-serve:<model dir>, omlx:<model dir> or "
               "mlxfast:<model dir>")


def unload_ollama() -> list[str]:
    freed = []
    with contextlib.suppress(OSError):
        ol = Ollama()
        for m in ol.loaded():
            _post(ol.root + "/api/generate", {"model": m["name"], "keep_alive": 0})
            freed.append(m["name"])
    return freed


GPU_BUSY_MAX_PCT = 25.0


def _busy_check(side: bool = False) -> tuple[dict, float | None, list[str]]:
    """Preflight problems. `side` (the side-model regime, workloads.SIDE_REGIME): another model running or resident,
    unknown residency, a loadable smol model, the smol server and stuck sessions are the measured condition (recorded
    per leg in system.contention / during), not refusals; only an unreadable GPU/CPU signal and swapping refuse."""
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
        if busy and not side:
            problems.append("another model is running: " + ", ".join(sysstats.proc_label(r) for r in busy)
                            + " (a run would be CONTENDED)")
    if summ.get("resident_unknown_samples", 0) and not side:
        problems.append("resident model state unknown during preflight; cannot prove no other model was resident")
    if cpu is None:
        problems.append("CPU signal unavailable (/usr/bin/top printed no CPU usage line)")
    if summ.get("swap_used_mb", {}).get("max", 0) > 1024:
        problems.append("swapping")
    if side:
        return summ, cpu, problems
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


def preflight(allow_busy: bool, wait_idle_s: float = 0, emit=None, side: bool = False) -> dict:
    """Refuse to measure while another model can run (a loadable smol model, a stuck session's fallback, another
    model's runner busy), while swapping, or when the GPU/CPU signal cannot be read. A busy machine is not refused:
    apps and user activity are the measured condition (system.during.load). load1 is recorded, not judged.
    With wait_idle_s, re-check every 30 s until the machine is idle or the wait runs out; the verdict is the
    same gate, applied to the state the run actually starts in (other agents share this host). `side`: the side-model
    regime's gate (_busy_check): no refusal, and no wait, for co-resident models."""
    t0 = time.time()
    while True:
        summ, cpu, problems = _busy_check(side)
        if not problems or allow_busy or time.time() - t0 >= wait_idle_s:
            break
        if emit:
            emit({"event": "preflight_wait", "problems": problems})
        time.sleep(30)
    if problems and not allow_busy:
        sys.exit("preflight refused: " + "; ".join(problems) + "  (rerun with --allow-busy to measure anyway; "
                 "such a run is non-proof)")
    return {"idle_check": summ, "cpu_busy_pct": cpu, "problems": problems, "allow_busy": allow_busy,
            "waited_s": round(time.time() - t0, 1), **({"regime": workloads.SIDE_REGIME} if side else {})}


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
            mem_rounds: int = MEM_ROUNDS, e2e_case: str | None = None,
            evaluation_campaign: dict | None = None, smol_model: str | None = None,
            side_regime: bool = False, watchdog_enabled: bool = False) -> dict:
    """One measurement of one model: preflight, isolate, tiers under the samplers, summary on disk. `smol_model`: a
    separate memory (smol-role) model for the mem and sess tiers' omp children, on the same ollama server; pinned
    as smol_model/smol_digest, warmed after isolation and watched by the sampler as the run's own second model.
    `side_regime` (workloads.SIDE_REGIME; ab --side-regime): no isolation: nothing resident is unloaded, the model
    (and smol model) are only warmed; co-resident models are recorded as contention as always; the leg is marked
    regime side in provenance and pins."""
    try:   # before the run dir: a smol model on a one-model server is a usage error, not a run
        smol = smol_pins(backend, smol_model)
    except ValueError as exc:
        _usage(f"--smol-model: {exc}")
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
    pre = preflight(allow_busy, wait_idle_s, emit, side=side_regime)
    freed = unload_ollama() if backend.name != "ollama" and not side_regime else []
    purged = sysstats.purge_file_cache() if purge else None
    if side_regime:
        evicted = []
        backends._warm(backend.base_url, model)   # loaded beside whatever else is resident; nothing is unloaded
    else:
        evicted = backend.isolate(model)
    if smol_model is not None:
        backends._warm(backend.base_url, smol_model)   # resident before the sampler starts: it is ours, not foreign
    fp = backend.fingerprint(model)
    pins = {**run_pins(backend, model, before["host"], mem_config), **smol,
            **({"regime": workloads.SIDE_REGIME} if side_regime else {})}
    emit({"event": "isolated", "evicted": evicted, "freed": freed, "purged": purged, "fingerprint": fp})
    tok_identity = backend.tokenizer_identity(model) if "replay" in tiers else None

    cancellation = RequestCancellation() if watchdog_enabled else None
    ctx = Ctx(backend=backend, model=model, repeats=repeats, run_dir=run_dir, emit=emit, pins=pins,
              loaded_context=fp.get("loaded_context"), tok_identity=tok_identity,
              mem_config=mem_config, mem_rounds=mem_rounds, e2e_case=e2e_case, smol_model=smol_model,
              cancellation=cancellation)
    results: list[Result] = []
    def on_contention(ev):
        emit({"event": "contention", **ev})
    target = (backend.name, model, *([smol_model] if smol_model else []))
    watchdog = (RunWatchdog(target, pins["omp_sha"], lambda: sha16(pins["omp_path"]), GPU_BUSY_MAX_PCT)
                if watchdog_enabled else None)
    watchdog_state: dict[str, str | float | None] = {"reason": None, "t": None}

    def on_sample(sample):
        if watchdog is None or watchdog_state["reason"] is not None:
            return
        try:
            violations = watchdog.violations(sample)
        except Exception as exc:
            violations = [f"watchdog_check_failed:{type(exc).__name__}:{exc}"]
        if violations:
            reason = "; ".join(violations)
            watchdog_state["reason"] = reason
            sample_time = sample.get("t")
            watchdog_state["t"] = sample_time if isinstance(sample_time, (int, float)) else None
            if cancellation:
                cancellation.cancel(reason)
            try:
                emit({"event": "watchdog_abort", "reasons": violations})
            except OSError:
                pass

    with sysstats.Sampler(1.0, target=target, on_contention=on_contention,
                          gpu_foreign_max_pct=GPU_BUSY_MAX_PCT,
                          on_sample=on_sample if watchdog_enabled else None) as smp, \
            sysstats.PowerSampler(1000) as pwr, sysstats.CpuSampler(2) as cpu:
        try:
            for tier in tiers:
                if cancellation:
                    cancellation.raise_if_cancelled()
                emit({"event": "tier", "tier": tier})
                results += TIERS[tier](ctx)
        except RunAborted as exc:
            if watchdog_state["reason"] is None:
                watchdog_state["reason"] = str(exc)
            if watchdog_state["t"] is None:
                watchdog_state["t"] = time.time()
    if cancellation and cancellation.reason is not None and watchdog_state["reason"] is None:
        watchdog_state["reason"] = cancellation.reason
    after = sysstats.snapshot()
    # A pin that moved during the run (omp was upgraded in place mid-campaign on 2026-09-23) means part of the
    # run measured another generation; the start pins would be a false label for it.
    pins_changed = golden.pin_diff(pins, {**run_pins(backend, model, after["host"], mem_config),
                                          **smol_pins(backend, smol_model)})

    metrics, conformance = golden.flatten(results)
    listed = golden.listed_discrepancies(backend.name, model)
    must_fail = sorted(c for c, e in conformance.items()
                       if e["level"] == "MUST" and e["verdict"] == "FAIL" and c not in listed)
    summary = {
        "provenance": {"pins": pins, "fingerprint": fp, "localbench_rev": _rev(), "created": stamp,
                       "tiers": tiers, "repeats": repeats, "label": label,
                       **({"regime": workloads.SIDE_REGIME} if side_regime else {})},
        "verdicts": {"contended": bool(smp.contention), "must_fail": must_fail,
                     "preflight_problems": pre["problems"], "allow_busy": allow_busy, "pins_changed": pins_changed,
                     "watchdog_abort": watchdog_state["reason"]},
        "metrics": metrics, "conformance": conformance,
        "results": [r.__dict__ for r in results],
        "system": {"before": before, "after": after, "preflight": pre, "during": smp.summary(),
                   "contention": smp.contention, "power": pwr.summary(), "cpu": cpu.summary()},
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    if evaluation_campaign:
        summary["evaluation_campaign"] = evaluation_campaign
    if watchdog_enabled:
        summary["watchdog"] = {"enabled": True, "aborted": watchdog_state["reason"] is not None,
                               "reason": watchdog_state["reason"], "sample_time": watchdog_state["t"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in ctx.samples))
    # 1 Hz machine series (GPU, memory, load, resident models on every local server): the contention evidence.
    (run_dir / "sampler.jsonl").write_text("".join(json.dumps(s) + "\n" for s in smp.series))
    done = {"event": "done", "contended": bool(smp.contention), "must_fail": must_fail}
    if watchdog_state["reason"] is not None:
        done["watchdog_abort"] = watchdog_state["reason"]
    emit(done)
    return summary


def _resident_unknown(summary: dict) -> int | None:
    """Sampler samples whose resident-model state was unreadable; None when the summary carries no such count (no
    sampler block, or an incomplete one), which proves isolation no more than an unreadable sample does."""
    n = ((summary.get("system") or {}).get("during") or {}).get("resident_unknown_samples")
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None


def unsound(summary: dict) -> list[str]:
    v = summary["verdicts"]
    reasons = []
    if summary.get("evaluation_campaign"):
        reasons.append("behavioral evaluation campaign: these runs are not performance or golden evidence")
    # The side-model regime (workloads.side_regime) records co-resident models and unreadable residency; they do not
    # make its legs unsound.
    side = workloads.side_regime(summary)
    if v["contended"] and not side:
        reasons.append("CONTENDED: another model was resident or running during the run (system.contention)")
    # An unreadable resident endpoint is missing evidence, not evidence of a competing model; so is a missing count.
    unknown = _resident_unknown(summary)
    if not side and unknown is None:
        reasons.append("residency sampling incomplete: no system.during.resident_unknown_samples; run is non-proof")
    elif not side and unknown:
        reasons.append(f"resident model state unknown in {unknown} sampler sample(s); run is non-proof")
    if v["must_fail"]:
        reasons.append(f"MUST FAIL: {', '.join(v['must_fail'])}")
    for case, entry in sorted(summary["conformance"].items()):
        if entry["level"] == "MUST" and entry["verdict"] == "VOID":
            reasons.append(f"MUST VOID: {case} (required conformance was not established)")
    if v["preflight_problems"]:
        reasons.append(f"preflight: {'; '.join(v['preflight_problems'])}")
    if v.get("pins_changed"):
        reasons.append(f"PINS CHANGED mid-run (another generation measured part of it): {v['pins_changed']}")
    if v.get("watchdog_abort"):
        reasons.append(f"watchdog aborted run: {v['watchdog_abort']}")
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


# Micro samples remain in runs/. Passing conf/replay diagnostics stay compact; non-PASS verdicts and replay
# perf rows retain their counts in receipts (perf rows have no verdict and are accessible via --path).
DETAILS_NOT_BANKED = frozenset({"micro"})


def _receipt_view(summary: dict) -> dict:
    """Bank non-PASS conf/replay diagnostics, including replay perf counts needed to audit fixture context."""
    details = {r["case"]: r["detail"] for r in summary.get("results", [])
               if r["tier"] not in DETAILS_NOT_BANKED
               and (r["tier"] not in ("conf", "replay") or r.get("verdict") != "PASS")}
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
                backend = SERVERS[gp["backend"]](receipt["runs"][0]["provenance"]["fingerprint"]["model_dir"])
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


STATUS_SNAPSHOT = Path.home() / ".localbench" / "status-snapshot.json"
STATUS_SNAPSHOT_TTL = 45.0


def _status_snapshot_read(now: float | None = None, host_id: str | None = None) -> dict | None:
    now = time.time() if now is None else now
    try:
        doc = json.loads(STATUS_SNAPSHOT.read_text())
        sampled = float(doc["sampled_at"])
        data = doc["data"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    age = max(0.0, now - sampled)
    if age > STATUS_SNAPSHOT_TTL or (host_id is not None and data.get("host_id") != host_id):
        return None
    return {**data, "status_snapshot": {"sampled_at": sampled, "age_s": age,
                                      "stale": False, "ttl_s": STATUS_SNAPSHOT_TTL}}


def _status_snapshot_write(data: dict, sampled: float) -> None:
    STATUS_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_SNAPSHOT.with_name(f".{STATUS_SNAPSHOT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"sampled_at": sampled, "data": data}, sort_keys=True, default=str) + "\n")
    os.replace(tmp, STATUS_SNAPSHOT)


def status_report() -> dict:
    """The facts `localbench status` shows, as data: per golden either why it is UNAVAILABLE or its tiers grouped by
    the pins that moved (CURRENT when none did), replay fixtures against the running omp, park and quiet state,
    ollama residency and auto-update, the smol server, sessions started while parked, and GPU users over 3 s."""
    host = sysstats.host()
    cached = _status_snapshot_read(host_id=host.get("host_id"))
    if cached is not None:
        return cached
    sampled = time.time()
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
    data = {"host_id": host["host_id"], "goldens": goldens,
            "running_omp": {"omp_version": running_gen[0], "omp_sha": running_gen[1]}, "fixtures": fixtures,
            "parked": json.loads(park.STATE.read_text()) if park.STATE.exists() else [],
            "ollama_loaded": None if residents is None else [{"model": m, "until": u} for m, u in residents],
            "ollama_auto_update": sysstats.ollama_auto_update(), "smol": smol_view,
            "quiet_paused": quiet.paused(), "started_while_parked": park.stuck_sessions(sysstats.omp_processes()),
            "gateway": gateway.status(resident_state=residents), "heavy_slot": heavyslot.holder(),
            "gpu_last_3s": sysstats.gpu_share(before, sysstats.gpu_time_by_pid(), time.time() - t0, min_pct=5.0)}
    _status_snapshot_write(data, sampled)
    return {**data, "status_snapshot": {"sampled_at": sampled, "age_s": 0.0, "stale": False, "ttl_s": STATUS_SNAPSHOT_TTL}}


def cmd_slot(args) -> int:
    """The heavy slot now: holder (pid, verb, repo, held-for, expected remaining = median wall_s of that verb's last 20
    holds, unknown without history) and the --wait-slot queue in enqueue order with each waiter's estimated start."""
    rep = heavyslot.report()
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    def secs(v):
        return "unknown" if v is None else f"{v:.0f}s"

    h = rep["holder"]
    print("holder: none (slot free)" if h is None else
          f"holder: pid {h.get('pid')} {h.get('verb')} in {h.get('repo') or '?'}, held {secs(h.get('held_s'))}, "
          f"expected remaining {secs(h.get('expected_remaining_s'))}")
    if not rep["queue"]:
        print("queue: empty")
        return 0
    print("\n".join(render.table(["pos", "pid", "verb", "repo", "waited", "est_start"],
                                 [[str(t["position"]), str(t.get("pid")), str(t.get("verb")), str(t.get("repo") or "-"),
                                   secs(t["waited_s"]), secs(t["estimated_start_s"])] for t in rep["queue"]])))
    return 0


def cmd_load(args) -> int:
    """Read-only LOAD_COMMAND process attribution: CPU, scheduler/system calls, spawns, memory, and OMP session trees."""
    from . import load

    try:
        report = load.collect(args.seconds)
    except load.LoadError as exc:
        return _fail(f"load: {exc}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(load.render(report))
    return 0


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
    residency = st["gateway"]
    service = residency["service"]
    health = "healthy" if service["health"] else "UNAVAILABLE"
    print(f"OMP Ollama gateway: {health} ({service['launchd_state']}; {service['bind']})")
    routes = ", ".join(f"{name}={url}" for name, url in residency["profiles"].items()) or "none"
    print("OMP Ollama profiles: " + routes)
    print(f"gateway requests in flight: {residency['active_requests']}")
    for lease in residency["leases"]:
        expires = max((t for t in (lease["idle_expires_at"], lease["manual_expires_at"]) if t is not None),
                      default=None)
        when = _when(expires) if expires is not None else "none"
        completed = _when(lease["last_completed_at"]) if lease["last_completed_at"] is not None else "never"
        print(f"  lease {lease['model']}: profiles={','.join(lease['profiles']) or 'unknown'} "
              f"active={lease['active_requests']} last_completed={completed} expires={when} "
              f"outcome={lease['last_outcome'] or 'active'}")
    unowned = residency["unowned_residents"]
    print("unowned Ollama residents: " +
          ("unknown (/api/ps unavailable)" if unowned is None else ", ".join(unowned) or "none"))
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
    slot = st["heavy_slot"]
    print("heavy slot: " + ("free" if slot is None else
                            f"{slot.get('verb')} pid {slot.get('pid')} since {slot.get('started_at')}"
                            + (f" in {slot['repo']}" if slot.get("repo") else "")))
    print("GPU, last 3 s: " + (", ".join(sysstats.proc_label(r) for r in st["gpu_last_3s"]) or "no process above 5%"))
    return rc


def cmd_gpu(args) -> int:
    """Who is using the GPU, and who can send it inference work: per-process GPU share over a window, models
    resident on each local server, processes connected to those servers, and, for each omp client, the features
    its profile routes to a local model (from omp's own resolved settings)."""
    conn0 = sysstats.connection_bytes()
    traffic_t0 = time.monotonic()
    device = sysstats.gpu_window(args.seconds)
    conn1 = sysstats.connection_bytes()
    traffic_window_s = time.monotonic() - traffic_t0
    report = {"window_s": device["window_s"], "traffic_window_s": traffic_window_s,
              "gpu_by_process": device["gpu_by_process"], "device": device,
              "resident": sysstats.resident_models(), "clients": sysstats.inference_clients()}
    for c in report["clients"]:
        c["traffic"] = sysstats.traffic(conn0, conn1, c["conns"])
    for s in park.stuck_sessions(report["clients"]):
        next(c for c in report["clients"] if c["pid"] == s["pid"])["started_while_parked"] = s["window"]
    routes: dict[str, dict] = {}
    for c in report["clients"]:
        if "omp_profile" in c:
            prof = c["omp_profile"]
            routes.setdefault(prof, models.local_routes(prof))   # a disabled subagent prints as disabled, not a route
            c["local_routes"] = routes[prof]
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    device_pct = device.get("device_pct")
    coverage = device.get("coverage")
    unattributed = device.get("unattributed_pct")
    device_text = "unavailable" if device_pct is None else f"{device_pct:.1f}%"
    coverage_text = ("n/a" if device["status"] == "IDLE" else "unavailable") if coverage is None else f"{coverage:.1f}%"
    unattributed_text = "n/a" if unattributed is None and device["status"] == "IDLE" else (
        "unavailable" if unattributed is None else f"{unattributed:.1f}%")
    print(f"GPU by process over {report['window_s']:g} s (% of wall time the process kept the GPU busy; a runner with several "
          f"queues can pass 100%; client traffic window {report['traffic_window_s']:g} s; "
          f"device {device_text} average, status {device['status']}, coverage {coverage_text}, "
          f"unattributed {unattributed_text} ({device['io_report_samples']} macmon IOReport samples):")
    for r in report["gpu_by_process"] or [{"pct": 0, "pid": "-", "name": "(nothing above 0.5%)", "cmd": ""}]:
        what = f"model {r['model']}" if r.get("model") else r["cmd"][:90]
        print(f"  {r['pct']:6.1f}%  pid {r['pid']:<6} {r['name']:<18} {what}")
    print("Resident models: " + "; ".join(f"{srv}: {', '.join(m) if m else '(none)' if m is not None else '(no answer)'}"
                                          for srv, m in report["resident"].items()))
    print(f"Clients of local inference servers (open connections; bytes sent up / received down in the {report['traffic_window_s']:g} s):")
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
    A window with no samples is an empty answer, not an error: exit 0, and the same shape under --json.
    `--by-purpose` / `--by-profile` instead sum the gateway's request counts and busy seconds (no request content);
    `--requests` lists its per-request rows (purpose, model, start/end, status, queue wait; never content)."""
    if args.by_purpose or args.by_profile:
        return _report_purposes(args)
    if args.requests:
        return _report_requests(args)
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
    print("Traffic by client (down = sampled response bytes; up = sampled request bytes; model = sample-time "
          "resident on that server, not per-request attribution):")
    for t in r["traffic"][:15]:
        print(f"  {_size(t['down']):>9} down {_size(t['up']):>9} up  {t['who']}  cwd={t['cwd']}  "
              f"-> {t['server']} ({t['while_resident']})")
    if not r["traffic"]:
        print("  none recorded (traffic is sampled since this watcher version; restart `localbench watch`)")
    return 0


def _report_requests(args) -> int:
    """Gateway per-request rows: status, wait, and abandonment reason. Busy duration is shown only when known.
    Metadata only; an empty window is an empty answer, not an error."""
    until = time.time()
    since = until - args.since
    path = gateway.database_path()
    rows = gateway.GatewayStore(path).requests_report(
        since, until, purpose=args.req_purpose, profile=args.req_profile, limit=args.limit) if path.is_file() else []
    if args.json:
        print(json.dumps({"since": since, "until": until, "rows": rows}, indent=2))
        return 0
    if not path.is_file():
        print(f"no gateway database at {path}; `localbench gateway start` records requests", file=sys.stderr)
    print("\n".join(render.table(
        ["started", "purpose", "model", "profile", "status", "abandon_reason", "queue_wait_s", "busy_s"],
        [[time.strftime("%m-%d %H:%M:%S", time.localtime(r["started_at"])), r["purpose"], r["model"],
          r["profile"] or "-", r["status"], r.get("abandon_reason") or "-",
          f"{r['queue_wait_s']:.2f}" if r["queue_wait_s"] is not None else "-",
          f"{r['busy_s']:.1f}" if r["busy_s"] is not None else "-"] for r in rows])))
    return 0


def _report_purposes(args) -> int:
    """Gateway purpose_stats over the window, grouped by purpose, by profile, or (both flags) by the pair. Hourly
    buckets are summed whole, so the window edges are approximate to the hour."""
    until = time.time()
    since = until - args.since
    path = gateway.database_path()
    rows = gateway.GatewayStore(path).purpose_report(since, until) if path.is_file() else []
    keys = [k for k, on in (("profile", args.by_profile), ("purpose", args.by_purpose)) if on]
    groups: dict[tuple, dict] = {}
    for r in rows:
        g = groups.setdefault(tuple(r[k] for k in keys), {**{k: r[k] for k in keys}, "requests": 0, "busy_s": 0.0})
        g["requests"] += r["requests"]
        g["busy_s"] += r["busy_s"]
    out = sorted(groups.values(), key=lambda g: (-g["busy_s"], -g["requests"], *[g[k] for k in keys]))
    if args.json:
        print(json.dumps({"since": since, "until": until, "by": keys, "rows": out}, indent=2))
        return 0
    if not path.is_file():
        print(f"no gateway database at {path}; `localbench gateway start` records purposes", file=sys.stderr)
    print("\n".join(render.table([*keys, "requests", "busy_s"],
                                 [[*(str(g[k]) for k in keys), str(g["requests"]), f"{g['busy_s']:.1f}"]
                                  for g in out])))
    return 0


def cmd_models(args) -> int:
    """Local models: installed on ollama, mlx-serve and Splash, plus the ones omp runs on the CPU itself (tiny models,
    mnemopi's embedding model); whether each is the newest build of its source, which omp profiles/features route to
    it, and what appeared upstream in the last `--days` days."""
    installed = models.ollama_models() + models.mlx_models()
    cpu = models.omp_cpu_models()
    names = models.profiles()
    disabled: dict[str, list[str]] = {}
    uses = models.routes_by_model(names, disabled)
    for m in installed + cpu:
        m["routes"] = uses.get(m.get("source", m["name"]), {})
    new = models.releases(args.days, installed)
    if args.json:
        # Profiles once; per model {feature: [profiles]} (was a flat "profile: feature" string per pair, which repeated
        # every feature name once per profile: 3.5k tokens, audit 2026-09-23).
        print(json.dumps({"profiles": names, "installed": installed, "omp_cpu": cpu, "releases": new,
                          "disabled": disabled}, separators=(",", ":")))
        return 0
    print(f"omp profiles ({len(names)}): {', '.join(names)}")
    for m in installed + cpu:
        label = m["name"] + (f"  [parked: {m['source']}]" if m.get("parked") else "")
        print(f"{m['server']:<9} {label:<48} {(m.get('digest') or m.get('installed_artifact') or 'unknown'):<12} "
              f"{m['gb']:>6.1f} GB  {m['freshness']}")
        if source := m.get("installed_artifact_source"):
            print(f"  installed identity source: {source}")
            date = (f"{m['upstream_date_source']} {m['upstream_modified']}"
                    if m.get("upstream_modified") else "date unavailable")
            print(f"  upstream {m.get('upstream_sha') or 'unavailable'} ({date})")
        elif m["server"] == "ollama" and not m["freshness"].startswith("cloud"):
            print("  upstream release date: unavailable")
        for feature, who in m["routes"].items():
            print(f"{'':12}{feature}: {render.members(who, names, 'profiles')}")
    for feature, who in disabled.items():
        print(f"disabled  {feature}: {render.members(who, names, 'profiles')} ({models.DISABLED})")
    print(f"\nNew or updated upstream in the last {args.days} days (publishers and families in use):")
    for r in new:
        print(f"  {r['modified']}  {r['id']}  ({r['task'] or '-'})")
    return 0


DECISION_BASE = "http://127.0.0.1:11434"   # the Ollama runtime itself, like every backend; never the gateway


@_held("decision run", run_only=True)
def cmd_decision(args) -> int:
    """`decision run ollama:<model> --suite <role|path>`: every suite item, `--repeats` times, against Ollama's POST
    /v1/systemone on loopback (localbench/decision.py), under a Sampler that records co-resident load. Side-model law:
    load is the measured condition, never a refusal or a void, so there is no one-model preflight and the receipt's
    verdicts.contended stays false. `--hosted-arm` adds hosted decision service on the same items in the same invocation (a
    comparison arm, never a fallback; refused without its key). `--feature <registry id>` stamps the receipt with that
    features.tsv row and the sha of its module in the installed omp, the pair a feature's proof is matched on. Writes
    runs/<UTC>__decision__<model>/summary.json and banks the receipt as run_suite returns it (verdict, problems,
    feature); exit 1 when it carries problems, a comparison that is not BETTER among them."""
    from . import decision
    if getattr(args, "decision_action", "run") == "derive":
        return cmd_decision_derive(args)
    if getattr(args, "decision_action", "run") == "paired":
        return cmd_decision_paired(args)
    if args.spec.startswith(decision.LAYA_PREFIX):
        return cmd_decision_laya(args)

    model = _ollama_model(args.spec)
    try:
        suite = decision.resolve_suite(args.suite)
    except (decision.SuiteError, OSError) as exc:
        _usage(f"decision run --suite {args.suite}: {exc}")
    hosted = decision.Hosted() if args.hosted_arm else None
    feature, module_sha, module_why = args.feature, None, None
    if feature is not None:
        module_sha, module_why = _feature_module_sha(feature)
    refuse = None
    if _run_alive():
        refuse = "a localbench run is alive; a decision run would share its GPU and perturb both"
    elif hosted is not None and not os.environ.get(hosted.api_key_env, "").strip():
        refuse = (f"--hosted-arm needs {hosted.api_key_env} in the environment; the hosted arm never runs without it "
                  "and never stands in for the local arm")
    elif feature is not None and module_sha is None:
        refuse = f"--feature {feature}: {module_why}; a receipt without its module sha proves nothing for the feature"
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{created}__decision__{golden.slug(model)}"
    rel_dir = run_dir.relative_to(ROOT)
    arms = f"{model}" + (f" and hosted {hosted.model}" if hosted else "")
    steps = [Step(f"ask {arms} every item of suite {suite.name} ({len(suite.items)} item(s), role {suite.role}) "
                  f"{args.repeats} time(s) via {DECISION_BASE}/v1/systemone",
                  "label log-probs, no sampling: quality is deterministic; latency is measured under the recorded load"),
             Step(f"write {rel_dir / 'summary.json'}", "the run as measured; the receipt is banked from it"),
             Step(f"bank {Path('docs/evidence/receipts') / f'decision__{model}__{suite.name}__<created>.json'}",
                  "the evidence a feature's proof cites; a run with problems banks too and exits 1"
                  + (f" (feature {feature}, omp_module_sha {module_sha[:12]})" if module_sha else ""))]
    m = _mut(args)
    if (rc := m.gate(steps, refuse=refuse)) is not None:
        return rc
    waited = _decision_wait_idle(args.wait_idle)
    sampler = sysstats.Sampler(target=("ollama", model))   # unentered: run_suite enters it around the requests
    try:
        receipt = decision.run_suite(DECISION_BASE, model, suite, repeats=args.repeats, hosted=hosted,
                                     sampler=sampler, rev=_rev(), feature=feature, omp_module_sha=module_sha,
                                     allow_evict=getattr(args, "allow_evict", False),
                                     in_use=None)   # None: decision.default_in_use, the live users of the model
    except decision.ContextError as exc:
        # The model cannot get the suite's context without evicting a user of it: refused before any item was sent.
        m.outcome, m.detail["reason"] = "refused", str(exc)
        return _fail(str(exc))
    except decision.RequestError as exc:
        return _fail(f"decision run: {exc}")
    run = receipt["run"]
    run["run_dir"] = str(rel_dir)
    run["system"]["wait_idle"] = waited
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    problems = list(receipt["problems"])
    path = _bank(f"decision__{model}__{suite.name}__{run['provenance']['created']}",
                 {**receipt, "problems": problems, "run": run})
    rel = str(path.relative_to(ROOT))
    m.detail.update(receipt=rel, run_dir=str(rel_dir), unsound=problems)
    print(f"banked decision receipt {rel}")
    for p in problems:
        print(f"UNSOUND  {p}", file=sys.stderr)
    m.outcome = "done"   # banked either way; exit 1 says the run has problems
    return 1 if problems else 0


def cmd_decision_paired(args) -> int:
    """`decision paired <receipt.json>`: per-question-type paired local-vs-hosted scoring of a banked decision
    receipt against the suite on disk (localbench/decision.py paired). Read-only: exit 1 on a scoring refusal,
    otherwise 0 with the table (exit code says nothing about the verdicts)."""
    from . import decision

    if not 0.0 < args.alpha < 1.0:
        _usage(f"decision paired --alpha {args.alpha}: a significance level strictly between 0 and 1")
    try:
        receipt = json.loads(Path(args.receipt).expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        _usage(f"decision paired {args.receipt}: {exc}")
    except ValueError as exc:
        return _fail(f"decision paired {args.receipt}: not JSON ({exc})")
    try:
        report = decision.paired(receipt, alpha=args.alpha, seed=args.seed)
    except decision.PairedError as exc:
        print(f"decision paired: refused: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"{report['suite']['name']}  alpha={report['alpha']} seed={report['seed']}")
    print("type n local hosted diff_pp b c mcnemar_p boot_mean [boot_lo,boot_hi] verdict")
    for kind, row in report["types"].items():
        boot = row["bootstrap"]
        print(f"{kind} {row['n']} {row['local_accuracy']:.4f} {row['hosted_accuracy']:.4f} "
              f"{row['diff_pp']:+.2f} {row['b']} {row['c']} {row['mcnemar_p']:.4g} "
              f"{boot['mean_pp']:+.2f} [{boot['lo_pp']:+.2f},{boot['hi_pp']:+.2f}] {row['verdict']}")
    return 0


def cmd_decision_laya(args) -> int:
    """`decision run laya:<hf repo>[@<subfolder>] --suite ...`: cmd_decision for a Laya-MLX checkpoint. A loopback
    shim (localbench/shims/laya_systemone.py, run by the laya venv's python3; decision.LayaShim) loads the checkpoint
    from the local Hugging Face cache, serves /v1/systemone on a free 127.0.0.1 port for this run only, and is always
    stopped afterwards; its log is runs/<UTC>__decision__<model>/laya-shim.log. No Ollama runner is touched: Laya
    truncates states over its max_len instead of refusing them, counted as decision.laya.truncated_items. A shim
    that cannot start is a refusal before any item (exit 1); the receipt is banked as for an ollama: spec."""
    from . import decision
    try:
        spec = decision.parse_laya_spec(args.spec)
    except ValueError as exc:
        _usage(f"decision run {exc}")
    model = spec.name
    try:
        suite = decision.resolve_suite(args.suite)
    except (decision.SuiteError, OSError) as exc:
        _usage(f"decision run --suite {args.suite}: {exc}")
    hosted = decision.Hosted() if args.hosted_arm else None
    feature, module_sha, module_why = args.feature, None, None
    if feature is not None:
        module_sha, module_why = _feature_module_sha(feature)
    venv = decision.laya_venv()
    refuse = None
    if _run_alive():
        refuse = "a localbench run is alive; a decision run would share its GPU and perturb both"
    elif not (venv / "bin" / "python3").is_file():
        refuse = f"no python3 in the laya venv {venv}; set LOCALBENCH_LAYA_VENV to the venv that has laya_mlx"
    elif hosted is not None and not os.environ.get(hosted.api_key_env, "").strip():
        refuse = (f"--hosted-arm needs {hosted.api_key_env} in the environment; the hosted arm never runs without it "
                  "and never stands in for the local arm")
    elif feature is not None and module_sha is None:
        refuse = f"--feature {feature}: {module_why}; a receipt without its module sha proves nothing for the feature"
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{created}__decision__{golden.slug(model)}"
    rel_dir = run_dir.relative_to(ROOT)
    arms = f"{model}" + (f" and hosted {hosted.model}" if hosted else "")
    receipt_name = f"decision__{golden.slug(model)}__{suite.name}"
    steps = [Step(f"start {venv / 'bin' / 'python3'} {decision.LAYA_SHIM.name} for {model} on a free 127.0.0.1 "
                  f"port, offline (log {rel_dir / 'laya-shim.log'})",
                  "Laya-MLX answers System One-shaped requests from the HF cache; stopped when the run ends, also "
                  "on failure"),
             Step(f"ask {arms} every item of suite {suite.name} ({len(suite.items)} item(s), role {suite.role}) "
                  f"{args.repeats} time(s) via the shim's /v1/systemone",
                  "label probabilities from one forward pass, no sampling: quality is deterministic; states over "
                  "Laya's max_len are truncated and counted"),
             Step(f"write {rel_dir / 'summary.json'}", "the run as measured; the receipt is banked from it"),
             Step(f"bank {Path('docs/evidence/receipts') / f'{receipt_name}__<created>.json'}",
                  "the evidence a feature's proof cites; a run with problems banks too and exits 1"
                  + (f" (feature {feature}, omp_module_sha {module_sha[:12]})" if module_sha else ""))]
    m = _mut(args)
    if (rc := m.gate(steps, refuse=refuse)) is not None:
        return rc
    waited = _decision_wait_idle(args.wait_idle)
    run_dir.mkdir(parents=True, exist_ok=True)
    # The backend's own GPU rows are matched by executable name in their command line (sysstats.gpu_is_ours): for
    # the shim that is the script the venv's python3 runs. Entered by run_suite around the requests.
    sampler = sysstats.Sampler(target=(decision.LAYA_SHIM.name, model))
    try:
        receipt = decision.run_laya(spec, suite, shim=decision.LayaShim(spec, log=run_dir / "laya-shim.log"),
                                    repeats=args.repeats, hosted=hosted, sampler=sampler, rev=_rev(),
                                    feature=feature, omp_module_sha=module_sha)
    except decision.ShimError as exc:
        m.outcome, m.detail["reason"] = "refused", str(exc)
        return _fail(f"decision run: {exc}")
    except decision.RequestError as exc:
        return _fail(f"decision run: {exc}")
    run = receipt["run"]
    run["run_dir"] = str(rel_dir)
    run["system"]["wait_idle"] = waited
    (run_dir / "summary.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    problems = list(receipt["problems"])
    path = _bank(f"{receipt_name}__{run['provenance']['created']}", {**receipt, "problems": problems, "run": run})
    rel = str(path.relative_to(ROOT))
    m.detail.update(receipt=rel, run_dir=str(rel_dir), unsound=problems)
    print(f"banked decision receipt {rel}")
    for p in problems:
        print(f"UNSOUND  {p}", file=sys.stderr)
    m.outcome = "done"   # banked either way; exit 1 says the run has problems
    return 1 if problems else 0


def cmd_decision_derive(args) -> int:
    """`decision derive ollama:<model> --num-ctx N [--name NAME]`: a model FROM <model> whose params layer sets
    num_ctx N (POST /api/create {model: NAME, from, parameters: {num_ctx}}), for /v1/systemone, which takes no options
    and loads a runner at the num_ctx the model ships (localbench/decision.py check_shipped_context). Read back via
    /api/show: num_ctx N, the base's weights blob and other parameters, the base's digest unchanged. NAME defaults to
    <model>-ctx<N>; an existing NAME is refused unless it already is this derivation (then a no-op). Never touches
    the base model."""
    from . import decision

    base = _ollama_model(args.spec)
    if args.num_ctx < 1:
        _usage(f"decision derive --num-ctx {args.num_ctx}: give a positive token count")
    m = _mut(args)
    try:
        plan = decision.plan_derive(DECISION_BASE, base, args.num_ctx, args.name)
    except decision.DeriveError as exc:
        return m.gate([], refuse=f"decision derive: {exc}")
    weights = (plan.get("base_weights") or "?")[:12]
    steps = [Step(f"POST {DECISION_BASE}/api/create {json.dumps(plan['request'])}",
                  f"a new model {plan['name']} FROM {base} (same weights blob {weights}) whose params layer sets "
                  f"num_ctx {args.num_ctx}: /v1/systemone loads runners at the shipped num_ctx"),
             Step(f"read back {plan['name']} via /api/show and /api/tags",
                  f"num_ctx {args.num_ctx}, weights {weights} and the base's other parameters, and {base} unchanged")]
    if (rc := m.gate(steps, refuse=plan["refuse"], noop=plan["noop"])) is not None:
        return rc
    try:
        out = decision.derive(DECISION_BASE, plan)
    except decision.DeriveError as exc:
        return _fail(f"decision derive: {exc}")
    m.detail.update(out)
    print(f"derived {out['name']} (digest {str(out['digest']).removeprefix('sha256:')[:12]}) from {base}: num_ctx "
          f"{out['num_ctx']}, weights {out['weights'][:12]} (the base's)")
    return 0


def _feature_module_sha(feature: str) -> tuple[str | None, str | None]:
    """(sha of the feature's registered module in the installed omp, None) or (None, why it cannot be read). An id
    that is not in registries/features.tsv is a usage error."""
    from . import features

    try:
        rows = features.load()
    except (OSError, ValueError) as exc:
        return None, f"feature registry unreadable ({exc})"
    row = next((r for r in rows if r["feature"] == feature), None)
    if row is None:
        _usage(f"decision run --feature {feature}: not a feature in {features.REGISTRY}; one of "
               f"{', '.join(sorted(r['feature'] for r in rows))} (`localbench features` lists them)")
    try:
        path = features.package_root(row["omp_package"], park.omp_package(omp_bin())) / row["omp_module"]
    except (OSError, ValueError, RuntimeError) as exc:
        return None, f"installed omp unreadable ({exc})"
    sha = features.module_sha(path)
    return (sha, None) if sha else (None, f"{row['omp_module']} is not in the installed {row['omp_package']} ({path})")


def _decision_wait_idle(seconds: float) -> dict | None:
    """`--wait-idle`: up to `seconds`, re-check the machine every 30 s and start once it is idle. Whatever is still
    busy is recorded, never a refusal (side-model law)."""
    if not seconds:
        return None
    t0 = time.time()
    while True:
        _summ, _cpu, problems = _busy_check()
        if not problems or time.time() - t0 >= seconds:
            return {"waited_s": round(time.time() - t0, 1), "still_busy": problems}
        time.sleep(min(30, max(0.0, seconds - (time.time() - t0))))


def cmd_features(args) -> int:
    """omp features that can route to a local model (registries/features.tsv) and the receipt proving each, per
    profile (localbench/features.py). Read-only. Exit 1 when any finding is FAIL: a local route without a current
    proof. --queue files or updates one open proof bead per unproven local route (audited)."""
    from . import features, proofqueue

    if getattr(args, "queue", False):
        units, open_beads, skipped = proofqueue.collect()
        planned = proofqueue.plan(units, open_beads)
        steps = [Step(*proofqueue.describe(a)) for a in planned if a["op"] != "noop"]
        m = _mut(args)
        noop = ("proof queue is empty: every local route already carries its trigger bead"
                if not steps else None)
        if (rc := m.gate(steps, noop=noop)) is not None:
            if not args.json:
                for entry in skipped:
                    print(f"skipped: {entry['preset']} ({entry['reason']})")
            return rc
        try:
            summary = proofqueue.apply([a for a in planned if a["op"] != "noop"])
        except proofqueue.BeadError as exc:
            return _fail(f"features --queue: {exc}")
        summary["skipped"] = skipped
        m.detail.update({key: summary[key] for key in ("filed", "updated", "adopted", "closed")})
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            for key in ("filed", "updated", "adopted", "closed"):
                for title in summary[key]:
                    print(f"{key}: {title}")
            for entry in skipped:
                print(f"skipped: {entry['preset']} ({entry['reason']})")
            print(f"proof queue: {len(summary['filed'])} filed, {len(summary['updated'])} updated, "
                  f"{len(summary['adopted'])} adopted, {len(summary['closed'])} closed, "
                  f"{len(summary['noop'])} unchanged, {len(skipped)} skipped")
        return 0

    names = models.profiles()
    rows = features.report(names)
    found = features.findings(rows)
    if args.json:
        print(json.dumps({"profiles": names, "features": rows}, default=str))
    else:
        from . import generation
        print("\n".join(features.lines(rows, names)))
        print("\n".join(generation.corpus_progress_lines()))
        for level, message, fix in found:
            if level != "PASS":
                print(f"{level}  {message}" + (f"\n      fix: {fix}" if fix else ""), file=sys.stderr)
    return 1 if any(level == "FAIL" for level, _, _ in found) else 0


@_held("prove")
def cmd_prove(args) -> int:
    """Run declarative proof specs (localbench/prove.py): `prove <spec>` runs one spec end to end
    (pre-reg commit, dataset pin, every candidate through its kind tier, assertions, banking,
    grading, bead updates); `--due` runs every spec with an open bead whose regime allows it.
    Audited mutation with a dry run that plans without inference or writes."""
    from . import prove

    if (getattr(args, "omp_frozen", False) and not getattr(args, "_omp_frozen_ready", False)
            and not _mut(args).dry_run):
        return _with_frozen_omp(cmd_prove, args, preserve_flag=True)
    if bool(args.spec) == bool(args.due):
        _usage("localbench prove <spec> | --due: exactly one")
    if args.due:
        try:
            pairs = [(path, prove.load_spec(path)) for path in prove.list_specs()]
        except prove.ProveError as exc:
            return _fail(f"prove --due: {exc}")
        try:
            due = prove.due_specs(pairs)
        except prove.ProveError as exc:
            return _fail(f"prove --due: {exc}")
        if not due:
            print("prove --due: no spec with an open bead and an allowing regime")
            return 0
        rc = 0
        for path, _ in due:
            rc = cmd_prove(_mimic(args, spec=str(path), due=False)) or rc
        return rc
    try:
        spec = prove.load_spec(args.spec)
    except prove.ProveError as exc:
        _usage(f"prove {args.spec}: {exc}")
    try:
        commit = prove.spec_commit(args.spec)
    except prove.ProveError as exc:
        return _fail(f"prove {args.spec}: {exc}")
    if spec.get("blocked"):
        return _fail(f"prove {args.spec}: spec is blocked ({spec['blocked'].get('reason')})")
    m = _mut(args)
    if m.dry_run:
        try:
            prove._refuse_rejected_screen(spec, args.spec)
        except prove.ProveError as exc:
            return _fail(str(exc))

    if m.dry_run and spec["kind"] == "generation":
        try:
            prove._generation_corpus(spec)
        except prove.ProveError as exc:
            return _fail(
                f"prove {args.spec}: {exc}; pinned corpus ids are write-once. "
                "Restore the registered bytes, or register a new corpus id and "
                "commit its matching hash.")
    try:
        steps = [Step(action, why) for action, why in
                 prove.plan_steps(spec, commit, omp_frozen=getattr(args, "omp_frozen", False))]
    except prove.ProveError as exc:
        return _fail(f"prove {args.spec}: {exc}")

    if (code := m.gate(steps)) is not None:
        return code
    try:
        buffered_br = getattr(args, "_frozen_proof_br", None)
        report = prove.prove_spec(args.spec, br=buffered_br) if buffered_br else prove.prove_spec(args.spec)
    except prove.ProveError as exc:
        return _fail(f"prove {args.spec}: {exc}")
    m.detail.update({c["route"]: c["grade"] for c in report.get("candidates", [])} or {"ran": True})
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        for candidate in report.get("candidates", []):
            print(f"{candidate['route']}: {candidate['grade']} ({candidate['reason']})")
        print(f"prove {args.spec}: " + ", ".join(
            f"{c['route']}={c['grade']}" for c in report.get("candidates", [])))
    return 0


def _mimic(args, **overrides):
    """A copy of parsed args with fields replaced (for --due dispatch without re-parsing)."""
    return argparse.Namespace(**{**vars(args), **overrides})


def cmd_watch_releases(args) -> int:
    """Upstream release watch (localbench/releasewatch.py). Bare: the queue of screens it filed (read-only). `--once`:
    one watch pass now (files beads, queues screens, marks seen); `--digest [--since 7d]`: the weekly new-model
    digest (files `screen <model> for <role>` beads, queue-capped); `--install-agent`: the weekly digest LaunchAgent
    (Monday 09:00 local); `--install-daily`: the daily `--once` one. State-changing verbs are audited mutations
    with a dry run."""
    from . import releasewatch

    if args.digest:
        try:
            since_days = releasewatch.parse_since(args.since)
        except releasewatch.WatchError as exc:
            _usage(f"watch-releases {exc}")
    else:
        since_days = None
    if args.once or args.install_agent or args.digest or args.install_daily:
        steps, refuse = [], None
        try:
            if args.install_agent:
                steps.append(Step(f"write {releasewatch.digest_plist_path()} and (re)load it with launchctl",
                                  "a weekly Monday 09:00 local digest pass; launchctl bootout and rm undo it"))
            if args.install_daily:
                steps.append(Step(f"write {releasewatch.plist_path()} and (re)load it with launchctl",
                                  f"a daily {releasewatch.SCHEDULE['Hour']:02d}:"
                                  f"{releasewatch.SCHEDULE['Minute']:02d} watch pass; launchctl bootout and rm undo it"))
            if args.once:
                steps.append(Step(f"one watch pass now: fetch upstream releases, file a bead per new release, queue "
                                  f"its screen and mark it seen under {releasewatch.watch_dir()}",
                                  "new releases of the model families in use become screen work, never an install"))
            if args.digest:
                steps.append(Step(f"one digest pass now: fetch the publishers' last-{args.since} models, file "
                                  f"`screen <model> for <role>` beads and mark them seen under "
                                  f"{releasewatch.watch_dir()}",
                                  "new models of any family become screen work, never an install"))
        except releasewatch.WatchError as exc:   # e.g. a watch dir inside the repo: refused, never written
            refuse = f"watch-releases: {exc}"
        if (rc := _mut(args).gate(steps, refuse=refuse)) is not None:
            return rc
    rc = releasewatch.cli(once=args.once, install=args.install_agent, as_json=args.json, digest_days=since_days,
                          install_daily=args.install_daily, replay_window=args.replay_window)
    if args.once or args.digest:
        from . import proofqueue
        try:
            queued = proofqueue.queue()
        except Exception as exc:  # the trigger must not take down the pass it follows
            print(f"proof queue: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            _mut(args).detail.update(proof_queue={key: queued[key] for key in ("filed", "updated", "adopted", "closed")})
            if not args.json:
                print(f"proof queue: {len(queued['filed'])} filed, {len(queued['updated'])} updated, "
                      f"{len(queued['adopted'])} adopted, {len(queued['closed'])} closed")
    return rc


def _profiles_arg(text: str) -> list[str]:
    profiles = [p.strip() for p in text.split(",") if p.strip()]
    if not profiles:
        raise argparse.ArgumentTypeError("name at least one omp profile, e.g. --profiles default,lab")
    bad = [p for p in profiles if "/" in p or "\\" in p or p in (".", "..") or ".." in p]
    if bad:   # a profile is a directory name under ~/.omp/profiles; a path would escape it
        raise argparse.ArgumentTypeError(f"profile name(s) {bad}: plain profile names only, no '/' or '..'")
    return profiles


def cmd_preset(args) -> int:
    """omp config presets (localbench/presets.py, registries/presets.json): `list`, `show`, `plan` (read-only),
    `apply` (backed up, read back through omp, restored on any mismatch; a local preset on the live target needs a
    PROVEN feature receipt unless --force), `rollback <id>` and `drift` (exit 1 when live config left an applied
    preset)."""
    from . import presets

    action = args.preset_action
    if action == "list":
        rows = presets.load()["presets"]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for p in rows:
                print(f"{p['name']:<32} {'local' if p['local'] else 'hosted':<6} {len(p['ops'])} op(s)")
        return 0
    if action == "show":
        print(json.dumps(presets.find(args.name), indent=2))
        return 0
    if action == "drift":
        rows = presets.drift(args.profiles)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                print(f"DRIFT {r['profile']}: {r['preset']} ({r['id']}) {r['key']} expected {json.dumps(r['expected'])}"
                      f", is {json.dumps(r['actual'])}")
            if not rows:
                print("no drift: every applied preset still holds")
        return 1 if rows else 0
    if action == "rollback":
        steps = [Step(f"restore every profile file preset apply {args.id} backed up, byte for byte",
                      "undoes exactly that apply; refused when a file changed after it unless --force")]
        if (rc := _mut(args).gate(steps)) is not None:
            return rc
        out = presets.rollback(args.id, force=args.force)
        print(f"rolled back {out['preset']} ({out['id']}): restored {', '.join(out['restored']) or 'nothing'}")
        return 0
    p = presets.plan(args.name, args.profiles, args.target, force=args.force)
    if action == "plan":
        print(json.dumps(p, indent=2, default=str) if args.json else "\n".join(presets.lines(p)))
        return 1 if p["refused"] else 0
    # apply: the plan's lines are the dry run; the refusal (an unproven local preset on live) is the gate's.
    why = "backed up under ~/.localbench/rollback/preset-<id>, read back through omp, restored on any mismatch"
    steps = [Step(line, why) for line in presets.lines(p)]
    noop = None if any(s["changed"] for s in p["steps"]) else f"preset {args.name} already holds on {args.profiles}"
    m = _mut(args)
    if (rc := m.gate(steps, refuse=p["refused"], noop=noop)) is not None:
        return rc
    try:
        manifest = presets.apply(args.name, args.profiles, args.target, force=args.force)
    except presets.PresetRefused as exc:   # the plan moved between the gate and the write
        return _fail(f"preset apply: {exc}")
    m.detail.update(id=manifest["id"], forced=manifest["forced"])
    print(f"applied {args.name} to {','.join(args.profiles)} ({args.target}); rollback id {manifest['id']}")
    for note in p["notes"]:
        print(f"note: {note}", file=sys.stderr)
    return 0


def _gate_arg(text: str) -> dict:
    """`--gate METRIC=OP:VALUE` (e.g. decision.noul.accuracy=min:0.8) -> {metric: {op: value}}."""
    metric, _, bound = text.partition("=")
    op, _, value = bound.partition(":")
    try:
        number = float(value)
    except ValueError:
        number = None
    if not metric or op not in ("min", "max") or number is None or not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"gate {text!r}: use METRIC=min:VALUE or METRIC=max:VALUE, e.g. "
                                         "decision.noul.accuracy=min:0.8")
    return {metric: {op: number}}


def _gates(values: list[dict]) -> dict:
    out: dict = {}
    for g in values:
        dup = set(out) & set(g)
        if dup:
            _usage(f"--gate {sorted(dup)[0]} given twice; one bound per metric")
        out |= g
    return out
@_held("generation replay")
def cmd_generation(args) -> int:
    """Replay a generation corpus through one arm (localbench/generation.py run_candidate):
    `generation replay --corpus <dir> --candidate route:ollama/<model>|builtin:<kind>`
    re-sends each captured body with the model set per arm (builtin arms reproduce omp's
    no-model fallback with no model call), runs FEATURE_CHECKS, and reports violations
    with latencies. Evidence only: prove.py banks and grades. Audited mutation with a
    dry run that plans without inference."""
    from . import generation

    if args.generation_action != "replay":
        _usage(f"generation {args.generation_action}: only replay")
    root = Path(args.corpus).expanduser()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return _fail(f"generation replay: no readable corpus manifest ({manifest_path}: {exc})")
    if manifest.get("kind") != "generation":
        return _fail(f"generation replay: {manifest_path} is not a generation corpus")
    candidate = args.candidate
    if candidate.startswith("route:"):
        candidate = {"route": candidate[len("route:"):]}
    elif candidate.startswith("builtin:"):
        candidate = {"builtin": candidate[len("builtin:"):]}
    else:
        return _fail("generation replay --candidate: route:ollama/<model> or builtin:<kind>")
    try:
        items = [json.loads(line) for line in (root / "items.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return _fail(f"generation replay: unreadable corpus items ({exc})")
    if args.max_items is not None:
        items = items[:args.max_items]
    m = _mut(args)
    steps = [Step(f"replay {len(items)} {manifest.get('feature')} requests through {args.candidate}",
                   "captured messages byte-faithful, model set per arm, through the managed gateway")]
    if (rc := m.gate(steps)) is not None:
        return rc
    try:
        out = generation.run_candidate({"kind": "generation", "feature": manifest.get("feature")},
                                       candidate, items)
    except generation.corpus.CorpusError as exc:
        return _fail(f"generation replay: {exc}")
    m.detail.update({"candidate": args.candidate, "items": len(items),
                     "violated": sum(1 for o in out["outcomes"] if o["violations"])})
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    bad = sum(1 for o in out["outcomes"] if o["violations"])
    lats = sorted(o["latency_s"] for o in out["outcomes"] if o["latency_s"] >= 0)
    p50 = lats[len(lats) // 2] if lats else float("nan")
    print(f"generation replay {args.candidate} on {len(items)} items: "
          f"{len(items) - bad} clean, {bad} with violations, latency p50 {p50:.2f} s")
    return 0


def cmd_corpus(args) -> int:
    """Private decision corpora (localbench/corpus.py, localbench/jevsuites.py) under ~/.localbench/corpora, never in
    the repo. `import` turns one omp profile's hosted judgments into a suite; `proj-b-build` builds a named suite from
    the proj-b checkout; both need an explicit --gate (no default bar). `capture on|off|status` sets the opt-in,
    capped, expiring capture of request bodies by purpose. import, proj-b-build and capture on/off are audited
    mutations with a dry run; list, stats and capture status read."""
    from . import corpus, decision, jevsuites

    action = args.corpus_action
    state = getattr(args, "capture_state", None)
    if action == "capture" and state == "on" and not (args.purpose and args.max_items and args.minutes):
        _usage("corpus capture on needs --purpose P (repeatable), --max-items N and --minutes M: capture is "
               "opt-in, capped and expiring")
    if action == "capture" and state != "on" and (args.purpose or args.max_items or args.minutes):
        _usage(f"corpus capture {state} takes no --purpose/--max-items/--minutes")
    steps, refuse = None, None
    if action == "import":
        try:
            source = corpus.cache_path(args.profile)
        except corpus.CorpusError as exc:
            source, refuse = args.profile, f"corpus import: {exc}"
        steps = [Step(f"read {source} and write a {args.role} suite under {corpus.CORPORA}/{args.role}/<sha>/ "
                      f"(gate {json.dumps(_gates(args.gate))})",
                      "hosted judgments become a private, content-addressed decision suite; never in the repo")]
    elif action == "proj-b-build":
        steps = [Step(f"build suite {args.name} under {jevsuites.OUT_ROOT} (gate {json.dumps(_gates(args.gate))})",
                      "a named suite from the proj-b checkout, with an explicit pass bar")]
    elif action == "capture" and state in ("on", "off"):
        what = (f"enable capture of {', '.join(args.purpose)} request bodies: at most {args.max_items} item(s), "
                f"expiring in {args.minutes:g} min" if state == "on" else "disable capture (purposes and cap kept)")
        steps = [Step(f"write {corpus.capture_spec_path()}: {what}",
                      "request bodies hold user code: capture is opt-in, capped and expires on its own")]
    if steps is not None and (rc := _mut(args).gate(steps, refuse=refuse)) is not None:
        return rc
    try:
        if action == "list":
            out = corpus.list_corpora()
            text = [f"{s['role'] or '-':<16} {s['items']:>6} item(s)  {s['name']}  {s['directory']}" for s in out]
        elif action == "stats":
            out = corpus.stats()
            text = [f"{out['suites']} suite(s), {out['items']} item(s)"] + [
                f"  {role}: {e['suites']} suite(s), {e['items']} item(s)" for role, e in sorted(out["roles"].items())]
        elif action == "capture":
            out = (corpus.capture_on(args.purpose, args.max_items, args.minutes) if state == "on" else
                   corpus.capture_off() if state == "off" else corpus.capture_status())
            text = [f"capture {state}: " + json.dumps(out, sort_keys=True, default=str)]
        elif action == "import":
            out = corpus.import_profile(args.profile, roles=(args.role,), model=args.model, gate=_gates(args.gate))
            text = [f"{role}: {s['items']} item(s) -> {s['suite'] or '(none)'} {s['directory'] or ''}".rstrip()
                    + f"  skipped {s['skipped']}" for role, s in out.items()]
        else:
            out = jevsuites.build(args.name, gate=_gates(args.gate))
            text = [f"built {out['suite']['name']} ({out['suite']['n_items']} item(s)) in {out['directory']}"] + [
                f"  excluded {reason}: {e['n']}" for reason, e in out["excluded"].items()]
    except (corpus.CorpusError, decision.SuiteError, jevsuites.BuildError, OSError) as exc:
        return _fail(f"corpus {action}: {exc}")
    print(json.dumps(out, indent=2, default=str) if args.json else "\n".join(text))
    return 0


def _memory_legs(sources: list[str]) -> list[dict]:
    """`memory-verdict` legs: execute summaries from run dirs (their summary.json) or summary files, or the run views
    a receipt banked (kind run: run; aa: runs; ab: legs)."""
    legs = []
    for src in sources:
        path = Path(src).expanduser()
        path = path / "summary.json" if path.is_dir() else path
        if not path.is_file():
            _usage(f"memory-verdict: no such run dir or receipt {src}")
        doc = _load_json("memory-verdict", path)
        kind = doc.get("kind")
        found = ([doc["run"]] if kind == "run" else doc.get("runs") if kind == "aa" else doc.get("legs")
                 if kind == "ab" else [doc] if "provenance" in doc else None)
        if not found:
            _usage(f"memory-verdict: {src} is neither a run summary nor a run/aa/ab receipt")
        legs += found
    return legs


def cmd_memory_verdict(args) -> int:
    """Bank a memory proof receipt (workloads.memory_verdict) comparing candidate legs of a memory configuration with
    baseline-route legs (mem+sess tiers, >= 2 per arm), stamped with the feature's installed omp_module_sha as
    `decision run --feature` does. Legs with unknown resident-model state are non-proof and refused, as aa/ab do.
    Exit 1 when the receipt carries problems (it is banked all the same, as evidence). A receipt `localbench validate`
    would call INVALID (validate_doc) is never written: exit 1, each reason on stderr."""
    candidate, baseline = _memory_legs(args.candidate), _memory_legs(args.baseline)
    module_sha, why = _feature_module_sha(args.feature)
    refuse = (f"--feature {args.feature}: {why}" if module_sha is None else
              "memory-verdict: a leg's resident-model state is unknown or unrecorded; cannot bank an unproven "
              "comparison" if _unknown_residency(*candidate, *baseline) else None)
    path = _receipt_path(args.bank)
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    steps = [Step(f"{'replace' if path.exists() else 'write'} {rel}: memory verdict of {len(candidate)} candidate vs "
                  f"{len(baseline)} baseline leg(s) for {args.feature}"
                  + (f" (omp_module_sha {module_sha[:12]})" if module_sha else ""),
                  "the receipt a memory feature's proof is matched on; a not-BETTER verdict banks as evidence")]
    m = _mut(args)
    if (rc := m.gate(steps, refuse=refuse)) is not None:
        return rc
    receipt = workloads.memory_verdict(candidate, baseline, feature=args.feature, omp_module_sha=module_sha)
    invalid = validate_doc(receipt)
    if invalid:
        m.outcome = "refused"
        m.detail.update(invalid=invalid)
        print(f"INVALID memory verdict for {rel}; nothing written:", file=sys.stderr)
        for reason in invalid:
            print(f"  {reason}", file=sys.stderr)
        return 1
    banked = _bank(args.bank, receipt)
    problems = list(receipt.get("problems") or [])
    m.detail.update(receipt=str(banked), unsound=problems)
    print(f"banked memory verdict {rel}: {(receipt.get('verdict') or {}).get('compare', receipt.get('verdict'))}")
    for p in problems:
        print(f"UNSOUND  {p}", file=sys.stderr)
    m.outcome = "done"   # banked either way; exit 1 says it is not proof
    return 1 if problems else 0


def _omp_watch_plist() -> Path:
    from . import ompupdate

    return Path.home() / "Library" / "LaunchAgents" / f"{ompupdate.DEFAULT_LABEL}.plist"


def _freeze_steps(p) -> list[Step]:
    """The plan of `omp freeze` for ompfreeze.plan() `p` (empty when its snapshot exists: reuse copies nothing)."""
    from . import ompfreeze

    if p.exists:
        return []
    steps = [Step(f"copy {len(p.packages)} packages ({p.size_bytes / 1e6:.0f} MB) from {p.install} into "
                  f"{p.target.parent}/.tmp-{p.snapshot_id}-<pid>, then rename it to {p.target}",
                  "omp and its runtime dependency closure as one read-only copy: an omp update (uca, about every 3 h) "
                  "cannot change it under a running study"),
             Step(f"copy bun {p.bun} to {p.target / 'bin' / 'bun'}",
                  "the runtime that runs omp's dist entry, pinned with it (its version goes into the marker)"),
             Step(f"write {ompfreeze.entry_path(p.target)}: exec the frozen bun on the frozen "
                  f"{p.entry.relative_to(p.package)}",
                  "the LOCALBENCH_OMP entry; its sha16 is the omp_sha every leg run through it pins")]
    steps += [Step(f"remove snapshot {d}", f"retention keeps the newest {ompfreeze.KEEP} snapshots this verb made "
                                           "(a run holding one keeps it)")
              for d in ompfreeze.prune_candidates(p.snapshot_id)]
    return steps


def _omp_freeze(args, m: Mutation) -> int:
    """`omp freeze`: snapshot the omp omp_bin() resolves into ~/.localbench/omp-frozen/<version>-<sha16>/ (localbench/
    ompfreeze.py) and print LOCALBENCH_OMP=<entry>. An existing snapshot of the same omp is reused."""
    from . import ompfreeze

    try:
        p = ompfreeze.plan()
    except FileNotFoundError as exc:
        return _fail(f"omp freeze: {exc}")
    entry = ompfreeze.entry_path(p.target)
    noop = f"omp snapshot {p.snapshot_id} exists at {p.target}: reused, nothing copied" if p.exists else None
    # A real reuse proceeds (its row says noop) so stdout carries only LOCALBENCH_OMP=, as for a fresh copy.
    if (rc := m.gate(_freeze_steps(p), refuse=p.refuse, noop=noop if m.dry_run else None)) is not None:
        return rc
    manifest = ompfreeze.freeze(p)
    pruned = [str(d) for d in ompfreeze.prune(p.snapshot_id)] if manifest.get("created") else []
    m.detail.update(snapshot_id=p.snapshot_id, entry=str(entry), omp_version=manifest.get("omp_version"),
                    bun_version=manifest.get("bun_version"), size_bytes=manifest.get("size_bytes"),
                    created=manifest.get("created"), pruned=pruned)
    if not manifest.get("created"):
        m.detail["noop"] = noop or f"omp snapshot {p.snapshot_id} was made by a concurrent freeze: reused"
    if args.json:
        print(json.dumps({**manifest, "entry": str(entry), "pruned": pruned}, indent=2, sort_keys=True))
        return 0
    print((f"omp {manifest.get('omp_version')} frozen at {p.target}" if manifest.get("created") else m.detail["noop"])
          + f" ({(manifest.get('size_bytes') or 0) / 1e6:.0f} MB, {len(p.packages)} packages, bun "
          f"{manifest.get('bun_version')})" + "".join(f"; removed {d}" for d in pruned), file=sys.stderr)
    print(f"LOCALBENCH_OMP={entry}")
    return 0


def _prove_receipts_snapshot() -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in RECEIPTS.glob("prove__*.json") if path.is_file()}


def _mark_prove_pin_drift(before: dict[str, bytes], message: str) -> None:
    for path in sorted(RECEIPTS.glob("prove__*.json")):
        if not path.is_file() or before.get(path.name) == path.read_bytes():
            continue
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink proof receipt {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        problems = doc.get("problems")
        if not isinstance(problems, list):
            raise RuntimeError(f"proof receipt {path} has no problems list")
        if message not in problems:
            problems.append(message)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
                           encoding="utf-8")
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()


def _buffer_proof_bead_actions(actions: list[list[str]]):
    from . import proofqueue

    def run(argv: list[str]) -> tuple[int, str, str]:
        if argv and argv[0] == "list":
            return proofqueue._br(argv)
        actions.append(list(argv))
        return 0, "", ""

    return run


def _flush_proof_bead_actions(actions: list[list[str]]) -> None:
    from . import proofqueue

    for argv in actions:
        rc, _out, err = proofqueue._br(argv)
        if rc:
            raise RuntimeError(f"br {' '.join(argv[:2])} failed rc={rc}: {err.strip()[-300:]}")


def _with_frozen_omp(handler, args, *, preserve_flag: bool = False) -> int:
    """Freeze and verify OMP before invocation; prove also rechecks at return before bead writes.

    The snapshot is held for the whole invocation. A changed proof snapshot annotates its receipts and discards
    buffered bead closes/comments instead of publishing a proof from a changed dependency closure."""
    from . import ompfreeze

    try:
        p = ompfreeze.plan()
    except FileNotFoundError as exc:
        return _fail(f"{args.cmd} --omp-frozen: {exc}")
    if p.refuse:
        audit.record("omp freeze", [args.cmd, "--omp-frozen"], [], "refused", {"reason": p.refuse})
        return _fail(f"{args.cmd} --omp-frozen: {p.refuse}")
    actions = [s.action for s in _freeze_steps(p)]
    try:
        manifest = ompfreeze.freeze(p)
        pruned = [str(d) for d in ompfreeze.prune(p.snapshot_id)] if manifest.get("created") else []
    except BaseException as exc:
        audit.record("omp freeze", [args.cmd, "--omp-frozen"], actions, "failed",
                     {"error": f"{type(exc).__name__}: {exc}"})
        raise
    entry = ompfreeze.entry_path(p.target)
    audit.record("omp freeze", [args.cmd, "--omp-frozen"], actions if manifest.get("created") else [], "done",
                 {"snapshot_id": p.snapshot_id, "entry": str(entry), "created": manifest.get("created"),
                  "pruned": pruned, **({} if manifest.get("created") else {"noop": "snapshot reused"})})
    print(f"omp {p.version} {'frozen' if manifest.get('created') else 'snapshot reused'}: LOCALBENCH_OMP={entry} "
          "for every leg", file=sys.stderr)
    if preserve_flag:
        args._omp_frozen_ready = True
    else:
        args.omp_frozen = False
    receipts_before = _prove_receipts_snapshot() if preserve_flag else {}
    pending_bead_actions: list[list[str]] = []
    if preserve_flag:
        args._frozen_proof_br = _buffer_proof_bead_actions(pending_bead_actions)
    with ompfreeze.hold(p.snapshot_id), _binaries_for_leg({"LOCALBENCH_OMP": entry}):
        try:
            ompfreeze.verify_snapshot(p.target, manifest)
        except RuntimeError as exc:
            audit.record("omp freeze", [args.cmd, "--omp-frozen"], [], "failed", {"error": str(exc)})
            return _fail(f"{args.cmd} --omp-frozen: {exc}")
        rc = handler(args)
        if preserve_flag:
            try:
                ompfreeze.verify_snapshot(p.target, manifest)
            except RuntimeError as exc:
                message = f"PINS CHANGED: frozen OMP changed while prove was running: {exc}"
                try:
                    _mark_prove_pin_drift(receipts_before, message)
                except (OSError, ValueError, RuntimeError) as write_exc:
                    return _fail(f"{message}; could not annotate receipt: {write_exc}")
                audit.record("omp freeze", [args.cmd, "--omp-frozen"], [], "failed", {"error": message})
                print(message, file=sys.stderr)
                return rc or 1
        if preserve_flag:
            try:
                _flush_proof_bead_actions(pending_bead_actions)
            except RuntimeError as exc:
                return _fail(f"prove --omp-frozen: could not apply bead updates: {exc}")
        return rc


def cmd_omp(args) -> int:
    """After an omp update (localbench/ompupdate.py). `refresh`: capture omp's feature requests against a local mock,
    diff them and the registered module shas against the stored baseline, and carry each feature's proof forward only
    when both are identical; every other feature goes STALE with its re-proof queued (exit 1). `watch install|remove|
    status`: the WatchPaths LaunchAgent that runs `localbench omp refresh` when omp's package.json changes."""
    from . import ompupdate

    m = _mut(args)
    if args.omp_action == "freeze":
        return _omp_freeze(args, m)
    if args.omp_action == "refresh":
        steps = [Step("capture omp's feature requests in an isolated session against a local mock server",
                      "the exact request bytes omp sends now; no model is called"),
                 Step(f"diff them and the registered module shas against {ompupdate.BASELINE_PATH}; record the "
                      "outcome per feature",
                      "a proof carries forward only for byte-identical requests from an unchanged module; the rest "
                      "go STALE and queue their re-proof")]
        if (rc := m.gate(steps)) is not None:
            return rc
        try:
            report = ompupdate.refresh()
        except (ompupdate.CaptureError, ValueError, OSError) as exc:
            return _fail(f"omp refresh: {exc}")
        outcomes = report.get("outcomes") or {}
        stale = sorted(name for name, o in outcomes.items() if o.get("status") != "CARRIED")
        m.detail.update(omp_version=report.get("omp_version"), stale=stale)
        m.outcome = "done"   # the refresh ran and recorded every outcome; exit 1 says a proof went STALE
        try:
            from . import proofqueue
            queued = proofqueue.queue()
        except Exception as exc:  # the trigger must not take down the refresh it follows
            print(f"proof queue: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            queued_counts = {key: queued[key] for key in ("filed", "updated", "adopted", "closed")}
            m.detail.update(proof_queue=queued_counts)
            report["proof_queue"] = queued_counts
            if not args.json:
                print(f"proof queue: {len(queued['filed'])} filed, {len(queued['updated'])} updated, "
                      f"{len(queued['adopted'])} adopted, {len(queued['closed'])} closed")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
        else:
            print(f"omp {report.get('omp_version')}: {len(outcomes) - len(stale)} carried, {len(stale)} STALE")
            for name, o in sorted(outcomes.items()):
                queue = ", ".join(o.get("queue") or [])
                print(f"  {name}: {o.get('status')}" + (f" (feature {o['feature']})" if o.get("feature") else "")
                      + (f"; re-prove: {queue}" if queue else ""))
        return 1 if stale else 0

    path, label = _omp_watch_plist(), ompupdate.DEFAULT_LABEL
    domain = f"gui/{os.getuid()}"
    existing = None
    if path.is_file() and not path.is_symlink():
        try:
            existing = plistlib.loads(path.read_bytes())
        except (OSError, ValueError, plistlib.InvalidFileException):
            existing = {}
    if args.watch_state == "status":
        loaded = subprocess.run(["launchctl", "print", f"{domain}/{label}"], capture_output=True, text=True,
                                check=False).returncode == 0
        view = {"plist": str(path), "installed": existing is not None and existing.get("Label") == label,
                "loaded": loaded, "watch_paths": (existing or {}).get("WatchPaths"),
                "program": (existing or {}).get("ProgramArguments")}
        if args.json:
            print(json.dumps(view))
        else:
            print(f"omp watch: {'installed' if view['installed'] else 'not installed'} ({path}), "
                  f"{'loaded' if loaded else 'not loaded'}; watches {view['watch_paths'] or '-'}")
        return 0
    refuse = None
    if path.is_symlink():
        refuse = f"{path} is a symlink; refusing"
    elif existing is not None and existing.get("Label") != label:
        refuse = f"{path} is not localbench's omp watch (Label {existing.get('Label')!r}); refusing"
    if args.watch_state == "install":
        body = ompupdate.render_watch_plist(localbench_bin=shutil.which("localbench") or ompupdate.DEFAULT_LOCALBENCH)
        steps = [Step(f"write {path}", f"WatchPaths {plistlib.loads(body)['WatchPaths']}: runs `localbench omp "
                                       "refresh` when omp's package.json changes (an omp update)"),
                 Step(f"launchctl bootout {domain}/{label}; launchctl bootstrap {domain} {path}",
                      "(re)loads the job so the new plist takes effect")]
        if (rc := m.gate(steps, refuse=refuse)) is not None:
            return rc
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(body)
            tmp.chmod(0o644)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], capture_output=True, text=True, check=False)
        p = subprocess.run(["launchctl", "bootstrap", domain, str(path)], capture_output=True, text=True, check=False)
        if p.returncode:
            return _fail(f"launchctl bootstrap {path} failed: {(p.stderr or p.stdout).strip()}")
        print(f"installed {path}; remove: localbench omp watch remove")
        return 0
    steps = [Step(f"launchctl bootout {domain}/{label}; remove {path}", "omp updates no longer trigger a refresh")]
    noop = None if path.exists() or path.is_symlink() else f"omp watch not installed ({path} absent)"
    if (rc := m.gate(steps, refuse=refuse if not noop else None, noop=noop)) is not None:
        return rc
    subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], capture_output=True, text=True, check=False)
    path.unlink()
    print(f"removed {path}")
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
    refuse = park.safety_refusal(plan)
    if (rc := _mut(args).gate([_park_step(e) for e in plan], refuse=refuse, noop=noop)) is not None:
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


RUN_PATTERN = "localbench (aa|run|ab|record|decision run)"   # a decision run shares the GPU like any measurement
KEEP_ALIASES = {"unload": 0, "0": 0}


def _finite_keep(value: str) -> str:
    if value not in KEEP_ALIASES:
        try:
            gateway.parse_finite_duration(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return value

def _run_alive() -> bool:
    """A measurement is in progress (loading a model or downloading one now would perturb it). This process and its
    parent (a `uv run` wrapper) are not another run: `decision run` checks this about itself."""
    out = subprocess.run(["pgrep", "-f", RUN_PATTERN], capture_output=True, text=True, check=False).stdout.split()
    return any(int(pid) not in (os.getpid(), os.getppid()) for pid in out)


def _ollama_model(spec: str) -> str:
    backend, _, model = spec.partition(":")
    if backend != "ollama" or not model:
        _usage(f"{spec}: this manages ollama models; give ollama:<name>")
    return model


def cmd_keep(args) -> int:
    """Set a finite Ollama keep-alive (default 5m); zero/unload requests a guarded unload."""
    model = _ollama_model(args.spec)
    duration = args.duration
    keep = KEEP_ALIASES.get(duration)
    if keep is None:
        try:
            keep = gateway.parse_finite_duration(duration)
        except ValueError as exc:
            _usage(str(exc))
    refuse = "a localbench run is alive; set keep-alive after it ends" if _run_alive() else None
    noop = None
    residents = None if refuse else sysstats.ollama_residents(Ollama().root, timeout=10.0)
    if keep == 0 and not refuse:
        if residents is None:
            refuse = "cannot verify Ollama resident state; unload refused"
        elif model not in dict(residents):
            noop = f"{model}: not loaded; nothing to unload"
        else:
            safe, reason = gateway.safe_to_unload(model)
            if safe is not True:
                refuse = reason or "external Ollama client activity is unknown; unload refused"
    step = (Step(f"unload ollama {model} (keep_alive 0)",
                 "frees its memory and GPU after active gateway/external clients are ruled out") if keep == 0 else
            Step(f"set ollama {model} keep_alive {duration} (finite lease)",
                 "loads it if needed; the finite lease is visible to localbench status"))
    mutation = _mut(args)
    if (rc := mutation.gate([step], refuse=refuse, noop=noop)) is not None:
        return rc
    if keep > 0:
        expires_at = gateway.keep_state(model, duration)
        mutation.detail["finite_lease_expires_at"] = datetime.fromtimestamp(expires_at, UTC).isoformat()
    body = {"model": model, "keep_alive": 0 if keep == 0 else duration}
    try:
        backends._post(Ollama().root + "/api/generate", body, timeout=900)
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode(errors='replace')[:200]
        finally:
            exc.close()
        if keep > 0:
            gateway.clear_manual_lease(model)
        return _fail(f"ollama refused {model}: HTTP {exc.code} {error_body}")
    residents = sysstats.ollama_residents(Ollama().root, timeout=120.0)
    if residents is None:
        return _fail(f"{model}: requested, but ollama did not answer /api/ps within 120 s; loaded state unknown")
    loaded = dict(residents)
    if keep == 0:
        if model in loaded:
            return _fail(f"{model}: still loaded")
        gateway.mark_unloaded(model)
        print(f"{model}: unloaded")
        return 0
    if model not in loaded:
        return _fail(f"{model}: not loaded after the request")
    print(f"{model}: loaded until {loaded[model]}")
    return 0


def cmd_gateway(args) -> int:
    if args.action == "serve":
        try:
            gateway.serve(host=args.host, port=args.port)
        except gateway.GatewayError as exc:
            return _fail(str(exc))
        return 0

    if args.action == "status":
        st = gateway.status()
        if args.json:
            print(json.dumps(st, indent=2, default=str))
            return 0
        service = st["service"]
        print(f"Ollama gateway: {'healthy' if service['health'] else 'UNAVAILABLE'} "
              f"({service['launchd_state']}, bind {service['bind']}, accepting={st['accepting_requests']})")
        print("OMP profiles: " + (", ".join(f"{name}={url}" for name, url in st["profiles"].items()) or "none"))
        print(f"requests in flight: {st['active_requests']}")
        for lease in st["leases"]:
            expiry = max((value for value in (lease["idle_expires_at"], lease["manual_expires_at"])
                          if value is not None), default=None)
            when = _when(expiry) if expiry is not None else "none"
            completed = _when(lease["last_completed_at"]) if lease["last_completed_at"] is not None else "never"
            print(f"lease {lease['model']}: profiles={','.join(lease['profiles']) or 'unknown'} "
                  f"active={lease['active_requests']} last_completed={completed} expires={when} "
                  f"outcome={lease['last_outcome'] or 'active'}")
            if lease["last_error"]:
                print(f"  unresolved: {lease['last_error']}")
        print(f"Ollama residency API: {st['ollama_state']}")
        print("unowned residents: " +
              ("unknown" if st["unowned_residents"] is None else ", ".join(st["unowned_residents"]) or "none"))
        if gateway.database_path().is_file():
            for f in gateway.GatewayStore(gateway.database_path()).fences():
                print(f"fence {f['fence_id']}: {f['model']} since {_when(f['created_at'])}")
        return 0

    if args.action in ("fence", "unfence"):
        return _gateway_fence(args)

    from . import omp_profiles

    mutation = _mut(args)
    service = gateway.service_status()
    refuse = service["plist_error"]
    noop = None
    steps = []
    profile_names = []
    try:
        if args.action == "install":
            manager = omp_profiles.ProfileManager.current(gateway.state_dir())
            profile_names = sorted(manager.dirs)
            manager.plan(args.port)
            manifest_path = gateway.state_dir() / omp_profiles.MANIFEST_NAME
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if set(manifest.get("profiles", {})) != set(manager.dirs):
                    refuse = "OMP profile set changed since gateway installation; reconcile before reinstalling"
                elif service["plist_installed"] and service["launchd_loaded"] and service["health"]:
                    noop = "gateway and OMP profile routes are already installed and healthy"
            if not refuse and not noop and _run_alive():
                refuse = "a localbench measurement is alive; install the gateway after it ends"
            if not refuse and not noop and not service["health"]:
                gateway.ensure_port_free(port=args.port)
            steps = [Step(f"route {len(profile_names)} OMP profiles through the loopback Ollama gateway",
                          "OMP provider changes are marker-owned and reversible; the gateway fails closed")]
        elif args.action == "start":
            if not service["plist_installed"]:
                refuse = "gateway LaunchAgent is not installed"
            elif service["launchd_loaded"] and service["health"]:
                service = gateway.service_status()
                if service["launchd_loaded"] and service["health"]:
                    noop = "gateway LaunchAgent is already healthy"
            steps = [Step("start the managed Ollama gateway LaunchAgent",
                          "the user-level service resumes profile-routed local Ollama requests")]
        elif args.action == "stop":
            if not service["plist_installed"]:
                noop = "gateway LaunchAgent is not installed"
            elif args.dry_run:
                refuse = gateway.stop_preflight()
            steps = [Step("drain and stop the managed Ollama gateway LaunchAgent",
                          "new requests are rejected; stale rows are abandoned only when their gateway PID is gone "
                          "or lsof finds no established client, otherwise stop is refused")]
        elif args.action == "remove":
            manifest_path = gateway.state_dir() / omp_profiles.MANIFEST_NAME
            if not manifest_path.exists() and not service["plist_installed"]:
                noop = "gateway is not installed"
            elif not manifest_path.exists():
                refuse = "gateway LaunchAgent exists without a reversible OMP profile manifest"
            else:
                profile_names = gateway.plan_remove()
                if gateway.active_request_count():
                    refuse = "gateway has in-flight requests; remove refused"
                elif _run_alive():
                    refuse = "a localbench measurement is alive; remove the gateway after it ends"
            steps = [Step("drain the gateway, restore original OMP routing, and remove its LaunchAgent",
                          "only the exact localbench-owned provider block is removed; later profile edits are preserved")]
    except (gateway.GatewayError, omp_profiles.ProfileConflict, OSError, ValueError) as exc:
        refuse = str(exc)

    if (rc := mutation.gate(steps, refuse=refuse, noop=noop)) is not None:
        return rc
    try:
        if args.action == "install":
            result = gateway.install(port=args.port)
            mutation.detail.update({"profiles": result["profiles"], "port": result["port"]})
            mutation.detail["omp_catalog_readback"] = result["readback"]
            print(f"installed gateway for {len(result['profiles'])} OMP profiles on "
                  f"{gateway.HOST}:{result['port']}; restart existing OMP sessions to load updated profiles")
        elif args.action == "start":
            gateway.start()
            print("Ollama gateway started")
        elif args.action == "stop":
            gateway.stop()
            print("Ollama gateway stopped; OMP requests fail closed until it starts")
        else:
            removed = gateway.remove()
            mutation.detail["profiles_reverted"] = removed
            print(f"removed gateway; restored OMP routing for {len(removed)} profiles")
    except (gateway.GatewayError, omp_profiles.ProfileConflict, OSError, ValueError) as exc:
        return _fail(str(exc))
    return 0

def _park_fence_ids() -> set[str]:
    """Fence ids that belong to `localbench park` (released only by unpark)."""
    ids = set()
    for entry in park.parked_now():
        ids.update(v for k, v in entry.items() if k in ("_park_fence_id", "_unpark_alias_fence_id") and v)
    return ids


def _gateway_fence(args) -> int:
    """`gateway fence --model M...`: the gateway refuses inference requests for M at once (a fast 503, so a client
    can fall back) until `gateway unfence --id`. For windows a local-route consumer asked for (proj-b 2026-10-01: its
    nimble gate falls back to paid decision service on a gateway error). Uses park's admission fence; it never unloads anything
    and never touches a fence that park holds."""
    m = _mut(args)
    store = gateway.GatewayStore(gateway.database_path())
    active = store.fences()
    if args.action == "fence":
        models = sorted(set(args.model))
        refuse = None if models else "name at least one --model to fence"
        already = sorted({f["model"] for f in active} & set(models))
        if already and not refuse:
            refuse = f"already fenced: {', '.join(already)}"
        steps = [Step(f"fence {', '.join(models)} at the gateway: their inference requests get an immediate error",
                      "until `localbench gateway unfence --id <id>`; nothing is unloaded")]
        if (rc := m.gate(steps, refuse=refuse)) is not None:
            return rc
        fence_id = f"manual-{uuid.uuid4().hex[:12]}"
        deadline = time.monotonic() + max(0.0, args.wait)
        while (refusal := store.acquire_park_fence(models, fence_id)) is not None:
            if "in flight" not in refusal or time.monotonic() >= deadline:
                m.outcome = "refused"
                return _fail(f"gateway fence: {refusal}")
            time.sleep(0.2)
        m.detail.update(fence_id=fence_id, models=models)
        print(f"fenced {', '.join(models)}; fence id {fence_id}; release: localbench gateway unfence --id {fence_id}")
        return 0
    fence_id = args.fence_id or ""
    held = [f["model"] for f in active if f["fence_id"] == fence_id]
    refuse = ("name the fence with --id" if not fence_id
              else "that fence belongs to localbench park; release it with localbench unpark"
              if fence_id in _park_fence_ids() else None)
    noop = None if refuse or held else f"no active fence {fence_id}"
    steps = [Step(f"release fence {fence_id} ({', '.join(held)})", "the gateway admits those models' requests again")]
    if (rc := m.gate(steps, refuse=refuse, noop=noop)) is not None:
        return rc
    store.release_park_fence(fence_id)
    m.detail.update(fence_id=fence_id, models=held)
    print(f"released fence {fence_id} ({', '.join(held)})")
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
        plan = _plan_pull_hf(repo, args.to, hub_cache=args.hub_cache)
        tool = (f"with `hf download --cache-dir {plan['root']}`" if args.hub_cache else "with `hf download`")
        shape = ("hub layout models--<org>--<name>/snapshots/<sha>, which the laya: backend reads offline"
                 if args.hub_cache else "plain files")
        steps = [Step(f"download {repo} @ {plan['sha'][:12]} ({plan['stored'] / 1e9:.1f} GB stored, "
                      f"{plan['need'] / 1e9:.1f} GB to fetch) into {plan['dest']} {tool}",
                      f"the revision is pinned to the sha the API reports now ({shape}); resumable: a rerun continues"),
                 Step(f"write {plan['dest'] / '.localbench-source.json'}", "records the repo and revision the files are")]
        if (rc := m.gate(steps, refuse=plan["refuse"])) is not None:
            return rc
        return _pull_hf(repo, plan, hub_cache=args.hub_cache)
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
# The Hugging Face hub cache the laya: backend reads offline (HF_HUB_OFFLINE=1): the shim inherits this process's
# environment, so it sees $HF_HUB_CACHE when set, else this default. pull --hub-cache writes the same layout.
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE")
                     or Path.home() / ".cache" / "huggingface" / "hub").expanduser()


def _plan_pull_hf(repo: str, to: Path | None, hub_cache: bool = False) -> dict:
    """What `pull hf:<repo>` would fetch and where, from the Hugging Face API (a read): {dest, sha, stored, need,
    gated, refuse} plus, for --hub-cache, {root, snapshot}. Plain mode writes files into dest (default
    HF_DIR/<org>/<repo>); hub-cache mode targets the hub layout root/models--<org>--<name>/snapshots/<sha> (default
    $HF_HUB_CACHE, else ~/.cache/huggingface/hub), which the laya: backend reads offline. `refuse` is set when the
    volume lacks the repo's stored size (the API's usedStorage, an upper bound) less what is already in the snapshot
    (hub-cache) or dest (plain), plus HF_HEADROOM_GB."""
    root = (to or HF_HUB_CACHE).expanduser().absolute() if hub_cache else None
    dest = (root / ("models--" + repo.replace("/", "--")) if hub_cache
            else (to or HF_DIR / repo).expanduser().absolute())
    info = backends._get(f"https://huggingface.co/api/models/{repo}?expand[]=usedStorage&expand[]=sha&expand[]=gated",
                         timeout=30)
    sha, stored = info["sha"], info.get("usedStorage") or 0
    snapshot = dest / "snapshots" / sha if hub_cache else None
    have_dir = snapshot if hub_cache else dest
    have = sum(f.stat().st_size for f in have_dir.rglob("*") if f.is_file()) if have_dir.exists() else 0
    need = max(stored - have, 0)
    anchor = next(p for p in (dest, *dest.parents) if p.exists())
    free = shutil.disk_usage(anchor).free
    refuse = None
    if free < need + HF_HEADROOM_GB * 1e9:
        where = ("--to or HF_HUB_CACHE picks another cache root" if hub_cache
                 else "--to or LOCALBENCH_HF_DIR picks another volume")
        refuse = (f"{repo}: needs {need / 1e9:.1f} GB plus {HF_HEADROOM_GB} GB headroom on {anchor}; "
                  f"{free / 1e9:.1f} GB free ({where})")
    return {"dest": dest, "sha": sha, "stored": stored, "need": need, "gated": bool(info.get("gated")),
            "refuse": refuse, "root": root, "snapshot": snapshot, "default_cache": to is None}


def _pull_hf(repo: str, plan: dict, hub_cache: bool = False) -> int:
    """Download a Hugging Face repo with the `hf` CLI (resumable: rerun to continue) as `plan` (_plan_pull_hf) says,
    the revision pinned to the sha the API reported and recorded in .localbench-source.json. Plain mode writes files
    into dest with --local-dir; hub-cache mode writes the hub layout with --cache-dir and fail-closes when
    snapshots/<sha> is missing afterwards (never claim a pinned layout the download did not produce). A hub-cache
    pull outside the default cache needs HF_HUB_CACHE set to its root for laya: runs to see it. A gated repo needs a
    token in the caller's environment, e.g. `HF_TOKEN=<token> localbench pull hf:<org>/<repo>`: `hf` reads it,
    localbench never reads or prints it."""
    dest, sha = plan["dest"], plan["sha"]
    print(f"{repo} @ {sha[:12]}: {plan['stored'] / 1e9:.1f} GB stored{' (gated)' if plan['gated'] else ''} -> {dest}",
          flush=True)
    if hub_cache:
        argv = ["hf", "download", repo, "--revision", sha, "--cache-dir", str(plan["root"])]
    else:
        dest.mkdir(parents=True, exist_ok=True)
        argv = ["hf", "download", repo, "--revision", sha, "--local-dir", str(dest)]
    rc = subprocess.run(argv, env={**os.environ, "HF_HUB_DISABLE_UPDATE_CHECK": "1"}, check=False).returncode
    if rc:
        return _fail(f"hf download failed (rc {rc}); a gated repo needs HF_TOKEN from an account that was granted access")
    if hub_cache:
        snapshot = plan["snapshot"]
        if not snapshot.is_dir() or not any(f.is_file() for f in snapshot.rglob("*")):
            return _fail(f"hf download wrote no files under snapshots/{sha} in {dest}; not claiming a hub-cache pull")
        if not plan["default_cache"]:
            print(f"note: laya: runs read $HF_HUB_CACHE (default ~/.cache/huggingface/hub); "
                  f"export HF_HUB_CACHE={plan['root']} to use this pull")
    files = [f for f in dest.rglob("*") if f.is_file() and ".cache" not in f.relative_to(dest).parts]
    (dest / ".localbench-source.json").write_text(json.dumps(
        {"repo": repo, "revision": sha, "layout": "hub-cache" if hub_cache else "plain",
         "downloaded": datetime.now(UTC).isoformat(timespec="seconds")}) + "\n")
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


E2E_PROFILE = "omp-e2e.v1"


def _evaluation_case_inputs(model: str) -> dict[str, str]:
    from .evaluation import canonical_sha256
    from .workloads import E2E_TASKS, child_flags

    return {
        name: canonical_sha256({"profile": E2E_PROFILE, "case": name, "prompt": prompt,
                                "fixture": {"answer.txt": "4817\n"} if name == "tool_read" else {},
                                "child_flags": child_flags(model)})
        for name, prompt, _ in E2E_TASKS
    }


def _localbench_source_hashes() -> dict[str, str]:
    from .evaluation import file_sha256

    return {source.name: file_sha256(source)
            for source in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def _evaluation_scorer_sha256() -> str:
    from .evaluation import canonical_sha256

    return canonical_sha256(_localbench_source_hashes())


def _evaluation_path(value: str) -> Path:
    from .evaluation import CampaignError

    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(RUNS.resolve())
    except ValueError as exc:
        raise CampaignError("campaign directory must be inside runs/") from exc
    return path


def _evaluation_score(campaign):
    from .evaluation import CampaignError
    from .workloads import score_e2e_case

    case_inputs = campaign.case_inputs
    cases = []
    for case_id in campaign.case_ids:
        row = campaign.completed_case(case_id)
        if row is None:
            cases.append({"case_id": case_id, "input_sha256": case_inputs[case_id], "status": "INCOMPLETE"})
            continue
        if row["status"] in ("VOID", "ERROR"):
            cases.append({"case_id": case_id, "input_sha256": row["input_sha256"], "status": row["status"]})
            continue
        run_dir = campaign.root / row["run_dir"]
        attempts = []
        try:
            for attempt in ("first", "repeat"):
                result = json.loads((run_dir / f"e2e.{case_id}.{attempt}.result.json").read_text(encoding="utf-8"))
                stdout = (run_dir / f"e2e.{case_id}.{attempt}.omp.jsonl").read_text(encoding="utf-8")
                if not isinstance(result, dict) or result.get("task") != case_id or result.get("attempt") != attempt:
                    raise CampaignError(f"case {case_id!r} has mismatched {attempt} result metadata")
                attempts.append({"stdout": stdout, "returncode": result.get("returncode")})
            scored = score_e2e_case(case_id, attempts)
        except CampaignError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CampaignError(f"case {case_id!r} cannot be re-scored: {exc}") from exc
        cases.append({"case_id": case_id, "input_sha256": row["input_sha256"], **scored})
    statuses = [case["status"] for case in cases]
    status = ("INCOMPLETE" if "INCOMPLETE" in statuses else "FAIL" if "FAIL" in statuses or "ERROR" in statuses
              else "VOID" if "VOID" in statuses else "PASS")
    document = {"identity_sha256": campaign.identity_sha256, "profile": campaign.manifest["profile"],
                "status": status, "expected": len(cases),
                "completed": sum(case["status"] not in ("INCOMPLETE",) for case in cases), "cases": cases}
    score_path = campaign.write_scores(_evaluation_scorer_sha256(), document)
    return document, score_path


VARIED_PROFILE = "omp-varied.v1"


def _evaluation_score_varied(campaign):
    """Rejudge saved trial exits, tool trajectories and final files without opening a backend."""
    from .evaluation import GRADER_VERSION, CampaignError, canonical_sha256, score_varied_trial, workspace_text
    from .workloads import _wilson

    specs = campaign.manifest["identity"].get("varied_specs")
    if not isinstance(specs, dict) or set(specs) != set(campaign.case_ids):
        raise CampaignError("varied campaign spec identity is incomplete")
    versions = {(s.get("grader_version") if isinstance(s, dict) else None) for s in specs.values()}
    stale = sorted(str(v) for v in versions if v != GRADER_VERSION)
    if stale:
        # Checked before any scoring, so write_scores never puts new-grader ERROR rows into an old campaign.
        raise CampaignError(f"specs carry grader_version {', '.join(stale)} but this localbench grades "
                            f"{GRADER_VERSION}; a fixed grader is a new campaign identity: run a new "
                            "`localbench eval varied` campaign instead of rescoring this one")
    cases = []
    for case_id in campaign.case_ids:
        spec = specs[case_id]
        if canonical_sha256(spec) != campaign.case_inputs[case_id]:
            raise CampaignError(f"case {case_id!r} spec differs from preregistered input hash")
        row = campaign.completed_case(case_id)
        if row is None:
            cases.append({"case_id": case_id, "status": "INCOMPLETE"})
            continue
        if row["status"] in ("VOID", "ERROR"):
            cases.append({"case_id": case_id, "status": row["status"]})
            continue
        attempt_dir = campaign.root / row["run_dir"] / "attempt"
        try:
            result = json.loads((attempt_dir / "result.json").read_text(encoding="utf-8"))
            stdout = (attempt_dir / "trajectory.jsonl").read_text(encoding="utf-8")
            final_files = json.loads((attempt_dir / "final_state.json").read_text(encoding="utf-8"))
            if not isinstance(final_files, dict):
                raise CampaignError(f"case {case_id!r} has no valid final file snapshot")
            if result.get("timed_out") is not True:
                for filename in ("target.txt", "decoy.txt"):
                    workspace_file = attempt_dir / "workspace" / filename
                    content = workspace_text(workspace_file)
                    if content is None or content != final_files.get(filename):
                        raise CampaignError(f"case {case_id!r} final snapshot differs from workspace: {filename}")
            score = score_varied_trial(spec, stdout=stdout, returncode=result["returncode"],
                                       timed_out=result["timed_out"], final_files=final_files,
                                       cwd=(attempt_dir / "workspace").resolve(), wall_s=result["wall_s"])
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            raise CampaignError(f"case {case_id!r} cannot be re-scored: {exc}") from exc
        cases.append({"case_id": case_id, "family": spec["family"], "phase": spec["phase"],
                      "seed": spec["seed"], **score})
    statuses = [case["status"] for case in cases]
    success_walls = [case["wall_s"] for case in cases if case["status"] == "PASS"]
    graded = statuses.count("PASS") + statuses.count("FAIL")
    status = ("INCOMPLETE" if "INCOMPLETE" in statuses else "FAIL" if "FAIL" in statuses or "ERROR" in statuses
              else "VOID" if "VOID" in statuses else "PASS")
    document = {"identity_sha256": campaign.identity_sha256, "profile": VARIED_PROFILE,
                "status": status, "expected": len(cases),
                "completed": sum(value != "INCOMPLETE" for value in statuses),
                "passed": statuses.count("PASS"), "failed": statuses.count("FAIL"),
                "void": statuses.count("VOID"), "error": statuses.count("ERROR"),
                "success": {"passed": statuses.count("PASS"), "graded": graded,
                            "wilson95": _wilson(statuses.count("PASS"), graded) if graded else None,
                            "wall_s_on_success": {"median": statistics.median(success_walls),
                                                  "range": [min(success_walls), max(success_walls)]}
                            if success_walls else None},
                "cases": cases}
    score_path = campaign.write_scores(_evaluation_scorer_sha256(), document)
    return document, score_path


def cmd_eval_rescore(args) -> int:
    """Re-score stored omp traces offline; this command never opens a backend."""
    from .evaluation import CampaignError, EvaluationCampaign

    try:
        path = _evaluation_path(args.campaign)
        campaign = EvaluationCampaign.open(path, root=ROOT)
        profile = campaign.manifest.get("profile")
        if profile == VARIED_PROFILE:
            document, score_path = _evaluation_score_varied(campaign)
        elif profile == E2E_PROFILE:
            document, score_path = _evaluation_score(campaign)
        else:
            raise CampaignError(f"unsupported campaign profile {profile!r}")
    except (CampaignError, OSError) as exc:
        print(f"localbench eval rescore: campaign: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"campaign": str(path.relative_to(ROOT)), "status": document["status"],
                      "score": str(score_path.relative_to(ROOT))}, sort_keys=True))
    return 0 if document["status"] == "PASS" else 1


def cmd_eval_varied(args) -> int:
    """Run preregistered changing-value read/edit trials; never bank performance goldens."""
    from .evaluation import CampaignError, EvaluationCampaign, canonical_sha256, make_varied_spec, score_varied_trial
    from .proxy import Proxy
    from .varied import run_trial
    from .workloads import child_flags

    if args.trials < 2 or args.seed < 0:
        _usage("eval varied: --trials must be at least 2 and --seed nonnegative")
    specs = {f"{family}-{args.seed + index}":
             make_varied_spec(family=family, seed=args.seed + index, phase=args.phase)
             for index in range(args.trials) for family in ("read", "edit")}
    case_inputs = {case_id: canonical_sha256(spec) for case_id, spec in specs.items()}
    steps = [Step(f"trial {case_id} input_sha256={case_inputs[case_id]}",
                  "fresh workspace, real omp tool trace, final file state, and immutable case record")
             for case_id in specs]
    if args.resume:
        steps.insert(0, Step(f"resume {_evaluation_path(args.resume)}",
                             "refuse if any backend, omp, child overlay, prompt, grader or fixture pin moved"))
    refused = None if park.parked_now() else "localbench eval varied: park omp's managed smol model first"
    gated = _mut(args).gate(steps, refuse=refused)
    if gated is not None:
        return gated

    try:
        pre = preflight(False, args.wait_idle)
        server_args = tuple(args.server_arg or ())
        with open_backend(args.backend, server_args) as (backend, model):
            if backend.name != "ollama":
                unload_ollama()
            before = sysstats.snapshot()
            fingerprint = backend.fingerprint(model)
            if not fingerprint.get("loaded_context"):
                return _fail("localbench eval varied: backend did not report its loaded context")
            pins = run_pins(backend, model, before["host"], args.mem_config)
            identity = {"profile": VARIED_PROFILE, "backend": backend.name, "model": model,
                        "pins": pins, "fingerprint": fingerprint, "server_args": list(server_args),
                        "source_sha256": _localbench_source_hashes(), "localbench_rev": _rev(),
                        "child_flags": child_flags(model, config=args.mem_config), "varied_specs": specs}
            if args.resume:
                path = _evaluation_path(args.resume)
                campaign = EvaluationCampaign.open(path, root=ROOT, expected_identity=identity)
                if campaign.manifest["profile"] != VARIED_PROFILE or campaign.case_inputs != case_inputs:
                    raise CampaignError("varied trial identities changed; refusing resume")
            else:
                path = RUNS / f"eval-varied-{time.time_ns()}-{golden.slug(model)}"
                campaign = EvaluationCampaign.create(path, root=ROOT, identity=identity, cases=case_inputs,
                                                     profile=VARIED_PROFILE)
            for case_id, spec in specs.items():
                if campaign.completed_case(case_id, input_sha256=case_inputs[case_id]) is not None:
                    continue
                run_dir = path / "trials" / case_id
                if run_dir.exists():
                    raise CampaignError(f"unrecorded attempt exists at {run_dir}; refusing to overwrite its evidence")
                run_dir.mkdir(parents=True)
                try:
                    backend.isolate(model)
                    calls = run_dir / "omp_calls.jsonl"
                    with sysstats.Sampler(1.0, target=(backend.name, model),
                                          gpu_foreign_max_pct=GPU_BUSY_MAX_PCT) as sampler, \
                            Proxy(backend.base_url, calls, save_dir=run_dir / "bodies", label="eval-varied"):
                        ensure_localbench_model(model, fingerprint["loaded_context"])
                        attempt, traces = run_trial(spec, model, run_dir / "attempt", config=args.mem_config)
                    after = sysstats.snapshot()
                    moved = golden.pin_diff(pins, run_pins(backend, model, after["host"], args.mem_config))
                    during = sampler.summary()
                    grade = score_varied_trial(spec, stdout=attempt["stdout"],
                                               returncode=attempt["returncode"], timed_out=attempt["timed_out"],
                                               final_files=attempt["final_files"], cwd=attempt["cwd"],
                                               wall_s=attempt["wall_s"])
                    no_model_call = not calls.is_file() or not calls.stat().st_size
                    nonproof = bool(sampler.contention or during.get("resident_unknown_samples") or moved)
                    status = "ERROR" if no_model_call else "VOID" if nonproof else grade["status"]
                    system = run_dir / "system.json"
                    system.write_text(json.dumps({"before": before, "after": after, "preflight": pre,
                                                  "during": during, "contention": sampler.contention,
                                                  "pins_changed": moved, "no_model_call": no_model_call,
                                                  "scorer": grade, "status": status}, default=str, sort_keys=True)
                                      + "\n")
                    samples = run_dir / "sampler.jsonl"
                    samples.write_text("".join(json.dumps(row, default=str) + "\n" for row in sampler.series))
                    traces += [system, samples]
                    if calls.is_file():
                        traces.append(calls)
                    bodies = run_dir / "bodies"
                    if bodies.is_dir():
                        traces.extend(p for p in bodies.iterdir() if p.is_file())
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    failure = run_dir / "infrastructure-error.json"
                    failure.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
                    evidence_root = run_dir.resolve()
                    traces = [p for p in sorted(run_dir.rglob("*"))
                              if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(evidence_root)]
                    status = "ERROR"
                campaign.record_case(case_id, input_sha256=case_inputs[case_id], status=status,
                                     run_dir=run_dir, trace_files=traces)
            document, score_path = _evaluation_score_varied(campaign)
    except (CampaignError, OSError) as exc:
        return _fail(f"localbench eval varied: {exc}")
    print(json.dumps({"campaign": str(path.relative_to(ROOT)), "status": document["status"],
                      "score": str(score_path.relative_to(ROOT)), "success": document["success"]}, sort_keys=True))
    return 0 if document["status"] == "PASS" else 1


def _e2e_case_status(summary: dict, case_id: str) -> str:
    verdicts = summary["verdicts"]
    voided = bool(verdicts["contended"] or verdicts["preflight_problems"] or verdicts["pins_changed"]
                  or _resident_unknown(summary) != 0)
    correctness = summary["conformance"].get(f"e2e.{case_id}.correct", {})
    return "VOID" if voided else correctness.get("verdict", "ERROR")


@_held("eval run")
def cmd_eval_run(args) -> int:
    """Run missing, identity-matched omp cases; no prior timings are reused."""
    from .evaluation import CampaignError, EvaluationCampaign

    if not park.parked_now():
        print("localbench eval run: park omp's managed smol model first with `localbench park`", file=sys.stderr)
        return 1
    server_args = tuple(args.server_arg or ())
    try:
        with open_backend(args.backend, server_args) as (backend, model):
            host = sysstats.snapshot()["host"]
            fingerprint = backend.fingerprint(model)
            pins = run_pins(backend, model, host)
            cases = _evaluation_case_inputs(model)
            source_sha256 = _localbench_source_hashes()
            identity = {
                "profile": E2E_PROFILE, "backend": backend.name, "model": model, "pins": pins,
                "fingerprint": fingerprint, "server_args": list(server_args), "localbench_rev": _rev(),
                "source_sha256": source_sha256,
            }
            if args.resume:
                path = _evaluation_path(args.resume)
                campaign = EvaluationCampaign.open(path, root=ROOT, expected_identity=identity)
                recorded_cases = {row["case_id"]: row["input_sha256"] for row in campaign.manifest["cases"]}
                if campaign.manifest.get("profile") != E2E_PROFILE or recorded_cases != cases:
                    raise CampaignError("campaign profile or test inputs changed; refusing resume")
            else:
                path = RUNS / f"eval-{time.time_ns()}-{golden.slug(model)}"
                campaign = EvaluationCampaign.create(path, root=ROOT, identity=identity, cases=cases,
                                                     profile=E2E_PROFILE)
            for case_id, input_sha256 in cases.items():
                if campaign.completed_case(case_id, input_sha256=input_sha256) is not None:
                    continue
                marker = {"profile": E2E_PROFILE, "campaign": path.relative_to(ROOT).as_posix(),
                          "case_id": case_id, "identity_sha256": campaign.identity_sha256}
                summary = execute(backend, model, tiers=["e2e"], repeats=1, allow_busy=False, purge=False,
                                  label=f"eval-{case_id}-{time.time_ns()}", wait_idle_s=args.wait_idle,
                                  e2e_case=case_id, evaluation_campaign=marker,
                                  watchdog_enabled=args.watchdog)
                if summary.get("watchdog", {}).get("aborted"):
                    print(f"localbench eval run: watchdog aborted {case_id}; completed campaign cases remain "
                          f"checkpointed under identical pins at {path.relative_to(ROOT)}; resume with "
                          f"`localbench eval run {args.backend} --resume {path.relative_to(ROOT)} --watchdog`",
                          file=sys.stderr)
                    return 1
                run_dir = ROOT / summary["run_dir"]
                status = _e2e_case_status(summary, case_id)

                traces = [run_dir / name for name in ("summary.json", "progress.jsonl", "samples.jsonl",
                                                       "sampler.jsonl", "omp_calls.jsonl",
                                                       f"e2e.{case_id}.first.omp.jsonl",
                                                       f"e2e.{case_id}.repeat.omp.jsonl",
                                                       f"e2e.{case_id}.first.result.json",
                                                       f"e2e.{case_id}.repeat.result.json")]
                traces = [trace for trace in traces if trace.is_file()]
                bodies = run_dir / "bodies"
                if bodies.is_dir():
                    traces.extend(trace for trace in bodies.iterdir() if trace.is_file())
                campaign.record_case(case_id, input_sha256=input_sha256, status=status, run_dir=run_dir,
                                     trace_files=traces)
            document, score_path = _evaluation_score(campaign)
    except (CampaignError, OSError) as exc:
        print(f"localbench eval run: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"campaign": str(path.relative_to(ROOT)), "status": document["status"],
                      "score": str(score_path.relative_to(ROOT))}, sort_keys=True))
    return 0 if document["status"] == "PASS" else 1


def cmd_eval(args) -> int:
    if args.eval_action == "run":
        return cmd_eval_run(args)
    if args.eval_action == "varied":
        return cmd_eval_varied(args)
    return cmd_eval_rescore(args)


@_held("run")
def cmd_run(args) -> int:
    if getattr(args, "omp_frozen", False):
        return _with_frozen_omp(cmd_run, args)
    with open_backend(args.backend, tuple(args.server_arg or ())) as (backend, model):
        s = execute(backend, model, tiers=args.tiers.split(","), repeats=args.repeats, allow_busy=args.allow_busy,
                    purge=args.purge, wait_idle_s=args.wait_idle, mem_config=args.mem_config,
                    mem_rounds=args.mem_rounds, smol_model=args.smol_model)
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
    """Bank an existing run's receipt view under docs/evidence/receipts/<name>.json (summary.json is immutable).
    Unknown residency cannot produce a banked receipt; other unsound runs retain diagnostic receipts with exit 1.
    Banking the same run under the same name again is a no-op."""
    s = _run_summary("bank", args.run_dir)
    problems = unsound(s)
    payload = {"kind": "run", "problems": problems, "run": _receipt_view(s)}
    path = _receipt_path(args.name)
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    same = path.is_file() and path.read_text() == _receipt_text(payload)
    step = Step(f"{'replace' if path.exists() else 'write'} {rel} (receipt of {s['run_dir']}"
                f"{f', UNSOUND: {len(problems)} problem(s)' if problems else ''})",
                "bank the receipt view (pins, verdicts, metrics and selected diagnostics); raw samples stay in runs/")
    m = _mut(args)
    m.detail["unsound"] = problems
    unknown = _resident_unknown(s)
    refuse = (None if unknown == 0 else
              "bank: residency sampling incomplete (no resident_unknown_samples); cannot bank an unproven run"
              if unknown is None else "bank: resident model state unknown during sampling; cannot bank an unproven run")
    rc = m.gate([] if same else [step], refuse=refuse,
                noop=f"already banked: {rel} holds this receipt" if same else None)
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
    if getattr(args, "json", False):
        print(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(render.show(doc, label, cursor=args.cursor, limit=args.limit), end="")
    return 0


@_held("aa")
def cmd_aa(args) -> int:
    """Run one config twice; bank the pair as a receipt and (with --write-golden) derive the golden from it. An unsound
    pair banks its receipt (verdict UNSOUND, with its problems) but leaves the golden unwritten and exits 1; a pair
    with unknown resident-model state is non-proof and is not banked at all."""
    tiers = args.tiers.split(",")
    m = _mut(args)
    refuse = None
    if args.write_golden and (args.server_arg or args.mem_config != MEM_CONFIG or args.mem_rounds != MEM_ROUNDS
                              or args.smol_model):
        refuse = ("refusing --write-golden with --server-arg, --mem-config, --mem-rounds or --smol-model: the golden "
                  "for a spec is its default launch; measure a variant (a separate smol model is a study leg) as the "
                  "B leg of `localbench ab`")
    steps = [Step(f"measure {args.backend} twice (aa1, aa2): tiers {args.tiers}, {args.repeats} repeat(s), each after "
                  "preflight", "the A/A pair is the null: its spread sets the band a later run is judged against"),
             Step("bank the pair as docs/evidence/receipts/aa__<backend>__<model>__<created>.json (an unsound pair "
                  "too, marked UNSOUND with its problems)",
                  "the receipt is the evidence the golden cites; an unsound receipt is diagnostic evidence only")]
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
                     mem_rounds=args.mem_rounds, smol_model=args.smol_model)
        s2 = execute(backend, model, tiers=tiers, repeats=args.repeats, allow_busy=args.allow_busy,
                     purge=args.purge, label="aa2", wait_idle_s=args.wait_idle, mem_config=args.mem_config,
                     mem_rounds=args.mem_rounds, smol_model=args.smol_model)
    pins = s1["provenance"]["pins"]
    problems = unsound(s1) + unsound(s2)
    if golden.pin_diff(pins, s2["provenance"]["pins"]):
        problems.append(f"pins changed between A/A runs: {golden.pin_diff(pins, s2['provenance']['pins'])}")
    if args.allow_busy:
        problems.append("--allow-busy: an A/A pair from a busy machine cannot derive a band")
    name = f"aa__{pins['backend']}__{pins['model']}__{s1['provenance']['created']}"
    for p in problems:
        print(f"UNSOUND  {p}", file=sys.stderr)
    if _unknown_residency(s1, s2):
        m.outcome, m.detail["unsound"] = "refused", problems
        return 1
    receipt = _bank(name, {"kind": "aa", "verdict": "UNSOUND" if problems else "SOUND",
                           "runs": [_receipt_view(s1), _receipt_view(s2)], "problems": problems})
    rel = str(receipt.relative_to(ROOT))
    m.detail["receipt"] = rel
    print(f"\nbanked A/A receipt {rel}\n")
    print(render.show(json.loads(receipt.read_text()), rel))
    if problems:
        print("golden left unwritten: an unsound pair is a diagnostic receipt, never a golden", file=sys.stderr)
        m.outcome, m.detail["unsound"] = "done", problems   # banked as cmd_bank does; exit 1 says it is unsound
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


@_held("ab")
def cmd_ab(args) -> int:
    """Same invocation, interleaved: A,B,...,A (`--pairs`). A's legs are the A/A null; each arm is the median of its
    legs. On a machine in use, more pairs spread bursts of activity across both arms."""
    tiers = args.tiers.split(",")
    order = ab_order(args.pairs)
    if args.side_regime:
        # The side-model regime is pre-registered for memory-route proofs only (bead kit-memory-study-vce DECISION).
        if not set(tiers) <= {"mem", "sess"}:
            _usage(f"ab --side-regime: only the mem and sess tiers run under the side-model regime, not {tiers}")
        if any(spec.partition(":")[0] != "ollama" for spec in (args.a, args.b)):
            _usage("ab --side-regime: both arms must be ollama: specs (models shared on one ollama server)")
    for arm, spec, smol_model in (("A", args.a, args.smol_model), ("B", args.b, args.b_smol_model or args.smol_model)):
        # Checked before the first leg: execute would refuse it only when that leg starts, after A legs ran.
        if smol_model and spec.partition(":")[0] != "ollama":
            _usage(f"ab: arm {arm} ({spec}) with smol model {smol_model}: a separate smol model needs an ollama: spec")
    if getattr(args, "omp_frozen", False):   # after the usage checks: a bad flag must not wait for a copy
        return _with_frozen_omp(cmd_ab, args)
    legs = []
    for arm, label in order:
        spec = args.a if arm == "A" else args.b
        extra = () if arm == "A" else tuple(args.b_server_arg or ())
        mem_config = args.mem_config if arm == "A" else (args.b_mem_config or args.mem_config)
        smol_model = args.smol_model if arm == "A" else (args.b_smol_model or args.smol_model)
        binaries = ({"LOCALBENCH_OMP": args.b_omp, "LOCALBENCH_MLX_SERVE": args.b_mlx_serve,
                     "LOCALBENCH_MLXFAST": args.b_mlxfast} if arm == "B" else {})
        with _binaries_for_leg(binaries), open_backend(spec, tuple(args.server_arg or ()) + extra) as (backend, model):
            legs.append(execute(backend, model, tiers=tiers, repeats=args.repeats, allow_busy=args.allow_busy,
                                purge=args.purge, label=label, wait_idle_s=args.wait_idle, mem_config=mem_config,
                                mem_rounds=args.mem_rounds, smol_model=smol_model, side_regime=args.side_regime))
    a_legs = [leg for leg, (arm, _) in zip(legs, order) if arm == "A"]
    b_legs = [leg for leg, (arm, _) in zip(legs, order) if arm == "B"]
    problems = [f"{leg['provenance']['label']}: {p}" for leg in legs for p in unsound(leg)]
    for p in problems:
        print(f"UNSOUND  {p}", file=sys.stderr)
    if _unknown_residency(*legs):
        return 1
    drift = golden.arm_pin_drift([leg["provenance"]["pins"] for leg in a_legs],
                                 [leg["provenance"]["pins"] for leg in b_legs], tiers)
    def busy(leg):
        return (((leg.get("system") or {}).get("cpu") or {}).get("busy_pct") or {}).get("mean")
    balance = golden.load_balance([busy(leg) for leg in a_legs], [busy(leg) for leg in b_legs])
    table = golden.ab_table([leg["metrics"] for leg in a_legs], [leg["metrics"] for leg in b_legs], void_tiers=drift,
                            load_favours=balance["favours"])
    a1, b = a_legs[0], b_legs[0]
    receipt = {"kind": "ab", "verdict": "UNSOUND" if problems else "SOUND", "a": args.a, "b": args.b,
               "order": [arm for arm, _ in order], "problems": problems,
               "pin_drift": drift, "load_balance": balance, "table": table,
               "legs": [_receipt_view(x) for x in legs]}
    name = args.bank or f"ab__{a1['provenance']['pins']['model']}__vs__{b['provenance']['pins']['model']}__{a1['provenance']['created']}"
    path = _bank(name, receipt)
    rel = str(path.relative_to(ROOT))
    print(f"\nbanked A/B receipt {rel}\n")
    print(render.show(json.loads(path.read_text()), rel))
    return 1 if problems else 0


def _unknown_residency(*summaries: dict) -> bool:
    """A leg sampled while the resident-model state was unreadable, or whose sampling count is missing, is non-proof
    (bead kit-unknown-residency-fail-closed-qqj): it is excluded from banking, unlike other unsound legs, which bank
    as diagnostic receipts."""
    return any(_resident_unknown(s) != 0 for s in summaries if not workloads.side_regime(s))


def cmd_record(args) -> int:
    """Capture omp's first chat request plus its sidecar (omp pins, prompt tokens, profile, provider path, backend pins,
    and tokenizer identity when the backend reports the actual GGML vocabulary)."""
    from .proxy import Proxy

    save = FIXTURES / "omp" / ".capture"
    dest = FIXTURES / "omp" / f"{args.label}.json"
    omp_flags = args.omp_flags[1:] if args.omp_flags[:1] == ["--"] else args.omp_flags
    rel = dest.relative_to(ROOT)
    steps = [Step(f"clear {save.relative_to(ROOT)}", "the capture dir holds only this recording's bodies"),
             Step(f"start {args.backend}, run omp once (flags: {' '.join(omp_flags) or '-'}) through the timing proxy",
                  "the fixture is the first chat request omp itself sends for this flag set, not a hand-built body"),
             Step(f"{'replace' if dest.exists() else 'write'} {rel} and {dest.with_suffix('.meta.json').name}",
                  "replay binds to the body's hash; the sidecar records omp, profile, backend/model and tokenizer "
                  "identity when the backend reports the full vocabulary; older or unsupported pins use backend-only "
                  "prompt-token comparison")]
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
        tok_identity = backend.tokenizer_identity(model)
        if tok_identity:
            pins["tokenizer_identity"] = tok_identity
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
            if not isinstance(prov.get("localbench_rev"), str) or not prov["localbench_rev"]:
                reasons.append(f"{where} provenance: no localbench_rev")
            fingerprint = prov.get("fingerprint")
            if not isinstance(fingerprint, dict) or not fingerprint:
                reasons.append(f"{where} provenance: no fingerprint")
            else:
                reasons += [f"{where} provenance: no fingerprint.{k}"
                            for k in ("backend", "model") if not fingerprint.get(k)]
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
    kind = render.kind_of(doc) if isinstance(doc, dict) else "json"
    if args.json:
        print(json.dumps({"valid": not reasons, "kind": kind, "path": str(path), "reasons": reasons},
                         ensure_ascii=False, sort_keys=True))
        return 1 if reasons else 0
    if reasons:
        print(f"INVALID {path} ({kind})", file=sys.stderr)
        for r in reasons:
            print(f"  {r}", file=sys.stderr)
        return 1
    print(f"valid {kind}: {path}")
    return 0


def cmd_ollama_app(args) -> int:
    """Ollama.app, the supervisor of `ollama serve` (localbench/ollama_app.py): `status` says whether it runs from
    /Applications and its server is up; `restart` quits it (SIGKILL when it refuses), stops its server, reopens it and
    waits until both are back. Refused while a run is alive or anything is parked: the restart unloads every model.
    `auto-update on|off|status` sets or shows the app's own updater switch (see cmd_ollama_auto_update)."""
    if args.action != "auto-update" and args.state is not None:
        _usage(f"ollama-app {args.action} takes no argument; `{args.state}` belongs to `ollama-app auto-update`")
    if args.action == "auto-update":
        return cmd_ollama_auto_update(args)
    st = ollama_app.state()
    if args.action == "status":
        view = {"app_pid": st.app_pid, "app_image": st.app_image, "serve_pid": st.serve_pid, "aligned": st.aligned}
        if args.json:
            print(json.dumps(view))
        else:
            print(f"Ollama.app pid {st.app_pid} image {st.app_image}; ollama serve pid {st.serve_pid}; "
                  + ("aligned" if st.aligned else "NOT aligned: `localbench ollama-app restart` relaunches it"))
        return 0 if st.aligned else 1
    refuse = ("a localbench run is alive; restart Ollama after it ends" if _run_alive() else
              "models are parked; `localbench unpark` first (a restart would drop the parked copies' residency)"
              if park.parked_now() else None)
    steps = [Step(a, w) for a, w in ollama_app.plan_restart(st)]
    if (rc := _mut(args).gate(steps, refuse=refuse)) is not None:
        return rc

    def up() -> bool:
        try:
            backends._get(Ollama().root + "/api/tags", timeout=5)
            return True
        except OSError:
            return False

    new = ollama_app.restart(up)
    print(f"Ollama.app back: pid {new.app_pid} from {new.app_image}; ollama serve pid {new.serve_pid}. Residency "
          "reset: re-lease models with a finite `localbench keep ollama:<m> <duration>` (e.g. 30m), or let the gateway "
          "(`localbench gateway start`) lease them per request")
    return 0


def cmd_ollama_auto_update(args) -> int:
    """`ollama-app auto-update on|off|status`: Ollama.app's settings.auto_update_enabled in its db.sqlite
    (LOCALBENCH_OLLAMA_APP_DB overrides the path). `on`/`off` copy the db to ~/.localbench/rollback/ollama-db-<UTC>.sqlite
    first, then write and read back; already in that state is a no-op. Manual upgrades are the policy (2026-09-30):
    an auto-installed ollama moves the pins under every golden. `status` also reports a staged update, which installs
    at the next app start whatever the switch says."""
    db = ollama_app.app_db()
    now = sysstats.ollama_auto_update(db)
    staged = ollama_app.staged_update()
    if args.state in (None, "status"):
        if args.json:
            print(json.dumps({"db": str(db), "auto_update": now, "staged_update": staged}))
        else:
            word = f"unknown (no Ollama.app settings in {db})" if now is None else "ON" if now else "OFF"
            print(f"auto-update: {word}")
            print("staged update: " + (f"{', '.join(staged)} in {ollama_app.UPDATES} (installs at the next app start "
                                       "regardless of the setting)" if staged else "none"))
        return 0 if now is False else 1
    want = args.state == "on"
    rollback = ollama_app.ROLLBACK
    steps = [Step(f"copy {db} to {rollback}/ollama-db-<UTC>.sqlite",
                  "the whole settings db as it was; copy it back with the app quit to undo"),
             Step(f"set settings.auto_update_enabled = {int(want)} in {db} and read it back",
                  "the app's updater re-reads this switch before every hourly download")]
    refuse = (f"no Ollama.app settings row with auto_update_enabled in {db} (no app, or an app older than that "
              "setting)") if now is None else None
    noop = f"auto-update already {args.state.upper()} in {db}" if now is want else None
    m = _mut(args)
    if (rc := m.gate(steps, refuse=refuse, noop=noop)) is not None:
        return rc
    backup = ollama_app.set_auto_update(want, db, rollback)
    m.detail.update(db=str(db), backup=str(backup), auto_update=want)
    print(f"auto-update: {args.state.upper()} (backup {backup})")
    if staged and not want:
        print(f"staged update still present: {', '.join(staged)} in {ollama_app.UPDATES} installs at the next app "
              "start regardless of the setting", file=sys.stderr)
    return 0


def cmd_capabilities(args) -> int:
    """Machine-readable CLI contract: verbs, exit codes and structured-output guidance."""
    doc = {"success": True, "version": _version(), "output_format": "json",
           "exit_codes": {"0": "success", "1": "unsound_or_refused", "2": "usage_or_safety", "141": "stdout_closed"},
           "read_json": ["stats", "status", "slot", LOAD_COMMAND, "gpu", "memory", "models", "report", "features", "show", "validate", "audit", "why"],
           "mutation_contract": "state-changing verbs accept --dry-run and --explain"}
    print(json.dumps(doc, sort_keys=True))
    return 0


def cmd_robot_docs(args) -> int:
    print("localbench agent guide: use capabilities --json; stdout is data, stderr diagnostics; exit 1 means refusal/unsound, 2 usage, 141 closed stdout.")
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
    from .lifecycle import install_cancel_handlers
    install_cancel_handlers()
    ap = argparse.ArgumentParser(prog="localbench", description=__doc__, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stats", help="print a full system snapshot (always JSON)")
    st.add_argument("--json", action="store_true", help="accepted; stats always prints JSON")
    sta = sub.add_parser("status", help="which configs have a live golden (generation check), park state, GPU users")
    sta.add_argument("--json", action="store_true", help="one JSON document instead of the text view")
    slt = sub.add_parser("slot", help="heavy slot: holder, held-for, expected remaining, the wait queue with ETAs")
    slt.add_argument("--json", action="store_true", help="one JSON document instead of the table")
    from . import load as load_cmd
    ld = sub.add_parser(LOAD_COMMAND, help="read-only CPU, syscall, spawn and OMP-session attribution")
    ld.add_argument("--seconds", type=int, choices=range(1, load_cmd.MAX_SECONDS + 1),
                    default=load_cmd.DEFAULT_SECONDS,
                    help=f"top delta window in seconds (1–{load_cmd.MAX_SECONDS})")
    ld.add_argument("--json", action="store_true", help="one JSON report instead of the text view")
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
    rp.add_argument("--by-purpose", action="store_true",
                    help="gateway request counts and busy seconds per purpose (no request content)")
    rp.add_argument("--by-profile", action="store_true",
                    help="the same per omp profile; with --by-purpose, per profile and purpose")
    rp.add_argument("--requests", action="store_true",
                    help="gateway per-request rows: purpose, model, start/end, status, queue wait (never content)")
    rp.add_argument("--req-purpose", help="with --requests: only this purpose")
    rp.add_argument("--req-profile", help="with --requests: only this profile")
    rp.add_argument("--limit", type=int, default=200, help="with --requests: newest rows to show (default 200)")
    dec = sub.add_parser("decision", help="side-model decision suites on Ollama's /v1/systemone: run (banks a receipt)")
    dec_sub = dec.add_subparsers(dest="decision_action", required=True)
    dec_run = dec_sub.add_parser("run", help="run a decision suite against a local model; bank the receipt")
    dec_run.add_argument("spec", help="ollama:<model>, e.g. ollama:nimble:latest")
    dec_run.add_argument("--suite", required=True, help="a suite manifest/dir path, or a name under ~/.localbench/corpora")
    dec_run.add_argument("--feature", metavar="ID",
                         help="the registries/features.tsv row this run proves; stamps its installed omp_module_sha")
    dec_run.add_argument("--hosted-arm", action="store_true",
                         help="also run hosted decision service on the same items (needs TYPESAFE_API_KEY; never a fallback)")
    dec_run.add_argument("--repeats", type=int, default=1, help="passes over the suite (default 1)")
    dec_run.add_argument("--wait-idle", type=float, default=0, metavar="SECONDS",
                         help="wait up to this long for an idle machine; still-busy load is recorded, not refused")
    dec_run.add_argument("--allow-evict", action="store_true", dest="allow_evict",
                         help="load even when Ollama's loaded-model cap would evict an in-use resident (recorded; "
                              "the eviction is still a problem)")
    dec_run.add_argument("--wait-slot", type=float, default=0, metavar="SECONDS",
                         help="queue for the heavy-job slot up to SECONDS instead of refusing when held or busy")
    dec_run.add_argument("--force-load", action="store_true",
                         help="take the heavy-job slot past a busy admission (recorded in the audit row)")
    dec_derive = dec_sub.add_parser("derive", help="a model FROM an ollama decision model with num_ctx N (audited)")
    dec_derive.add_argument("spec", help="ollama:<base model>, e.g. ollama:tev1:latest (never modified)")
    dec_derive.add_argument("--num-ctx", type=int, required=True, help="the context the derived model ships")
    dec_derive.add_argument("--name", help="the derived model's name (default <model>-ctx<N>)")
    dec_paired = dec_sub.add_parser("paired", help="paired local-vs-hosted scoring of a banked decision receipt")
    dec_paired.add_argument("receipt", help="banked decision receipt path")
    dec_paired.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")
    dec_paired.add_argument("--seed", type=int, default=20261001, help="bootstrap seed (default 20261001)")
    dec_paired.add_argument("--json", action="store_true", help="the per-type table as one JSON document")
    feat = sub.add_parser("features", help="omp features that can route local, per profile, and the receipt proving each")
    feat.add_argument("--json", action="store_true")
    feat.add_argument("--queue", action="store_true",
                      help="file or update one open proof bead per unproven local route (audited)")
    wr = sub.add_parser("watch-releases", help="upstream release watch: queued screens; --once a pass; --digest new models; --install-agent")
    wr.add_argument("--once", action="store_true", help="one watch pass now (files beads, queues screens; audited)")
    wr.add_argument("--digest", action="store_true", help="one weekly-style new-model digest pass now (files `screen <model> for <role>` beads; audited)")
    wr.add_argument("--since", default="7d", help="digest window: Nd or Nh of days/hours back (default 7d)")
    wr.add_argument("--replay-window", action="store_true", help="print already-seen in-window digest models without refiling")
    wr.add_argument("--install-agent", action="store_true", help="install the weekly digest LaunchAgent (Monday 09:00 local; audited)")
    wr.add_argument("--install-daily", action="store_true", help="install the daily --once LaunchAgent (audited)")
    wr.add_argument("--json", action="store_true", help="the queue or the pass report as JSON; with --dry-run the plan")
    pv = sub.add_parser("prove", help="run declarative proof specs (no orchestrator one-offs)")
    pv.add_argument("spec", nargs="?", help="registries/proofs/<feature>__<slug>.json")
    pv.add_argument("--due", action="store_true", help="run every spec with an open bead whose regime allows it")
    pv.add_argument("--json", action="store_true", help="the report as one JSON document")
    pv.add_argument("--wait-slot", type=float, default=0, metavar="SECONDS",
                    help="queue for the heavy-job slot up to SECONDS instead of refusing when held or busy")
    pv.add_argument("--force-load", action="store_true",
                    help="take the heavy-job slot past a busy admission (recorded in the audit row)")
    pv.add_argument("--omp-frozen", action="store_true",
                    help="required for memory proof planning; pin the omp binary before live legs")
    gn = sub.add_parser("generation", help="generation-proof corpora: replay captured requests per arm")
    gn_sub = gn.add_subparsers(dest="generation_action", required=True)
    gn_replay = gn_sub.add_parser("replay", help="replay a generation corpus through one arm (audited)")
    gn_replay.add_argument("--corpus", required=True, help="generation corpus directory")
    gn_replay.add_argument("--candidate", required=True,
                           help="route:ollama/<model> or builtin:<kind>")
    gn_replay.add_argument("--max-items", type=int, default=None, metavar="N",
                           help="replay only the first N items")
    gn_replay.add_argument("--json", action="store_true", help="per-item outcomes as one JSON document")
    gn_replay.add_argument("--wait-slot", type=float, default=0, metavar="SECONDS",
                           help="queue for the heavy-job slot up to SECONDS instead of refusing when held or busy")
    gn_replay.add_argument("--force-load", action="store_true",
                           help="take the heavy-job slot past a busy admission (recorded in the audit row)")
    pre = sub.add_parser("preset", help="omp config presets: list, show, plan, apply, rollback, drift")
    pre_sub = pre.add_subparsers(dest="preset_action", required=True)
    pre_list = pre_sub.add_parser("list", help="the presets in registries/presets.json")
    pre_list.add_argument("--json", action="store_true")
    pre_show = pre_sub.add_parser("show", help="one preset's ops")
    pre_show.add_argument("name")
    pre_show.add_argument("--json", action="store_true", help="the preset object as JSON")
    pre_plan = pre_sub.add_parser("plan", help="what apply would change, read from omp now (writes nothing)")
    pre_apply = pre_sub.add_parser("apply", help="apply a preset: backed up, read back, restored on mismatch (audited)")
    for p in (pre_plan, pre_apply):
        p.add_argument("name")
        p.add_argument("--profiles", type=_profiles_arg, required=True, help="comma-separated omp profiles")
        p.add_argument("--target", choices=["test", "live"], required=True,
                       help="test: only the registry's test profiles; live: a local preset needs a PROVEN receipt")
        p.add_argument("--force", action="store_true", help="live: proceed past a missing proof (recorded as forced)")
    pre_plan.add_argument("--json", action="store_true")
    pre_rb = pre_sub.add_parser("rollback", help="restore the files one apply backed up (audited)")
    pre_rb.add_argument("id", help="the rollback id apply printed")
    pre_rb.add_argument("--force", action="store_true", help="restore even files changed after that apply")
    pre_drift = pre_sub.add_parser("drift", help="applied presets live config no longer matches (exit 1 if any)")
    pre_drift.add_argument("--profiles", type=_profiles_arg, help="only these profiles (default: every applied one)")
    pre_drift.add_argument("--json", action="store_true")
    cp = sub.add_parser("corpus", help="private decision corpora under ~/.localbench/corpora: import, proj-b-build, list, "
                                       "stats")
    cp_sub = cp.add_subparsers(dest="corpus_action", required=True)
    cp_import = cp_sub.add_parser("import", help="one omp profile's hosted judgments as a decision suite")
    cp_import.add_argument("--profile", required=True, help="omp profile whose cache/judgment-cache.db is read")
    cp_import.add_argument("--role", required=True, choices=["decision.noul", "decision.choice"])
    cp_import.add_argument("--model", help="pin the answering hosted model when the cache holds several")
    cp_jev = cp_sub.add_parser("proj-b-build", help="build a named suite from the proj-b checkout")
    cp_jev.add_argument("name")
    for p in (cp_import, cp_jev):
        p.add_argument("--gate", type=_gate_arg, action="append", required=True, metavar="METRIC=OP:VALUE",
                       help="the suite's pass bar, e.g. decision.noul.accuracy=min:0.8 (required; repeatable)")
    cp_capture = cp_sub.add_parser("capture", help="opt-in, capped, expiring capture of request bodies by purpose")
    cp_capture.add_argument("capture_state", choices=["on", "off", "status"])
    cp_capture.add_argument("--purpose", action="append", metavar="P", help="on: a request purpose (repeatable)")
    cp_capture.add_argument("--max-items", type=int, metavar="N", help="on: stop after N captured bodies")
    cp_capture.add_argument("--minutes", type=float, metavar="M", help="on: expire after M minutes")
    for p in (cp_import, cp_jev, cp_capture, cp_sub.add_parser("list", help="every suite under the corpora root"),
              cp_sub.add_parser("stats", help="suite and item counts by role")):
        p.add_argument("--json", action="store_true")
    mv = sub.add_parser("memory-verdict", help="bank a memory proof receipt: candidate vs baseline mem+sess legs")
    mv.add_argument("--candidate", nargs="+", required=True, metavar="RUN",
                    help="candidate legs: run dirs, summary.json files or run/aa/ab receipts (>= 2 legs)")
    mv.add_argument("--baseline", nargs="+", required=True, metavar="RUN",
                    help="baseline-route legs, same forms (>= 2 legs)")
    mv.add_argument("--feature", required=True, metavar="ID", help="the registries/features.tsv row it proves, "
                                                                  "e.g. mnemopi-extraction")
    mv.add_argument("--bank", required=True, metavar="NAME", help="receipt name under docs/evidence/receipts/")
    omp = sub.add_parser("omp", help="after an omp update: refresh (carry or STALE each proof); watch the package")
    omp_sub = omp.add_subparsers(dest="omp_action", required=True)
    omp_refresh = omp_sub.add_parser("refresh", help="capture, diff against the baseline, carry forward or STALE")
    omp_refresh.add_argument("--json", action="store_true", help="the report as JSON; with --dry-run the plan")
    omp_watch = omp_sub.add_parser("watch", help="the LaunchAgent that runs `omp refresh` on an omp update")
    omp_watch.add_argument("watch_state", choices=["install", "remove", "status"])
    omp_watch.add_argument("--json", action="store_true", help="status as JSON; with --dry-run the plan")
    omp_freeze = omp_sub.add_parser("freeze", help="copy the current omp (package, its dependencies, bun) to "
                                                   "~/.localbench/omp-frozen/<version>-<sha16>/; print LOCALBENCH_OMP=")
    omp_freeze.add_argument("--json", action="store_true", help="the snapshot manifest as JSON; with --dry-run the plan")

    def measured(p, *specs):
        for s in specs:
            p.add_argument(s, help="backend spec: ollama:<model> | mlx-serve:<model dir> | omlx:<model dir> | "
                                   "mlxfast:<model dir>")
        p.add_argument("--tiers", type=_unique_tiers, default="conf,micro,replay,e2e")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--allow-busy", action="store_true", help="measure anyway; the run is non-proof")
        p.add_argument("--purge", action="store_true", help="drop the file cache first (needs sudoers grant)")
        p.add_argument("--server-arg", action="append", help="extra mlx-serve flag (repeatable)")
        p.add_argument("--wait-idle", type=float, default=0, metavar="SECONDS",
                       help="re-check a busy machine every 30 s for up to SECONDS before refusing")
        p.add_argument("--wait-slot", type=float, default=0, metavar="SECONDS",
                       help="queue for the heavy-job slot up to SECONDS instead of refusing when held or busy")
        p.add_argument("--force-load", action="store_true",
                       help="take the heavy-job slot past a busy admission (recorded in the audit row)")
        p.add_argument("--mem-config", type=_overlay, default=MEM_CONFIG, metavar="OVERLAY",
                       help="omp config overlay for the mem tier's children (default fixtures/omp/child-config-mem.yml)")
        p.add_argument("--mem-rounds", type=_mem_rounds, default=MEM_ROUNDS, metavar="N",
                       help=f"mem-tier rounds (default {MEM_ROUNDS}); each round plants three fresh facts")
        p.add_argument("--smol-model", metavar="MODEL",
                       help="a separate ollama memory (smol-role) model for the mem/sess tiers' omp children, on the "
                            "same server; pinned as smol_model/smol_digest (study legs only: no golden)")
        return p

    measured(sub.add_parser("run", help="measure one model and compare to its golden"), "backend")
    ev = sub.add_parser("eval", help="run or offline re-score a versioned omp behavioral campaign")
    evs = ev.add_subparsers(dest="eval_action", required=True)
    ev_run = evs.add_parser("run", help="run missing exact-identity cases on the real omp local path")
    ev_run.add_argument("backend", help="backend spec: ollama:<model> | mlx-serve:<model dir> | omlx:<model dir>")
    ev_run.add_argument("--resume", metavar="CAMPAIGN", help="resume only cases with matching runtime and input pins")
    ev_run.add_argument("--watchdog", action="store_true",
                        help="abort on sampled run-invariant violations; completed cases remain identity-pinned "
                             "and resumable with --resume")
    ev_run.add_argument("--server-arg", action="append", help="extra backend server flag (repeatable)")
    ev_run.add_argument("--wait-idle", type=float, default=1800, metavar="SECONDS",
                        help="wait for local model conditions before refusing (default 1800)")
    ev_run.add_argument("--wait-slot", type=float, default=0, metavar="SECONDS",
                        help="queue for the heavy-job slot up to SECONDS instead of refusing when held or busy")
    ev_run.add_argument("--force-load", action="store_true",
                        help="take the heavy-job slot past a busy admission (recorded in the audit row)")
    ev_varied = evs.add_parser("varied", help="changing-value read/edit campaign; not performance-golden evidence")
    ev_varied.add_argument("backend", help="one local backend spec, held fixed within this campaign")
    ev_varied.add_argument("--phase", choices=("exploratory", "heldout"), required=True)
    ev_varied.add_argument("--seed", type=int, required=True, help="preregister the first trial seed")
    ev_varied.add_argument("--trials", type=int, default=2, help="independent seeds per read/edit family (minimum 2)")
    ev_varied.add_argument("--mem-config", type=_overlay, default=CHILD_CONFIG, metavar="OVERLAY",
                           help="pinned omp child overlay (default: memory off)")
    ev_varied.add_argument("--resume", metavar="CAMPAIGN", help="never replace a recorded trial")
    ev_varied.add_argument("--server-arg", action="append", help="extra backend server flag (repeatable)")
    ev_varied.add_argument("--wait-idle", type=float, default=1800, metavar="SECONDS")
    ev_score = evs.add_parser("rescore", help="offline re-score an existing campaign from its preserved traces")
    ev_score.add_argument("campaign", help="a runs/eval-* campaign directory")
    ev_score.add_argument("--json", action="store_true", help="the score object as JSON")

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
    ab.add_argument("--side-regime", action="store_true",
                    help="mem/sess memory-route proofs under the side-model regime: nothing is unloaded or parked, "
                         "co-resident models are recorded per leg, never a refusal or a void (ollama arms only)")
    ab.add_argument("--b-smol-model", metavar="MODEL",
                    help="separate ollama smol (memory) model for the B leg only (default: --smol-model)")
    ab.add_argument("--b-omp", type=_exe_path, metavar="PATH",
                    help="omp executable for the B legs only (A uses LOCALBENCH_OMP or PATH), e.g. an isolated "
                         "install of an older release to A/B omp itself")
    ab.add_argument("--b-mlx-serve", type=_exe_path, metavar="PATH",
                    help="mlx-serve executable for the B legs only (A uses LOCALBENCH_MLX_SERVE or PATH), e.g. a "
                         "release extracted beside Homebrew's to A/B mlx-serve itself")
    ab.add_argument("--b-mlxfast", type=_exe_path, metavar="PATH",
                    help="mlx-server (mlxfast) executable for the B legs only (A uses LOCALBENCH_MLXFAST or PATH), e.g. "
                         "a build of another engine commit")
    for p in (sub.choices["run"], ab):
        p.add_argument("--omp-frozen", action="store_true",
                       help="freeze the current omp first (`localbench omp freeze`, or reuse its snapshot) and run every "
                            "leg through it, so an omp update mid-run cannot split the legs across omp versions")
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
    sh.add_argument("--json", action="store_true", help="the parsed receipt/golden/run document as JSON")
    rec = sub.add_parser("record", help="record omp's request body + sidecar into fixtures/omp/")
    rec.add_argument("backend", help="backend spec: ollama:<model> | mlx-serve:<model dir> | omlx:<model dir>")
    rec.add_argument("--label", required=True)
    rec.add_argument("omp_flags", nargs=argparse.REMAINDER)
    pk = sub.add_parser("park", help="move omp smol models out of reach for a test window (memory may fail)")
    pk.add_argument("--status", action="store_true", help="print park state as JSON; parks nothing")
    pk.add_argument("--json", action="store_true", help="with --status: accepted (--status always prints JSON)")
    unpk = sub.add_parser("unpark", help="restore models parked by `localbench park`")
    kp = sub.add_parser("keep", help="set a finite Ollama residency lease (default 5m; 0 unloads)")
    kp.add_argument("spec", help="ollama:<model>")
    kp.add_argument("duration", nargs="?", type=_finite_keep, default="5m",
                    help="finite duration such as 5m, 30m, 2h; 0 or unload (default 5m)")
    gw = sub.add_parser("gateway", help="manage loopback OMP Ollama routing and residency leases")
    gw.add_argument("action", choices=["serve", "status", "start", "stop", "install", "remove", "fence", "unfence"])
    gw.add_argument("--model", action="append", default=[],
                    help="fence: an Ollama model name as clients request it (repeatable); the gateway refuses its "
                         "inference requests at once until `gateway unfence --id`")
    gw.add_argument("--id", dest="fence_id", help="unfence: the fence id `gateway fence` printed")
    gw.add_argument("--wait", type=float, default=120.0,
                    help="fence: seconds to wait for in-flight requests of the model(s) to finish (default 120)")
    gw.add_argument("--host", default=gateway.HOST, help="serve only; loopback 127.0.0.1 is required")
    gw.add_argument("--port", type=int, default=gateway.PORT, help="serve/install port (default 11300)")
    gw.add_argument("--json", action="store_true", help="status JSON or mutation dry-run plan JSON")
    pl = sub.add_parser("pull", help="download an ollama model (library tag or hf.co/<org>/<repo>:<quant>), or a Hugging "
                                     "Face repo's files (hf:<org>/<repo>; token from HF_TOKEN in the environment)")
    pl.add_argument("spec", help="ollama:<model> | hf:<org>/<repo>")
    pl.add_argument("--to", type=Path, help="hf: download directory (default $LOCALBENCH_HF_DIR/<org>/<repo>, "
                                            "LOCALBENCH_HF_DIR defaulting to ~/.cache/localbench/hf; with --hub-cache, "
                                            "the cache root, default $HF_HUB_CACHE or ~/.cache/huggingface/hub)")
    pl.add_argument("--hub-cache", action="store_true", help="hf: write the Hugging Face hub cache layout "
                    "(root/models--<org>--<name>/snapshots/<sha>) the laya: backend reads offline, instead of plain "
                    "files")
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
    va.add_argument("--json", action="store_true", help="validation result as one JSON object")
    oa = sub.add_parser("ollama-app", help="Ollama.app, the supervisor of ollama serve: status, restart, auto-update")
    oa.add_argument("action", choices=["status", "restart", "auto-update"])
    oa.add_argument("state", nargs="?", choices=["on", "off", "status"],
                    help="auto-update only: turn Ollama.app's own updater on or off, or show it (default status)")
    oa.add_argument("--json", action="store_true",
                    help="status, auto-update status: one JSON object; restart/auto-update --dry-run: the plan")
    cap = sub.add_parser("capabilities", help="stable machine-readable CLI contract")
    cap.add_argument("--json", action="store_true", help="contract as JSON (default)")
    rd = sub.add_parser("robot-docs", help="paste-ready agent operating guide")
    rd.set_defaults()
    dr = sub.add_parser("doctor", help="PASS/WARN/FAIL per subsystem with the command that fixes each")
    dr.add_argument("--fix", action="store_true", help="perform the safe, reversible repairs (each is audited)")
    dr.add_argument("--json", action="store_true", help="the rows as one JSON array")
    for p, has_json in ((pk, True), (unpk, False), (kp, False), (gw, True), (pl, False), (cr, False), (sm, True), (qt, False),
                        (mem, True), (bk, False), (aa, False), (rec, False), (oa, True), (ev_varied, False),
                        (dec_run, False), (wr, True), (feat, True), (pv, True), (pre_apply, False), (pre_rb, False), (cp_import, True),
                        (dec_derive, False),
                        (cp_jev, True), (cp_capture, True), (mv, False), (omp_refresh, True), (omp_watch, True),
                        (omp_freeze, True)):
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
    handler = {"stats": cmd_stats, "status": cmd_status, "slot": cmd_slot,
               LOAD_COMMAND: cmd_load, "gpu": cmd_gpu, "run": cmd_run, "eval": cmd_eval,
               "aa": cmd_aa, "ab": cmd_ab,
               "record": cmd_record, "memory": cmd_memory, "models": cmd_models, "watch": cmd_watch,
               "report": cmd_report, "compare": cmd_compare, "bank": cmd_bank, "park": cmd_park, "unpark": cmd_unpark,
               "show": cmd_show, "keep": cmd_keep, "pull": cmd_pull, "create": cmd_create, "smol": cmd_smol,
               "quiet": cmd_quiet, "audit": cmd_audit, "why": cmd_why, "validate": cmd_validate,
               "doctor": cmd_doctor, "capabilities": cmd_capabilities, "robot-docs": cmd_robot_docs,
               "ollama-app": cmd_ollama_app, "gateway": cmd_gateway,
               "decision": cmd_decision, "features": cmd_features, "watch-releases": cmd_watch_releases,
               "prove": cmd_prove, "generation": cmd_generation,
               "preset": cmd_preset, "corpus": cmd_corpus, "memory-verdict": cmd_memory_verdict,
               "omp": cmd_omp}[args.cmd]
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
    except RuntimeError as exc:
        if mut:
            mut.finish(None, exc)
        if type(exc) is not RuntimeError:   # RecursionError, NotImplementedError: bugs keep their traceback
            raise
        # The code raises plain RuntimeError for refusals it can explain (a resolver that cannot run, a server that
        # exited, a digest that moved); the user gets the reason, not a stack (exit-code contract: 1 = refused).
        print(f"localbench {args.cmd}: {exc}", file=sys.stderr)
        return 1
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
    if c == "eval":
        return "eval varied" if args.eval_action == "varied" else None
    if c == "gateway":
        return None if args.action in {"serve", "status"} else f"gateway {args.action}"

    if c == "watch-releases":
        return c if args.once or args.install_agent else None
    if c == "corpus":
        if args.corpus_action == "capture":
            return None if args.capture_state == "status" else f"corpus capture {args.capture_state}"
        return f"corpus {args.corpus_action}" if args.corpus_action in ("import", "proj-b-build") else None
    if c == "omp":
        if args.omp_action == "freeze":
            return "omp freeze"
        if args.omp_action == "watch":
            return None if args.watch_state == "status" else f"omp watch {args.watch_state}"
        return "omp refresh"
    if c == "memory-verdict":
        return c
    if c == "preset":
        return f"preset {args.preset_action}" if args.preset_action in ("apply", "rollback") else None
    if c == "features":
        return "features --queue" if getattr(args, "queue", False) else None
    if c == "decision":
        return None if args.decision_action == "paired" else f"decision {args.decision_action}"
    if c == "park":
        return None if args.status else c
    if c == "ollama-app":
        if args.action == "auto-update":
            return None if args.state in (None, "status") else "ollama-app auto-update"
        return None if args.action == "status" else "ollama-app restart"
    if c == "smol":
        return None if args.action == "status" else f"smol {args.action}"
    if c == "memory":
        return "memory --prune" if args.prune else None
    if c == "quiet":
        return "quiet --resume" if args.resume else c
    if c == "aa":
        return "aa --write-golden" if args.write_golden else c
    if c == "prove":
        return "prove"
    if c == "generation":
        return "generation replay"
    return c if c in ("unpark", "keep", "pull", "create", "bank", "record") else None


if __name__ == "__main__":
    sys.exit(main())
