"""`localbench doctor`: one PASS / WARN / FAIL row per subsystem this harness depends on, each probed in isolation, so a
dead subsystem is one FAIL row, never a crash. FAIL means localbench cannot measure here; WARN means it can, with a
gap the row names; every row that has a remedy names the exact command in `fix`.

`--fix` performs one repair, the only one that is safe, reversible and idempotent: removing a mutation lock
(runs/.mutation.lock, held by scripts/mutate.py) whose owner pid is provably dead. Everything else (unpark, smol start,
re-banking goldens, sudoers) restarts servers, writes goldens or needs root, so it is reported with its command and
left to the user. Each repair appends an audit row (localbench audit / why).

A live measurement has no marker to go stale: `_run_alive()` asks pgrep for the running process itself."""

from __future__ import annotations

import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path

from . import backends, golden, park, smol, sysstats, workloads

MUTATION_LOCK = workloads.ROOT / "runs" / ".mutation.lock"
WRITE_GOLDEN = "localbench aa <spec> --write-golden"


def _row(status: str, detail: str, fix: str | None = None) -> dict:
    return {"status": status, "detail": detail, "fix": fix, "fixed": False}


def _cli():
    """The CLI module, imported at call time: it imports this one."""
    from . import __main__ as cli
    return cli


@cache
def _host() -> dict:
    return sysstats.host()


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


# ------------------------------------------------------------------ probes (each: fix flag -> row)


def check_platform(_fix: bool) -> dict:
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine == "arm64":
        return _row("PASS", f"macOS {platform.mac_ver()[0]} on {machine}")
    return _row("FAIL", f"{system} {machine}: localbench measures macOS on Apple Silicon only (ioreg GPU counters, "
                        "sysctl, pmset)")


def check_python(_fix: bool) -> dict:
    """The interpreter this install runs on. requires-python (>= 3.12) is enforced at install, so there is no failing
    branch to probe: the row names which interpreter and where, for a user with several."""
    return _row("PASS", f"{platform.python_version()} ({sys.executable})")


def check_data_root(_fix: bool) -> dict:
    root = workloads.ROOT
    if not (workloads.FIXTURES / "omp").is_dir():
        return _row("FAIL", f"no data root at {root} (no fixtures/omp): every verb but stats, memory, keep, pull and "
                            "create exits 2", "export LOCALBENCH_HOME=<path to a localbench clone>")
    host_id = _host()["host_id"]
    n = len(list((golden.GOLDENS / host_id).glob("*.json")))
    if not n:
        return _row("WARN", f"{root}; no goldens for this host ({host_id}): no regression gate here yet", WRITE_GOLDEN)
    return _row("PASS", f"{root}; {n} golden(s) for {host_id}")


def check_omp(_fix: bool) -> dict:
    install = "bun install -g @oh-my-pi/pi-coding-agent  # or: export LOCALBENCH_OMP=<omp executable>"
    try:
        binary = workloads.omp_bin()
    except FileNotFoundError:
        return _row("FAIL", "omp is not on PATH and LOCALBENCH_OMP is unset: runs cannot pin it, the e2e, rel, mem "
                            "and sess tiers cannot drive it", install)
    version = backends._first_line(binary, "--version").removeprefix("omp/").strip()
    if not version:
        return _row("FAIL", f"{binary} --version printed nothing: not a runnable omp", install)
    via = " (LOCALBENCH_OMP)" if os.environ.get("LOCALBENCH_OMP") else ""
    return _row("PASS", f"omp {version} at {binary}{via}")


def check_ollama(_fix: bool) -> dict:
    ollama = backends.Ollama()
    try:
        # The pins every ollama run records: the server's model list answers, and `ollama --version` names it.
        version = ollama.pins("")["backend_version"]
    except (OSError, ValueError) as exc:
        return _row("WARN", f"not answering on {ollama.root} ({exc}); optional when mlx-serve or oMLX serves the "
                            "model", "open -a Ollama")
    auto = sysstats.ollama_auto_update()
    detail = f"ollama {version} on {ollama.root}; Ollama.app auto-update " + (
        "unknown (no Ollama.app settings database)" if auto is None else "ON" if auto else "off")
    if auto:
        return _row("WARN", detail + ": the app can install a new ollama under every golden (turn it off in "
                                     "Ollama.app Settings)")
    return _row("PASS", detail)


def check_mlx(_fix: bool) -> dict:
    found, missing = [], []
    for name, exe, install in (
            ("mlx-serve", backends.mlx_serve_bin(),
             "brew install ddalcu/mlx-serve/mlx-serve  # or: export LOCALBENCH_MLX_SERVE=<mlx-serve executable>"),
            ("oMLX", "omlx", "uv tool install omlx")):
        path = shutil.which(exe)
        (found if path else missing).append((name, path or exe, install))
    detail = "; ".join([f"{n} at {p}" for n, p, _ in found] + [f"{n} not found ({p})" for n, p, _ in missing])
    if missing:
        return _row("WARN", detail + " (optional: needed only for mlx-serve: / omlx: specs)",
                    "; ".join(i for _, _, i in missing))
    return _row("PASS", detail)


def _sudo_granted(command: str) -> bool:
    """True when sudoers lets this user run `command` without a password (listed, never run)."""
    return subprocess.run(["sudo", "-n", "-l", command], capture_output=True, check=False).returncode == 0


def check_sudoers(_fix: bool) -> dict:
    grants = {"powermetrics": sysstats.powermetrics_available(), "purge": _sudo_granted("/usr/sbin/purge")}
    detail = ", ".join(f"{k} {'granted' if v else 'not granted'}" for k, v in grants.items())
    if all(grants.values()):
        return _row("PASS", detail + " (sudo -n)")
    return _row("WARN", detail + ": runs record no powermetrics power/frequency and cannot --purge (optional)",
                "sudo scripts/install-sudoers.sh")


def check_park(_fix: bool) -> dict:
    parked = json.loads(park.STATE.read_text()) if park.STATE.exists() else []
    if not parked:
        return _row("PASS", "not parked")
    names = ", ".join(f"{p['name']} as {p['parked_as']}" for p in parked)
    if _cli()._run_alive():
        return _row("PASS", f"parked for the live run: {names}")
    # Never repaired here: unparking restarts servers and reloads models.
    return _row("WARN", f"parked while no run is alive: {names}; omp sessions' smol/memory calls get the parked "
                        "fallback until unpark", "localbench unpark")


def check_smol(_fix: bool) -> dict:
    st = smol.load_state()
    if not st:
        return _row("PASS", "no managed smol server (`localbench smol set` manages one)")
    where = f"{st['selector']} on :{st['port']}"
    autostart = smol.autostart_on()
    if autostart:
        argv = plistlib.loads(smol.plist_path().read_bytes()).get("ProgramArguments")
        if argv != smol.server_argv(st):
            return _row("WARN", f"{where}: the LaunchAgent {smol.plist_path()} starts {argv}, not the recorded "
                                "server; a login would start the wrong one", "localbench smol autostart on")
    boot = f"; LaunchAgent {'present (autostart on)' if autostart else 'absent (autostart off)'}"
    if smol.server_up(st["port"]):
        return _row("PASS", f"{where} up{boot}")
    if _cli()._smol_parked():
        return _row("PASS", f"{where} parked (stopped by `localbench park`){boot}")
    return _row("WARN", f"{where} DOWN: every omp session's smol/memory calls fail{boot}", "localbench smol start")


def _owner(lock: Path) -> tuple[str | None, int | None]:
    """The lock's owner line (`pid <n> since <ctime>`, scripts/mutate.py) and its pid; None for what is absent."""
    try:
        owner = (lock / "owner").read_text().strip()
    except OSError:
        return None, None
    words = owner.split()
    return owner, int(words[1]) if len(words) > 1 and words[0] == "pid" and words[1].isdigit() else None


def _pid_dead(pid: int) -> bool:
    """True only when no process has this pid (ESRCH). Permission denied means a process exists: not dead."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def check_mutation_lock(fix: bool) -> dict:
    lock = MUTATION_LOCK
    if not lock.exists():
        return _row("PASS", f"no mutation lock ({lock})")
    owner, pid = _owner(lock)
    if pid is None:
        return _row("WARN", f"{lock} has no readable owner pid: cannot prove its holder dead; scripts/mutate.py waits "
                            "on it", f"rm -rf {lock}  # only after checking no mutate.py runs (pgrep -f mutate.py)")
    if not _pid_dead(pid):
        return _row("PASS", f"{lock} held by live pid {pid}")
    if not fix:
        return _row("WARN", f"{lock} held by dead pid {pid}: scripts/mutate.py would wait on it, then time out",
                    "localbench doctor --fix")
    from . import audit
    (lock / "owner").unlink()
    lock.rmdir()
    action = f"removed {lock} (owner {owner!r}: pid {pid} is dead)"
    audit.record("doctor --fix", sys.argv[1:], [action], "done", {"path": str(lock), "pid": pid, "owner": owner})
    return {**_row("PASS", action), "fixed": True}


def _spec(path: Path) -> str:
    """The backend spec that re-banks a golden: ollama:<model>, else <backend>:<model dir from its A/A receipt>."""
    g = golden.load(path)
    if g["pins"]["backend"] == "ollama":
        return f"ollama:{g['pins']['model']}"
    receipt = json.loads((workloads.ROOT / g["aa_receipt"]).read_text())
    return f"{g['pins']['backend']}:{receipt['runs'][0]['provenance']['fingerprint']['model_dir']}"


def check_goldens(_fix: bool) -> dict:
    host = _host()
    states = _cli().golden_states(host)
    stale = [(g["golden"], t) for g in states for t in g.get("tiers", []) if t["state"] == "GENERATION-MISMATCH"]
    unavailable = sum("unavailable" in g for g in states)
    counts = f"{len(states)} golden(s), {unavailable} unavailable now"
    if not stale:
        return _row("PASS", f"{counts}; every available tier CURRENT")
    detail = "; ".join(f"{name} {','.join(t['tiers'])} GENERATION-MISMATCH (moved: {', '.join(t['moved'])})"
                       for name, t in stale)
    fixes = [f"localbench aa {_spec(golden.GOLDENS / host['host_id'] / name)} --tiers {','.join(t['tiers'])} "
             "--write-golden" for name, t in stale]
    return _row("WARN", f"{counts}; {detail}", "; ".join(fixes))


def _free(path: Path) -> tuple[Path, int]:
    anchor = next(p for p in (path, *path.parents) if p.exists())
    return anchor, shutil.disk_usage(anchor).free


def check_disk(_fix: bool) -> dict:
    cli = _cli()
    floor = cli.HF_HEADROOM_GB * 1e9
    home, home_free = _free(Path.home())
    hf, hf_free = _free(cli.HF_DIR)
    parts = [f"home ({home}) {_gb(home_free)} free"]
    if os.stat(hf).st_dev != os.stat(home).st_dev:
        parts.append(f"HF dir {cli.HF_DIR} ({hf}) {_gb(hf_free)} free")
    if min(home_free, hf_free) >= floor:
        return _row("PASS", "; ".join(parts))
    fix = "export LOCALBENCH_HF_DIR=<dir on a volume with room>" if hf_free < floor else None
    return _row("WARN", "; ".join(parts) + f": under {cli.HF_HEADROOM_GB} GB free", fix)


CHECKS: list[tuple[str, Callable[[bool], dict]]] = [
    ("platform", check_platform), ("python", check_python), ("data root", check_data_root), ("omp", check_omp),
    ("ollama", check_ollama), ("mlx-serve/oMLX", check_mlx), ("sudoers", check_sudoers), ("park", check_park),
    ("smol server", check_smol), ("mutation lock", check_mutation_lock), ("goldens", check_goldens), ("disk", check_disk),
]


def checks(fix: bool = False) -> list[dict]:
    """Every probe in order, each isolated: an exception inside one is a FAIL row for that subsystem, and the rest
    still run. Rows: {subsystem, status, detail, fix, fixed}."""
    out = []
    for subsystem, probe in CHECKS:
        try:
            row = probe(fix)
        except Exception as exc:  # noqa: BLE001 - a dead subsystem is a row, never a crash
            row = _row("FAIL", f"probe raised {type(exc).__name__}: {exc}"[:300])
        out.append({"subsystem": subsystem, **row})
    return out
