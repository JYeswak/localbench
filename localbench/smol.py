"""omp's `smol` role on a dedicated local server, switched and reverted as one operation.

the owner, 2026-09-25: "we dont need 4 hour test - go live with it" after the stage 1 screen of the Qwen3.8-27B MLX-Serve
4-bit pack with MTP on mlx-serve 26.9.5 (ledger KEEP (SCREEN), receipts/ab-screen-qwen38-4bit-mtp-20260925.json).

Three parts, each reversible:
- a server: mlx-serve on 127.0.0.1:PORT serving one model dir. Without autostart it runs in its own session from
  the caller's lineage and is down after a reboot. `autostart on` makes it the LaunchAgent LABEL (RunAtLoad, no
  KeepAlive, so park's stop sticks); that needs Full Disk Access granted to the mlx-serve binary itself, because a
  launchd job reading /Volumes/Models without it died with `error: PermissionDenied` (macOS privacy; the
  terminal's grant is not inherited). `localbench status` says when it is down.
- every omp profile's `modelRoles.smol` line, pointed at PROVIDER/<model id>; nothing else in config.yml changes.
- a marked `PROVIDER` block in each profile's models.yml (static model entry, so the role resolves whatever the
  server state), inserted under `providers:`; the file is created when a profile has none.

`revert` undoes exactly those edits (the previous smol line comes back; the block goes; a models.yml that `set`
created is removed), so other edits made since survive. Whole-file backups are kept under BACKUPS as well.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .backends import _get
from .sysstats import SMOL_PORT
from .workloads import omp_bin, omp_env

PROVIDER = "mlx-smol"
HOST, PORT = "127.0.0.1", SMOL_PORT
STATE_DIR = Path.home() / ".localbench" / "smol"
BACKUPS = Path.home() / ".localbench" / "rollback"
LABEL = "com.localbench.smol"
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
BEGIN = f"  # >>> localbench smol: {PROVIDER} provider (localbench smol set / revert)"
END = "  # <<< localbench smol"
SMOL_LINE = re.compile(r"^(?P<indent>[ \t]+)smol:[ \t]*(?P<value>\S+)[ \t]*$", re.MULTILINE)


def state_path(state_dir: Path | None = None) -> Path:
    """Resolved at call time from STATE_DIR, so the test package can point it away from the live server's state."""
    return (state_dir or STATE_DIR) / "state.json"


def load_state(state_dir: Path | None = None) -> dict | None:
    p = state_path(state_dir)
    return json.loads(p.read_text()) if p.exists() else None


def _save_state(state: dict, state_dir: Path | None = None) -> None:
    (state_dir or STATE_DIR).mkdir(parents=True, exist_ok=True)
    state_path(state_dir).write_text(json.dumps(state, indent=2) + "\n")


def profile_dirs(home: Path | None = None) -> dict[str, Path]:
    """profile -> agent dir, for every omp profile with a config.yml (the default profile is ~/.omp/agent)."""
    home = home or Path.home()
    dirs = {"default": home / ".omp" / "agent"} if (home / ".omp" / "agent" / "config.yml").is_file() else {}
    for cfg in sorted((home / ".omp" / "profiles").glob("*/agent/config.yml")):
        dirs[cfg.parent.parent.name] = cfg.parent
    return dirs


def provider_block(model_id: str, context_window: int, port: int = PORT) -> str:
    """The models.yml block for the smol server: the same compat and static entry shape as the localbench provider
    the screen measured through (workloads.ensure_localbench_model), pointed straight at the server."""
    return "\n".join([
        BEGIN,
        f"  {PROVIDER}:",
        f"    baseUrl: http://{HOST}:{port}/v1",
        "    api: openai-completions",
        "    auth: none",
        "    compat:",
        "      supportsDeveloperRole: false",
        "      supportsReasoningEffort: true",
        "      maxTokensField: max_tokens",
        "      thinkingFormat: qwen",
        "    models:",
        f'      - id: "{model_id}"',
        f"        contextWindow: {context_window}",
        "        maxTokens: 32768",
        "        reasoning: true",
        "        input: [text]",
        END,
    ]) + "\n"


def _strip_block(text: str) -> str:
    return re.sub(rf"^{re.escape(BEGIN)}\n.*?^{re.escape(END)}\n", "", text, flags=re.MULTILINE | re.DOTALL)


def with_block(text: str | None, block: str) -> str:
    """models.yml text with `block` under `providers:` (replacing an earlier block of ours); a new file when None."""
    if text is None:
        return "# Local providers written by `localbench smol set`; `localbench smol revert` removes this file.\nproviders:\n" + block
    text = _strip_block(text)
    m = re.search(r"^providers:[ \t]*\n", text, re.MULTILINE)
    if m:
        return text[:m.end()] + block + text[m.end():]
    return text.rstrip("\n") + "\nproviders:\n" + block


def set_profiles(selector: str, block: str, dirs: dict[str, Path], backup_dir: Path,
                 prior: dict[str, dict] | None = None) -> dict[str, dict]:
    """Point every profile's smol line at `selector` and add `block` to its models.yml. Returns per profile what
    revert needs: the previous smol value and whether models.yml was created. A profile without a smol line is
    skipped (reported with previous None). Whole files are copied to backup_dir first. `prior` is an earlier set's
    record: a profile that already says `selector` keeps its original previous value and created flag from it, so a
    second set does not make revert forget the ollama model."""
    prior = prior or {}
    done: dict[str, dict] = {}
    for name, agent in dirs.items():
        cfg, mdl = agent / "config.yml", agent / "models.yml"
        dest = backup_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg, dest / "config.yml")
        if mdl.exists():
            shutil.copy2(mdl, dest / "models.yml")
        text = cfg.read_text()
        m = SMOL_LINE.search(text)
        if not m:
            done[name] = {"previous": None, "created_models_yml": False}
            continue
        created = not mdl.exists()
        mdl.write_text(with_block(None if created else mdl.read_text(), block))
        if m.group("value") == selector and name in prior:
            done[name] = prior[name]
            continue
        cfg.write_text(text[:m.start("value")] + selector + text[m.end("value"):])
        done[name] = {"previous": m.group("value"), "created_models_yml": created}
    return done


def plan_revert(record: dict[str, dict], selector: str, dirs: dict[str, Path]) -> list[tuple[str, str | None]]:
    """What revert_profiles would do, changing nothing: (profile, the smol value it gets back), or (profile, None)
    when its smol line no longer says `selector` (changed by hand since set; it is left alone). Profiles set skipped
    or that no longer exist are not listed."""
    out = []
    for name, rec in record.items():
        agent = dirs.get(name)
        if agent is None or rec.get("previous") is None:
            continue
        m = SMOL_LINE.search((agent / "config.yml").read_text())
        out.append((name, rec["previous"] if m and m.group("value") == selector else None))
    return out


def revert_profiles(record: dict[str, dict], selector: str, dirs: dict[str, Path]) -> list[str]:
    """Undo set_profiles per plan_revert: the recorded smol value comes back where the line still says `selector`,
    the block goes, and a models.yml that set created is removed when nothing but our block is left in it. Returns
    profiles whose smol line no longer said `selector` (changed by hand since; left alone)."""
    untouched = []
    for name, previous in plan_revert(record, selector, dirs):
        cfg, mdl = dirs[name] / "config.yml", dirs[name] / "models.yml"
        if previous is None:
            untouched.append(name)
        else:
            text = cfg.read_text()
            m = SMOL_LINE.search(text)
            cfg.write_text(text[:m.start("value")] + previous + text[m.end("value"):])
        if mdl.exists():
            rest = _strip_block(mdl.read_text())
            if record[name].get("created_models_yml") and not re.sub(r"(?m)^#.*\n|^providers:[ \t]*\n", "", rest).strip():
                mdl.unlink()
            else:
                mdl.write_text(rest)
    return untouched


def context_length(model_id: str, port: int = PORT) -> int | None:
    """The context the server loaded for `model_id` (the field MlxServe.fingerprint reads)."""
    try:
        rows = _get(f"http://{HOST}:{port}/v1/models", timeout=5).get("data", [])
    except OSError:
        return None
    row = next((r for r in rows if r.get("id") == model_id), {})
    return row.get("context_length") or (row.get("meta") or {}).get("context_length")


# ------------------------------------------------------------------ server


def server_up(port: int = PORT) -> bool:
    try:
        _get(f"http://{HOST}:{port}/v1/models", timeout=2)
        return True
    except OSError:
        return False


def served_ids(port: int = PORT) -> list[str]:
    try:
        return [m["id"] for m in _get(f"http://{HOST}:{port}/v1/models", timeout=2).get("data", [])]
    except OSError:
        return []


def server_bin(state: dict) -> str:
    """The mlx-serve binary the smol server runs (recorded by `set`: a side-by-side release, not PATH's)."""
    return state["binary"]


def server_argv(state: dict) -> list[str]:
    return [server_bin(state), "--model", state["model_dir"], "--serve", "--host", HOST, "--port", str(state["port"]),
            "--metrics", *state.get("server_args", [])]


def listener_pid(port: int = PORT, run=subprocess.run) -> int | None:
    """The pid listening on the port now. A pid saved in the state file goes stale across a reboot or a launchd
    restart and could by then name an unrelated process, so stop signals this one instead."""
    out = run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


# ------------------------------------------------------------------ launchd autostart


def plist_path() -> Path:
    """Resolved at call time from LAUNCH_AGENTS, so the test package can point it away from the live job."""
    return LAUNCH_AGENTS / f"{LABEL}.plist"


def launchd_plist(state: dict) -> bytes:
    log = str(STATE_DIR / "server.log")
    return plistlib.dumps({"Label": LABEL, "ProgramArguments": server_argv(state), "RunAtLoad": True,
                           "KeepAlive": False, "ProcessType": "Interactive",
                           "StandardOutPath": log, "StandardErrorPath": log})


def autostart_on() -> bool:
    return plist_path().exists()


def _target() -> str:
    return f"gui/{os.getuid()}/{LABEL}"


def _job_running(run=subprocess.run) -> bool:
    out = run(["launchctl", "print", _target()], capture_output=True, text=True).stdout
    return bool(re.search(r"^\s*state = running\s*$", out, re.MULTILINE))


def install_autostart(state: dict, run=subprocess.run) -> None:
    """Write the LaunchAgent for `state` and (re)load it; loading starts the server (RunAtLoad). The caller stops a
    server the job would collide with on the port first."""
    p = plist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    run(["launchctl", "bootout", _target()], capture_output=True)
    p.write_bytes(launchd_plist(state))
    run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(p)], check=True, capture_output=True)


def remove_autostart(run=subprocess.run) -> None:
    """Unload the job (launchd stops its server) and delete the plist."""
    run(["launchctl", "bootout", _target()], capture_output=True)
    plist_path().unlink(missing_ok=True)


# ------------------------------------------------------------------ start / stop


def start_server(state: dict, ready_timeout: float = 900) -> int:
    """Start the recorded server and wait until it serves; returns its pid. With autostart on, launchd starts it (the
    plist is rewritten first when `set` changed binary, model or args); otherwise it starts in its own session and
    outlives this process. A server already answering on the port is taken as running (`set` then checks the model
    it serves); a port held by something that does not answer makes the new server exit, which raises here."""
    if server_up(state["port"]):
        return listener_pid(state["port"]) or state.get("pid") or 0
    log = STATE_DIR / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = server_argv(state)
    launchd = autostart_on()
    with log.open("a") as fh:
        fh.write(f"\n=== {datetime.now(UTC).isoformat(timespec='seconds')} start{' (launchd)' if launchd else ''} "
                 f"{' '.join(argv)}\n")
    if launchd:
        if plist_path().read_bytes() != launchd_plist(state):
            install_autostart(state)
        else:
            subprocess.run(["launchctl", "kickstart", _target()], check=True, capture_output=True)
        alive = _job_running
    else:
        with log.open("a") as fh:
            proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        alive = lambda: proc.poll() is None
    started = time.time()
    deadline = started + ready_timeout
    while not server_up(state["port"]):
        if time.time() - started > 5 and not alive():   # launchd reports the job a moment after bootstrap/kickstart
            tail = log.read_text(errors="replace").splitlines()[-3:]
            hint = (" (launchd job: grant Full Disk Access to that binary, see smol.py)"
                    if launchd and any("PermissionDenied" in t for t in tail) else "")
            raise RuntimeError(f"smol server exited{hint}; {log} ends: {' | '.join(tail)}")
        if time.time() > deadline:
            raise TimeoutError(f"smol server not ready after {ready_timeout:.0f}s; see {log}")
        time.sleep(2)
    state["pid"] = listener_pid(state["port"])
    _save_state(state)
    return state["pid"]


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_server(state: dict, timeout: float = 60) -> None:
    """Stop whatever serves the smol port (see listener_pid) and wait until that process has exited, not only until
    the port closes: mlx-serve closes its socket and then spends seconds shutting down, and a launchd kickstart in
    that gap is a no-op on the still-running job, so unpark's start found the job gone (2026-09-25). Under launchd
    the job has no KeepAlive, so it stays down until start_server kickstarts it."""
    pid = listener_pid(state["port"])
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid = None
    deadline = time.time() + timeout
    while (server_up(state["port"]) or _alive(pid)) and time.time() < deadline:
        time.sleep(1)
    if (server_up(state["port"]) or _alive(pid)) and pid:
        os.kill(pid, signal.SIGKILL)
        time.sleep(2)
    if server_up(state["port"]):
        raise RuntimeError(f"smol server on port {state['port']} still answers after SIGTERM/SIGKILL of pid {pid}")


# ------------------------------------------------------------------ verification


def omp_smol(profile: str, run=subprocess.run) -> str | None:
    """The smol role omp itself reports for a profile (`omp config list --json`, defaults applied)."""
    args = [omp_bin(), *([] if profile == "default" else ["--profile", profile]), "config", "list", "--json"]
    raw = run(args, capture_output=True, text=True, timeout=60, env=omp_env(), check=False).stdout
    roles = (json.loads(raw or "{}").get("modelRoles") or {}).get("value") or {}
    return roles.get("smol")


def omp_lists(profile: str, model_id: str, run=subprocess.run) -> bool:
    """True when the profile's model registry lists PROVIDER/model_id (`omp models PROVIDER --json`)."""
    args = [omp_bin(), *([] if profile == "default" else ["--profile", profile]), "models", PROVIDER, "--json"]
    raw = run(args, capture_output=True, text=True, timeout=60, env=omp_env(), check=False).stdout
    try:
        rows = json.loads(raw or "{}").get("models") or []
    except ValueError:
        return False
    return any(r.get("provider") == PROVIDER and r.get("id") == model_id for r in rows)
