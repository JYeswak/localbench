"""Park local `smol` models while measuring.

Every omp profile on this host sets `modelRoles.smol: ollama/qwen3.8:27b-mlx`, and the default profile
routes mnemopi memory through smol, so any omp session (including localbench's own `omp -p` children)
loads that model in the background and competes for the GPU mid-run. Unloading is not enough: the next
smol call reloads it. Parking moves the NAME out of the way — `ollama cp <name> localbench-parked:<digest12>`
then delete `<name>` — so smol calls fail (user decision 2026-09-22: memory may fail while testing) and
nothing can reload it. The parked name must not contain the original: omp resolves an unknown
`ollama/<id>` by provider-scoped fuzzy match, and `localbench-parked/qwen3.8:27b-mlx` was matched and loaded
by a smol call on 2026-09-22 (ledger row). omp's resolver also hands a missing smol model's role to another
installed model (qwen3.8-uncensored:latest, 2026-09-23), and a session that starts inside a park window keeps
it after unpark, so park() parks those fallbacks too (`fallbacks`). Weights are shared blobs, so no bytes are
copied and `unpark`
restores the original name with the same digest. To benchmark a parked model, use its parked name, e.g.
`ollama:localbench-parked:5642e97495e1` (same digest, pinned in the golden).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from . import smol
from .backends import _get, _post
from .workloads import ROOT, omp_bin, omp_env

OLLAMA = "http://127.0.0.1:11434"
PREFIX = "localbench-parked:"
STATE = ROOT / "runs" / "PARKED.json"
RESOLVER = ROOT / "scripts" / "omp-resolve.ts"
HISTORY = ROOT / "runs" / "park-history.jsonl"
THINKING = {"off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"}
SMOL_SERVER = "smol-server"   # a PARKED.json entry for the dedicated smol server (localbench smol), not an ollama name


def _tags() -> dict[str, str]:
    return {m["name"]: m["digest"] for m in _get(OLLAMA + "/api/tags").get("models", [])}


def _delete(name: str) -> None:
    req = urllib.request.Request(OLLAMA + "/api/delete", json.dumps({"model": name}).encode(),
                                 {"Content-Type": "application/json"}, method="DELETE")
    with urllib.request.urlopen(req, timeout=60):
        pass


def _copy(src: str, dst: str) -> None:
    req = urllib.request.Request(OLLAMA + "/api/copy", json.dumps({"source": src, "destination": dst}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60):
        pass


def smol_targets() -> list[str]:
    """Ollama model names that any omp profile's smol role points at."""
    configs = [Path.home() / ".omp/agent/config.yml", *sorted(Path.home().glob(".omp/profiles/*/agent/config.yml"))]
    names = set()
    for cfg in configs:
        if not cfg.is_file():
            continue
        m = re.search(r"^\s+smol:\s*ollama/(\S+)\s*$", cfg.read_text(), re.MULTILINE)
        if m:
            sel = m.group(1)
            base, _, last = sel.rpartition(":")
            names.add(base if base and last in THINKING and ":" in base else sel)
    return sorted(names)


def reachable_smol() -> list[str]:
    """Smol targets an omp session could load right now (present under their own name)."""
    tags = _tags()
    return [n for n in smol_targets() if n in tags or f"{n}:latest" in tags]


def omp_resolves(selector: str, ids: list[str]) -> str | None:
    """The ollama id omp's own resolver returns for a role `selector` when `ids` are the installed ollama models;
    None when it resolves to nothing. Runs scripts/omp-resolve.ts under bun against the omp package `omp_bin()`
    runs, so a new omp release answers for itself. Measured 2026-09-23 (omp 18.2.11): with qwen3.8:27b-mlx gone it
    returns qwen3.8-uncensored:latest; with that gone too, nothing."""
    pkg = Path(os.path.realpath(omp_bin())).parent.parent
    p = subprocess.run(["bun", "run", str(RESOLVER), str(pkg), selector, *ids], capture_output=True, text=True,
                       timeout=60, check=False)
    lines = p.stdout.strip().splitlines()
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"omp resolver failed (rc {p.returncode}): {p.stderr.strip()[-300:]}")
    return json.loads(lines[-1])["picked"]


def fallbacks(resolve=omp_resolves, tags: dict | None = None) -> list[str]:
    """Installed models an omp session gets for its smol role while the smol targets are parked, in the order it
    reaches them: resolve each target's selector with the targets gone, take the pick away too, and repeat until
    omp resolves to nothing. A pick that is a parked copy cannot be parked away, so that raises."""
    tags = _tags() if tags is None else tags
    targets = smol_targets()
    gone = set(targets) | {f"{t}:latest" for t in targets}
    out: list[str] = []
    for target in targets:
        while True:
            pick = resolve(f"ollama/{target}", [n for n in tags if n not in gone])
            if pick is None or pick in gone:
                break
            if pick.startswith(PREFIX):
                raise RuntimeError(f"omp resolves ollama/{target} to the parked copy {pick}; parking cannot hide it")
            out.append(pick)
            gone.add(pick)
    return out


LOCAL_PROVIDERS = ("ollama/", "localbench/", "mlx-serve/", f"{smol.PROVIDER}/", "local/")


def local_routes(profile: str = "default") -> dict[str, str]:
    """The features of an omp session under `profile` that send work to a local model, each with its model.
    Settings come from omp itself (`omp config list --json`, defaults included). The fallback chains are omp
    18.2.11's: config/model-resolver.ts (memory -> tiny -> smol; the judge chain tries typesafe/proj-b-latest when
    TypeSafe is credentialed, then @tiny, @smol), utils/title-generator.ts (tiny, commit, smol) and
    commit/model-selection.ts (commit, smol). The bundled `scout` subagent (prompts/agents/scout.md) runs on
    `@smol`. `local/<key>` is omp's in-process tiny model: ONNX on the CPU unless
    providers.tinyModelDevice is `mlx`, so it is not GPU work."""
    args = [omp_bin(), *([] if profile == "default" else ["--profile", profile]), "config", "list", "--json"]
    raw = subprocess.run(args, capture_output=True, text=True, timeout=60, env=omp_env(), check=False).stdout
    cfg = {k: (v or {}).get("value") for k, v in json.loads(raw or "{}").items()}
    roles = cfg.get("modelRoles") or {}

    def first(*chain: str) -> str | None:
        return next((roles[r] for r in chain if roles.get(r)), None)

    uses = {
        "main turns": first("default"),
        "scout subagents (task tool; whole agent turns)": first("smol"),
        "auto-thinking classifier (every user turn, before the main call, 4 s cap)":
            first("judge", "tiny", "smol") if cfg.get("defaultThinkingLevel") == "auto" else None,
        "mnemopi memory LLM (retain/consolidate)":
            first("memory", "tiny", "smol")
            if cfg.get("memory.backend") == "mnemopi" and cfg.get("mnemopi.llmMode") == "smol" else None,
        "session titles": first("tiny", "commit", "smol"),
        "commit messages": first("commit", "smol"),
        "edit auto-repair, eval completion('smol')": first("smol"),
        "eval judge(), find judge, TTSR question rules": first("judge", "tiny", "smol"),
        "mnemopi embeddings (recall + retain, worker per memory-on process)":
            f"local/{fastembed_name(cfg)}"
            if cfg.get("memory.backend") == "mnemopi" and not cfg.get("mnemopi.noEmbeddings") else None,
    }
    return {k: v for k, v in uses.items() if v and v.startswith(LOCAL_PROVIDERS)}


def fastembed_name(cfg: dict) -> str:
    """The fastembed cache dir mnemopi loads for a profile's settings: omp 18.2.11 mnemopi/config.ts (explicit
    mnemopi.embeddingModel, else the variant default) mapped by pi-mnemopi core/embeddings.ts (`BAAI/bge-base-en-v1.5`
    -> `fast-bge-base-en-v1.5`). The MNEMOPI_EMBEDDING_MODEL env override is not visible here."""
    model = (cfg.get("mnemopi.embeddingModel") or "").strip() or (
        "intfloat/multilingual-e5-large" if cfg.get("mnemopi.embeddingVariant") == "multilingual"
        else "BAAI/bge-base-en-v1.5")
    return "fast-" + model.split("/")[-1]


def parked_now() -> list[dict]:
    """The PARKED.json entries: what `unpark` would restore."""
    return json.loads(STATE.read_text()) if STATE.exists() else []


def plan_park(resolve=None) -> list[dict]:
    """What park() would do now, changing nothing: every loadable smol target, then every model omp would hand
    their role instead (`fallbacks`, taken from the full installed set before anything moves), each
    {name, parked_as, digest, role}; then the dedicated smol server ({kind: SMOL_SERVER}) when it is up and not
    parked yet. Empty when everything is parked already."""
    tags = _tags()
    loadable = reachable_smol()
    plan = [{"name": name, "parked_as": PREFIX + tags[name].removeprefix("sha256:")[:12], "digest": tags[name],
             "role": "smol" if name in loadable else "fallback"}
            for name in dict.fromkeys([*loadable, *fallbacks(resolve or omp_resolves, tags)])]
    live = smol.load_state()
    if live and smol.server_up(live["port"]) and not any(p.get("kind") == SMOL_SERVER for p in parked_now()):
        # Since 2026-09-25 smol can live on its own mlx-serve (localbench smol): stopping it is its park, and every
        # profile's static PROVIDER entry keeps resolving, so smol calls fail at request time instead of moving.
        plan.append({"name": f"{smol.PROVIDER}/{live['model_id']}", "parked_as": "(server stopped)", "digest": None,
                     "role": "smol", "kind": SMOL_SERVER})
    return plan


def park(resolve=None, plan: list[dict] | None = None) -> list[dict]:
    """Carry out `plan` (default plan_park(resolve)) and return every parked entry, earlier parks included. Each
    ollama name is copied to its parked name, the copy's digest checked, then the name unloaded and deleted. A plan
    with nothing in it writes nothing."""
    plan = plan_park(resolve) if plan is None else plan
    parked = parked_now()
    for entry in plan:
        if entry.get("kind") == SMOL_SERVER:
            smol.stop_server(smol.load_state())
            parked.append(entry)
            continue
        name, dst = entry["name"], entry["parked_as"]
        if dst not in _tags():
            _copy(name, dst)
        if _tags().get(dst) != entry["digest"]:
            raise RuntimeError(f"parked copy {dst} digest {_tags().get(dst)} != {entry['digest']}; refusing to delete {name}")
        _post(OLLAMA + "/api/generate", {"model": name, "keep_alive": 0})
        _delete(name)
        parked.append(entry)
    if plan:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(parked, indent=2) + "\n")
        # sealed: omp's resolver has nothing left to hand a parked smol role (fallbacks parked above).
        append_history("park", [e["name"] for e in plan], sealed=True)
    return parked


def unpark() -> list[dict]:
    """Restore every entry parked_now() lists (the ollama name from its parked copy, digest checked; the smol server
    restarted), then forget them. Nothing parked: nothing happens, and [] is returned."""
    parked = parked_now()
    if not parked:
        return []
    for p in parked:
        if p.get("kind") == SMOL_SERVER:
            smol.start_server(smol.load_state())
            continue
        tags = _tags()
        if p["name"] not in tags:
            _copy(p["parked_as"], p["name"])
        if _tags().get(p["name"]) != p["digest"]:
            raise RuntimeError(f"restored {p['name']} digest differs from {p['digest']}; keeping {p['parked_as']}")
        for loaded in _get(OLLAMA + "/api/ps").get("models", []):
            if loaded["name"] == p["parked_as"]:
                _post(OLLAMA + "/api/generate", {"model": p["parked_as"], "keep_alive": 0})
        _delete(p["parked_as"])
    STATE.unlink()
    append_history("unpark", [p["name"] for p in parked])
    return parked


def ollama_ids(raw: str) -> set[str]:
    """Ollama model ids in `omp models ollama --json` output ({"models": [{"provider", "id", ...}]})."""
    try:
        rows = json.loads(raw or "{}").get("models") or []
    except (ValueError, AttributeError):
        return set()
    return {r.get("id") for r in rows if r.get("provider") == "ollama"}


def refresh_catalogs(names: list[str], profiles: list[str], run=subprocess.run) -> dict[str, list[str]]:
    """Make every omp profile see the tags an unpark restored, and say which still cannot.

    omp caches its implicit ollama discovery per profile for 24 h (pi-coding-agent model-registry.ts), so a profile
    whose cache was taken during a park keeps a catalog without the parked tag: on 2026-09-24 six of the nine smol
    profiles did not list qwen3.8:27b-mlx, and the default profile still listed a parked copy deleted hours earlier.
    Each profile's ollama catalog is refreshed through omp itself, then read back. Returns profile -> restored names
    that profile still does not list."""
    missing = {}
    for profile in profiles:
        pre = [omp_bin(), *([] if profile == "default" else ["--profile", profile]), "models"]
        run([*pre, "refresh", "ollama"], capture_output=True, text=True, timeout=120, env=omp_env(), check=False)
        listed = ollama_ids(run([*pre, "ollama", "--json"], capture_output=True, text=True, timeout=60,
                                env=omp_env(), check=False).stdout)
        missing[profile] = [n for n in names if n not in listed and f"{n}:latest" not in listed]
    return missing


def append_history(event: str, names: list[str], t: float | None = None, sealed: bool = False) -> dict:
    """One park or unpark. Nothing here prints. A caller that parked nothing does not open a window. A `sealed`
    park also parked every model omp's resolver would hand the role instead (rows before 2026-09-23 20:39 are not)."""
    row = {"event": event, "t": time.time() if t is None else t, "names": list(names)}
    if sealed:
        row["sealed"] = True
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return row


def read_history() -> list[dict]:
    if not HISTORY.is_file():
        return []
    return [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]


def park_windows(rows: list[dict] | None = None, now: float | None = None, leaky_only: bool = False) -> list[list[float]]:
    """Closed [park, unpark) spans, plus a still-open park ending at `now`. A second park before unpark
    does not move the start: the window is the whole time the name was out of the way. With `leaky_only`, only
    the spans in which omp still had another model to hand a parked smol role: a sealed park opens none and
    ends an open one."""
    rows = read_history() if rows is None else rows
    now = time.time() if now is None else now
    windows: list[list[float]] = []
    open_at = None
    for row in rows:
        sealed = leaky_only and row.get("sealed")
        if row.get("event") == "park" and open_at is None and row.get("names") and not sealed:
            open_at = row["t"]
        elif open_at is not None and (row.get("event") == "unpark" or (row.get("event") == "park" and sealed)):
            windows.append([open_at, row["t"]])
            open_at = None
    if open_at is not None:
        windows.append([open_at, now])
    return windows


def process_started(pid: int) -> float | None:
    """Epoch seconds of a process start, from `ps -o lstart=` in the C locale. None if the pid is gone."""
    out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
                         env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"}, check=False)
    text = out.stdout.strip()
    if not text:
        return None
    return datetime.strptime(text, "%a %b %d %H:%M:%S %Y").astimezone().timestamp()


def stuck_sessions(clients: list[dict]) -> list[dict]:
    """Omp clients whose start falls inside a leaky park window (omp could hand their smol role another local
    model, which they keep after unpark). Each row is pid, cwd, started, window. A start at the unpark instant is
    outside. Nothing here prints."""
    windows = park_windows(leaky_only=True)
    out = []
    for c in clients:
        cmd = c.get("cmd") or ""
        if not (re.search(r"\bomp\b", cmd) or "omp_profile" in c or "agent_dir" in c):
            continue
        started = process_started(c["pid"])
        if started is None:
            continue
        for window in windows:
            if window[0] <= started < window[1]:
                out.append({"pid": c["pid"], "cwd": c.get("cwd"), "started": started, "window": window})
                break
    return out
