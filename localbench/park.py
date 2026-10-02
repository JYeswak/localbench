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

import fcntl
import functools
import json
import os
import re
import subprocess
import time
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import gateway, smol
from .backends import _get, _post
from .workloads import ROOT, omp_bin, omp_env

OLLAMA = "http://127.0.0.1:11434"
PREFIX = "localbench-parked:"
STATE = ROOT / "runs" / "PARKED.json"
RESOLVER = ROOT / "scripts" / "omp-resolve.ts"
HISTORY = ROOT / "runs" / "park-history.jsonl"
THINKING = {"off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"}
SMOL_SERVER = "smol-server"


@contextmanager
def _park_operation_lock():
    """Serialize park/unpark journal and model mutations across localbench processes."""
    lock_path = STATE.with_name(STATE.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("park/unpark is already in progress; retry after it finishes") from exc
        yield
    finally:
        os.close(fd)


def _serialize_park_operation(operation):
    @functools.wraps(operation)
    def locked(*args, **kwargs):
        with _park_operation_lock():
            return operation(*args, **kwargs)
    return locked


def _parked_alias_refusal(plan: list[dict]) -> str | None:
    """Only a journal that already owns an alias may reuse a pre-existing tag."""
    journal_aliases = {entry["parked_as"]: entry["digest"] for entry in parked_now()
                       if entry.get("kind") != SMOL_SERVER}
    tags = _tags()
    for entry in plan:
        if entry.get("kind") == SMOL_SERVER:
            continue
        alias = entry["parked_as"]
        existing_digest = tags.get(alias)
        if existing_digest is None:
            continue
        owned_digest = journal_aliases.get(alias)
        if owned_digest is None:
            return f"pre-existing parked alias {alias} is not owned by the park journal; refusing to park"
        if existing_digest != owned_digest:
            return f"parked alias collision: {alias} has digest {existing_digest}, journal owns {owned_digest}; refusing to park"
    return None


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


OMP_TRUST_CONFIG = Path.home() / ".config" / "omp-trust" / "config"


def omp_package(binary: str) -> Path:
    """The pi-coding-agent package directory behind an `omp` path. ~/.local/bin/omp can be proj-c's omp trust
    guard (a compiled wrapper, installed 2026-09-26, that execs OMP_REAL_BIN from OMP_TRUST_CONFIG): it has no
    package beside it, so its target is followed. Anything else without src/ is refused with the fix, never answered
    with some other install's resolver."""
    def has_src(p: Path) -> bool:
        return (p / "src" / "config" / "model-resolver.ts").is_file()

    pkg = Path(os.path.realpath(binary)).parent.parent
    if has_src(pkg):
        return pkg
    if OMP_TRUST_CONFIG.is_file():
        for line in OMP_TRUST_CONFIG.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "OMP_REAL_BIN" and value.strip():
                real = Path(os.path.realpath(value.strip())).parent.parent
                if has_src(real):
                    return real
    raise RuntimeError(f"omp at {binary} has no package source ({pkg}/src/config/model-resolver.ts) and no trust-guard "
                       f"OMP_REAL_BIN in {OMP_TRUST_CONFIG} leads to one, so omp's model resolver cannot be run; point "
                       "LOCALBENCH_OMP at a package install (e.g. LOCALBENCH_OMP=~/.bun/bin/omp)")


def omp_resolves(selector: str, ids: list[str]) -> str | None:
    """The ollama id omp's own resolver returns for a role `selector` when `ids` are the installed ollama models;
    None when it resolves to nothing. Runs scripts/omp-resolve.ts under bun against the omp package `omp_bin()`
    runs (omp_package), so a new omp release answers for itself. Measured 2026-09-23 (omp 18.2.11): with
    qwen3.8:27b-mlx gone it returns qwen3.8-uncensored:latest; with that gone too, nothing."""
    pkg = omp_package(omp_bin())
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


def _write_state(entries: list[dict]) -> None:
    """Atomically persist a recoverable park journal before relying on it across a crash."""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_name(f".{STATE.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(entries, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, STATE)
        directory_fd = os.open(STATE.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


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


def safety_refusal(plan: list[dict]) -> str | None:
    """Refuse alias conflicts with the current journal or unsafe Ollama activity before park mutates state."""
    if refusal := _parked_alias_refusal(plan):
        return refusal
    aliases: dict[str, str] = {}
    for entry in parked_now():
        if entry.get("kind") == SMOL_SERVER:
            continue
        alias = entry["parked_as"]
        previous_digest = aliases.get(alias)
        if previous_digest is not None and previous_digest != entry["digest"]:
            return f"parked alias collision: {alias} maps to {previous_digest} and {entry['digest']}; refusing to park"
        aliases[alias] = entry["digest"]
    for entry in plan:
        if entry.get("kind") == SMOL_SERVER:
            continue
        alias = entry["parked_as"]
        previous_digest = aliases.get(alias)
        if previous_digest is not None and previous_digest != entry["digest"]:
            return f"parked alias collision: {alias} maps to {previous_digest} and {entry['digest']}; refusing to park"
        aliases[alias] = entry["digest"]
        safe, reason = gateway.safe_to_unload(entry["name"])
        if safe is not True:
            return f"cannot park {entry['name']}: {reason or 'external Ollama client activity is unknown'}"
    return None


@_serialize_park_operation
def park(resolve=None, plan: list[dict] | None = None) -> list[dict]:
    """Fence every planned Ollama model, then checkpoint each reversible park step."""
    current_plan = plan_park(resolve)
    if plan is not None and plan != current_plan:
        raise RuntimeError("park plan changed before mutation; rerun the command to review the current plan")
    plan = current_plan
    if refusal := safety_refusal(plan):
        raise RuntimeError(refusal)
    if not plan:
        return parked_now()

    state_existed = STATE.exists()
    prior = parked_now()
    models = [entry["name"] for entry in plan if entry.get("kind") != SMOL_SERVER]
    fence_id = uuid.uuid4().hex if models else None
    additions = []
    for entry in plan:
        checkpoint = {**entry, "_park_phase": "planned"}
        if fence_id is not None and entry.get("kind") != SMOL_SERVER:
            checkpoint["_park_fence_id"] = fence_id
        additions.append(checkpoint)
    parked = [*prior, *additions]
    _write_state(parked)

    if fence_id is not None:
        acquired, reason = gateway.acquire_park_fence(models, fence_id)
        if not acquired:
            if state_existed:
                _write_state(prior)
            else:
                STATE.unlink(missing_ok=True)
            raise RuntimeError(f"cannot park: {reason or 'gateway admission fence refused'}")

    for index, entry in enumerate(plan):
        state_index = len(prior) + index
        if entry.get("kind") == SMOL_SERVER:
            smol.stop_server(smol.load_state())
            parked[state_index]["_park_phase"] = "parked"
            _write_state(parked)
            continue
        name, dst = entry["name"], entry["parked_as"]
        if dst not in _tags():
            _copy(name, dst)
        if _tags().get(dst) != entry["digest"]:
            raise RuntimeError(f"parked copy {dst} digest {_tags().get(dst)} != {entry['digest']}; refusing to delete {name}")
        parked[state_index]["_park_phase"] = "copied"
        _write_state(parked)
        _post(OLLAMA + "/api/generate", {"model": name, "keep_alive": 0})
        parked[state_index]["_park_phase"] = "unloaded"
        _write_state(parked)
        _delete(name)
        parked[state_index]["_park_phase"] = "parked"
        _write_state(parked)
    # sealed: omp's resolver has nothing left to hand a parked smol role (fallbacks parked above).
    append_history("park", [e["name"] for e in plan], sealed=True)
    return parked


def _prepare_unpark_alias_cleanup(
    parked: list[dict],
) -> tuple[dict[str, set[str]], set[str], str | None]:
    aliases: dict[str, set[str]] = {}
    owned: set[str] = set()
    for entry in parked:
        if entry.get("kind") == SMOL_SERVER:
            continue
        alias = entry["parked_as"]
        aliases.setdefault(alias, set()).add(entry["digest"])
        if entry.get("_park_phase") in {"copied", "unloaded", "parked", "restored"}:
            owned.add(alias)

    tags = _tags()
    to_clean = []
    for alias, expected_digests in aliases.items():
        digest = tags.get(alias)
        if digest is None:
            continue
        if digest not in expected_digests:
            raise RuntimeError(f"parked copy {alias} digest {digest} differs from all "
                               "journaled digests; keeping park state")
        if alias not in owned:
            raise RuntimeError(f"parked alias {alias} is present, but journal does not own it; "
                               "preserving alias and park state")
        to_clean.append(alias)

    recorded_ids = {entry["_unpark_alias_fence_id"] for entry in parked
                    if entry.get("_unpark_alias_fence_id")}
    if len(recorded_ids) > 1:
        raise RuntimeError("park journal has conflicting alias cleanup fence ids; keeping park state")
    fence_id = next(iter(recorded_ids), None)
    created_id = False
    if to_clean and fence_id is None:
        fence_id = uuid.uuid4().hex
        for entry in parked:
            entry["_unpark_alias_fence_id"] = fence_id
        _write_state(parked)
        created_id = True

    if fence_id is not None and to_clean:
        try:
            store = gateway.GatewayStore(gateway.database_path())
            unfenced = []
            for alias in to_clean:
                current_id = store.park_fence(alias)
                if current_id is None:
                    unfenced.append(alias)
                elif current_id != fence_id:
                    raise RuntimeError(f"parked alias {alias} has a different active fence; keeping park state")
                else:
                    safe, reason = gateway.safe_to_unload(alias, park_fence_id=fence_id)
                    if safe is not True:
                        raise RuntimeError(f"cannot remove parked alias {alias}: "
                                           f"{reason or 'external Ollama client activity is unknown'}")
            if unfenced:
                acquired, reason = gateway.acquire_park_fence(unfenced, fence_id)
                if not acquired:
                    raise RuntimeError(f"cannot remove parked alias: "
                                       f"{reason or 'external Ollama client activity is unknown'}")
        except Exception:
            if created_id:
                for entry in parked:
                    entry.pop("_unpark_alias_fence_id", None)
                _write_state(parked)
            raise
    return aliases, set(to_clean), fence_id


@_serialize_park_operation
def unpark() -> list[dict]:
    """Restore journaled tags only after proving and fencing ownership of each parked alias."""
    parked = parked_now()
    if not parked:
        return []
    aliases, fenced_aliases, alias_fence_id = _prepare_unpark_alias_cleanup(parked)
    for index, entry in enumerate(parked):
        if entry.get("kind") == SMOL_SERVER:
            smol.start_server(smol.load_state())
        else:
            name, parked_as, digest = entry["name"], entry["parked_as"], entry["digest"]
            tags = _tags()
            original_digest = tags.get(name)
            parked_digest = tags.get(parked_as)
            if original_digest is not None and original_digest != digest:
                raise RuntimeError(f"restored {name} digest differs from {digest}; keeping {parked_as}")
            if parked_digest is not None and parked_digest != digest and original_digest != digest:
                raise RuntimeError(f"parked copy {parked_as} digest differs from {digest}; keeping park state")
            if original_digest is None:
                if parked_digest is None:
                    raise RuntimeError(f"cannot restore {name}: neither original nor parked copy has digest {digest}")
                _copy(parked_as, name)
                if _tags().get(name) != digest:
                    raise RuntimeError(f"restored {name} digest differs from {digest}; keeping {parked_as}")
            if _tags().get(name) != digest:
                raise RuntimeError(f"restored {name} digest differs from {digest}; keeping {parked_as}")
        parked[index]["_park_phase"] = "restored"
        _write_state(parked)

    # Multiple original tags can share one digest-derived alias; keep it until all restores succeed.
    for parked_as, expected_digests in aliases.items():
        parked_digest = _tags().get(parked_as)
        if parked_digest is None:
            continue
        if parked_as not in fenced_aliases:
            raise RuntimeError(f"parked alias {parked_as} appeared during unpark without a cleanup fence; "
                               "keeping park state")
        if parked_digest not in expected_digests:
            raise RuntimeError(f"parked copy {parked_as} digest {parked_digest} differs from all "
                               "journaled digests; keeping park state")
        safe, reason = gateway.safe_to_unload(parked_as, park_fence_id=alias_fence_id)
        if safe is not True:
            raise RuntimeError(f"cannot remove parked alias {parked_as}: "
                               f"{reason or 'external Ollama client activity is unknown'}")
        # /api/ps stalls while the scheduler loads a model; wait as backends.Ollama.loaded does.
        for loaded in _get(OLLAMA + "/api/ps", timeout=120).get("models", []):
            if loaded["name"] == parked_as:
                _post(OLLAMA + "/api/generate", {"model": parked_as, "keep_alive": 0})
        safe, reason = gateway.safe_to_unload(parked_as, park_fence_id=alias_fence_id)
        if safe is not True:
            raise RuntimeError(f"cannot remove parked alias {parked_as}: "
                               f"{reason or 'external Ollama client activity is unknown'}")
        _delete(parked_as)
        if parked_as in _tags():
            raise RuntimeError(f"parked copy {parked_as} remains after deletion; keeping park state")

    fence_ids = {entry.get("_park_fence_id") for entry in parked if entry.get("_park_fence_id")}
    fence_ids.update(entry.get("_unpark_alias_fence_id") for entry in parked
                     if entry.get("_unpark_alias_fence_id"))
    for fence_id in sorted(fence_ids):
        gateway.release_park_fence(fence_id)
    STATE.unlink(missing_ok=True)
    append_history("unpark", [entry["name"] for entry in parked])
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
