"""Feature presets: named, reversible switches of the omp settings localbench owns (registries/presets.json).

Ownership (bead kit-presets-switcher-r3l, settled with omp-test 2026-10-01; one writer per key): localbench writes
only OWNED_KEYS, and inside `modelRoles` only OWNED_ROLES, plus its own managed System One provider block
(PROVIDER) in a profile's models.yml. omp-kit writes ttsr.*; a preset naming any other key does not load.

A preset is a family:choice name (`judge:nimble`) and a list of ops:
- set          {key, value}           the whole value of an owned key
- role         {role, selector}       one owned entry of modelRoles; the other roles are kept
- list_add     {key, item}            add item to an owned list setting; other items are kept
- list_remove  {key, item}            remove item from it
- provider     {model}                the managed `localbench-sys1` provider in models.yml, api typesafe, baseUrl the
                                      residency gateway (http://127.0.0.1:11300/omp-profile/<profile>); model null
                                      removes the block
A preset with `local: true` sets a route the PROOF CONTRACT governs: it routes work to a local model, or turns a local
model route off at a setting (memory:none: mnemopi.llmMode none, a no-model route, features.NO_MODEL_ROUTE). On the
live target it needs every feature that registries/features.tsv tags with its family (the `preset` column) PROVEN
(features.proof_status); `force=True` is the owner's override and is recorded. The test target needs no proof but only
writes registry `test_profiles`.

apply(): plan, back up every profile's config.yml and models.yml byte for byte under
~/.localbench/rollback/preset-<UTC>-<id>/ (manifest.json beside them), write through `omp config set` with
OMP_PROFILE per profile (unset for default), read every written key back through `omp config get --json`; any
error or mismatch restores every file from the backup and raises. rollback(id) restores those bytes. The applied
record (~/.localbench/presets/applied.json, profile -> family -> expectations) is what drift() compares to a fresh
readback. Every apply and rollback appends a row to ~/.localbench/presets/audit.jsonl.

Running sessions: omp never applies a defaultThinkingLevel change to a running session (agent-session.ts:2316-2319,
#watchSessionSettings: it only seeds new sessions), so a plan names every running omp session on a changed profile
that keeps its old level.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import features, models, park, sysstats
from .omp_profiles import provider_url
from .workloads import ROOT, omp_bin, omp_env

REGISTRY = ROOT / "registries" / "presets.json"
# mnemopi.noEmbeddings: embeddings:fts / embeddings:on, the recall-embeddings route and its declared alternative.
OWNED_KEYS = ("modelRoles", "task.disabledAgents", "defaultThinkingLevel", "mnemopi.llmMode",
              "mnemopi.noEmbeddings")
OWNED_ROLES = ("smol", "judge")
OPS = ("set", "role", "list_add", "list_remove", "provider")
TARGETS = ("test", "live")
PROVIDER = "localbench-sys1"
PROVIDER_API = "typesafe"
BEGIN = "  # >>> localbench preset provider (managed)"
END = "  # <<< localbench preset provider (managed)"
FILES = ("config.yml", "models.yml")
# Settings a running omp session never picks up from config.yml: key -> why.
KEEPS_OLD = {"defaultThinkingLevel": "omp seeds defaultThinkingLevel into new sessions only (agent-session.ts:2316-2319)"}
_NAME = re.compile(r"[a-z0-9]+:[a-z0-9][a-z0-9.-]*")
_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Manifest states rollback() restores from: applied, and an apply that died or whose own restore failed.
ROLLBACKABLE = ("applied", "applying", "restore_failed")


class PresetError(RuntimeError):
    """A preset could not be planned, applied, read back or rolled back."""


class PresetRefused(PresetError):
    """The plan refuses: an unproven local preset on the live target, or a non-test profile on the test target."""


def _check(preset: dict) -> None:
    name = preset.get("name", "")
    if not _NAME.fullmatch(name) or not isinstance(preset.get("local"), bool) or not preset.get("ops"):
        raise ValueError(f"preset {name!r}: needs a family:choice name, a boolean `local` and ops")
    for op in preset["ops"]:
        kind = op.get("op")
        if kind not in OPS:
            raise ValueError(f"preset {name}: unknown op {kind!r}")
        if kind == "role" and (op.get("role") not in OWNED_ROLES or not isinstance(op.get("selector"), str)):
            raise ValueError(f"preset {name}: modelRoles.{op.get('role')} is not localbench's to write")
        if kind in ("set", "list_add", "list_remove") and op.get("key") not in OWNED_KEYS[1:]:
            raise ValueError(f"preset {name}: {op.get('key')} is not localbench's to write")
        if (kind == "set" and "value" not in op) or (kind in ("list_add", "list_remove")
                                                      and not isinstance(op.get("item"), str)):
            raise ValueError(f"preset {name}: {kind} {op.get('key')} needs a value/item")
        if kind == "provider" and not (op.get("model") is None or isinstance(op.get("model"), str)):
            raise ValueError(f"preset {name}: provider model must be a model id or null")


def load(path: Path = REGISTRY) -> dict:
    """The registry, validated: unique names, ops only over localbench-owned keys. Raises ValueError."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [p.get("name") for p in data.get("presets", [])]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate preset names")
    for preset in data["presets"]:
        _check(preset)
    data.setdefault("test_profiles", [])
    return data


def find(name: str, registry: Path = REGISTRY) -> dict:
    for preset in load(registry)["presets"]:
        if preset["name"] == name:
            return preset
    raise PresetError(f"unknown preset {name!r}; `localbench preset list` names them")


def family(name: str) -> str:
    return name.split(":", 1)[0]


# --- omp I/O -------------------------------------------------------------------------------------------------------

def _omp(profile: str, *args: str, run=None) -> str:
    _agent_dir(profile)
    env = omp_env()
    if profile != "default":
        env["OMP_PROFILE"] = profile
    argv = [omp_bin(), "config", *args]
    try:
        p = (run or subprocess.run)(argv, capture_output=True, text=True, timeout=60, env=env,
                                    cwd=str(Path.home()), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PresetError(f"{profile}: `omp config {' '.join(args)}` failed: {exc}") from exc
    if p.returncode != 0:
        raise PresetError(f"{profile}: `omp config {' '.join(args)}` exited {p.returncode}: {p.stderr.strip()[-300:]}")
    return p.stdout


def omp_get(profile: str, key: str, run=None):
    """omp's effective value of `key` for `profile` (`omp config get <key> --json`)."""
    out = _omp(profile, "get", key, "--json", run=run)
    try:
        doc = json.loads(out)
    except ValueError as exc:
        raise PresetError(f"{profile}: `omp config get {key} --json` was not JSON") from exc
    if not isinstance(doc, dict) or "value" not in doc:
        raise PresetError(f"{profile}: `omp config get {key} --json` has no value")
    return doc["value"]


def omp_set(profile: str, key: str, value, run=None) -> None:
    _omp(profile, "set", key, json.dumps(value, separators=(",", ":")), run=run)


def _agent_dir(profile: str) -> Path:
    """The agent dir of `profile`, the one gate every profile name passes before localbench reads or writes it: a
    plain name (_PROFILE, no `..`) whose dir resolves to exactly ~/.omp/agent (default) or ~/.omp/profiles/<name>/agent.
    A `./x` alias, a `../x` escape or an agent dir symlinked out of ~/.omp raises PresetError."""
    if not isinstance(profile, str) or not _PROFILE.fullmatch(profile) or ".." in profile:
        raise PresetError(f"invalid omp profile name {profile!r}")
    omp = (Path.home() / ".omp").resolve()
    allowed = omp / "agent" if profile == "default" else omp / "profiles" / profile / "agent"
    agent = models.agent_config(profile).parent
    if agent.resolve() != allowed:
        raise PresetError(f"{profile}: agent dir {agent} resolves to {agent.resolve()}, outside {allowed}")
    return agent


def _path(profile: str, name: str) -> Path:
    return _agent_dir(profile) / name


# --- managed provider block ----------------------------------------------------------------------------------------

def provider_block(profile: str, model: str) -> str:
    return "\n".join((BEGIN, f"  {PROVIDER}:", f"    baseUrl: {provider_url(profile, sysstats.OLLAMA_GATEWAY_PORT)}",
                      f"    api: {PROVIDER_API}", "    apiKey: localbench-gateway-no-key", "    models:",
                      f"      - id: {model}", END, ""))


def _managed(text: str) -> str | None:
    start = text.find(BEGIN + "\n")
    end = text.find(END + "\n", start + 1)
    if start < 0 or end < 0 or text.count(BEGIN + "\n") != 1:
        if BEGIN in text or END in text:
            raise PresetError("the localbench preset provider block is damaged; restore models.yml from a rollback")
        return None
    return text[start:end + len(END) + 1]


def _models_text(text: str, block: str | None) -> str:
    """models.yml `text` with the managed provider block replaced by `block` (None removes it)."""
    existing = _managed(text)
    rest = text.replace(existing, "", 1) if existing else text
    if re.search(rf"^\s+['\"]?{re.escape(PROVIDER)}['\"]?:", rest, re.MULTILINE):
        raise PresetError(f"models.yml declares `{PROVIDER}` outside localbench's managed block; refusing")
    if block is None:
        return rest
    if re.search(r"^providers:[ \t]+[^\s#]", rest, re.MULTILINE):
        raise PresetError("models.yml has an inline providers mapping that cannot be edited safely")
    m = re.search(r"^providers:[ \t]*(?:#[^\n]*)?$", rest, re.MULTILINE)
    if m is None:
        return rest + ("" if not rest or rest.endswith("\n") else "\n") + "providers:\n" + block
    head = rest[:m.end()] + ("\n" if m.end() == len(rest) else rest[m.end()])
    return head + block + rest[m.end() + 1:]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


# --- running sessions ----------------------------------------------------------------------------------------------

def running_sessions() -> list[dict]:
    """Running interactive omp sessions on this host: pid, profile, command. localbench's isolated children
    (PI_CODING_AGENT_DIR pinned) and `omp config` calls are not sessions."""
    out = []
    for proc in sysstats.omp_processes():
        if re.search(r"\bomp\s+config\b", proc["cmd"]):
            continue
        p = subprocess.run(["ps", "eww", "-o", "command=", "-p", str(proc["pid"])], capture_output=True, text=True,
                           timeout=10, check=False)
        ident = sysstats.omp_client_identity(proc["cmd"], p.stdout)
        if "agent_dir" in ident or not ident.get("omp_profile"):
            continue
        out.append({"pid": proc["pid"], "profile": ident["omp_profile"], "cmd": proc["cmd"]})
    return out


# --- plan ----------------------------------------------------------------------------------------------------------

def _route_model(row: dict, roles: dict, provs: dict[str, dict[str, str]]) -> str | None:
    """The model feature `row`'s role chain reaches first on a local provider under `roles`/`provs` (features.route's
    chain rules, gating settings ignored); None when it reaches no local model."""
    r = features.route({**row, "route_kind": "model_role"}, {"modelRoles": roles}, provs)
    if not r["local"]:
        return None
    return features.route_model(next(t for t in r["target"].split(" > ") if features.is_local(t, provs)))


def proof_statuses(fam: str, routes: dict[str, tuple[dict, dict]], *, registry: Path = features.REGISTRY,
                   receipts_dir: Path = features.RECEIPTS, package: Path | None = None,
                   digests: dict[str, str] | None = None,
                   settings: dict[str, dict] | None = None) -> dict[str, dict[str, dict]]:
    """feature -> profile -> {model, digest, setting, proof, reason} for every features.tsv row whose preset family is
    `fam`, graded by features.proof_status (its PROOF CONTRACT) against the model each profile's role chain reaches once
    the preset is applied (`routes`: profile -> (modelRoles, providers) after apply), with the profile's current route
    (features.incumbent of `omp config list` now, before the flip) as the incumbent the receipt's baseline must name.
    A chain that reaches no local model is graded without the model check. When the preset's own set ops (`settings`:
    profile -> {key: value after apply}) turn a setting row's route off (e.g. mnemopi.llmMode none), the route targets
    no model: it is graded as off at that gating value (`setting`), model None. `digests` (installed model -> digest)
    defaults to features.ollama_digests()."""
    rows = [r for r in features.load(registry) if r["preset"] == fam]
    pkg = package or park.omp_package(omp_bin())
    found = features.receipts(receipts_dir)
    digests = features.ollama_digests() if digests is None else digests
    out: dict[str, dict[str, dict]] = {}
    current = {prof: (features.omp_settings(prof), features.providers(prof)) for prof in routes}
    for row in rows:
        sha = features.module_sha(features.package_root(row["omp_package"], pkg) / row["omp_module"])
        gating = ({name for alt in features._alternatives(row["route_key"]) for name, _, _ in alt}
                  if row["route_kind"] == "setting" else set())
        for prof, (roles, provs) in routes.items():
            after = (settings or {}).get(prof) or {}
            cfg_after = {**current[prof][0], **after, "modelRoles": roles}
            setting = None
            if set(after) & gating and features.route(row, cfg_after, provs)["disabled"]:
                setting = features.incumbent(row, cfg_after, provs, "after apply")["setting"]
            model = None if setting is not None else _route_model(row, roles, provs)
            digest = None if model is None else features.installed_digest(row, model, digests)
            graded = features.proof_status(found.get(row["feature"], []), sha, model, digest,
                                           features.incumbent(row, *current[prof], "current route"), setting)
            out.setdefault(row["feature"], {})[prof] = {"model": model, "digest": digest, "setting": setting,
                                                        "proof": graded["proof"], "reason": graded["reason"]}
    return out


def _fold(op: dict, value):
    kind = op["op"]
    if kind == "set":
        return op["value"]
    if kind == "role":
        return {**(value or {}), op["role"]: op["selector"]}
    items = list(value or [])
    if kind == "list_add":
        return items if op["item"] in items else [*items, op["item"]]
    return [i for i in items if i != op["item"]]


def _expect(op: dict) -> dict:
    kind = op["op"]
    if kind == "set":
        return {"key": op["key"], "equals": op["value"]}
    if kind == "role":
        return {"key": "modelRoles", "role": op["role"], "equals": op["selector"]}
    return {"key": op["key"], ("contains" if kind == "list_add" else "excludes"): op["item"]}


def _holds(expect: dict, value) -> bool:
    if "role" in expect:
        return isinstance(value, dict) and value.get(expect["role"]) == expect["equals"]
    if "contains" in expect:
        return isinstance(value, list) and expect["contains"] in value
    if "excludes" in expect:
        return isinstance(value, list) and expect["excludes"] not in value
    return value == expect["equals"]


def plan(name: str, profiles: list[str], target: str, *, force: bool = False, registry: Path = REGISTRY,
         features_registry: Path = features.REGISTRY, receipts_dir: Path = features.RECEIPTS,
         package: Path | None = None, sessions: list[dict] | None = None, digests: dict[str, str] | None = None,
         run=None) -> dict:
    """What applying preset `name` to `profiles` on `target` would do, read from omp now: one step per profile and
    key (`before`, `after`, `changed`, `expect`), the proof gate (`proof`: feature -> profile -> grade, `refused`),
    running sessions that keep an old value (`keeps_old`) and `notes`. Writes nothing."""
    if target not in TARGETS:
        raise PresetError(f"target must be one of {TARGETS}")
    if not profiles or len(set(profiles)) != len(profiles):
        raise PresetError("name each profile once")
    reg = load(registry)
    preset = find(name, registry)
    missing = [p for p in profiles if not _path(p, "config.yml").is_file()]
    if missing:
        raise PresetError(f"no omp profile config.yml for {missing}")
    out = {"preset": name, "target": target, "profiles": list(profiles), "force": force, "forced": None,
           "proof": {}, "refused": None, "steps": [], "keeps_old": [], "notes": []}
    if target == "test":
        outside = [p for p in profiles if p not in reg["test_profiles"]]
        if outside:
            out["refused"] = f"test target writes only test profiles {reg['test_profiles']}; not {outside}"
    for profile in profiles:
        keys: dict[str, list[dict]] = {}
        for op in preset["ops"]:
            if op["op"] == "provider":
                path = _path(profile, "models.yml")
                current = _read_text(path)
                block = None if op["model"] is None else provider_block(profile, op["model"])
                new = _models_text(current, block)
                out["steps"].append({"profile": profile, "kind": "provider", "key": f"providers.{PROVIDER}",
                                     "file": str(path), "before": _managed(current), "after": block,
                                     "changed": new != current, "expect": [{"provider": PROVIDER, "block": block}]})
            else:
                keys.setdefault(op.get("key", "modelRoles"), []).append(op)
        for key, ops in keys.items():
            before = omp_get(profile, key, run=run)
            after = before
            for op in ops:
                after = _fold(op, after)
            out["steps"].append({"profile": profile, "kind": "set", "key": key, "file": str(_path(profile, "config.yml")),
                                 "before": before, "after": after, "changed": after != before,
                                 "expect": [_expect(op) for op in ops]})
    if target == "live" and preset["local"]:
        routes = {}
        for prof in profiles:
            mine = {s["key"]: s for s in out["steps"] if s["profile"] == prof}
            roles = mine["modelRoles"]["after"] if "modelRoles" in mine else omp_get(prof, "modelRoles", run=run)
            provs = features.providers(prof)
            if f"providers.{PROVIDER}" in mine:
                provs.pop(PROVIDER, None)
                if mine[f"providers.{PROVIDER}"]["after"] is not None:
                    provs[PROVIDER] = {"baseUrl": provider_url(prof, sysstats.OLLAMA_GATEWAY_PORT), "api": PROVIDER_API}
            routes[prof] = (roles or {}, provs)
        after = {prof: {s["key"]: s["after"] for s in out["steps"]
                        if s["profile"] == prof and s["kind"] == "set" and s["key"] != "modelRoles"}
                 for prof in profiles}
        out["proof"] = proof_statuses(family(name), routes, registry=features_registry, receipts_dir=receipts_dir,
                                      package=package, digests=digests, settings=after)
        if not out["proof"]:
            out["refused"] = f"no feature in features.tsv carries preset family {family(name)!r}; nothing proves it"
        unproven = {f"{f}@{prof}": f"{q['proof']}: {q['reason']}" for f, per in out["proof"].items()
                    for prof, q in per.items() if q["proof"] != "PROVEN"}
        if unproven:
            out["refused"] = f"live {name} routes to a local model without proof: {unproven}"
        if out["refused"] and force:
            out["forced"], out["refused"] = out["refused"], None
            out["notes"].append(f"FORCED past: {out['forced']}")
    changed = {(s["profile"], s["key"]): s for s in out["steps"] if s["changed"]}
    live = running_sessions() if sessions is None else sessions
    for session in live:
        for key, why in KEEPS_OLD.items():
            step = changed.get((session["profile"], key))
            if step:
                out["keeps_old"].append({"pid": session["pid"], "profile": session["profile"], "key": key,
                                         "old": step["before"], "new": step["after"], "why": why})
    for k in out["keeps_old"]:
        out["notes"].append(f"running omp pid {k['pid']} (profile {k['profile']}) keeps {k['key']}={k['old']!r}, "
                            f"not {k['new']!r}: {k['why']}; restart it to pick up the change")
    return out


def lines(p: dict) -> list[str]:
    """Text form of a plan()."""
    out = [f"preset {p['preset']} -> {p['target']} {','.join(p['profiles'])}"
           + (f"  REFUSED: {p['refused']}" if p["refused"] else "")]
    out += [f"  proof {f} @{prof}: {q['proof']} {q['model'] or '-'}" + (f" ({q['reason']})" if q["reason"] else "")
            for f, per in p["proof"].items() for prof, q in per.items()]
    for s in p["steps"]:
        mark = "set " if s["changed"] else "same"
        out.append(f"  {mark} {s['profile']}: {s['key']} {json.dumps(s['before'])} -> {json.dumps(s['after'])}")
    out += [f"  note: {n}" for n in p["notes"]]
    return out


# --- state, backup, restore ----------------------------------------------------------------------------------------

def _home() -> Path:
    return Path.home() / ".localbench"


def rollback_root() -> Path:
    return _home() / "rollback"


def _state_dir() -> Path:
    return _home() / "presets"


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_json(path: Path, data) -> None:
    _atomic_write(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def applied() -> dict:
    """profile -> family -> {preset, id, at, expect} of every applied preset."""
    return _read_json(_state_dir() / "applied.json", {})


def _audit(row: dict) -> None:
    path = _state_dir() / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _utc(), **row}, sort_keys=True) + "\n")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _backup(profiles: list[str], backup_dir: Path) -> dict:
    files: dict[str, dict] = {}
    for profile in profiles:
        for name in FILES:
            src = _path(profile, name)
            if src.is_symlink():
                raise PresetError(f"{profile}: {name} is a symlink; refusing to replace it")
            record = {"path": str(src), "existed": src.is_file(), "sha256": _sha(src), "backup": None, "mode": 0o600}
            if record["existed"]:
                record["mode"] = src.stat().st_mode & 0o777
                record["backup"] = str(backup_dir / f"{profile}.{name}")
                _atomic_write(Path(record["backup"]), src.read_bytes(), 0o600)
            files[f"{profile}/{name}"] = record
    return files


def _restore(files: dict) -> list[str]:
    """Put every recorded file back to its backed-up bytes (or absent); returns what changed. Raises when a
    restored file does not hash to the recorded original."""
    restored = []
    for label, record in files.items():
        profile, _, fname = label.partition("/")
        if fname not in FILES or Path(record["path"]) != _path(profile, fname):
            raise PresetError(f"manifest entry {label} points at {record['path']}, not that profile's {fname}")
    for label, record in files.items():
        path = Path(record["path"])
        if _sha(path) == record["sha256"]:
            continue
        if record["existed"]:
            data = Path(record["backup"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise PresetError(f"backup of {label} does not match its recorded sha256")
            _atomic_write(path, data, record["mode"])
        else:
            path.unlink(missing_ok=True)
        if _sha(path) != record["sha256"]:
            raise PresetError(f"{label} did not restore byte-identically")
        restored.append(label)
    return restored


def _readback(steps: list[dict], run=None) -> list[str]:
    """Mismatches between what each step wrote and what omp (or models.yml) reads back now."""
    bad = []
    for s in steps:
        if s["kind"] == "provider":
            actual = _managed(_read_text(Path(s["file"])))
        else:
            actual = omp_get(s["profile"], s["key"], run=run)
        if actual != s["after"]:
            bad.append(f"{s['profile']}: {s['key']} reads back {json.dumps(actual)}, wrote {json.dumps(s['after'])}")
    return bad


# --- apply / rollback / drift --------------------------------------------------------------------------------------

def apply(name: str, profiles: list[str], target: str, *, force: bool = False, run=None, **kw) -> dict:
    """Apply preset `name`: plan (refusal raises PresetRefused), back up, write, read back. Any failure or
    readback mismatch restores every backed-up file and raises PresetError. Returns the manifest (its `id` is what
    rollback() takes)."""
    p = plan(name, profiles, target, force=force, run=run, **kw)
    if p["refused"]:
        raise PresetRefused(p["refused"])
    rid = f"{_utc()}-{secrets.token_hex(3)}"
    backup_dir = rollback_root() / f"preset-{rid}"
    backup_dir.mkdir(parents=True, mode=0o700)
    state = applied()
    fam = family(name)
    manifest = {"id": rid, "preset": name, "target": target, "forced": p["forced"], "status": "applying", "plan": p,
                "files": _backup(profiles, backup_dir),
                "previous_applied": {prof: state.get(prof, {}).get(fam) for prof in profiles}}
    _write_json(backup_dir / "manifest.json", manifest)
    try:
        for s in p["steps"]:
            if not s["changed"]:
                continue
            if s["kind"] == "provider":
                path = Path(s["file"])
                if _managed(_read_text(path)) != s["before"]:
                    raise PresetError(f"{s['profile']}: models.yml changed since the plan")
                mode = path.stat().st_mode & 0o777 if path.is_file() else 0o600
                _atomic_write(path, _models_text(_read_text(path), s["after"]).encode("utf-8"), mode)
            else:
                omp_set(s["profile"], s["key"], s["after"], run=run)
        mismatches = _readback(p["steps"], run=run)
        if mismatches:
            raise PresetError("readback mismatch: " + "; ".join(mismatches))
    except BaseException as exc:
        try:
            _restore(manifest["files"])
        except BaseException as failed:
            manifest["status"] = "restore_failed"
            manifest["error"] = f"{exc}; restore failed: {failed}"
            _write_json(backup_dir / "manifest.json", manifest)
            _audit({"action": "apply", "id": rid, "preset": name, "target": target, "profiles": profiles,
                    "result": "restore_failed", "error": manifest["error"]})
            raise PresetError(f"{name}: {manifest['error']}; profile files may be half-written: run "
                              f"`localbench preset rollback {rid}` to restore them from {backup_dir}") from failed
        manifest["status"] = "rolled-back"
        manifest["error"] = str(exc)
        _write_json(backup_dir / "manifest.json", manifest)
        _audit({"action": "apply", "id": rid, "preset": name, "target": target, "profiles": profiles,
                "result": "rolled-back", "error": str(exc)})
        if isinstance(exc, PresetError):
            raise PresetError(f"{name}: {exc}; every profile file restored from {backup_dir}") from exc
        raise
    manifest["status"] = "applied"
    manifest["applied_sha"] = {label: _sha(Path(r["path"])) for label, r in manifest["files"].items()}
    _write_json(backup_dir / "manifest.json", manifest)
    for prof in profiles:
        expect = [e for s in p["steps"] if s["profile"] == prof for e in s["expect"]]
        state.setdefault(prof, {})[fam] = {"preset": name, "id": rid, "at": _utc(), "expect": expect}
    _write_json(_state_dir() / "applied.json", state)
    _audit({"action": "apply", "id": rid, "preset": name, "target": target, "profiles": profiles,
            "result": "applied", "forced": manifest["forced"]})
    return manifest


def rollback(rid: str, *, force: bool = False) -> dict:
    """Restore every file apply `rid` backed up, byte for byte (sha256-verified). An applied preset: refuses (unless
    force) when a file changed after the apply, since restoring would silently undo that later change. An apply that
    died mid-write (`applying`) or whose own restore failed (`restore_failed`) is restored without that check."""
    if not re.fullmatch(r"[0-9TZ]+-[0-9a-f]+", rid):
        raise PresetError(f"invalid rollback id {rid!r}")
    backup_dir = rollback_root() / f"preset-{rid}"
    manifest = _read_json(backup_dir / "manifest.json", None)
    if manifest is None:
        raise PresetError(f"no preset rollback {rid} under {rollback_root()}")
    if manifest["status"] not in ROLLBACKABLE:
        raise PresetError(f"rollback {rid} is {manifest['status']}; only {ROLLBACKABLE} can be rolled back")
    later = [label for label, r in manifest["files"].items()
             if manifest["status"] == "applied" and _sha(Path(r["path"])) != manifest["applied_sha"][label]]
    if later and not force:
        raise PresetError(f"changed since apply {rid}: {later}; rollback would undo those changes (force to proceed)")
    restored = _restore(manifest["files"])
    state = applied()
    fam = family(manifest["preset"])
    for prof, previous in manifest["previous_applied"].items():
        if state.get(prof, {}).get(fam, {}).get("id") != rid:
            continue
        if previous is None:
            state[prof].pop(fam)
        else:
            state[prof][fam] = previous
    _write_json(_state_dir() / "applied.json", state)
    manifest["status"] = "rolled-back"
    _write_json(backup_dir / "manifest.json", manifest)
    _audit({"action": "rollback", "id": rid, "preset": manifest["preset"], "restored": restored, "forced": force})
    return {"id": rid, "preset": manifest["preset"], "restored": restored}


def drift(profiles: list[str] | None = None, run=None) -> list[dict]:
    """Per applied preset on `profiles` (default: every profile with one), the expectations live config no longer
    meets: {profile, preset, id, key, expected, actual}. Empty when live config still matches."""
    out = []
    for prof, fams in sorted(applied().items()):
        if profiles is not None and prof not in profiles:
            continue
        values: dict[str, object] = {}
        for entry in fams.values():
            for e in entry["expect"]:
                if "provider" in e:
                    actual = _managed(_read_text(_path(prof, "models.yml")))
                    ok, key, expected = actual == e["block"], f"providers.{e['provider']}", e["block"]
                else:
                    if e["key"] not in values:
                        values[e["key"]] = omp_get(prof, e["key"], run=run)
                    actual = values[e["key"]]
                    ok = _holds(e, actual)
                    key = e["key"] + (f".{e['role']}" if "role" in e else "")
                    expected = {k: v for k, v in e.items() if k not in ("key", "role")}
                    if "role" in e:
                        actual = actual.get(e["role"]) if isinstance(actual, dict) else actual
                if not ok:
                    out.append({"profile": prof, "preset": entry["preset"], "id": entry["id"], "key": key,
                                "expected": expected, "actual": actual})
    return out
