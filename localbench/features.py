"""The omp feature map: every omp feature that can send work to a local model (registries/features.tsv), tied to its
call site in the installed omp package (sha256 of the module), the route each omp profile gives it right now, and the
proof receipt that covers it.

Routes come from omp's own resolved settings per profile (`omp config list --json`, defaults included, the same call
park.local_routes makes): a feature's role chain (`judge>tiny>smol`) resolves to the first role the profile sets, an
`agent` row is DISABLED while the profile's `task.disabledAgents` lists it, and a `setting` row only while its gating
settings hold (see _alternatives). A target is local when its provider is built-in local (park.LOCAL_PROVIDERS) or the
profile's models.yml points the provider at this host (`ollama-sys1`, api typesafe, at 127.0.0.1:11434, 2026-09-30).

Proof is read from receipts under docs/evidence/receipts/*.json, indexed by their top-level `feature`
(FEATURE_FIELD). The PROOF CONTRACT (grade()) a receipt must meet to prove a feature on one profile's route:
  kind == "run"; run.label in PROOF_LABELS (decision, memory, generation); feature == the row's feature;
  omp_module_sha (SHA_FIELD) == sha256 of the feature's omp module as installed now; problems empty;
  verdict.compare == "BETTER" with verdict.baseline {kind, id} naming this profile's incumbent route: the route it had
  before the local preset of the feature's family was applied (that apply's recorded `before` settings), or, with no
  local preset applied, its current route (a proof graded before the flip); hosted -> the hosted model id,
  fixed -> the gating setting's value, route -> the selector (baseline_matches); never the routed model itself;
  run.provenance.pins.model_digest == the installed digest of the model that profile's live route targets; or, for a
  route that targets no model (a setting row off at a gating value, e.g. mnemopi-extraction under mnemopi.llmMode
  none), pins.route (NO_MODEL_ROUTE) {kind fixed, id <that value>} matching the profile's route now or after the preset
  being applied, with pins.model_calls == 0.
  An already-local route with no local preset apply record (its current route is the incumbent) may instead name, as
  verdict.baseline, a declared alternative (alternatives(): a preset of its family routing no local model, e.g.
  memory:none); a local model route is never such an alternative.
A proof therefore applies only to profiles whose route targets the model (or the no-model setting) it measured. Per
profile the best receipt wins, in STATUSES order, each with its reason:
- PROVEN    every check holds.
- CARRIED   every check holds and the receipt has a truthy `carried_forward` (CARRIED_FIELD): the omp-update refresh
            carried an older proof forward after its mock-backed re-record of the new module matched, writing e.g.
            {"from_sha": "<omp_module_sha the proof measured>", "at": "<ISO time>", "omp_version": "18.4.5"}. A
            carried proof stands until the queued re-proof lands.
- BAD       the sha matches but the receipt lists `problems`, or its verdict.compare is not BETTER.
- STALE     a proof receipt names the feature, but its sha is not the installed module's (or the module is gone).
- UNPROVEN  no receipt names the feature, or the best one is not a decision/memory/generation run receipt, has no
            verdict.compare or baseline, or pins no model digest / another model's.
A feature's `proof` is its worst local profile's; its `status` is DISABLED when every listed profile has its route off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from . import models, omp_profiles, park, render, sysstats
from .workloads import ROOT, fastembed_digest, omp_bin, omp_env

REGISTRY = ROOT / "registries" / "features.tsv"
RECEIPTS = ROOT / "docs" / "evidence" / "receipts"
COLUMNS = ("consumer", "feature", "omp_package", "omp_module", "omp_symbol", "role", "route_kind", "route_key",
           "proof_suite", "preset", "notes")
ROUTE_KINDS = ("model_role", "agent", "setting")
EMBEDDING_ROLE = "fastembed"
FEATURE_FIELD = "feature"
SHA_FIELD = "omp_module_sha"
CARRIED_FIELD = "carried_forward"
# run.provenance.pins key of a receipt proving a route that targets no model: {kind fixed, id <gating-setting value>}
# in verdict.baseline's vocabulary (baseline_matches `fixed`), pinned by pins.model_calls == 0.
NO_MODEL_ROUTE = "route"
# incumbent_settings' source when no local preset apply is recorded: the incumbent is the profile's current route.
CURRENT_ROUTE = "current route"
STATUSES = ("PROVEN", "CARRIED", "BAD", "STALE", "UNPROVEN")
FAILING = ("UNPROVEN", "STALE", "BAD")
PROOF_LABELS = ("decision", "memory", "generation")
OLLAMA_PORT = urllib.parse.urlsplit(park.OLLAMA).port
GATEWAY_PORT = sysstats.OLLAMA_GATEWAY_PORT
NATIVE_JUDGE = "native-judge"
# omp's System One judgment APIs (pi-ai judgment/typesafe.ts JUDGMENT_ROUTES); the built-in `typesafe` provider uses its
# own name as its API.
NATIVE_APIS = ("typesafe", "openrouter-decisions")
LOOPBACK = ("127.0.0.1", "localhost", "::1", "0.0.0.0")

_SLUG = re.compile(r"[a-z0-9][a-z0-9.-]*")
_CHAIN = re.compile(r"[a-z]+(?:>[a-z]+)*")
_CONDITION = re.compile(r"([A-Za-z][\w.]*)(!=|=)([^&|=!]+)")


def _alternatives(route_key: str) -> list[list[tuple[str, str, str]]]:
    """A setting row's route_key: alternatives joined by `|`, each a set of conditions joined by `&`; a condition is
    `key=value`, `key!=value`, or `native-judge` (the row's role chain resolves first to a native System One model,
    omp's judgment/index.ts hasNativeJudge). `find.enabled=on|find.enabled=auto&native-judge` -> two alternatives.
    ValueError on any other part."""
    out = []
    for alternative in route_key.split("|"):
        conditions = []
        for part in (p.strip() for p in alternative.split("&")):
            if part == NATIVE_JUDGE:
                conditions.append((NATIVE_JUDGE, "", ""))
                continue
            m = _CONDITION.fullmatch(part)
            if not m:
                raise ValueError(f"setting condition {part!r} is not key=value, key!=value or {NATIVE_JUDGE}")
            conditions.append((m.group(1), m.group(2), m.group(3).strip()))
        out.append(conditions)
    return out


def _check(row: dict) -> None:
    if not _SLUG.fullmatch(row["feature"]):
        raise ValueError(f"feature {row['feature']!r} is not a lowercase id")
    if not _SLUG.fullmatch(row["omp_package"]):
        raise ValueError(f"omp_package {row['omp_package']!r} is not a package dir name")
    module = PurePosixPath(row["omp_module"])
    if module.is_absolute() or ".." in module.parts:
        raise ValueError(f"omp_module {row['omp_module']!r} must be a path inside the package root")
    if row["role"] != EMBEDDING_ROLE and not _CHAIN.fullmatch(row["role"]):
        raise ValueError(f"role {row['role']!r} is not a role chain like judge>tiny>smol (or {EMBEDDING_ROLE})")
    kind, key = row["route_kind"], row["route_key"]
    if kind not in ROUTE_KINDS:
        raise ValueError(f"route_kind {kind!r} is not one of {', '.join(ROUTE_KINDS)}")
    if kind == "model_role" and key != row["role"].split(">")[0]:
        raise ValueError(f"model_role route_key {key!r} must be the first role of {row['role']!r}")
    if kind == "agent" and not _SLUG.fullmatch(key):
        raise ValueError(f"agent route_key {key!r} is not an agent name")
    if kind == "setting":
        _alternatives(key)


def load(path: Path = REGISTRY) -> list[dict]:
    """The registry rows, validated: the header is COLUMNS, every row has exactly that many non-empty tab-separated
    fields, features are unique, and each field has its column's shape. Any violation raises ValueError naming the
    line, so a malformed row never becomes a silently missing feature."""
    rows: list[dict] = []
    header = False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = [c.strip() for c in line.split("\t")]
        if not header:
            if tuple(cells) != COLUMNS:
                raise ValueError(f"{path}:{n}: header must be the tab-separated columns {', '.join(COLUMNS)}")
            header = True
            continue
        if len(cells) != len(COLUMNS):
            raise ValueError(f"{path}:{n}: {len(cells)} tab-separated fields, need {len(COLUMNS)}")
        row = dict(zip(COLUMNS, cells, strict=True))
        if empty := [k for k, v in row.items() if not v]:
            raise ValueError(f"{path}:{n}: empty {', '.join(empty)} (write - for none)")
        try:
            _check(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from None
        if any(r["feature"] == row["feature"] for r in rows):
            raise ValueError(f"{path}:{n}: feature {row['feature']} is listed twice")
        rows.append(row)
    if not header:
        raise ValueError(f"{path}: no header row")
    return rows


def package_root(name: str, coding_agent: Path) -> Path:
    """The installed @oh-my-pi/<name> package that the omp package `coding_agent` (park.omp_package) loads: itself,
    a sibling in the same scope dir (bun's global install), or one nested under its own node_modules."""
    if name == coding_agent.name:
        return coding_agent
    nested = coding_agent / "node_modules" / coding_agent.parent.name / name
    return nested if nested.is_dir() and not (coding_agent.parent / name).is_dir() else coding_agent.parent / name


def module_sha(path: Path) -> str | None:
    """sha256 hex of a module file; None when it is not there."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def omp_settings(profile: str) -> dict:
    """omp's resolved settings for `profile` (`omp config list --json`, defaults included), key -> value. Raises when
    omp does not answer: routes are never guessed from a missing config."""
    args = [omp_bin(), *([] if profile == "default" else ["--profile", profile]), "config", "list", "--json"]
    p = subprocess.run(args, capture_output=True, text=True, timeout=60, env=omp_env(), check=False)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"`{' '.join(args[1:])}` failed (rc {p.returncode}): {p.stderr.strip()[-300:]}")
    return {k: (v or {}).get("value") for k, v in json.loads(p.stdout).items()}


def _setting_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def _expand(roles: dict, value, depth: int = 0) -> list[str]:
    """A role value (a selector, a chain list, or `@role` aliases) as the selectors it stands for."""
    out: list[str] = []
    for item in value if isinstance(value, list) else [value]:
        if not isinstance(item, str) or not item:
            continue
        if item.startswith("@"):
            if depth < 4:
                out += _expand(roles, roles.get(item[1:]), depth + 1)
        else:
            out.append(item)
    return out


def providers(profile: str) -> dict[str, dict[str, str]]:
    """provider -> {baseUrl, api} as the profile's models.yml declares them (top-level `providers:`, two-space
    provider keys, four-space fields): what makes a custom provider such as `ollama-sys1` (api typesafe, baseUrl
    127.0.0.1) local and native."""
    path = models.agent_config(profile).parent / "models.yml"
    out: dict[str, dict[str, str]] = {}
    inside, current = False, None
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            inside, current = re.fullmatch(r"providers:\s*(#.*)?", line) is not None, None
        elif inside and indent == 2 and (m := re.fullmatch(r"\s+['\"]?([\w.-]+)['\"]?:\s*(#.*)?", line)):
            current = out.setdefault(m.group(1), {})
        elif inside and indent == 4 and current is not None and (
                m := re.fullmatch(r"\s+(baseUrl|api):\s*['\"]?([^'\"#\s]+)['\"]?\s*(#.*)?", line)):
            current[m.group(1)] = m.group(2)
    return out


def is_local(selector: str, provs: dict[str, dict[str, str]]) -> bool:
    """A built-in local provider (park.LOCAL_PROVIDERS), or a declared provider whose baseUrl is on this host."""
    if selector.startswith(park.LOCAL_PROVIDERS):
        return True
    base = provs.get(selector.split("/", 1)[0], {}).get("baseUrl", "")
    return urllib.parse.urlsplit(base).hostname in LOOPBACK


def is_native(selector: str, provs: dict[str, dict[str, str]]) -> bool:
    """A System One judgment model: its provider's API is one of NATIVE_APIS (omp judgment kindOf == native)."""
    provider = selector.split("/", 1)[0]
    return provs.get(provider, {}).get("api", provider) in NATIVE_APIS


def route(row: dict, cfg: dict, provs: dict[str, dict[str, str]] | None = None) -> dict:
    """The live route a profile with resolved settings `cfg` and declared providers `provs` gives the feature `row`:
    {target, local, disabled}. `disabled` is the reason when the route is off (target None); `target` is None too when
    the profile sets no role of the chain (omp then falls back to its own built-in choice). A chain-valued role is
    local when any selector omp can fall through to is local; for a judge chain, as in omp's judgeRoleChain, no
    prompted model is reached after the first native one."""
    provs = provs or {}
    if row["role"] == EMBEDDING_ROLE:
        targets = [f"local/{park.fastembed_name(cfg)}"]
    else:
        roles = cfg.get("modelRoles") or {}
        chain = row["role"].split(">")
        targets = next((t for r in chain if (t := _expand(roles, roles.get(r)))), [])
        native = [is_native(t, provs) for t in targets]
        if chain[0] == "judge" and any(native):
            first = native.index(True)
            targets = [t for i, t in enumerate(targets) if i < first or native[i]]
    kind, key = row["route_kind"], row["route_key"]
    off = None
    if kind == "agent" and key in (cfg.get("task.disabledAgents") or []):
        off = f"task.disabledAgents lists {key}"
    elif kind == "setting":
        unmet: list[str] = []
        for conditions in _alternatives(key):
            failed = None
            for name, op, want in conditions:
                if name == NATIVE_JUDGE:
                    if not (targets and is_native(targets[0], provs)):
                        failed = f"judge {targets[0] if targets else 'unset'} is not native System One"
                else:
                    have = _setting_text(cfg.get(name)) or "unset"
                    if (have == want) != (op == "="):
                        failed = f"{name}={have} (route needs {name}{op}{want})"
                if failed:
                    break
            if failed is None:
                unmet = []
                break
            unmet.append(failed)
        off = "; ".join(unmet) or None
    if off:
        return {"target": None, "local": False, "disabled": off}
    return {"target": " > ".join(targets) or None, "local": any(is_local(t, provs) for t in targets),
            "disabled": None}


def receipts(directory: Path = RECEIPTS) -> dict[str, list[tuple[str, dict]]]:
    """feature -> [(receipt file name, receipt)] for every *.json receipt whose top-level `feature` names one."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for f in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get(FEATURE_FIELD), str):
            out.setdefault(data[FEATURE_FIELD], []).append((f.name, data))
    return out


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def same_digest(a, b) -> bool:
    """Two model digests name the same build: `sha256:` dropped, case folded, and the shorter (at least 12 hex, the
    length run pins record) a prefix of the longer."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    a, b = (d.lower().removeprefix("sha256:") for d in (a, b))
    short, long = sorted((a, b), key=len)
    return len(short) >= 12 and long.startswith(short)


def _bare(selector: str) -> str:
    """A selector without its thinking suffix: `ollama/qwen3.8:27b-mlx:high` -> ollama/qwen3.8:27b-mlx."""
    base, _, last = selector.rpartition(":")
    return base if base and last in park.THINKING and ":" in base else selector


def incumbent(row: dict, cfg: dict, provs: dict[str, dict[str, str]], source: str) -> dict:
    """The route a profile with settings `cfg`/`provs` gives feature `row`, as a decision baseline can name it:
    {selector (first selector of the live route, None when off or unset), local, setting (when a setting row is off:
    the value of its first unmet setting, e.g. defaultThinkingLevel `high`), source (where `cfg` came from)}."""
    r = route(row, cfg, provs)
    setting = None
    if r["disabled"] and row["route_kind"] == "setting":
        for name, op, want in _alternatives(row["route_key"])[0]:
            have = _setting_text(cfg.get(name)) or "unset"
            if name != NATIVE_JUDGE and (have == want) != (op == "="):
                setting = have
                break
    return {"selector": r["target"].split(" > ")[0] if r["target"] else None, "local": r["local"],
            "setting": setting, "source": source}


def baseline_matches(baseline: dict, inc: dict) -> bool:
    """Whether a receipt's verdict.baseline {kind, id} names incumbent route `inc`: kind `hosted` -> the hosted model
    id of a non-local selector (`typesafe/proj-b-latest` or `proj-b-latest`); `route` -> the selector (thinking suffix
    optional); `fixed` -> the gating setting's value while the route is off; `builtin` -> the profile route
    has no role in the chain (selector None, setting None: omp falls back to its built-in choice)."""
    kind, bid, sel = baseline.get("kind"), baseline.get("id"), inc["selector"]
    if kind == "hosted":
        return bool(sel) and not inc["local"] and bid in {sel, _bare(sel), _bare(sel).split("/", 1)[-1]}
    if kind == "route":
        return bool(sel) and bid in {sel, _bare(sel)}
    if kind == "fixed":
        return inc["setting"] is not None and bid == inc["setting"]
    if kind == "builtin":
        return sel is None and inc.get("setting") is None
    return False


def grade(receipt: dict, sha: str | None, model: str | None = None, digest: str | None = None,
          inc: dict | None = None, setting: str | None = None,
          alternatives: list[dict] | None = None) -> tuple[str, str]:
    """(status, reason) of one receipt for a feature whose installed module has `sha`, on a route to `model` whose
    installed digest is `digest` and whose incumbent (the route the profile had before the local one, incumbent())
    is `inc` (model None: no local route to match, the baseline-route and model checks are skipped), or on a route
    that targets no model: off at gating-setting value `setting` (incumbent()['setting'] of the profile's settings now,
    or after the preset being applied). The PROOF CONTRACT, checked in this order; the first failure is the status:
    kind == "run" and run.label in PROOF_LABELS (else UNPROVEN); omp_module_sha == sha (else STALE); problems empty
    (else BAD); verdict.baseline {kind, id} present (else UNPROVEN) and naming `inc` (baseline_matches; else UNPROVEN, also when `inc` is unknown);
    a `builtin` baseline names this feature and matches a route with no role in the chain (selector and setting None:
    omp's built-in fallback, e.g. titles/commit-messages/skill-compression on qwen3.8 against no local preset);
    run.provenance.pins.model_digest present and the route model's digest (else UNPROVEN); then CARRIED when
    carried_forward is set, else PROVEN.
    A receipt proving a no-model route (NO_MODEL_ROUTE) instead pins run.provenance.pins.route {kind fixed, id
    <setting value>} and pins.model_calls 0. It is BAD on a route to a model, or off at another setting, or with any
    model call pinned; UNPROVEN where the route is not off at a setting; its baseline is checked as above and may not be
    the no-model route itself. A model receipt on a no-model route is UNPROVEN.
    An already-local route (`inc` is the routed model itself: no local preset apply records the route it replaced)
    is also proven by a receipt whose baseline names one of `alternatives` (alternatives(): a declared preset of the
    row's family that routes no local model); the digest check is unchanged. Callers pass `alternatives` only when no
    local apply record exists, so a recorded pre-apply route stays the only baseline."""
    run = _dict(receipt.get("run"))
    if receipt.get("kind") != "run" or run.get("label") not in PROOF_LABELS:
        return "UNPROVEN", (f"not a {'/'.join(PROOF_LABELS)} run receipt (kind {receipt.get('kind')!r}, "
                            f"run.label {run.get('label')!r})")
    if sha is None or receipt.get(SHA_FIELD) != sha:
        return "STALE", f"proved {SHA_FIELD} {receipt.get(SHA_FIELD)}, installed is {sha}"
    if receipt.get("problems"):
        return "BAD", f"lists problems: {receipt['problems']}"[:200]
    verdict = _dict(receipt.get("verdict"))
    if verdict.get("compare") is None:
        return "UNPROVEN", "no verdict.compare"
    if verdict["compare"] != "BETTER":
        return "BAD", f"verdict.compare is {verdict['compare']}, not BETTER"
    baseline = _dict(verdict.get("baseline"))
    if not baseline.get("kind") or not baseline.get("id"):
        return "UNPROVEN", "verdict names no baseline"
    if baseline["kind"] == "builtin" and baseline["id"] != receipt.get(FEATURE_FIELD):
        return "UNPROVEN", f"builtin baseline names {baseline['id']!r}, not this feature"
    pins = _dict(_dict(run.get("provenance")).get("pins"))
    pinned, proved = pins.get("model_digest"), _dict(pins.get(NO_MODEL_ROUTE))
    named = f"{baseline['kind']} {baseline['id']}"
    if proved or setting is not None:
        if not proved:
            return "UNPROVEN", f"this profile's route targets no model (off at {setting}); the receipt proves a model"
        route_text = f"{proved.get('kind')} {proved.get('id')}"
        if model is not None:
            return "BAD", f"proves no-model route {route_text}; this profile's route targets model {model}"
        if setting is None:
            return "UNPROVEN", f"proves no-model route {route_text}; this profile's route is not off at a setting"
        off = {"selector": None, "local": False, "setting": setting, "source": "routed"}
        if proved.get("kind") != "fixed" or not baseline_matches(proved, off):
            return "BAD", f"proves no-model route {route_text}; this profile's route is off at {setting}"
        calls = pins.get("model_calls")
        if not isinstance(calls, int) or isinstance(calls, bool) or calls != 0:
            return "BAD", f"no-model route {route_text} pins model_calls {calls!r}, not 0"
        if inc is None:
            return "UNPROVEN", f"baseline {named}: this profile's incumbent route is unknown"
        if not baseline_matches(baseline, inc):
            inc_text = inc["selector"] or (f"off ({inc['setting']})" if inc["setting"] else "unset")
            return "UNPROVEN", f"baseline {named} is not this profile's route {inc_text} ({inc['source']})"
        if baseline_matches(baseline, off):
            return "UNPROVEN", f"baseline {named} is the routed no-model route itself ({inc['source']})"
    elif model is not None:
        if inc is None:
            return "UNPROVEN", f"baseline {named}: this profile's incumbent route is unknown"
        # An already-local route with no local apply record (inc is the routed model itself): its proof is BETTER than
        # a declared alternative of its preset family that routes no local model (alternatives()).
        own = bool(inc["local"] and inc["selector"] and route_model(inc["selector"]) == model)
        alt = next((a for a in alternatives or () if own and not a["local"] and baseline_matches(baseline, a)), None)
        if alt is None and not baseline_matches(baseline, inc):
            route_text = inc["selector"] or (f"off ({inc['setting']})" if inc["setting"] else "unset")
            declared = "; ".join(f"{a['selector'] or 'off (' + str(a['setting']) + ')'} ({a['source']})"
                                 for a in alternatives or () if own and not a["local"])
            return "UNPROVEN", (f"baseline {named} is not this profile's route {route_text} ({inc['source']})"
                                + (f" nor a declared alternative: {declared}" if declared else ""))
        if alt is None and own:
            return "UNPROVEN", (f"baseline {named} is the routed model itself ({inc['source']}): no local preset "
                                "apply records the route it replaced, and it is no declared non-local alternative")
        if not pinned:
            return "UNPROVEN", "run.provenance.pins has no model_digest"
        if digest is None:
            return "UNPROVEN", f"route model {model} has no installed digest to match {pinned}"
        if not same_digest(pinned, digest):
            return "UNPROVEN", f"proves model digest {pinned}; the route's {model} is {digest[:12]}"
    if receipt.get(CARRIED_FIELD):
        return "CARRIED", f"carried forward: {receipt[CARRIED_FIELD]}"
    return "PROVEN", ""


def proof_status(found: list[tuple[str, dict]], sha: str | None, model: str | None = None,
                 digest: str | None = None, inc: dict | None = None, setting: str | None = None,
                 alternatives: list[dict] | None = None) -> dict:
    """{proof, receipt, receipt_sha, reason} of the best receipt in `found` (STATUSES order) under grade()."""
    graded = [(*grade(data, sha, model, digest, inc, setting, alternatives), name, data.get(SHA_FIELD))
              for name, data in found]
    status, reason, name, receipt_sha = min(graded, key=lambda g: STATUSES.index(g[0]),
                                            default=("UNPROVEN", "no receipt names it", None, None))
    return {"proof": status, "receipt": name, "receipt_sha": receipt_sha, "reason": reason}


def alternatives(row: dict, cfg: dict, provs: dict[str, dict[str, str]], profile: str,
                 registry: Path | None = None) -> list[dict]:
    """The declared alternatives of feature `row` on a profile with settings `cfg`/`provs`: incumbent() of the route
    each preset of the row's family (registries/presets.json, `preset` column; none for `-`) gives it when applied to
    those settings, for every preset whose route routes no local model (off, e.g. memory:none, or non-local). Source
    `alternative <preset>`."""
    if row["preset"] == "-":
        return []
    from . import presets  # presets imports this module
    out = []
    for p in presets.load(registry or presets.REGISTRY)["presets"]:
        if presets.family(p["name"]) != row["preset"]:
            continue
        c, pv = dict(cfg), {k: dict(v) for k, v in provs.items()}
        for op in p["ops"]:
            if op["op"] == "provider":
                pv.pop(presets.PROVIDER, None)
                if op["model"] is not None:
                    pv[presets.PROVIDER] = {"baseUrl": omp_profiles.provider_url(profile, GATEWAY_PORT),
                                            "api": presets.PROVIDER_API}
            else:
                key = op.get("key", "modelRoles")
                c[key] = presets._fold(op, c.get(key))
        # A preset that keeps or moves the route to a local model is never an alternative (only a proof replaces it).
        if not route(row, c, pv)["local"]:
            out.append(incumbent(row, c, pv, f"alternative {p['name']}"))
    return out


def incumbent_settings(profile: str, family: str, cfg: dict,
                       provs: dict[str, dict[str, str]]) -> tuple[dict, dict[str, dict[str, str]], str] | str:
    """The settings the baseline must be measured against for `profile` and preset family `family`: when the presets
    applied record (~/.localbench/presets/applied.json) shows a local preset of that family on the profile, the
    settings from before that apply (each step's recorded `before` in its rollback manifest); otherwise `cfg`/`provs`,
    the profile's current route (a proof graded before the flip). Returns (cfg, provs, source), or the reason the
    pre-apply settings cannot be read."""
    from . import presets  # presets imports this module
    entry = presets.applied().get(profile, {}).get(family) if family != "-" else None
    if not entry:
        return cfg, provs, CURRENT_ROUTE
    try:
        local = presets.find(entry["preset"])["local"]
    except (presets.PresetError, OSError, ValueError) as exc:
        return f"applied preset {entry.get('preset')} cannot be read: {exc}"
    if not local:
        return cfg, provs, CURRENT_ROUTE
    manifest = presets.rollback_root() / f"preset-{entry['id']}" / "manifest.json"
    try:
        steps = json.loads(manifest.read_text(encoding="utf-8"))["plan"]["steps"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"apply {entry['id']} of {entry['preset']} has no readable manifest ({manifest}: {exc})"
    before, before_provs = dict(cfg), {k: dict(v) for k, v in provs.items()}
    for s in (s for s in steps if s.get("profile") == profile):
        if s.get("kind") == "provider":
            before_provs.pop(presets.PROVIDER, None)
            if s.get("before") is not None:
                before_provs[presets.PROVIDER] = {"baseUrl": omp_profiles.provider_url(profile, GATEWAY_PORT),
                                                  "api": presets.PROVIDER_API}
        else:
            before[s["key"]] = s.get("before")
    return before, before_provs, f"before preset {entry['preset']} apply {entry['id']}"


def ollama_digests() -> dict[str, str]:
    """Installed ollama model name -> digest (GET /api/tags); empty when ollama does not answer, so every local route
    then reads UNPROVEN with the reason, never PROVEN on an unchecked model."""
    try:
        return park._tags()
    except (OSError, ValueError):
        return {}


def route_model(selector: str) -> str:
    """The model a selector names, thinking suffix dropped: `ollama/qwen3.8:27b-mlx:high` -> qwen3.8:27b-mlx."""
    model = selector.split("/", 1)[-1]
    base, _, last = model.rpartition(":")
    return base if base and last in park.THINKING else model


def _laya_snapshot(repo: str) -> str | None:
    """The Hugging Face cache snapshot sha for `repo` (<org>/<name>, subfolder stripped): the single
    models--<org>--<name>/snapshots/<sha> under HF_HUB_CACHE (else ~/.cache/huggingface/hub), or None
    when missing or ambiguous (zero or several snapshots cannot pin a build)."""
    base = repo.split("@", 1)[0]
    if not base or ".." in base or base.count("/") != 1:
        return None
    roots = [os.environ.get("HF_HUB_CACHE"), str(Path.home() / ".cache" / "huggingface" / "hub")]
    for root in roots:
        if not root:
            continue
        snaps = Path(root) / ("models--" + base.replace("/", "--")) / "snapshots"
        if not snaps.is_dir():
            continue
        shas = sorted(p.name for p in snaps.iterdir()
                      if len(p.name) == 40 and all(c in "0123456789abcdef" for c in p.name.lower()))
        if len(shas) == 1:
            return shas[0]
        return None
    return None


def installed_digest(row: dict, model: str, digests: dict[str, str]) -> str | None:
    """The installed digest of the model a route of `row` targets: ollama's (`digests`, name or name:latest);
    laya: routes resolve to the hub cache snapshot sha; or for mnemopi's embedding model (`fastembed` role,
    `local/<fastembed name>`) workloads.fastembed_digest of its on-disk files, the digest embedding_pins
    records in a run's pins (an entry in `digests` overrides it)."""
    if model.startswith("laya:"):
        return _laya_snapshot(model[len("laya:"):])
    if row["role"] == EMBEDDING_ROLE:
        return digests.get(model) or fastembed_digest(model)
    return digests.get(model) or (None if ":" in model else digests.get(f"{model}:latest"))


def gateway_bypass(selector: str, provs: dict[str, dict[str, str]]) -> str | None:
    """The baseUrl when a local selector's provider talks to ollama directly on this host (port OLLAMA_PORT) instead of
    through the localbench gateway (port GATEWAY_PORT); omp's built-in `ollama` provider, undeclared, is park.OLLAMA."""
    provider = selector.split("/", 1)[0]
    base = provs.get(provider, {}).get("baseUrl") or (park.OLLAMA if provider == "ollama" else "")
    url = urllib.parse.urlsplit(base)
    return base if url.hostname in LOOPBACK and url.port == OLLAMA_PORT else None


def report(profiles: list[str] | None = None, *, registry: Path = REGISTRY, receipts_dir: Path = RECEIPTS,
           digests: dict[str, str] | None = None) -> list[dict]:
    """One dict per registry row: the row's columns, the installed module's path and sha, the route per profile
    (`routes`: profile -> {target, local, disabled}), `local_profiles`, and per local profile `proofs`: profile ->
    {model, digest, bypass, proof, receipt, receipt_sha, reason}, a proof applying only to the model that profile's
    live route targets (its first local selector). The feature's `proof`/`receipt`/`receipt_sha`/`reason` are its
    worst local profile's (with no local route: the best receipt, model check skipped), and `status` is DISABLED when
    every profile has the route off, else `proof`. `profiles` defaults to every omp profile on this host
    (models.profiles); `digests` (installed model name -> digest) to ollama_digests()."""
    rows = load(registry)
    pkg = park.omp_package(omp_bin())
    names = models.profiles() if profiles is None else list(profiles)
    with ThreadPoolExecutor(max_workers=4) as pool:  # one `omp config list` is ~1.5 s; 13 profiles serially is ~20 s
        settings = dict(zip(names, pool.map(omp_settings, names), strict=True))
    provs = {p: providers(p) for p in names}
    digests = ollama_digests() if digests is None else digests
    found = receipts(receipts_dir)
    out = []
    for row in rows:
        path = package_root(row["omp_package"], pkg) / row["omp_module"]
        sha = module_sha(path)
        routes = {p: route(row, cfg, provs[p]) for p, cfg in settings.items()}
        local = [p for p, r in routes.items() if r["local"]]
        candidates = found.get(row["feature"], [])
        proofs = {}
        for p in local:
            selector = next(t for t in routes[p]["target"].split(" > ") if is_local(t, provs[p]))
            model = route_model(selector)
            digest = installed_digest(row, model, digests)
            before = incumbent_settings(p, row["preset"], settings[p], provs[p])
            inc = (incumbent(row, *before) if not isinstance(before, str)
                   else {"selector": None, "local": False, "setting": None, "source": before})
            alts = (alternatives(row, settings[p], provs[p], p)
                    if not isinstance(before, str) and before[2] == CURRENT_ROUTE else None)
            proofs[p] = {"model": model, "digest": digest, "bypass": gateway_bypass(selector, provs[p]),
                         "incumbent": inc, "alternatives": alts,
                         **proof_status(candidates, sha, model, digest, inc, alternatives=alts)}
        worst = max(proofs.values(), key=lambda q: STATUSES.index(q["proof"]), default=None)
        overall = ({k: worst[k] for k in ("proof", "receipt", "receipt_sha", "reason")} if worst
                   else proof_status(candidates, sha))
        live = any(r["disabled"] is None for r in routes.values())
        out.append({**row, "module_path": str(path), SHA_FIELD: sha, "routes": routes, "local_profiles": local,
                    "proofs": proofs, **overall, "status": "DISABLED" if routes and not live else overall["proof"]})
    return out


def _fix(r: dict) -> str:
    if r["proof_suite"] == "-":
        return (f"no localbench suite proves {r['feature']} yet: route it off the local model"
                + (f" (preset {r['preset']})" if r["preset"] != "-" else ""))
    return (f"prove {r['feature']} with the {r['proof_suite']} suite on the routed model and bank a BETTER run receipt "
            f"carrying {FEATURE_FIELD}={r['feature']} {SHA_FIELD}={r[SHA_FIELD]}, or route it off the local model"
            + (f" (preset {r['preset']})" if r["preset"] != "-" else ""))


def findings(rows: list[dict]) -> list[tuple[str, str, str | None]]:
    """(PASS|WARN|FAIL, message, fix) per feature of a report(), each message starting with the feature id. FAIL: a
    profile routes the feature to a local model and that profile's proof is UNPROVEN, STALE or BAD, or the module it
    is registered at is gone from the installed omp while routed local. WARN: a local route standing on a CARRIED
    proof, or a registered module that is gone (not local). Plus one WARN per feature whose local route bypasses the
    localbench gateway (provider baseUrl on ollama's own port, e.g. ollama-sys1 at 127.0.0.1:11434)."""
    out: list[tuple[str, str, str | None]] = []
    for r in rows:
        if r[SHA_FIELD] is None:
            out.append(("FAIL" if r["local_profiles"] else "WARN",
                        f"{r['feature']}: {r['omp_module']} is not in the installed {r['omp_package']} "
                        f"({r['module_path']}); the registry row no longer names omp's call site",
                        "point the row in registries/features.tsv at the installed omp's call site"))
            continue
        proofs = r["proofs"]
        failing = {p: q for p, q in proofs.items() if q["proof"] in FAILING}
        carried = {p: q for p, q in proofs.items() if q["proof"] == "CARRIED"}
        if not proofs:
            out.append(("PASS", f"{r['feature']}: {r['status']}; no profile routes it to a local model", None))
        elif failing:
            detail = "; ".join(f"{p}: {r['routes'][p]['target']} {q['proof']} ({q['reason']}"
                               f"{', ' + q['receipt'] if q['receipt'] else ''})" for p, q in failing.items())
            out.append(("FAIL", f"{r['feature']} routes local without a current proof: {detail}", _fix(r)))
        elif carried:
            out.append(("WARN", f"{r['feature']} routes local on a proof carried forward ("
                        + "; ".join(f"{p}: {q['receipt']}" for p, q in carried.items())
                        + "); its re-proof is still owed", _fix(r)))
        else:
            out.append(("PASS", f"{r['feature']} routes local, PROVEN: "
                        + "; ".join(f"{p}: {q['model']} by {q['receipt']}" for p, q in proofs.items()), None))
        if bypass := {p: q["bypass"] for p, q in proofs.items() if q["bypass"]}:
            out.append(("WARN", f"{r['feature']} bypasses the localbench gateway (:{GATEWAY_PORT}): "
                        + "; ".join(f"{p}: {r['routes'][p]['target']} via {url}" for p, url in bypass.items()),
                        "repoint the provider's baseUrl in each profile's models.yml at "
                        + ", ".join(omp_profiles.provider_url(p, GATEWAY_PORT) for p in bypass)))
    return out


def doctor_findings(profiles: list[str] | None = None, *, registry: Path = REGISTRY, receipts_dir: Path = RECEIPTS,
                    digests: dict[str, str] | None = None) -> list[tuple[str, str, str | None]]:
    """findings(report(...)); a registry, omp or settings failure is one FAIL finding, never a crash."""
    try:
        rows = report(profiles, registry=registry, receipts_dir=receipts_dir, digests=digests)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return [("FAIL", f"omp feature map unavailable: {exc}",
                 f"fix {registry} or point LOCALBENCH_OMP at an omp package install, then `localbench features`")]
    return findings(rows)


def lines(rows: list[dict], names: list[str]) -> list[str]:
    """Text form of a report() over profiles `names`: per feature, status, call site and module sha, then each
    distinct route with the profiles that have it."""
    out = []
    for r in rows:
        sha = (r[SHA_FIELD] or "missing")[:12]
        out.append(f"{r['feature']:<30} {r['status']:<9} {r['omp_package']}/{r['omp_module']} "
                   f"{r['omp_symbol']}  sha {sha}  receipt {r['receipt'] or '-'}")
        by: dict[str, list[str]] = {}
        for p, rt in r["routes"].items():
            q = r["proofs"].get(p)
            text = f"DISABLED ({rt['disabled']})" if rt["disabled"] else (
                f"{rt['target']}{'  [local]' if rt['local'] else ''}" if rt["target"]
                else "no role of the chain set (omp's built-in fallback)")
            if q:
                text += f"  {q['proof']}" + (f" ({q['reason']})" if q["reason"] else f" by {q['receipt']}")
                text += f"  BYPASSES GATEWAY via {q['bypass']}" if q["bypass"] else ""
            by.setdefault(text, []).append(p)
        for text, who in by.items():
            out.append(f"    {render.members(who, names, 'profiles')}: {text}")
    return out
