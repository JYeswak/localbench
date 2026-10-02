"""Declarative proof specs plus one runner: no proof is ever an orchestrator one-off.

A spec (registries/proofs/<feature>__<slug>.json) declares WHAT is proven;
`prove` runs it through the existing tiers, banks the receipts, grades them
through features.grade, and files the proof beads through proofqueue. Kinds:
decision (run_suite + paired), memory (legs + memory_verdict), generation
(corpus + %30's assertions), agent (no backend yet: planned, refused live).
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "registries" / "proofs"
RECEIPTS_DIR = REPO_ROOT / "docs" / "evidence" / "receipts"

KINDS = ("decision", "generation", "memory", "agent")
STAGES = ("screen", "proof")
REGIMES = ("side",)

REQUIRED_FIELDS = ("feature", "kind", "candidates", "dataset", "assertions", "stage")
OPTIONAL_FIELDS = ("repeats", "seed", "regime", "wins", "wins_needed", "fresh_per_round",
                   "allow_errors", "blocked", "profile", "hosted", "baseline", "arms", "verdicts",
                   "legs", "main_model", "smol_model", "tiers", "pairs", "mem_rounds", "wait_idle",
                   "description")


class ProveError(RuntimeError):
    """A spec or run refusal: invalid spec, uncommitted spec, unresolvable candidate, missing backend."""


def _fail(path, detail: str) -> ProveError:
    return ProveError(f"{path}: {detail}")


def load_spec(path: str | Path) -> dict:
    """Read and validate one proof spec. Unknown top-level keys, bad enums, empty lists and
    duplicated assertion ids are refused; everything else is checked at run time against live state."""
    from . import features

    target = Path(path).expanduser()
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _fail(target, f"unreadable ({exc})") from None
    except ValueError as exc:
        raise _fail(target, f"not JSON ({exc})") from None
    if not isinstance(doc, dict):
        raise _fail(target, "top level is not an object")
    unknown = sorted(set(doc) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise _fail(target, f"unknown fields {unknown}")
    missing = [key for key in REQUIRED_FIELDS if key not in doc]
    if missing:
        raise _fail(target, f"missing fields {missing}")
    try:
        rows = features.load()
    except (OSError, ValueError) as exc:
        raise _fail(target, f"feature registry unreadable ({exc})") from None
    if doc["feature"] not in {row["feature"] for row in rows}:
        raise _fail(target, f"feature {doc['feature']!r} is not in registries/features.tsv")
    if doc["kind"] not in KINDS:
        raise _fail(target, f"kind {doc['kind']!r} is not one of {list(KINDS)}")
    if doc["stage"] not in STAGES:
        raise _fail(target, f"stage {doc['stage']!r} is not one of {list(STAGES)}")
    if doc.get("regime", "side") not in REGIMES:
        raise _fail(target, f"regime {doc.get('regime')!r} is not one of {list(REGIMES)}")
    if not isinstance(doc["candidates"], list) or not doc["candidates"]:
        raise _fail(target, "candidates is empty")
    dataset = doc["dataset"]
    if not isinstance(dataset, dict) or not dataset.get("suite") or not dataset.get("items_sha256"):
        raise _fail(target, "dataset needs {suite, items_sha256}")
    if not isinstance(doc["assertions"], list) or not doc["assertions"]:
        raise _fail(target, "assertions must be a non-empty list")
    seen: set = set()
    for assertion in doc["assertions"]:
        if not isinstance(assertion, dict) or not assertion.get("id") or not assertion.get("type"):
            raise _fail(target, f"assertion needs {{id, type}}: {assertion!r}")
        if assertion["id"] in seen:
            raise _fail(target, f"duplicated assertion id {assertion['id']!r}")
        seen.add(assertion["id"])
    if doc["kind"] == "memory" and (not doc.get("arms") or not doc.get("verdicts")):
        raise _fail(target, "memory specs need arms and verdicts")
    if "allow_errors" in doc:
        allowed = doc["allow_errors"]
        if not isinstance(allowed, (int, float)) or isinstance(allowed, bool) or not 0 <= allowed <= 0.05:
            raise _fail(target, "allow_errors must be a number in [0, 0.05]")
    return doc



def spec_slug(path: str | Path) -> str:
    """<feature>__<slug> from the spec filename."""
    stem = Path(path).stem
    if "__" not in stem:
        raise ProveError(f"{path}: spec filename must be <feature>__<slug>.json")
    return stem


# A git hook exports GIT_DIR, GIT_WORK_TREE and friends for the OUTER repo; any git child that
# inherits them reads the wrong index. Every git call in this module scrubs them (and the test
# fixtures do the same).
_HOOK_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
             "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX")


def _git(repo: Path, *argv: str) -> tuple[int, str]:
    env = {key: value for key, value in os.environ.items() if key not in _HOOK_ENV}
    proc = subprocess.run(["git", *argv], cwd=repo, env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    return proc.returncode, proc.stdout.strip()


def spec_commit(path: str | Path, repo: Path = REPO_ROOT) -> str:
    """The commit that froze this spec: tracked, clean, and its last-touching sha.

    Pre-registration is the commit: editing a spec after data is impossible to hide (forbidden
    pattern 9), so prove() refuses anything else before planning anything."""
    target = str(Path(path).expanduser())
    root = Path(repo).expanduser()
    rc, _ = _git(root, "ls-files", "--error-unmatch", target)
    if rc != 0:
        raise ProveError(f"{target}: spec is uncommitted; commit it for %20's review first")
    rc, out = _git(root, "status", "--porcelain", "--", target)
    if rc != 0:
        raise ProveError(f"{target}: cannot status the spec")
    if out:
        raise ProveError(f"{target}: spec modified since its commit; commit it for %20's review first")
    rc, sha = _git(root, "log", "-1", "--format=%H", "--", target)
    if rc != 0 or not sha:
        raise ProveError(f"{target}: no commit touches the spec")
    return sha


def resolve_candidate(candidate: dict) -> dict:
    """A candidate route spec to {backend, model}: ollama:/laya: routes, or a preset's model from its ops."""
    from . import features, presets

    if not isinstance(candidate, dict):
        raise ProveError(f"candidate is not an object: {candidate!r}")
    if "route" in candidate:
        route = candidate["route"]
        for prefix, backend in (("ollama:", "ollama"), ("laya:", "laya")):
            if isinstance(route, str) and route.startswith(prefix):
                model = route[len(prefix):]
                if not model:
                    raise ProveError(f"candidate route {route!r} names no model")
                return {"backend": backend, "model": model, "via": "route"}
        raise ProveError(f"candidate route {route!r}: ollama: or laya: only")
    if "preset" in candidate:
        try:
            preset = presets.find(candidate["preset"])
        except (presets.PresetError, OSError, ValueError) as exc:
            raise ProveError(f"candidate preset {candidate['preset']!r}: {exc}") from None
        for op in preset.get("ops", []):
            if op.get("op") == "provider" and op.get("model"):
                return {"backend": "ollama", "model": op["model"], "via": f"preset {preset['name']}"}
            if op.get("op") == "role" and isinstance(op.get("selector"), str):
                return {"backend": "ollama", "model": features.route_model(op["selector"]),
                        "via": f"preset {preset['name']}"}
        raise ProveError(f"candidate preset {preset['name']!r} routes no model")
    raise ProveError(f"candidate needs route or preset: {candidate!r}")


def bank_receipt(receipt: dict, name: str, directory: Path = RECEIPTS_DIR) -> Path:
    """Write one receipt atomically under docs/evidence/receipts/."""
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ProveError(f"refusing symlink receipts dir: {target}")
    path = target / name
    if path.is_symlink():
        raise ProveError(f"refusing symlink receipt path: {path}")
    text = json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n"
    tmp = target / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def receipt_name(kind: str, feature: str, candidate_slug: str, created: str) -> str:
    """prove__<kind>__<feature>__<candidate>__<UTC>.json, shell- and filename-safe."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in candidate_slug)[:80]
    return f"prove__{kind}__{feature}__{safe}__{created}.json"


def utc_stamp() -> str:
    """Compact UTC stamp for receipt names."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _judge_family(model: str, base_url: str, timeout: float = 20.0) -> str:
    """The model family from Ollama /api/show at runtime (never typed): the `details.family`
    of the installed model (e.g. qwen3, llama). Refuses when the model is missing or unparseable."""
    body = json.dumps({"model": model}).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/api/show", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            doc = json.load(response)
    except (OSError, ValueError) as exc:
        raise ProveError(f"judge {model!r}: /api/show failed ({exc})") from None
    family = (doc.get("details") or {}).get("family") if isinstance(doc, dict) else None
    if not family or not isinstance(family, str):
        raise ProveError(f"judge {model!r}: /api/show names no family")
    return family


def check_assertion(assertion: dict, context: dict) -> tuple[bool, str]:
    """One spec assertion against a candidate's evidence. Returns (passed, detail).

    Types: `paired` (the decision.paired verdict must satisfy the rule);
    `metric` (a receipt metric within [min, max]); anything else is refused
    (generation assertions land with %30)."""
    kind = assertion.get("type")
    params = assertion.get("params", {})
    if kind == "paired":
        table = context.get("paired")
        if table is None:
            return False, f"{assertion['id']}: no paired table (decision paired did not run)"
        rule = params.get("rule", "BETTER")
        allowed = [rule] if isinstance(rule, str) else list(rule)
        if table["verdict"] not in allowed:
            return False, f"{assertion['id']}: paired verdict {table['verdict']} not in {allowed}"
        if table["n"] < int(params.get("min_n", 1)):
            return False, f"{assertion['id']}: n={table['n']} below min_n"
        return True, f"{assertion['id']}: {table['verdict']} (diff {table['diff_pp']:+.2f}pp, p={table['mcnemar_p']:.3g})"
    if kind == "metric":
        value = context.get("metrics", {}).get(params.get("metric"))
        if value is None:
            return False, f"{assertion['id']}: metric {params.get('metric')} unmeasured"
        lo, hi = params.get("min"), params.get("max")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            return False, f"{assertion['id']}: {params.get('metric')}={value} outside [{lo}, {hi}]"
        return True, f"{assertion['id']}: {params.get('metric')}={value} within [{lo}, {hi}]"
    raise ProveError(f"assertion {assertion.get('id')!r}: unknown type {kind!r}")


def _candidate_slug(candidate: dict) -> str:
    return str(candidate.get("route") or candidate.get("preset") or candidate.get("arm", "candidate"))


def run_decision_candidate(spec: dict, candidate: dict, suite, *, run_suite=None,
                           corpora=None) -> dict:
    """Run one decision candidate through run_suite + paired, returning evidence (no banking)."""
    from . import decision

    resolved = resolve_candidate(candidate)
    run = run_suite or _live_decision_run
    receipt = run(spec, candidate, resolved, suite)
    table = decision.paired({"kind": "run", "run": {"decision": {"local": receipt["run"]["decision"]["local"]}}},
                            corpora=corpora)
    arm = receipt["run"]["decision"]["local"]
    metrics = {name: entry.get("value") for name, entry in arm.get("metrics", {}).items()}
    spreads = {name: entry.get("spread") for name, entry in arm.get("metrics", {}).items()
               if isinstance(entry, dict)}
    return {"candidate": candidate, "resolved": resolved, "receipt": receipt,
            "paired": table["types"], "metrics": metrics, "spreads": spreads}


def _live_decision_run(spec: dict, candidate: dict, resolved: dict, suite):
    """The live decision tier call (inference): run_suite on the candidate backend. Not used in tests."""
    from . import decision

    if resolved["backend"] == "laya":
        shim_spec = decision.parse_laya_spec(resolved["model"])
        with decision.LayaShim(shim_spec) as shim:
            return decision.run_suite(shim.url, resolved["model"], suite,
                                      repeats=int(spec.get("repeats", 1)),
                                      feature=spec["feature"],
                                      omp_module_sha=_module_sha(spec["feature"]))
    from . import sysstats
    from .__main__ import DECISION_BASE
    sampler = sysstats.Sampler(target=("ollama", resolved["model"]))
    receipt = decision.run_suite(DECISION_BASE, resolved["model"], suite,
                                 repeats=int(spec.get("repeats", 1)),
                                 sampler=sampler, feature=spec["feature"],
                                 omp_module_sha=_module_sha(spec["feature"]))
    receipt["run"]["decision"]["judge"] = {"model": resolved["model"],
                                           "family": _judge_family(resolved["model"], DECISION_BASE)}
    return receipt


def _module_sha(feature: str) -> str:
    """The feature's registered module sha in the installed omp (refuses when unreadable)."""
    from . import features, park
    from .workloads import omp_bin

    rows = features.load()
    row = next((r for r in rows if r["feature"] == feature), None)
    if row is None:
        raise ProveError(f"feature {feature!r} is not in registries/features.tsv")
    path = features.package_root(row["omp_package"], park.omp_package(omp_bin())) / row["omp_module"]
    sha = features.module_sha(path)
    if sha is None:
        raise ProveError(f"module {row['omp_module']} is gone from the installed omp")
    return sha

def run_memory_verdicts(spec: dict, legs: list[dict]) -> list[dict]:
    """Every spec verdict entry over leg subsets: memory_verdict is real, legs come from the caller.

    Leg labels carry their arm (`ab_a1`, `ab_b`, ...): each verdict entry names its candidate and
    baseline arms, and legs whose label starts with `<arm>_` fall in that arm."""
    from .workloads import memory_verdict

    if spec.get("blocked"):
        raise ProveError(f"spec is blocked: {spec['blocked'].get('reason', spec['blocked'])}")
    by_arm: dict[str, list[dict]] = {}
    arms_needed: list[str] = []
    for entry in spec["verdicts"]:
        for arm in (entry.get("candidate"), entry.get("baseline")):
            if not arm:
                raise ProveError(f"verdict entry names no candidate and baseline: {entry!r}")
            if arm not in arms_needed:
                arms_needed.append(arm)
    for leg in legs:
        label = leg.get("label", "")
        hits = [arm for arm in arms_needed if label == arm or label.startswith(arm)]
        if not hits:
            raise ProveError(f"leg {leg.get('id', label)!r} matches no verdict arm {arms_needed}")
        by_arm.setdefault(max(hits, key=len), []).append(leg)
    sha = _module_sha(spec["feature"])
    out = []
    for entry in spec["verdicts"]:
        candidate, baseline = entry.get("candidate"), entry.get("baseline")
        if candidate not in by_arm or baseline not in by_arm:
            raise ProveError(f"verdict needs legs for arms {candidate!r} and {baseline!r}; "
                             f"have {sorted(by_arm)}")
        receipt = memory_verdict(by_arm[candidate], by_arm[baseline], feature=spec["feature"],
                                 omp_module_sha=sha)
        out.append({"candidate": candidate, "baseline": baseline, "bank": entry.get("bank"),
                    "receipt": receipt})
    return out


def grade_proof(spec: dict, proof: dict, model: str | None, digest: str | None,
                inc: dict | None = None) -> tuple[str, str]:
    """Grade one banked proof through features.grade (the single grading rule)."""
    from . import features

    rows = features.load()
    if spec["feature"] not in {row["feature"] for row in rows}:
        raise ProveError(f"feature {spec['feature']!r} is not in registries/features.tsv")
    return features.grade(proof, _module_sha(spec["feature"]), model, digest, inc)

def update_beads(spec: dict, outcomes: list[dict], br=None) -> dict:
    """Comment on or close the proof bead: a passing grade closes with the receipt,
    anything else comments the status. Returns a proofqueue-style summary."""

    actions = []
    for outcome in outcomes:
        title = f"prove {spec['feature']} on {outcome['route']}"
        if outcome["grade"] == "PROVEN":
            actions.append({"op": "close", "title": title, "unit": {"feature": spec["feature"]},
                            "reason": f"route proven: receipt {outcome['receipt']} grades PROVEN"})
        else:
            actions.append({"op": "comment", "title": title, "unit": {"feature": spec["feature"]},
                            "note": f"proof {Path(outcome['spec']).stem}: {outcome['grade']} "
                                    f"({outcome['reason']})"})
    return _apply_to_beads(actions, br)


def _apply_to_beads(actions: list[dict], br=None) -> dict:
    """Match actions to open beads by title (exact or proofqueue adoption) and apply."""
    from . import proofqueue

    runner = br if br is not None else proofqueue._br
    rc, out, err = runner(["list", "--status", "open", "--json"])
    if rc != 0:
        raise ProveError(f"br list failed rc={rc}: {err.strip()[-300:]}")
    try:
        issues = json.loads(out).get("issues", [])
    except ValueError as exc:
        raise ProveError(f"br list returned non-JSON: {exc}") from exc
    by_title = {issue.get("title", ""): issue for issue in issues}
    summary: dict = {"commented": [], "closed": [], "missing": []}
    for action in actions:
        bead = by_title.get(action["title"])
        if bead is None:
            candidates = [issue for issue in issues
                          if issue.get("title", "").startswith(action["title"].split(" on ")[0] + " on")
                          and action["title"].split(" on ", 1)[1] in issue.get("title", "")]
            bead = sorted(candidates, key=lambda b: b.get("id", ""))[0] if candidates else None
        if bead is None:
            summary["missing"].append(action["title"])
            continue
        if action["op"] == "close":
            argv = ["close", bead["id"], "--reason", action["reason"]]
            key = "closed"
        else:
            argv = ["comments", "add", bead["id"], "-m", action["note"]]
            key = "commented"
        rc, out, err = runner(argv)
        if rc != 0:
            raise ProveError(f"br {' '.join(argv[:2])} failed rc={rc}: {err.strip()[-300:]}")
        summary[key].append(bead["title"])
    return summary


def regime_allows(spec: dict) -> bool:
    """Whether the spec's regime runs now. Side runs anywhere; anything else is refused at load."""
    return spec.get("regime", "side") == "side"


def due_specs(pairs: list[tuple[str | Path, dict]], br=None) -> list[tuple[str | Path, dict]]:
    """(path, spec) pairs with an open covering bead whose regime allows it now. A bead covers
    a spec when its title starts with `prove <feature> on` and names one of the spec's candidate routes."""
    from . import proofqueue

    runner = br if br is not None else proofqueue._br
    rc, out, err = runner(["list", "--status", "open", "--json"])
    if rc != 0:
        raise ProveError(f"br list failed rc={rc}: {err.strip()[-300:]}")
    try:
        issues = json.loads(out).get("issues", [])
    except ValueError as exc:
        raise ProveError(f"br list returned non-JSON: {exc}") from exc
    titles = [issue.get("title", "") for issue in issues]
    due = []
    for path, spec in pairs:
        if spec.get("blocked") or not regime_allows(spec):
            continue
        routes = [_candidate_slug(c) for c in spec["candidates"]]
        if any(title.startswith(f"prove {spec['feature']} on") and any(r in title for r in routes)
               for title in titles):
            due.append((path, spec))
    return due

def _grade_candidate(spec: dict, row: dict, profile: str, proof: dict, model: str, *,
                     cfg=None, provs=None, digests=None) -> tuple[str, str]:
    """Grade one fresh proof exactly like report() grades banked ones: the profile's incumbent
    route is what the receipt's baseline must name. cfg/provs/digests inject for tests."""
    from . import features

    cfg = cfg if cfg is not None else features.omp_settings(profile)
    provs = provs if provs is not None else features.providers(profile)
    digests = digests if digests is not None else features.ollama_digests()
    before = features.incumbent_settings(profile, row["preset"], cfg, provs)
    inc = (features.incumbent(row, *before) if not isinstance(before, str)
           else {"selector": None, "local": False, "setting": None, "source": before})
    digest = features.installed_digest(row, model, digests)
    return features.grade(proof, _module_sha(spec["feature"]), model, digest, inc)


def _verify_privacy(feature: str, endpoints: list[str], profile: str = "default", *,
                    cfg=None, provs=None) -> tuple[bool, str, dict]:
    """Privacy counts as a win only when verified: the profile's incumbent route (derived by
    features.py, never the spec) is hosted (a set non-local selector, not off/fixed/local), and
    every candidate arm endpoint is loopback, checked with decision.endpoint on the receipt's
    endpoints, never the spec. Returns (verified, reason, evidence for the receipt)."""
    from . import decision, features

    rows = {row["feature"]: row for row in features.load()}
    row = rows.get(feature)
    if row is None:
        return False, f"feature {feature!r} is not in registries/features.tsv", {}
    try:
        cfg = cfg if cfg is not None else features.omp_settings(profile)
        provs = provs if provs is not None else features.providers(profile)
        before = features.incumbent_settings(profile, row["preset"], cfg, provs)
        inc = (features.incumbent(row, *before) if not isinstance(before, str)
               else {"selector": None, "local": False, "setting": None, "source": before})
    except (OSError, ValueError) as exc:
        return False, f"incumbent route unreadable: {exc}", {"endpoints": list(endpoints)}
    evidence = {"incumbent": inc.get("selector"), "endpoints": list(endpoints)}
    if not inc.get("selector") or inc.get("local"):
        return False, f"incumbent route is not hosted ({inc.get('selector') or 'off'})", evidence
    try:
        for endpoint in endpoints:
            decision.endpoint(endpoint)
    except decision.RequestError as exc:
        return False, f"candidate arm not loopback: {exc}", evidence
    return True, f"incumbent {inc['selector']}; arms {', '.join(endpoints)}", evidence


def _metric_band(spreads: dict | None, metric: str) -> tuple[float, float] | None:
    """The noise band for one metric: the receipt [min, max] spread across repeats. A metric
    with no spread (a single unrepeated point) has no band: its wins are unmeasurable."""
    band = (spreads or {}).get(metric)
    if (isinstance(band, (list, tuple)) and len(band) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in band)):
        return (min(band), max(band))
    return None


def _check_wins(spec: dict, evidence: dict, hosted: dict | None) -> list[str]:
    """Wins the proof needs beyond quality: verified structural privacy plus measured metric wins.

    Privacy counts only when _verify_privacy says so (hosted incumbent, loopback arms). A win on
    a paired question needs its paired verdict BETTER (the McNemar rigor); any other win needs
    non-overlapping [min, max] spread bands in the win direction (within-noise is not a win).
    Without a hosted arm, or without spreads on either side, the win is unmeasured (a problem:
    wins must be measured, never claimed)."""
    problems = []
    needed = int(spec.get("wins_needed", 1))
    wins = list(spec.get("wins", []))
    measured = 0
    for win in wins:
        if win.get("metric") == "privacy":
            if evidence.get("privacy_verified") is True:
                measured += 1
            else:
                problems.append("privacy win unverified (incumbent not hosted, or an arm not loopback)")
            continue
        name, direction = win.get("metric"), win.get("direction", "lower")
        value = evidence.get("metrics", {}).get(name)
        base = (hosted or {}).get("metrics", {}).get(name)
        if value is None or base is None:
            problems.append(f"win {name} unmeasured (no hosted arm to compare)")
            continue
        paired = (evidence.get("paired") or {}).get(name)
        if paired is not None:
            if paired.get("verdict") == "BETTER":
                measured += 1
            else:
                problems.append(f"win {name}: paired {paired.get('verdict')} "
                                f"(diff {paired.get('diff_pp'):+.2f}pp, p={paired.get('mcnemar_p'):.3g})")
            continue
        band = _metric_band(evidence.get("spreads"), name)
        base_band = _metric_band((hosted or {}).get("spreads"), name)
        if band is None or base_band is None:
            problems.append(f"win {name} unmeasured (no repeat spread on "
                            f"{'candidate' if band is None else 'hosted'} arm)")
            continue
        won = band[1] < base_band[0] if direction == "lower" else band[0] > base_band[1]
        if won:
            measured += 1
        else:
            problems.append(f"win {name}: candidate band [{band[0]}, {band[1]}] overlaps hosted "
                            f"[{base_band[0]}, {base_band[1]}]")
    if measured < needed:
        problems.append(f"only {measured} measured wins, need {needed}")
    return problems


def _evaluate(spec: dict, evidence: dict, hosted: dict | None) -> tuple[list[str], list[str]]:
    """Run every spec assertion over the evidence. Returns (problems, passed details)."""
    problems, passed = [], []
    for assertion in spec["assertions"]:
        kind = assertion.get("type")
        if kind == "paired":
            tables = evidence.get("paired") or {}
            name = assertion.get("params", {}).get("question")
            if name is None:
                if len(tables) != 1:
                    raise ProveError(f"assertion {assertion.get('id')!r}: names no question and "
                                     f"the evidence has {sorted(tables)}")
                table = next(iter(tables.values()))
            else:
                table = tables.get(name)
            context: dict = {"paired": table}
        elif kind == "metric":
            context = {"metrics": evidence.get("metrics", {})}
        else:
            raise ProveError(f"assertion {assertion.get('id')!r}: unknown type {kind!r}")
        ok, detail = check_assertion(assertion, context)
        (passed if ok else problems).append(detail)
    problems.extend(_check_wins(spec, evidence, hosted))
    return problems, passed

def _check_error_kinds(spec: dict, suite, outcomes: list[dict]) -> tuple[dict, list[str]]:
    """Per-question-type error rates from not-ok outcomes, listed in the receipt; kinds over
    allow_errors (capped 0.05 at load, default 0) are problems. Missing outcomes were already
    refused by paired scoring, so every item is present here."""
    allowed = float(spec.get("allow_errors", 0.0))
    outs = {o["id"]: o for o in outcomes if isinstance(o, dict) and isinstance(o.get("id"), str)}
    totals: dict[str, int] = {}
    for item in suite.items:
        for question in item["questions"].values():
            totals[question["type"]] = totals.get(question["type"], 0) + 1
    failed: dict[str, list[str]] = {}
    for item in suite.items:
        outcome = outs.get(item["id"], {})
        if not outcome.get("ok", False):
            for name, question in item["questions"].items():
                failed.setdefault(question["type"], []).append(f"{item['id']}.{name}")
    kinds, problems = {}, []
    for kind in sorted(totals):
        ids = sorted(failed.get(kind, []))
        rate = len(ids) / totals[kind]
        kinds[kind] = {"error_rate": rate, "items": ids}
        if rate > allowed:
            problems.append(f"error rate {rate:.3f} on {kind} exceeds allow_errors {allowed}")
    return kinds, problems

def list_specs(directory: Path = SPEC_DIR) -> list[Path]:
    """Every proof spec file, sorted. A directory that is missing is no specs, not an error."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return sorted(root.glob("*__*.json"))


def plan_steps(spec: dict, commit: str) -> list[tuple[str, str]]:
    """Dry-run lines for one spec: what prove() would run, bank, grade and file. No inference, no writes."""
    steps = [(f"resolve dataset {spec['dataset']['suite']} pin {spec['dataset']['items_sha256'][:12]}",
               "refuse when the suite on disk differs")]
    for candidate in spec["candidates"]:
        resolved = resolve_candidate(candidate)
        slug = _candidate_slug(candidate)
        if spec["kind"] == "decision":
            steps.append((f"run {resolved['backend']}:{resolved['model']} on the suite "
                          f"({spec.get('repeats', 1)} repeats, seed {spec.get('seed', 20261001)})",
                          "existing decision.run_suite tier; laya: candidates start the loopback shim"))
        elif spec["kind"] == "memory":
            source = (f"{len(spec['legs'])} banked run sources" if spec.get("legs")
                      else "a live mem/sess campaign (refused without one)")
            steps.append((f"run mem/sess legs for arms {sorted(spec.get('arms', {}))} from {source}",
                          "existing mem/sess tiers; legs split by ab_a/ab_b label prefix"))
        elif spec["kind"] == "generation":
            steps.append((f"run generation candidate {slug} on the corpus",
                          "corpus builder exists; assertions land with %30"))
        else:
            steps.append((f"run agent candidate {slug}", "no agent tier exists yet"))
        steps.append((f"bank prove__{spec['kind']}__{spec['feature']}__{slug} receipt "
                      f"(spec commit {commit[:12]})",
                      "receipts stay in-repo under docs/evidence/receipts/"))
        steps.append((f"grade {spec['feature']} on {slug} through features.grade; "
                      "comment on or close the proof bead",
                      "proofqueue: PROVEN closes citing the receipt, anything else comments"))
    return steps


def prove_spec(spec_path: str | Path, *, dry_run: bool = False, br=None, run_suite_fn=None,
               legs_fn=None, grade_fn=None, receipts_dir: Path = RECEIPTS_DIR,
               repo: Path = REPO_ROOT, corpora=None) -> dict:
    """Run one proof spec end to end (or plan it with dry_run): pre-reg commit, dataset pin,
    every candidate through its kind tier, assertions, banking, grading, bead updates.

    Generation and agent kinds plan but refuse live runs (no backend yet: %30's assertions,
    no agent tier). legs_fn provides memory legs; grade_fn replaces grading."""
    from . import decision

    spec = load_spec(spec_path)
    spec_slug(spec_path)
    if not Path(spec_path).stem.startswith(spec["feature"] + "__"):
        raise ProveError(f"{spec_path}: filename must be <feature>__<slug>.json")
    commit = spec_commit(spec_path, repo)
    if spec.get("blocked"):
        raise ProveError(f"{spec_path}: spec is blocked ({spec['blocked'].get('reason')})")
    if dry_run:
        return {"spec": str(spec_path), "commit": commit, "dry_run": True,
                "steps": plan_steps(spec, commit)}
    kind = spec["kind"]
    if kind == "generation":
        raise ProveError("generation run backend lands with %30's assertions; dry-run plans it")
    if kind == "agent":
        raise ProveError("no agent tier exists yet; dry-run plans it")
    report: dict = {"spec": str(spec_path), "commit": commit, "kind": kind, "candidates": []}
    if kind == "decision":
        suite = decision.resolve_suite(spec["dataset"]["suite"])
        if suite.items_sha256 != spec["dataset"]["items_sha256"]:
            raise ProveError(f"suite {suite.name} items_sha256 {suite.items_sha256} does not match "
                             f"the spec pin {spec['dataset']['items_sha256']}")
        for candidate in spec["candidates"]:
            entry = run_decision_candidate(spec, candidate, suite, run_suite=run_suite_fn, corpora=corpora)
            hosted_arm = entry["receipt"]["run"]["decision"].get("hosted")
            hosted = ({"metrics": {name: metric.get("value") for name, metric in
                                   hosted_arm.get("metrics", {}).items()},
                       "spreads": {name: metric.get("spread") for name, metric in
                                   hosted_arm.get("metrics", {}).items() if isinstance(metric, dict)}}
                      if hosted_arm else None)
            receipt = entry["receipt"]
            endpoint = receipt["run"]["decision"]["local"].get("endpoint")
            verified, why, priv = _verify_privacy(spec["feature"], [endpoint] if endpoint else [],
                                                  spec.get("profile", "default"))
            receipt["privacy"] = {"verified": verified, "reason": why, **priv}
            entry["privacy_verified"] = verified
            problems, passed = _evaluate(spec, entry, hosted)
            kinds, err_problems = _check_error_kinds(
                spec, suite, receipt["run"]["decision"]["local"].get("outcomes", []))
            receipt["error_kinds"] = kinds
            problems.extend(err_problems)
            receipt["proof_spec"] = {"path": str(spec_path), "commit": commit, "stage": spec["stage"],
                                     "passed": passed}
            created = utc_stamp()
            path = bank_receipt(receipt, receipt_name(kind, spec["feature"],
                                                     _candidate_slug(candidate), created), receipts_dir)
            grade = grade_fn or _grade_default(spec)
            resolved = entry["resolved"]
            model = f"laya:{resolved['model']}" if resolved["backend"] == "laya" else resolved["model"]
            status, reason = grade(receipt, model)
            report["candidates"].append({"candidate": candidate, "resolved": entry["resolved"],
                                        "route": _candidate_slug(candidate),
                                        "receipt": str(path), "grade": status, "reason": reason,
                                        "passed": passed, "problems": problems})
    else:
        if legs_fn is not None:
            legs = legs_fn(spec)
        elif spec.get("legs"):
            legs = _legs_from_sources(spec["legs"])
        else:
            legs = _live_memory_legs(spec)
        for verdict in run_memory_verdicts(spec, legs):
            receipt = verdict["receipt"]
            receipt["proof_spec"] = {"path": str(spec_path), "commit": commit, "stage": spec["stage"],
                                     "candidate": verdict["candidate"], "baseline": verdict["baseline"]}
            created = utc_stamp()
            path = bank_receipt(receipt, receipt_name(kind, spec["feature"],
                                                     str(verdict["candidate"]), created), receipts_dir)
            grade = grade_fn or _grade_default(spec)
            status, reason = grade(receipt, verdict["candidate"])
            report["candidates"].append({"candidate": verdict["candidate"], "baseline": verdict["baseline"],
                                        "route": str(verdict["candidate"]),
                                        "receipt": str(path), "grade": status, "reason": reason})
    report["beads"] = update_beads(spec, [{"route": _candidate_slug(c.get("candidate", c)),
                                           "grade": c["grade"], "reason": c["reason"],
                                           "receipt": c["receipt"], "spec": str(spec_path)}
                                          for c in report["candidates"]], br)
    return report


def _grade_default(spec: dict, *, cfg=None, provs=None, digests=None):
    """Grade fresh proofs on the default profile, like report() grades banked ones.
    cfg/provs/digests inject for tests."""
    from . import features

    profile = spec.get("profile", "default")
    cfg = cfg if cfg is not None else features.omp_settings(profile)
    provs = provs if provs is not None else features.providers(profile)
    digests = digests if digests is not None else features.ollama_digests()
    rows = {row["feature"]: row for row in features.load()}
    row = rows[spec["feature"]]

    def grade(receipt: dict, model: str) -> tuple[str, str]:
        before = features.incumbent_settings(profile, row["preset"], cfg, provs)
        inc = (features.incumbent(row, *before) if not isinstance(before, str)
               else {"selector": None, "local": False, "setting": None, "source": before})
        return features.grade(proof=receipt, sha=_module_sha(spec["feature"]),
                              model=model, digest=features.installed_digest(row, model, digests), inc=inc)

    return grade


def _live_memory_legs(spec: dict) -> list[dict]:
    """Legs for a memory spec come from its campaigns; running them is a live window job."""
    raise ProveError("memory legs need a live mem/sess campaign; dry-run plans it")


def _legs_from_sources(sources: list[str]) -> list[dict]:
    """Memory legs from banked run dirs, summaries or receipts (the memory-verdict leg rule):
    kind run -> [run], aa -> runs, ab -> legs, else a bare leg dict."""
    legs = []
    for source in sources:
        path = Path(source).expanduser()
        path = path / "summary.json" if path.is_dir() else path
        if not path.is_file():
            raise ProveError(f"no such run dir or receipt {source}")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ProveError(f"{source}: not JSON ({exc})") from None
        if not isinstance(doc, dict):
            raise ProveError(f"{source}: not an object")
        kind = doc.get("kind")
        found = ([doc["run"]] if kind == "run" else doc.get("runs") if kind == "aa" else doc.get("legs")
                 if kind == "ab" else [doc] if "provenance" in doc else None)
        if not found:
            raise ProveError(f"{source}: neither a run summary nor a run/aa/ab receipt")
        legs.extend(found)
    return legs
