"""Declarative proof specs plus one runner: no proof is ever an orchestrator one-off.

A spec (registries/proofs/<feature>__<slug>.json) declares WHAT is proven;
`prove` runs it through the existing tiers, banks the receipts, grades them
through features.grade, and files the proof beads through proofqueue. Kinds:
decision (run_suite + paired), memory (legs + memory_verdict), generation
(corpus + %pane's assertions), agent (no backend yet: planned, refused live).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "registries" / "proofs"
RECEIPTS_DIR = REPO_ROOT / "docs" / "evidence" / "receipts"
NEGATIVE_EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "NEGATIVE_EVIDENCE.md"

KINDS = ("decision", "generation", "memory", "agent")
STAGES = ("screen", "proof")
REGIMES = ("side",)

REQUIRED_FIELDS = ("feature", "kind", "candidates", "dataset", "assertions", "stage")
OPTIONAL_FIELDS = ("repeats", "seed", "regime", "wins", "wins_needed", "fresh_per_round",
                   "allow_errors", "blocked", "profile", "hosted", "baseline", "arms", "verdicts",
                   "legs", "main_model", "smol_model", "tiers", "pairs", "mem_rounds", "wait_idle",
                   "description", "gold_set", "analysis_sha", "retry_of", "new_hypothesis")
# Assertion types with an evaluator per kind (an assertion with no evaluator is a load error, never a
# silent pass). decision reuses its tier verdicts; memory evaluates verdict-delta rows and leg design sizes;
# generation runs the named FEATURE_CHECKS over produced outputs. agent has no evaluator yet.
RATE_TESTS = ("fisher-pooled", "pooled-fisher", "cmh")
LATENCY_TESTS = ("unpaired-t-log", "welch-t-log", "drift-paired-t")


def _memory_metrics() -> set[str]:
    """Every metric a memory verdict row can carry: quality keys plus win keys."""
    from .workloads import MEMORY_QUALITY, MEMORY_WINS

    return set(MEMORY_QUALITY) | {key for key, _ in MEMORY_WINS}


_CHECK_KEY = {"check_title": "titles", "check_commit_message": "commit-messages",
              "check_skill_compression": "skill-description-compression"}


def _generation_check_key(name) -> str | None:
    """The FEATURE_CHECKS key for a check-function name, or None when no check answers to it."""
    return _CHECK_KEY.get(name) if isinstance(name, str) else None


def _unknown_assertion(kind: str, feature: str, assertion: dict) -> str | None:
    """Why (kind, assertion) has no evaluator, or None when it dispatches. Memory metric assertions name a
    rate or latency test family; memory deterministic is the sized-design leg count; generation deterministic
    names a generation check, a pass rate, or the builtin comparison."""
    type_ = assertion.get("type")
    params = assertion.get("params", {})
    if not isinstance(params, dict):
        return f"params must be an object, not {params!r}"
    if kind == "decision":
        return None if type_ in ("paired", "metric") else f"decision assertions cannot be {type_!r}"
    if kind == "memory":
        if type_ == "metric":
            metric = params.get("metric")
            if metric not in _memory_metrics():
                return f"unknown memory metric {metric!r}"
            if params.get("test") not in (*RATE_TESTS, *LATENCY_TESTS):
                return f"unknown memory test {params.get('test')!r}"
            if params.get("direction") not in ("not-worse", "lower"):
                return f"memory metric direction must be not-worse or lower, not {params.get('direction')!r}"
            margin = params.get("margin", 0)
            if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0:
                return f"margin must be a number >= 0, not {margin!r}"
            return None
        if type_ == "deterministic" and assertion.get("id") == "sized-design":
            for key in ("min_pairs", "min_recall_facts_b", "min_derail_facts_b"):
                value = params.get(key, 0)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    return f"sized-design {key} must be a number >= 0, not {value!r}"
            return None
        return f"memory assertions cannot be {type_!r} id {assertion.get('id')!r}"
    if kind == "generation":
        if type_ != "deterministic":
            return f"generation assertions cannot be {type_!r} id {assertion.get('id')!r}"
        aid = assertion.get("id")
        if aid == "beats-builtin":
            if feature not in _CHECK_KEY.values():
                return f"beats-builtin needs a checked feature, not {feature!r}"
            return None
        if _generation_check_key(params.get("check")) is None:
            return f"generation assertion {aid!r} names no check"
        if aid == "compression-pass-rate":
            minimum = params.get("min_pass_rate", 1)
            if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
                return f"min_pass_rate must be a number, not {minimum!r}"
            if "only" in params and params["only"] != "answerable":
                return f"pass rate covers answerable outputs only, not {params['only']!r}"
        return None
    return None


class ProveError(RuntimeError):
    """A spec or run refusal: invalid spec, uncommitted spec, unresolvable candidate, missing backend."""


def _fail(path, detail: str) -> ProveError:
    return ProveError(f"{path}: {detail}")


def analysis_sha(repo: Path = REPO_ROOT) -> str:
    """Digest verdict implementation files that pre-registered specs may pin."""
    digest = hashlib.sha256()
    for name in ("localbench/stats.py", "localbench/generation.py"):
        source = repo / name
        try:
            digest.update(name.encode() + b"\0" + source.read_bytes())
        except OSError as exc:
            raise ProveError(f"analysis source unavailable: {source}: {exc}") from None
    return digest.hexdigest()


def check_analysis_sha(doc: dict, repo: Path = REPO_ROOT) -> None:
    pinned = doc.get("analysis_sha")
    if pinned is None:
        if doc.get("stage") == "proof":
            raise ProveError("proof spec missing analysis_sha; register analysis_sha before proving")
        return
    if not isinstance(pinned, str) or len(pinned) != 64 or any(c not in "0123456789abcdef" for c in pinned):
        raise ProveError("analysis_sha must be 64 lowercase hex characters")
    actual = analysis_sha(repo)
    if pinned != actual:
        raise ProveError(f"analysis_sha mismatch: spec {pinned}, installed {actual}; re-register the spec")


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
    if "wins_needed" in doc:
        need, wins = doc["wins_needed"], doc.get("wins", [])
        if isinstance(need, bool) or not isinstance(need, int) or need < 1:
            raise _fail(target, "wins_needed must be an int >= 1")
        if not isinstance(wins, list) or need > len(wins):
            raise _fail(target, f"wins_needed {need} needs that many declared wins")
    if "retry_of" in doc or "new_hypothesis" in doc:
        retry_of, hypothesis = doc.get("retry_of"), doc.get("new_hypothesis")
        if not isinstance(retry_of, str) or not retry_of.strip():
            raise _fail(target, "retry_of must name a REJECT ledger heading")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise _fail(target, "new_hypothesis must be a non-empty rationale when retry_of is set")
    for assertion in doc["assertions"]:
        why = _unknown_assertion(doc["kind"], doc["feature"], assertion)
        if why is not None:
            raise _fail(target, f"assertion {assertion['id']!r} has no evaluator: {why}")
    if "gold_set" in doc:
        reference = doc["gold_set"]
        if doc["kind"] != "generation":
            raise _fail(target, "gold_set is only valid on generation specs")
        if not isinstance(reference, dict) or set(reference) != {"file", "sha256"}:
            raise _fail(target, "gold_set needs exactly {file, sha256}")
        file_name, digest = reference.get("file"), reference.get("sha256")
        if (not isinstance(file_name, str) or Path(file_name).name != file_name
                or not file_name.endswith(".json")):
            raise _fail(target, "gold_set file must be a JSON basename beside the corpus")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise _fail(target, "gold_set sha256 must be 64 lowercase hex characters")
    check_analysis_sha(doc)
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


def _git(repo: Path, *argv: str, strip_stdout: bool = True) -> tuple[int, str]:
    env = {key: value for key, value in os.environ.items() if key not in _HOOK_ENV}
    proc = subprocess.run(["git", *argv], cwd=repo, env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    return proc.returncode, proc.stdout.strip() if strip_stdout else proc.stdout


def spec_commit(path: str | Path, repo: Path = REPO_ROOT) -> str:
    """The commit that froze this spec: tracked, clean, and its last-touching sha.

    Pre-registration is the commit: editing a spec after data is impossible to hide (forbidden
    pattern 9), so prove() refuses anything else before planning anything."""
    target = str(Path(path).expanduser())
    root = Path(repo).expanduser()
    rc, _ = _git(root, "ls-files", "--error-unmatch", target)
    if rc != 0:
        raise ProveError(f"{target}: spec is uncommitted; commit it for %pane's review first")
    rc, out = _git(root, "status", "--porcelain", "--", target)
    if rc != 0:
        raise ProveError(f"{target}: cannot status the spec")
    if out:
        raise ProveError(f"{target}: spec modified since its commit; commit it for %pane's review first")
    rc, sha = _git(root, "log", "-1", "--format=%H", "--", target)
    if rc != 0 or not sha:
        raise ProveError(f"{target}: no commit touches the spec")
    return sha


def resolve_candidate(candidate: dict) -> dict:
    """A candidate route spec to {backend, model}: ollama:/laya: routes, or a preset's model from its ops."""
    from . import features, presets

    if not isinstance(candidate, dict):
        raise ProveError(f"candidate is not an object: {candidate!r}")
    if "builtin" in candidate:
        builtin = candidate["builtin"]
        if not isinstance(builtin, str) or not builtin:
            raise ProveError(f"candidate builtin {builtin!r}: name required")
        return {"backend": "builtin", "model": builtin, "via": "builtin"}
    if "route" in candidate:
        route = candidate["route"]
        for prefix, backend in (("ollama:", "ollama"), ("mlx-serve:", "mlx-serve"), ("laya:", "laya")):
            if isinstance(route, str) and route.startswith(prefix):
                model = route[len(prefix):]
                if not model:
                    raise ProveError(f"candidate route {route!r} names no model")
                if backend == "mlx-serve":
                    model = str(Path(model).expanduser())
                return {"backend": backend, "model": model, "via": "route"}
        raise ProveError(f"candidate route {route!r}: ollama:, laya:, or mlx-serve: only")
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
    raise ProveError(f"candidate needs route, builtin, or preset: {candidate!r}")


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
    (generation assertions land with %pane)."""
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


def evaluate_memory_metric(assertion: dict, row) -> tuple[bool, str]:
    """One memory metric assertion against its verdict-delta row. Returns (passed, detail).

    The row already ran the right test (Fisher/CMH pooled rates, Welch/drift-paired log latencies);
    the assertion checks its judgement plus effect: not-worse needs no loss within margin, lower needs
    a measured gain. Anything else (including an unmeasured row or a test family the row cannot
    satisfy) fails, never passes silently."""
    params = assertion.get("params", {})
    metric, direction = params.get("metric"), params.get("direction")
    if not isinstance(row, dict):
        return False, f"{assertion['id']}: no {metric} row in the verdict"
    test = params.get("test")
    if test in RATE_TESTS and row.get("class") != "quality":
        return False, f"{assertion['id']}: {test} needs a quality row, not {row.get('class')}"
    if test in LATENCY_TESTS and row.get("class") != "win":
        return False, f"{assertion['id']}: {test} needs a win row, not {row.get('class')}"
    judgement = row.get("judgement")
    if judgement == "unmeasured":
        return False, f"{assertion['id']}: {metric} unmeasured"
    if direction == "not-worse":
        margin = params.get("margin", 0)
        delta = row.get("delta")
        if judgement == "loss" or not isinstance(delta, (int, float)) or delta < -margin:
            return False, f"{assertion['id']}: loss beyond margin (delta {delta}, {judgement})"
        return True, f"{assertion['id']}: no loss (delta {delta:+.3f}, {judgement})"
    if direction == "lower":
        if judgement != "gain":
            return False, f"{assertion['id']}: no measured gain ({judgement})"
        return True, f"{assertion['id']}: gain (delta {row.get('delta'):+.3f})"
    raise ProveError(f"assertion {assertion['id']!r}: unknown memory direction {direction!r}")


def _memory_arm_facts(legs: list[dict], metric: str) -> int | None:
    """Total observation count behind a rate metric over one arm's legs; None when any leg lacks n."""
    total = 0
    for leg in legs:
        entry = (leg.get("metrics") or {}).get(metric) or {}
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            return None
        total += n
    return total


def evaluate_sized_design(assertion: dict, candidate_legs: list[dict],
                           baseline_legs: list[dict]) -> tuple[bool, str]:
    """The study-size gate: min_pairs legs per arm (each side needs its own A/A null) plus fact totals
    for recall and derail on the baseline arm (the _b params count the arm the verdict measures against).
    Returns (passed, detail)."""
    params = assertion.get("params", {})
    wants = (("min_pairs", None), ("min_recall_facts_b", "mem.recall.hit_rate"),
             ("min_derail_facts_b", "mem.derail.ok_rate"))
    shorts = []
    for key, metric in wants:
        want = params.get(key, 0)
        if metric is None:
            have = min(len(candidate_legs), len(baseline_legs))
        else:
            have = _memory_arm_facts(baseline_legs, metric)
            if have is None:
                return False, f"{assertion['id']}: {metric} count unrecorded on the baseline arm"
        if have < want:
            shorts.append(f"{key} {have} < {want}")
    if shorts:
        return False, f"{assertion['id']}: undersized ({'; '.join(shorts)})"
    return True, f"{assertion['id']}: sized ({len(candidate_legs)}+{len(baseline_legs)} legs)"


def _split_legs_by_arm(spec: dict, legs: list[dict]) -> dict[str, list[dict]]:
    """Legs keyed by verdict arm from label prefixes (`ab_a1` falls in `ab_a`); a leg matching no arm,
    and a verdict entry naming no arms, refuse instead of silently dropping legs."""
    arms_needed: list[str] = []
    for entry in spec["verdicts"]:
        for arm in (entry.get("candidate"), entry.get("baseline")):
            if not arm:
                raise ProveError(f"verdict entry names no candidate and baseline: {entry!r}")
            if arm not in arms_needed:
                arms_needed.append(arm)
    by_arm: dict[str, list[dict]] = {}
    for leg in legs:
        label = leg.get("label", "")
        hits = [arm for arm in arms_needed if label == arm or label.startswith(arm)]
        if not hits:
            raise ProveError(f"leg {leg.get('id', label)!r} matches no verdict arm {arms_needed}")
        by_arm.setdefault(max(hits, key=len), []).append(leg)
    return by_arm


def _candidate_slug(candidate: dict) -> str:
    return str(candidate.get("route") or candidate.get("preset") or candidate.get("arm", "candidate"))


def candidate_evidence(spec: dict, candidate: dict, suite, *, run_suite=None,
                           corpora=None) -> dict:
    """Run one decision candidate through run_suite + paired, returning evidence (no banking)."""
    from . import decision

    resolved = resolve_candidate(candidate)
    run = run_suite or _live_decision_run
    receipt = run.__call__(spec, candidate, resolved, suite)
    table = decision.paired({"kind": "run", "run": {"decision": {"local": receipt["run"]["decision"]["local"]}}},
                            corpora=corpora)
    arm = receipt["run"]["decision"]["local"]
    metrics = {name: entry.get("value") for name, entry in arm.get("metrics", {}).items()}
    spreads = {name: entry.get("spread") for name, entry in arm.get("metrics", {}).items()
               if isinstance(entry, dict)}
    return {"candidate": candidate, "resolved": resolved, "receipt": receipt,
            "paired": table["types"], "metrics": metrics, "spreads": spreads}


def screen_verdict(receipt: dict, *, problems: list[str], error_kinds: dict,
                   allow_errors: float) -> str:
    """Void screens only when infrastructure errors exceed their budget; reject failed quality gates."""
    if any(row.get("infra_error_rate", 0.0) > allow_errors for row in error_kinds.values()):
        return "VOID"
    run = receipt.get("run") or {}
    if "decision.early_stop" in (run.get("details") or {}):
        return "REJECT"
    if any(entry.get("level") == "MUST" and entry.get("verdict") == "FAIL"
           for entry in (run.get("conformance") or {}).values()):
        return "REJECT"
    return "REJECT" if problems else "ADVANCE"




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

    by_arm = _split_legs_by_arm(spec, legs)
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


def evaluate_memory_entry(spec: dict, verdict: dict, by_arm: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    """Every spec assertion matching this verdict entry (an assertion candidate unset matches all entries),
    over its verdict-delta rows plus arm legs for sized-design. Returns (problems, passed)."""
    problems, passed = [], []
    candidate, baseline = verdict["candidate"], verdict["baseline"]
    deltas = (((verdict.get("receipt") or {}).get("run") or {}).get("memory", {}).get("compare", {})
              or {}).get("deltas", {})
    for assertion in spec["assertions"]:
        if assertion.get("candidate", candidate) != candidate:
            continue
        kind = assertion.get("type")
        if kind == "metric":
            ok, detail = evaluate_memory_metric(assertion, deltas.get(assertion.get("params", {}).get("metric")))
        elif kind == "deterministic":
            ok, detail = evaluate_sized_design(assertion, by_arm.get(candidate, []), by_arm.get(baseline, []))
        else:
            raise ProveError(f"assertion {assertion.get('id')!r}: no memory evaluator for {kind!r}")
        (passed if ok else problems).append(detail)
    return problems, passed


def _generation_candidate_key(candidate: dict) -> str:
    return str(candidate.get("route") or ("builtin:" + str(candidate.get("builtin", "?"))))


def _generation_corpus(spec: dict) -> dict:
    """Load and pin the generation corpus: manifest + items under corpora/generation/<suite>."""
    from .decision import CORPORA

    name = spec["dataset"]["suite"]
    root = (CORPORA / "generation" / name).expanduser()
    manifest_path, items_path = root / "manifest.json", root / "items.jsonl"
    if not manifest_path.is_file() or not items_path.is_file():
        raise ProveError(f"generation corpus {name}: no manifest/items under {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProveError(f"{manifest_path}: not JSON ({exc})") from None
    if manifest.get("items_sha256") != spec["dataset"]["items_sha256"]:
        raise ProveError(f"corpus {name} items_sha256 {manifest.get('items_sha256')} does not match "
                         f"the spec pin {spec['dataset']['items_sha256']}")
    items = []
    for lineno, line in enumerate(items_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ProveError(f"{items_path}:{lineno}: not JSON ({exc})") from None
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ProveError(f"{items_path}:{lineno}: no item id")
        items.append(item)
    return {"manifest": manifest, "items": items, "root": root}
def load_gold_set(spec: dict, *, corpus_root: Path | None = None) -> dict:
    """Load the spec-pinned blind PPI gold set beside its generation corpus.

    Format v1: pair_ids are unique corpus item ids; labels maps each pair id to
    candidate, baseline, or tie; order_seed freezes blind presentation; labeler
    names the trusted label source; kappa and agreement record independent review.
    """
    reference = spec.get("gold_set")
    if reference is None:
        raise ProveError("generation spec has no gold_set pin")
    if corpus_root is None:
        from .decision import CORPORA
        corpus_root = CORPORA / "generation" / spec["dataset"]["suite"]
    path = Path(corpus_root) / reference["file"]
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != reference["sha256"]:
            raise ProveError(f"gold set {path}: sha256 {digest} does not match spec pin")
        doc = json.loads(raw)
    except OSError as exc:
        raise ProveError(f"gold set {path}: unreadable ({exc})") from None
    except ValueError as exc:
        raise ProveError(f"gold set {path}: not JSON ({exc})") from None
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise ProveError(f"gold set {path}: expected format version 1")
    pair_ids, labels = doc.get("pair_ids"), doc.get("labels")
    if (not isinstance(pair_ids, list) or not pair_ids
            or any(not isinstance(item_id, str) or not item_id for item_id in pair_ids)
            or len(set(pair_ids)) != len(pair_ids)):
        raise ProveError(f"gold set {path}: pair_ids must be nonempty, unique strings")
    labeler = doc.get("labeler")
    if (not isinstance(labeler, list) or len(labeler) != 2
            or any(not isinstance(name, str) or not name.strip() for name in labeler)
            or labeler[0] == labeler[1]):
        raise ProveError(f"gold set {path}: labeler must name two independent reviewers")
    if not isinstance(labels, dict) or set(labels) != set(pair_ids):
        raise ProveError(f"gold set {path}: labels must cover every pair_id exactly once")
    valid = ("candidate", "baseline", "tie")
    human_votes = []
    for item_id, row in labels.items():
        if not isinstance(row, dict) or row.get("label") not in valid:
            raise ProveError(f"gold set {path}: invalid label row for {item_id}")
        if row.get("source") == "human":
            votes = row.get("votes")
            if not isinstance(votes, list) or len(votes) != 2 or any(vote not in valid for vote in votes):
                raise ProveError(f"gold set {path}: human pair {item_id} needs two blind votes")
            if votes[0] == votes[1]:
                if row["label"] != votes[0]:
                    raise ProveError(f"gold set {path}: agreed votes disagree with final label for {item_id}")
            elif row.get("adjudicated") != row["label"]:
                raise ProveError(f"gold set {path}: disagreement for {item_id} needs matching adjudication")
            human_votes.append(votes)
        elif row.get("source") != "deterministic" or "votes" in row or "adjudicated" in row:
            raise ProveError(f"gold set {path}: pair {item_id} needs human votes or deterministic source")
    if not human_votes:
        raise ProveError(f"gold set {path}: at least one pair needs independent human labels")
    if not isinstance(doc.get("order_seed"), str) or not doc["order_seed"]:
        raise ProveError(f"gold set {path}: order_seed must be nonempty")
    observed = sum(a == b for a, b in human_votes) / len(human_votes)
    marginals = []
    for column in (0, 1):
        marginals.append({label: sum(votes[column] == label for votes in human_votes) / len(human_votes)
                          for label in valid})
    expected = math.fsum(marginals[0][label] * marginals[1][label] for label in valid)
    computed_kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    agreement, kappa = doc.get("agreement"), doc.get("kappa")
    if (not isinstance(agreement, (int, float)) or isinstance(agreement, bool)
            or not math.isfinite(agreement) or not 0.0 <= agreement <= 1.0):
        raise ProveError(f"gold set {path}: agreement must be finite in [0, 1]")
    if agreement < 0.90:
        raise ProveError(f"gold set {path}: independent blind agreement must be at least 90%")
    if abs(agreement - observed) > 1e-9:
        raise ProveError(f"gold set {path}: agreement does not match independent human votes")
    if (not isinstance(kappa, (int, float)) or isinstance(kappa, bool)
            or not math.isfinite(kappa) or not -1.0 <= kappa <= 1.0
            or abs(kappa - computed_kappa) > 1e-6):
        raise ProveError(f"gold set {path}: kappa does not match independent human votes")
    return doc


def normalize_generation_outputs(
        outputs: dict) -> tuple[dict[str, list[dict]], dict[str, dict], dict[str, dict]]:
    """Separate outcomes, calibrated judge evidence, and pinned backend identity."""
    from . import generation

    if not isinstance(outputs, dict):
        raise ProveError("generation outputs must map candidate keys to evidence")
    normalized, calibrations, backend_pins = {}, {}, {}
    for key, value in outputs.items():
        if not isinstance(key, str) or not key:
            raise ProveError(f"generation output key is invalid: {key!r}")
        if isinstance(value, list):
            normalized[key] = value
            continue
        if not isinstance(value, dict):
            raise ProveError(f"generation output {key}: expected outcomes or evidence object")
        if "raw_win_rate" in value or "win_rate" in value:
            raise ProveError(f"generation output {key}: raw win rate is refused; provide judge_pairs")
        allowed = {"outcomes", "judge_pairs", "backend_pins"}
        if set(value) - allowed:
            raise ProveError(f"generation output {key}: unknown evidence fields "
                             f"{sorted(set(value) - allowed)}")
        outcomes = value.get("outcomes")
        if not isinstance(outcomes, list):
            raise ProveError(f"generation output {key}: outcomes must be a list")
        normalized[key] = outcomes
        if "judge_pairs" in value:
            try:
                calibration = generation.pairwise_calibration(value["judge_pairs"])
                generation.require_pairwise_calibration({"judge_calibration": calibration})
            except ValueError as exc:
                raise ProveError(f"generation output {key}: {exc}") from exc
            calibrations[key] = calibration
        if "backend_pins" in value:
            pins = value["backend_pins"]
            required_pins = ("model", "model_dir", "model_digest", "backend_version", "backend_sha")
            if (not isinstance(pins, dict) or pins.get("backend") != "mlx-serve"
                    or any(not isinstance(pins.get(field), str) or not pins[field]
                           for field in required_pins)):
                raise ProveError(f"generation output {key}: incomplete mlx-serve backend pins")
            backend_pins[key] = pins
    return normalized, calibrations, backend_pins


def _generation_provenance(resolved: dict, key: str, backend_pins: dict, *,
                           created: str, host: dict, rev: str, module_sha: str,
                           model_digest: str | None = None) -> dict:
    """Carry runtime identity and installed model digest into a generation receipt."""
    runtime_pins = backend_pins.get(key) or {}

    pins = {
        "backend": runtime_pins.get("backend", resolved["backend"]),
        "backend_version": runtime_pins.get("backend_version"),
        "backend_sha": runtime_pins.get("backend_sha"),
        "model": runtime_pins.get("model", resolved["model"]),
        "model_digest": runtime_pins.get("model_digest") or model_digest,
        "candidate_route": key,
        "macos_build": host.get("macos_build"),
        "host_id": host.get("host_id"),
    }
    for field in ("model_dir", "backend_args"):
        if field in runtime_pins:
            pins[field] = runtime_pins[field]
    return {
        "label": "generation",
        "pins": pins,
        "fingerprint": {**pins, "candidate": key, "omp_module_sha": module_sha},
        "tiers": ["generation"],
        "created": created,
        "localbench_rev": rev,
    }


def _generation_baseline(context: dict | None, feature: str) -> dict:
    """Name the profile incumbent; custom graders retain an explicit unknown baseline."""
    if context is None:
        return {"kind": "unknown", "id": "unknown"}
    if not context.get("incumbent_known"):
        return {"kind": "unknown", "id": "unknown"}
    inc = context["inc"]
    if inc["selector"]:
        return {"kind": "route" if inc["local"] else "hosted", "id": inc["selector"]}
    if inc["setting"] is not None:
        return {"kind": "fixed", "id": inc["setting"]}
    return {"kind": "builtin", "id": feature}


def _check_generation_outputs(assertion: dict, check: str, outputs: list[dict]) -> list[dict]:
    """Per-item check violations for produced outputs: [{item_id, violations, answerable}]. A missing
    output or a missing text (the error outcome, never an empty string standing in for one) violates;
    unanswerable items are listed but never fail. A raising check violates that item, not the run.
    Outputs carrying pre-computed violations (the replay producer ran FEATURE_CHECKS with the right
    context) keep them; the asserted check runs only otherwise."""
    from . import generation

    fn = generation.FEATURE_CHECKS[check]
    rows = []
    for output in outputs:
        item_id = output.get("item_id")
        answerable = output.get("answerable", True)
        text = output.get("text")
        if text is None:
            rows.append({"item_id": item_id, "violations": ["no text (error outcome)"],
                         "answerable": answerable})
            continue
        if isinstance(output.get("violations"), list):
            rows.append({"item_id": item_id, "violations": list(output["violations"]),
                         "answerable": answerable})
            continue
        try:
            violations = fn(text, output.get("context") or {})
        except Exception as exc:
            violations = [f"check raised {type(exc).__name__}: {exc}"]
        rows.append({"item_id": item_id, "violations": list(violations), "answerable": answerable})
    return rows


def replay_generation_outputs(spec: dict, items: list[dict], **kw) -> dict[str, list[dict] | dict]:
    """Replay corpus items via the selected loopback backend; carry MLX fingerprint pins."""
    from . import generation

    out = {}
    for candidate in spec["candidates"]:
        key = _generation_candidate_key(candidate)
        evidence: dict = generation.run_candidate(spec, candidate, items, **kw)
        rows = [{"item_id": o.get("id"), "text": o.get("text"), "answerable": True,
                 "violations": list(o.get("violations") or [])} for o in evidence["outcomes"]]
        if evidence.get("pins") is not None:
            out[key] = {"outcomes": rows, "backend_pins": evidence["pins"]}
        else:
            out[key] = rows
    return out


def evaluate_generation_assertions(spec: dict, key: str, outputs: list[dict],
                                   all_outputs: dict[str, list[dict]]) -> tuple[list[str], list[str], dict]:
    """Every spec assertion for one candidate's produced outputs. Check-bearing assertions (shape, retention,
    pass-rate) run the named check; beats-builtin compares violation rates against the builtin arm's rows.
    Returns (problems, passed, rows_by_assertion)."""
    problems, passed, by_id = [], [], {}
    for assertion in spec["assertions"]:
        aid, params = assertion.get("id"), assertion.get("params", {})
        if aid == "beats-builtin":
            from . import generation

            if spec["feature"] not in generation.FEATURE_CHECKS:
                raise ProveError(f"assertion {aid!r}: feature {spec['feature']!r} names no check")
            rows = _check_generation_outputs(assertion, spec["feature"], outputs)
            builtin = next((c for c in spec["candidates"] if "builtin" in c), None)
            if builtin is None:
                raise ProveError(f"assertion {aid!r}: no builtin candidate to compare against")
            base_rows = _check_generation_outputs(assertion, spec["feature"],
                                                  all_outputs.get(_generation_candidate_key(builtin), []))
            cand_rate = _violation_rate(rows)
            base_rate = _violation_rate(base_rows)
            by_id[aid] = {"candidate": rows, "builtin": base_rows}
            if cand_rate is None or base_rate is None:
                problems.append(f"{aid}: no answerable outputs to compare")
            elif cand_rate < base_rate:
                passed.append(f"{aid}: violation rate {cand_rate:.3f} < builtin {base_rate:.3f}")
            else:
                problems.append(f"{aid}: violation rate {cand_rate:.3f} not lower than builtin {base_rate:.3f}")
            continue
        check = _generation_check_key(params.get("check"))
        if check is None:
            raise ProveError(f"assertion {aid!r}: no generation check answers to it")
        rows = _check_generation_outputs(assertion, check, outputs)
        by_id[aid] = rows
        if aid == "compression-pass-rate":
            rate = _violation_rate(rows, invert=True)
            minimum = params.get("min_pass_rate", 1)
            if rate is None:
                problems.append(f"{aid}: no answerable outputs")
            elif rate >= minimum:
                passed.append(f"{aid}: pass rate {rate:.3f} >= {minimum}")
            else:
                problems.append(f"{aid}: pass rate {rate:.3f} < {minimum}")
            continue
        failed = [r for r in rows if r["answerable"] and r["violations"]]
        if failed:
            shown = ", ".join(f"{r['item_id']}: {r['violations'][0]}" for r in failed[:3])
            problems.append(f"{aid}: {len(failed)}/{len([r for r in rows if r['answerable']])} answerable "
                            f"outputs violate ({shown})")
        else:
            passed.append(f"{aid}: all {len([r for r in rows if r['answerable']])} answerable outputs pass")
    return problems, passed, by_id


def _violation_rate(rows: list[dict], invert: bool = False) -> float | None:
    """Fraction of answerable rows with violations (or passing, inverted); None when no answerable rows."""
    answerable = [r for r in rows if r["answerable"]]
    if not answerable:
        return None
    bad = sum(1 for r in answerable if r["violations"])
    return 1 - bad / len(answerable) if invert else bad / len(answerable)


def grade_proof(spec: dict, proof: dict, model: str | None, digest: str | None,
                inc: dict | None = None) -> tuple[str, str]:
    """Grade one banked proof through features.grade (the single grading rule)."""
    from . import features

    rows = features.load()
    if spec["feature"] not in {row["feature"] for row in rows}:
        raise ProveError(f"feature {spec['feature']!r} is not in registries/features.tsv")
    return features.grade(proof, _module_sha(spec["feature"]), model, digest, inc)

def update_beads(spec: dict, outcomes: list[dict], br=None) -> dict:
    """Close proven non-screen routes; screen outcomes always comment with their verdict."""
    actions = []
    for outcome in outcomes:
        title = f"prove {spec['feature']} on {outcome['route']}"
        screen_verdict = outcome.get("screen_verdict")
        if outcome["grade"] == "PROVEN" and not screen_verdict:
            actions.append({"op": "close", "title": title, "unit": {"feature": spec["feature"]},
                            "reason": f"route proven: receipt {outcome['receipt']} grades PROVEN"})
        else:
            note = (f"proof {Path(outcome['spec']).stem}: {outcome['grade']} "
                    f"({outcome['reason']})")
            if screen_verdict:
                note += f" screen_verdict={screen_verdict}"
            actions.append({"op": "comment", "title": title, "unit": {"feature": spec["feature"]},
                            "note": note})
    return _apply_to_beads(actions, br)

def _apply_to_beads(actions: list[dict], br=None) -> dict:
    """Match actions to open beads by exact title. A lookalike title is never a match: commenting on or
    closing the wrong bead is worse than reporting the action missing."""
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
    a spec when its title is exactly `prove <feature> on <route>` for one of the spec's candidate routes."""
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
        wanted = {f"prove {spec['feature']} on {r}" for r in routes}
        if any(title in wanted for title in titles):
            due.append((path, spec))
    return sorted(due, key=_due_order_key)


def _due_order_key(item: tuple[str | Path, dict]) -> tuple[int, str]:
    """Order --due work by declared cheap-round shape only; never changes proof execution."""
    path, spec = item
    repeats = spec.get("repeats", 1)
    pairs = spec.get("pairs", 1)
    mem_rounds = spec.get("mem_rounds", 1)
    def positive_int(value: object) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1
    cost = positive_int(repeats) * positive_int(pairs) * positive_int(mem_rounds)
    return cost, str(path)


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
    """Report per-question failure and infrastructure rates; failures over allow_errors are problems."""
    allowed = float(spec.get("allow_errors", 0.0))
    infra_errors = {"http", "timeout", "unavailable"}
    outs = {o["id"]: o for o in outcomes if isinstance(o, dict) and isinstance(o.get("id"), str)}
    totals: dict[str, int] = {}
    for item in suite.items:
        for question in item["questions"].values():
            totals[question["type"]] = totals.get(question["type"], 0) + 1
    failed: dict[str, list[str]] = {}
    infra_failed: dict[str, list[str]] = {}
    for item in suite.items:
        outcome = outs.get(item["id"], {})
        if not outcome.get("ok", False):
            error = outcome.get("error")
            for name, question in item["questions"].items():
                kind = question["type"]
                item_id = f"{item['id']}.{name}"
                failed.setdefault(kind, []).append(item_id)
                if isinstance(error, str) and error in infra_errors:
                    infra_failed.setdefault(kind, []).append(item_id)
    kinds, problems = {}, []
    for kind in sorted(totals):
        ids = sorted(failed.get(kind, []))
        infra_ids = sorted(infra_failed.get(kind, []))
        rate = len(ids) / totals[kind]
        kinds[kind] = {"error_rate": rate, "items": ids,
                       "infra_error_rate": len(infra_ids) / totals[kind], "infra_items": infra_ids}
        if rate > allowed:
            problems.append(f"error rate {rate:.3f} on {kind} exceeds allow_errors {allowed}")
    return kinds, problems

def list_specs(directory: Path = SPEC_DIR) -> list[Path]:
    """Every proof spec file, sorted. A directory that is missing is no specs, not an error."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return sorted(root.glob("*__*.json"))


def plan_steps(spec: dict, commit: str, *, omp_frozen: bool = False) -> list[tuple[str, str]]:
    """Dry-run lines for one spec: what prove() would run, bank, grade and file. No inference, no writes.

    Memory plans require an explicitly frozen omp because their live legs otherwise can cross an
    omp update between arms. Blocked specs refuse before any plan is constructed.
    """
    if spec.get("blocked"):
        raise ProveError(f"spec is blocked: {spec['blocked'].get('reason', spec['blocked'])}")
    if spec.get("kind") == "memory" and not omp_frozen:
        raise ProveError("memory proof planning requires --omp-frozen")
    capabilities = {"decision": {"ollama", "laya"},
                    "generation": {"ollama", "builtin", "mlx-serve"},
                    "memory": {"ollama"}}
    for candidate in spec["candidates"]:
        resolved = resolve_candidate(candidate)
        if resolved["backend"] not in capabilities.get(spec["kind"], set()):
            raise ProveError(f"{spec['kind']} proof does not support backend {resolved['backend']!r}; "
                             f"supported: {sorted(capabilities.get(spec['kind'], set()))}")
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
            runs = 2 * spec.get("pairs", 1) + 1
            design = (f"{runs} interleaved A/B runs ({spec.get('pairs', 1)} pairs plus final A)")
            campaign = (f"{spec.get('mem_rounds', 1)} rounds, {spec.get('repeats', 1)} repeats")
            steps.append((f"launch mem/sess legs for arms {sorted(spec.get('arms', {}))} "
                          f"from {source}; {design}; {campaign}",
                          "existing mem/sess tiers; legs split by ab_a/ab_b label prefix"))
        elif spec["kind"] == "generation":
            steps.append((f"run generation candidate {slug} on the corpus via "
                          "outputs_fn (replay_generation_outputs)",
                          "per-candidate corpus outputs; deterministic assertions run the named checks"))
        else:
            steps.append((f"run agent candidate {slug}", "no agent tier exists yet"))
        steps.append((f"bank prove__{spec['kind']}__{spec['feature']}__{slug} receipt "
                      f"(spec commit {commit[:12]})",
                      "receipts stay in-repo under docs/evidence/receipts/"))
        steps.append((f"grade {spec['feature']} on {slug} through features.grade; "
                      "comment on or close the proof bead",
                      "proofqueue: PROVEN closes citing the receipt, anything else comments"))
    return steps


def _proof_ollama_preflight(spec: dict, suite, base_url: str) -> dict | None:
    """Check every Ollama decision candidate against one unchanged resident baseline before any item runs."""
    from . import decision

    candidates = [(candidate, resolve_candidate(candidate)) for candidate in spec["candidates"]]
    ollama = [(candidate, resolved) for candidate, resolved in candidates if resolved["backend"] == "ollama"]
    if not ollama:
        return None
    try:
        required = decision.required_context(suite)["required_tokens"]
        initial = decision.resident_contexts(base_url)
        if initial is None:
            raise ProveError("decision admission preflight: Ollama resident state is unreachable")
        cap = None
        for candidate, resolved in ollama:
            model = resolved["model"]
            shipped = decision.check_shipped_context(base_url, model, required)
            tag = decision._tagged(model)
            if tag in initial:
                context = initial[tag]
                sufficient = context == shipped if shipped is not None else context is not None and context >= required
                if not sufficient:
                    raise ProveError(f"decision admission preflight candidate {model}: existing runner context "
                                     f"{context} cannot satisfy {required} tokens without reloading a pre-existing "
                                     "resident")
            else:
                if cap is None:
                    cap_info = decision.loaded_model_cap()
                    cap = cap_info.get("cap") if isinstance(cap_info, dict) else None
                    if type(cap) is not int or cap < 1:
                        raise ProveError("decision admission preflight: Ollama loaded-model cap is unreadable")
                if len(initial) >= cap:
                    raise ProveError(f"decision admission preflight candidate {model}: {len(initial)} residents already "
                                     f"meet Ollama's loaded-model cap ({cap}); prove will not evict a pre-existing "
                                     "resident")
    except decision.ContextError as exc:
        raise ProveError(f"decision admission preflight: {exc}") from None
    return {"base_url": base_url, "initial": initial}


def _release_proof_ollama_candidate(base_url: str, model: str, initial: dict,
                                    context_action: str | None) -> None:
    """Unload only a runner the proof loaded, after the gateway guard, then prove the baseline was restored."""
    from . import decision, gateway

    tag = decision._tagged(model)
    try:
        current = decision.resident_contexts(base_url)
        if current is None:
            raise ProveError(f"decision admission cleanup for {model}: Ollama resident state is unreachable")
        if tag not in initial and tag in current:
            if context_action not in {"loaded", "reloaded"}:
                raise ProveError(f"decision admission cleanup for {model}: runner appeared without a proof load "
                                 "receipt; refusing to unload an unowned resident")
            safe, reason = gateway.safe_to_unload(model)
            if safe is not True:
                raise ProveError(f"decision admission cleanup for {model}: guarded unload refused "
                                 f"({reason or 'activity is unknown'})")
            decision._native(base_url, "/api/generate", {"model": model, "keep_alive": 0})
            current = decision.resident_contexts(base_url)
            if current is None:
                raise ProveError(f"decision admission cleanup for {model}: Ollama resident state is unreachable "
                                 "after unload")
    except ProveError:
        raise
    except Exception as exc:
        raise ProveError(f"decision admission cleanup for {model} failed: {exc}") from None
    if current != initial:
        added = sorted(set(current) - set(initial))
        removed = sorted(set(initial) - set(current))
        changed = sorted(name for name in set(initial) & set(current) if initial[name] != current[name])
        raise ProveError(f"decision admission cleanup for {model} did not restore pre-existing Ollama residents "
                         f"(added={added}, removed={removed}, context_changed={changed})")




def _refuse_dirty_code_tree(repo: Path) -> None:
    """Non-dry proof runs must execute code pinned by the committed spec, never dirty shared WIP."""
    rc, out = _git(repo, "status", "--porcelain", "--", "localbench", strip_stdout=False)
    if rc != 0:
        raise ProveError(f"cannot inspect code tree under {repo}; run git status --porcelain localbench")
    if out:
        dirty = ", ".join(line[3:] for line in out.splitlines())
        raise ProveError(f"proof requires a clean localbench tree; dirty files: {dirty}; use scripts/run_plan.py")


def _refuse_rejected_screen(spec: dict, spec_path: str | Path) -> None:
    """Refuse a screen already rejected for this feature, corpus and installed model digest."""
    if spec.get("stage") != "screen":
        return
    ledger_path = NEGATIVE_EVIDENCE_PATH
    try:
        text = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProveError(f"{spec_path}: negative-evidence ledger unavailable: {ledger_path}: {exc}") from None

    rows = []
    for section in re.split(r"(?m)(?=^## )", text):
        lines = section.splitlines()
        if not lines or not lines[0].startswith("## "):
            continue
        if re.search(r"(?m)^\s*-\s+\*\*Verdict:\*\*\s+REJECT\b", section):
            rows.append((lines[0][3:].strip(), section))

    feature = spec["feature"]

    def names_feature(section: str) -> bool:
        return re.search(rf"(?<![\w-]){re.escape(feature)}(?![\w-])",
                         section, re.IGNORECASE) is not None

    retry_of = spec.get("retry_of")
    retry_heading = retry_of.strip() if retry_of is not None else None
    if retry_heading is not None and not any(
            heading == retry_heading and names_feature(section) for heading, section in rows):
        raise ProveError(f"{spec_path}: retry_of {retry_of!r} does not name a REJECT ledger row "
                         f"for feature {feature}")

    items_sha256 = spec["dataset"]["items_sha256"]
    relevant = []
    ledger_root = ledger_path.parents[2]
    for heading, section in rows:
        if not names_feature(heading) or heading == retry_heading:
            continue
        for receipt_rel in re.findall(r"`(docs/evidence/receipts/[^`\s]+\.json)`", section):
            receipt_path = ledger_root / receipt_rel
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ProveError(f"{spec_path}: cannot read REJECT receipt {receipt_rel}: {exc}") from None
            run = receipt.get("run") if isinstance(receipt, dict) else None
            provenance = run.get("provenance") if isinstance(run, dict) else None
            pins = provenance.get("pins") if isinstance(provenance, dict) else None
            if not isinstance(pins, dict) or pins.get("suite_items_sha256") != items_sha256:
                continue
            digest = pins.get("model_digest")
            if isinstance(digest, str) and digest:
                relevant.append((receipt_rel, digest))

    if not relevant:
        return

    from . import features

    feature_row = next((row for row in features.load() if row["feature"] == feature), None)
    if feature_row is None:
        raise ProveError(f"{spec_path}: feature {feature!r} is not in the feature registry")
    digests = features.ollama_digests()
    for candidate in spec["candidates"]:
        resolved = resolve_candidate(candidate)
        if resolved["backend"] != "ollama":
            continue
        installed_digest = features.installed_digest(feature_row, resolved["model"], digests)
        if not installed_digest:
            continue
        for receipt_rel, rejected_digest in relevant:
            if installed_digest == rejected_digest:
                raise ProveError(
                    f"{spec_path}: REJECT screen for {feature} and items_sha256 {items_sha256} "
                    f"matches candidate {resolved['model']} installed digest {installed_digest}; "
                    f"receipt {receipt_rel}. Set retry_of to the ledger heading and provide "
                    "a new_hypothesis rationale to retry intentionally.")


def prove_spec(spec_path: str | Path, *, dry_run: bool = False, br=None, run_suite_fn=None,
               legs_fn=None, outputs_fn=None, grade_fn=None, receipts_dir: Path = RECEIPTS_DIR,
               repo: Path = REPO_ROOT, corpora=None) -> dict:
    """Run one proof spec end to end (or plan it with dry_run): pre-reg commit, dataset pin,
    every candidate through its kind tier, assertions, banking, grading, bead updates.
    Decision proofs preflight all Ollama candidates and restore the resident baseline through the guarded unload path.

    Generation needs outputs_fn (produced outputs per candidate key; replay_generation_outputs replays
    corpus items through each route plus the builtin arm) and refuses live runs without it; no agent
    tier exists. legs_fn provides memory legs; grade_fn replaces grading."""
    from . import decision

    spec = load_spec(spec_path)
    spec_slug(spec_path)
    if not dry_run:
        _refuse_dirty_code_tree(repo)
    if not Path(spec_path).stem.startswith(spec["feature"] + "__"):
        raise ProveError(f"{spec_path}: filename must be <feature>__<slug>.json")
    commit = spec_commit(spec_path, repo)
    if spec.get("blocked"):
        raise ProveError(f"{spec_path}: spec is blocked ({spec['blocked'].get('reason')})")
    _refuse_rejected_screen(spec, spec_path)
    if dry_run:
        return {"spec": str(spec_path), "commit": commit, "dry_run": True,
                "steps": plan_steps(spec, commit)}
    kind = spec["kind"]
    if kind == "generation" and outputs_fn is None:
        raise ProveError("generation run backend lands with %pane's replay; dry-run plans it, "
                         "or pass outputs_fn")
    if kind == "agent":
        raise ProveError("no agent tier exists yet; dry-run plans it")
    report: dict = {"spec": str(spec_path), "commit": commit, "kind": kind, "candidates": []}
    if kind == "decision":
        suite = decision.resolve_suite(spec["dataset"]["suite"])
        if suite.items_sha256 != spec["dataset"]["items_sha256"]:
            raise ProveError(f"suite {suite.name} items_sha256 {suite.items_sha256} does not match "
                             f"the spec pin {spec['dataset']['items_sha256']}")
        from .__main__ import DECISION_BASE
        admission = _proof_ollama_preflight(spec, suite, DECISION_BASE)
        for candidate in spec["candidates"]:
            resolved_candidate = resolve_candidate(candidate)
            entry = None
            try:
                entry = candidate_evidence(spec, candidate, suite, run_suite=run_suite_fn, corpora=corpora)
            finally:
                if admission is not None and resolved_candidate["backend"] == "ollama":
                    decision_doc = ((entry or {}).get("receipt", {}).get("run", {}).get("decision", {}))
                    context = decision_doc.get("context") or {}
                    _release_proof_ollama_candidate(
                        admission["base_url"], resolved_candidate["model"], admission["initial"],
                        context.get("action"))
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
            if spec.get("stage") == "screen":
                receipt["screen_verdict"] = screen_verdict(
                    receipt, problems=problems, error_kinds=kinds,
                    allow_errors=float(spec.get("allow_errors", 0.0)))
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
                                        "passed": passed, "problems": problems,
                                        "screen_verdict": receipt.get("screen_verdict")})
    elif kind == "generation":
        corpus = _generation_corpus(spec)
        gold_set = load_gold_set(spec, corpus_root=corpus["root"]) if spec.get("gold_set") else None
        if gold_set is not None:
            corpus_ids = {item["id"] for item in corpus["items"]}
            absent = sorted(set(gold_set["pair_ids"]) - corpus_ids)
            if absent:
                raise ProveError(f"gold set names pair ids absent from corpus: {absent[:3]}")
        if grade_fn is None:
            grade, grade_context = _grade_default_context(spec)
            module_sha = grade_context["module_sha"]

        else:
            grade, grade_context = grade_fn, None
            module_sha = _module_sha(spec["feature"])

        raw_outputs = outputs_fn(spec, corpus["items"])
        outputs, calibrations, backend_pins = normalize_generation_outputs(raw_outputs)
        candidate_keys = {_generation_candidate_key(candidate): candidate for candidate in spec["candidates"]}
        if set(outputs) - set(candidate_keys):
            raise ProveError(f"generation outputs contain undeclared candidates: {sorted(set(outputs) - set(candidate_keys))}")
        resolved_candidates = {}
        for key, candidate in candidate_keys.items():
            resolved = resolve_candidate(candidate)
            resolved_candidates[key] = resolved
            if resolved["backend"] == "mlx-serve":
                pins = backend_pins.get(key)
                if pins is None:
                    raise ProveError(f"generation candidate {key}: missing mlx-serve backend fingerprint")
                if pins["model_dir"] != resolved["model"]:
                    raise ProveError(f"generation candidate {key}: backend fingerprint model_dir does not match route")
            elif key in backend_pins:
                raise ProveError(f"generation candidate {key}: mlx-serve pins on non-mlx route")
        from . import sysstats
        from .__main__ import _rev
        host, rev = sysstats.host(), _rev()

        for candidate in spec["candidates"]:
            key = _generation_candidate_key(candidate)
            resolved = resolved_candidates[key]
            model = (None if resolved["backend"] == "builtin"
                     else f"laya:{resolved['model']}" if resolved["backend"] == "laya"
                     else resolved["model"])
            runtime_pins = backend_pins.get(key) or {}
            installed_digest = (grade_context["digest_for"](model)
                                if grade_context is not None and model is not None else None)
            pinned_digest = runtime_pins.get("model_digest") or installed_digest
            grade_digest = installed_digest or runtime_pins.get("model_digest")
            problems, passed, by_id = evaluate_generation_assertions(
                spec, key, outputs.get(key, []), outputs)
            if spec["stage"] == "proof":
                if key not in calibrations:
                    problems.append("proof generation requires calibrated judge pairs")
                if gold_set is None:
                    problems.append("proof generation requires a sealed gold set")
            created = utc_stamp()
            run = {
                "label": "generation",
                "feature": spec["feature"],
                "suite": spec["dataset"]["suite"],
                "items_sha256": spec["dataset"]["items_sha256"],
                "manifest": {k: corpus["manifest"].get(k) for k in ("seed", "version", "n_items")},
                "checks": by_id,
                "metrics": {},
                "verdicts": {},
                "provenance": _generation_provenance(
                    resolved, key, backend_pins, created=created, host=host,
                    rev=rev, module_sha=module_sha, model_digest=pinned_digest,
                ),
            }
            if key in calibrations:
                run["judge_calibration"] = calibrations[key]
            if key in backend_pins:
                run["backend_pins"] = backend_pins[key]
            if gold_set is not None:
                run["gold_set"] = {
                    "sha256": spec["gold_set"]["sha256"], "n_pairs": len(gold_set["pair_ids"]),
                    "order_seed": gold_set["order_seed"], "labeler": gold_set["labeler"],
                    "kappa": gold_set["kappa"], "agreement": gold_set["agreement"]}
            if "judge_calibration" in run:
                from . import generation
                generation.require_pairwise_calibration(run)
            compare = "BETTER" if passed and not problems else "NOT_BETTER"
            baseline = _generation_baseline(grade_context, spec["feature"])
            run["verdicts"] = {
                "compare": compare,
                "assertions_passed": bool(passed) and not problems,
                "judge_calibrated": key in calibrations,
                "sealed_gold_set": gold_set is not None,
            }
            receipt = {
                "kind": "run",
                "feature": spec["feature"],
                "omp_module_sha": module_sha,
                "run": run,
                "problems": problems,
                "verdict": {"compare": compare, "baseline": baseline},
            }
            receipt["proof_spec"] = {"path": str(spec_path), "commit": commit, "stage": spec["stage"],
                                     "candidate": key, "passed": passed}
            path = bank_receipt(receipt, receipt_name(kind, spec["feature"], key, created), receipts_dir)
            if grade_context is None:
                status, reason = grade(receipt, key)
            else:
                status, reason = grade(receipt, model, grade_digest)
            report["candidates"].append({"candidate": candidate, "route": key,
                                        "receipt": str(path), "grade": status, "reason": reason,
                                        "passed": passed, "problems": problems})
    else:
        if legs_fn is not None:
            legs = legs_fn(spec)
        elif spec.get("legs"):
            legs = _legs_from_sources(spec["legs"])
        else:
            legs = _live_memory_legs(spec)
        by_arm = _split_legs_by_arm(spec, legs)
        for verdict in run_memory_verdicts(spec, legs):
            receipt = verdict["receipt"]
            problems, passed = evaluate_memory_entry(spec, verdict, by_arm)
            receipt["proof_spec"] = {"path": str(spec_path), "commit": commit, "stage": spec["stage"],
                                     "candidate": verdict["candidate"], "baseline": verdict["baseline"],
                                     "passed": passed}
            created = utc_stamp()
            path = bank_receipt(receipt, receipt_name(kind, spec["feature"],
                                                     str(verdict["candidate"]), created), receipts_dir)
            grade = grade_fn or _grade_default(spec)
            status, reason = grade(receipt, verdict["candidate"])
            report["candidates"].append({"candidate": verdict["candidate"], "baseline": verdict["baseline"],
                                        "route": str(verdict["candidate"]),
                                        "receipt": str(path), "grade": status, "reason": reason,
                                        "passed": passed, "problems": problems})
    report["beads"] = update_beads(spec, [{"route": c.get("route") or _candidate_slug(c.get("candidate", c)),
                                           "grade": c["grade"], "reason": c["reason"],
                                           "receipt": c["receipt"], "spec": str(spec_path),
                                           "screen_verdict": c.get("screen_verdict")}
                                          for c in report["candidates"]], br)
    return report


def _grade_default(spec: dict, *, cfg=None, provs=None, digests=None):
    """Grade fresh proofs against the active profile."""
    grade, _ = _grade_default_context(spec, cfg=cfg, provs=provs, digests=digests)
    return grade


def _grade_default_context(spec: dict, *, cfg=None, provs=None, digests=None):
    """Capture the profile route and model pins used to grade one proof run."""
    from . import features

    profile = spec.get("profile", "default")
    cfg = cfg if cfg is not None else features.omp_settings(profile)
    provs = provs if provs is not None else features.providers(profile)
    digests = digests if digests is not None else features.ollama_digests()
    rows = {row["feature"]: row for row in features.load()}
    row = rows[spec["feature"]]
    before = features.incumbent_settings(profile, row["preset"], cfg, provs)
    inc = (features.incumbent(row, *before) if not isinstance(before, str)
           else {"selector": None, "local": False, "setting": None, "source": before})
    module_sha = _module_sha(spec["feature"])

    def digest_for(model: str | None) -> str | None:
        return features.installed_digest(row, model, digests) if model is not None else None

    def grade(receipt: dict, model: str | None, digest: str | None = None) -> tuple[str, str]:
        return features.grade(receipt, sha=module_sha, model=model,
                              digest=digest if digest is not None else digest_for(model), inc=inc)

    context = {"profile": profile, "row": row, "inc": inc,
               "incumbent_known": not isinstance(before, str), "digests": digests,
               "module_sha": module_sha, "digest_for": digest_for}
    return grade, context


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
