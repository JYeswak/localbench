"""Opt-in private corpus: import omp's recorded hosted judgments as decision suites.

omp records every hosted judgment it makes in ~/.omp/profiles/<profile>/cache/judgment-cache.db
(tables states/oracle/usage). The find tool's per-candidate relevance questions land there as noul
items; choice judgments land as choice items. This module turns those rows into decision-suite items
(state, questions, labels, reference = the hosted answer) and writes the suites with
localbench.decision.write_suite under ~/.localbench/corpora/<role>/, content-addressed.

Privacy law: judged states embed the user's code, so suite output is REFUSED anywhere inside this
repo, and no real cache content may enter tests (tests/test_corpus.py builds synthetic fixtures).
Only hashes, counts and aggregate receipts may enter git.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

from . import decision, jevsuites

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES = Path.home() / ".omp" / "profiles"
CORPORA = decision.CORPORA

ROLE_TYPES = {"decision.noul": "noul", "decision.choice": "choice"}


class CorpusError(ValueError):
    """A corpus that cannot be trusted or written: unreadable cache, ambiguous labels, repo path."""


def cache_path(profile: str) -> Path:
    """The judgment cache a profile's hosted judgments live in (read, never written)."""
    if not profile or profile in {".", ".."} or "/" in profile or "\\" in profile:
        raise CorpusError(f"profile {profile!r} is not a plain profile name")
    return PROFILES / profile / "cache" / "judgment-cache.db"


def read_cache(path: Path | str) -> list[dict]:
    """Every judged state with its oracle rows, from a read-only connection. The caller passes a
    fixture path in tests; import_profile passes the profile's cache. Content stays in memory and
    in the corpora dir, never in the repo."""
    db_path = Path(path).expanduser()
    if not db_path.is_file():
        raise CorpusError(f"judgment cache {db_path} is missing")
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise CorpusError(f"judgment cache {db_path} is unreadable ({exc})") from None
    try:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if {"states", "oracle"} - tables:
            raise CorpusError(f"judgment cache {db_path} has no states/oracle tables")
        rows = []
        for state in db.execute("SELECT id, state, created_at FROM states ORDER BY id"):
            oracle = [dict(row) for row in db.execute(
                "SELECT id, model, name, type, instruction, criteria, answer, created_at FROM oracle "
                "WHERE state=? ORDER BY created_at, id", (state["id"],))]
            rows.append({"state_id": state["id"], "state": state["state"],
                         "created_at": state["created_at"], "oracle": oracle})
        return rows
    except sqlite3.Error as exc:
        raise CorpusError(f"judgment cache {db_path} read failed ({exc})") from None
    finally:
        db.close()


def _parse_json(value, where: str):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CorpusError(f"{where}: not JSON ({exc})") from None


def _reference(question: dict, answer: dict, where: str) -> dict:
    try:
        return decision.validate_answer(question, answer, where)
    except decision.DecisionError as exc:
        raise CorpusError(f"{where}: hosted answer invalid ({exc.detail})") from None


# omp 18.4.9 question literals in wire order (criteria order is semantic: Ollama assigns
# choice codes in criteria order). Verified byte-exact in Bun against the installed
# package (rendered instructions) and its TypeScript literals. Prompt text is omp's
# shipped package, not user content.
_STOP_QUESTION = {"type": "noul", "instructions": "Classify whether this assistant message is an unexpected stop: it says it will act, continue working, or call a tool, then ends without doing so.",
                    "criteria": {"true": "Unexpected stops:\n- \"I should do the same for the JS eval worker. Doing that now.\"\n- \"Let me run the tests next.\"\n- \"I'll fix that now.\"\n- \"Should I do that for you?\"", "false": "Not an unexpected stop:\n- \"I've completed the task.\"\n- \"Is there anything else I can help with?\"\n- \"The fix is done and tests pass.\""}}
_LEVEL_QUESTION = {"type": "choice", "instructions": "The state is a user's request to a coding agent. Judge how open-ended its problem is: whether the fix or design is given, or which causes or designs remain open. Choose the reasoning effort that needs, judging inherent difficulty rather than phrasing politeness or verbosity. Volume of work never raises it. If torn between levels, choose the lower one.",
                   "criteria": {"low": "One obvious solution, mechanically applied: target, mapping, or fix given.", "medium": "A few candidates in a localized area, or one small trap: which line breaks a test, one boundary case.", "high": "Several viable designs or candidate causes: API shape, policy choice, a known cause whose fix needs a design choice.", "xhigh": "Open cause of flaky, concurrent, or stale behavior; solutions that are easy to get subtly wrong (races, invariants, cross-version compatibility)."}}
_BUCKET_QUESTION = {"type": "choice", "instructions": "Classify the coding request by how open-ended its problem is. Volume of work never raises it.\n\nExamples:\n<request>rename a local constant and its two uses</request>\ntrivial\n\n<request>make the failing pagination test pass</request>\nmoderate\n\n<request>convert the fixtures in five repos to the new JSON format</request>\ntrivial\n\n<request>diagnose an intermittent deadlock across two services</request>\nhard\n",
                    "criteria": {"trivial": "One obvious solution, mechanically applied: target, mapping, or fix given.", "moderate": "A few candidate causes or several viable designs: which line breaks a test, API shape, policy choice.", "hard": "Open cause of flaky, concurrent, or stale behavior; solutions that are easy to get subtly wrong (races, invariants, cross-version compatibility)."}}
_TTSR_INSTRUCTIONS = "Does this output write source code in a Dicklesworthstone-owned repository, including eidetic_engine_cli, instead of preparing an upstream issue?"

# (question name, type) -> the 18.4.9 omp question in wire order. The cache stores
# criteria via stableStringify (alphabetical), so rows matching a known question use the
# literal (order restored); rows with a known name but different text are drifted (order
# unrecoverable) and skipped. Anything else keeps the generic path (order as cached).
_KNOWN_QUESTIONS = {
    ("stopped", "noul"): _STOP_QUESTION,
    ("level", "choice"): _LEVEL_QUESTION,
    ("bucket", "choice"): _BUCKET_QUESTION,
}


def _stable(value) -> str:
    """pi-utils stableStringifyJson for these values: sorted keys, compact, non-ASCII kept."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _omp_state(state: str):
    """Cache states are stored stable-stringified (sorted keys); restore omp's construction
    order: TTSR sends {output, content} (export/ttsr.ts:86). Single-key, already-sorted and
    non-JSON states pass through untouched (parsed objects stay objects, as the reference
    builder emits)."""
    try:
        parsed = json.loads(state)
    except (ValueError, UnicodeDecodeError):
        return state
    if not isinstance(parsed, dict):
        return state
    if set(parsed) == {"content", "output"}:
        return {"output": parsed["output"], "content": parsed["content"]}
    return parsed


def to_items(rows: list[dict], *, profile: str, role: str, model: str | None = None) -> tuple[list[dict], dict]:
    """Judged states -> suite items for one role. One item per state, questions of the role's type
    only, labels + reference from the hosted model answer. Rows matching a known 18.4.9 omp question
    use its wire-order literal (choice codes follow criteria order); known names with different text
    are drifted and skipped (counted), as are states whose hosted answer does not validate (hosted
    choice probabilities are rescaled to sum 1 first, as jevsuites does) and states left with no valid
    question. Multi-key states are reordered to omp's construction order (TTSR {output, content}).
    `model` pins the answering model; without it the cache must hold exactly one."""
    if role not in ROLE_TYPES:
        raise CorpusError(f"role {role!r} is not one of {sorted(ROLE_TYPES)}")
    kind = ROLE_TYPES[role]
    models = {row["model"] for item in rows for row in item["oracle"]}
    if model is None:
        if len(models) != 1:
            raise CorpusError(f"profile {profile!r} answers come from {len(models)} models "
                              f"({sorted(models)}); pass model= to pin one")
        model = next(iter(models))
    items, skipped = [], {"invalid_answer": 0, "empty_state": 0, "oversize_state": 0, "drifted_question": 0}
    for item in rows:
        candidates: dict[str, dict] = {}
        for row in item["oracle"]:
            if row["model"] != model or row["type"] != kind:
                continue
            previous = candidates.get(row["name"])
            if previous is not None and (row["created_at"], row["id"]) <= previous[0]:
                continue
            candidates[row["name"]] = ((row["created_at"], row["id"]), row)
        questions, labels, reference = {}, {}, {}
        for name in sorted(candidates):
            row = candidates[name][1]
            where = f"{profile}:{item['state_id']} question {name!r}"
            known = _KNOWN_QUESTIONS.get((row["name"], row["type"]))
            if known is not None:
                if row["instruction"] != known["instructions"] or _stable(
                        _parse_json(row["criteria"], f"{where} criteria")) != _stable(known.get("criteria")):
                    skipped["drifted_question"] += 1
                    continue
                question = {"type": known["type"], "instructions": known["instructions"]}
                if "criteria" in known:
                    question["criteria"] = dict(known["criteria"])
            elif (row["name"], row["type"]) == ("q0", "noul") and row["instruction"] == _TTSR_INSTRUCTIONS and (
                    row["criteria"] or "null") == "null":
                question = {"type": "noul", "instructions": _TTSR_INSTRUCTIONS}
            else:
                criteria = _parse_json(row["criteria"], f"{where} criteria")
                question = {"type": kind, "instructions": row["instruction"]}
                if criteria is not None:
                    question["criteria"] = criteria
            try:
                raw = _parse_json(row["answer"], f"{where} answer") or {}
                if kind == "choice" and isinstance(raw, dict) and isinstance(
                        question.get("criteria"), dict):
                    try:
                        ps, _ = jevsuites._normalized(raw.get("probabilities"), list(question["criteria"]))
                    except jevsuites._Skip as exc:
                        raise CorpusError(f"{where}: hosted probabilities ({exc})") from None
                    raw = {**raw, "probabilities": ps}
                ref = _reference(question, raw, where)
            except CorpusError:
                skipped["invalid_answer"] += 1
                continue
            questions[name] = question
            reference[name] = ref
            labels[name] = ref["noul"] > 0.5 if kind == "noul" else ref["choice"]
        if not questions:
            skipped["empty_state"] += 1
            continue
        state = _omp_state(item["state"])
        try:
            decision.build_request("suite", state, questions)
        except decision.RequestError:
            skipped["oversize_state"] += 1
            continue
        items.append({"id": item["state_id"], "state": state, "questions": questions,
                      "labels": labels, "reference": reference, "source": f"{profile}:{item['state_id']}"})
    return items, skipped


def refuse_repo_path(directory: Path | str) -> Path:
    """Content-addressed corpora live outside git; refuse this checkout and any git work tree.
    Shared with the gateway capture path, which must never write request content here either."""
    target = Path(directory).expanduser().resolve()
    if target == REPO_ROOT or REPO_ROOT in target.parents:
        raise CorpusError(f"corpus output {target} is inside the repo {REPO_ROOT}; refusing")
    for parent in (target, *target.parents):
        if (parent / ".git").exists():
            raise CorpusError(f"corpus output {target} is inside the git work tree {parent}; refusing")
    return target


def secure_dir(path: Path) -> Path:
    """Create a corpora directory with owner-only access. Refuses up front (creating nothing),
    then re-resolves after creation and refuses again: a nested symlink planted before creation
    must not redirect writes into a repo."""
    refuse_repo_path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise CorpusError(f"cannot secure {path} ({exc})") from None
    return refuse_repo_path(path.resolve())


_refuse_repo_path = refuse_repo_path


def snapshot_source(source_db: Path, directory: Path) -> Path:
    """Pin a read-only copy of the judgment cache beside the suite: the live cache keeps
    appending, so a suite pinned to it would stop loading. Returns the snapshot path,
    named sources/<sha256>.sqlite."""
    sources = secure_dir(directory / "sources")
    staging = sources / "snapshot.sqlite.tmp"
    if staging.exists():
        staging.unlink()
    source = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(staging)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    staging.chmod(0o600)
    digest = hashlib.sha256(staging.read_bytes()).hexdigest()
    snapshot = sources / f"{digest}.sqlite"
    if snapshot.exists():
        staging.unlink()
    else:
        staging.rename(snapshot)
    snapshot.chmod(0o600)
    return snapshot


def import_profile(profile: str, *, dest_root: Path | str | None = None,
                   roles: tuple[str, ...] = ("decision.noul",),
                   model: str | None = None, gate: dict | None = None,
                   name: str | None = None) -> dict:
    """Import one profile's hosted judgments as one content-addressed suite per role and return
    {role: suite-name, items, skipped, directory}. Re-importing the same content lands on the same
    directory (the address is the items' sha256). The gate must be passed explicitly (metric plus
    threshold); without one the build is refused before anything is written."""
    if gate is None:
        raise decision.SuiteError("corpus import refuses without an explicit gate "
                                  "(metric plus threshold, e.g. {\"decision.noul.accuracy\": {\"min\": 0.8}})")
    source_db = cache_path(profile)
    rows = read_cache(source_db)
    root = _refuse_repo_path(Path(dest_root).expanduser() if dest_root is not None else CORPORA)
    summary: dict[str, dict] = {}
    for role in roles:
        items, skipped = to_items(rows, profile=profile, role=role, model=model)
        if not items:
            summary[role] = {"suite": None, "items": 0, "skipped": skipped, "directory": None}
            continue
        payload = "".join(json.dumps(it, ensure_ascii=False, separators=(",", ":"),
                                     sort_keys=True) + "\n" for it in items)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        directory = root / role / digest
        snapshot = snapshot_source(source_db, directory)
        suite = decision.write_suite(directory, name=name or f"{profile}-{role}-{digest}", role=role,
                                     items=items, sources=[snapshot], gate=gate)
        summary[role] = {"suite": suite.name, "items": len(items), "skipped": skipped,
                         "directory": str(directory)}
    return summary


def list_corpora(root: Path | str | None = None) -> list[dict]:
    """Every suite manifest under the corpora root, without validating pins (load_suite does that
    at eval time, when a drifted source must fail loudly rather than silently vanish here)."""
    base = Path(root).expanduser() if root is not None else CORPORA
    suites = []
    if not base.is_dir():
        return suites
    for manifest in sorted(base.rglob("manifest.json")):
        try:
            doc = json.loads(manifest.read_bytes())
        except (ValueError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        try:
            lines = (manifest.parent / doc.get("items", "")).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        lines = [line for line in lines if line.strip()]
        suites.append({"name": doc.get("name"), "role": doc.get("role"), "items": len(lines),
                       "directory": str(manifest.parent),
                       "has_reference": bool(lines) and all('"reference"' in line for line in lines)})
    return suites


def stats(root: Path | str | None = None) -> dict:
    """Counts by role over list_corpora plus capture counts: suite/item totals, captured pairs,
    and whether a capture spec is currently active. Pure read, no validation."""
    by_role: dict[str, dict] = {}
    for suite in list_corpora(root):
        entry = by_role.setdefault(suite["role"], {"suites": 0, "items": 0, "names": []})
        entry["suites"] += 1
        entry["items"] += suite["items"]
        entry["names"].append(suite["name"])
    status = capture_status(root)
    return {"roles": by_role,
            "suites": sum(entry["suites"] for entry in by_role.values()),
            "items": sum(entry["items"] for entry in by_role.values()),
            "capture": {"items": status["items"], "active": status["active"]}}


def capture_spec_path(root: Path | str | None = None) -> Path:
    """The opt-in capture spec: <root>/capture.json (default ~/.localbench/corpora/)."""
    base = Path(root).expanduser() if root is not None else CORPORA
    return base / "capture.json"


def _is_request_file(path: Path) -> bool:
    """A captured request document: <sha256>.json (responses are <sha256>.response.json)."""
    stem = path.name[:-len(".json")]
    return len(stem) == 64 and all(c in "0123456789abcdef" for c in stem)


def capture_items(root: Path | str | None = None) -> int:
    """Captured request documents (a request+response pair counts as one item) under
    <root>/captured/. Read-only count for the cap and status."""
    captured = (Path(root).expanduser() if root is not None else CORPORA) / "captured"
    try:
        return sum(1 for path in captured.rglob("*.json") if _is_request_file(path))
    except OSError:
        return 0


def capture_purpose_items(root: Path | str | None, purpose: str) -> int:
    """Request documents captured for one purpose. The per-purpose cap counts pairs, so a
    max_items=1 budget still captures the response."""
    captured = (Path(root).expanduser() if root is not None else CORPORA) / "captured"
    try:
        return sum(1 for path in (captured / purpose).glob("*.json") if _is_request_file(path))
    except OSError:
        return 0


def valid_capture_purpose(purpose: str) -> bool:
    """A purpose the gateway can actually capture: captured bodies land in a
    directory named for the purpose, so path separators and dot-names are
    refused. Shared with gateway._capture_body; capture_on fails fast here
    instead of arming a spec that silently captures nothing."""
    return (isinstance(purpose, str) and bool(purpose) and "/" not in purpose
            and "\\" not in purpose and purpose not in {".", ".."})


def _write_spec(path: Path, spec: dict) -> dict:
    secure_dir(path.parent)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staging.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        staging.chmod(0o600)
        os.replace(staging, path)
    finally:
        try:
            staging.unlink()
        except OSError:
            pass
    return spec


def capture_on(purposes: list[str], max_items: int, minutes: float, *,
               root: Path | str | None = None, now: float | None = None) -> dict:
    """Enable capture for `purposes` request purposes, at most `max_items` request documents per
    purpose (a request+response pair counts as one), expiring `minutes` from now. Fail-closed
    validation; the spec carries no request content."""
    if (not isinstance(purposes, list) or not purposes
            or any(not isinstance(p, str) or not p.strip() or len(p) > 256 for p in purposes)):
        raise CorpusError("capture purposes must be a non-empty list of purpose strings")
    if any(not valid_capture_purpose(purpose) for purpose in purposes):
        raise CorpusError("capture purposes must be plain directory names: no /, \\, . or ..")
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        raise CorpusError("capture max_items must be a positive integer")
    if not isinstance(minutes, (int, float)) or isinstance(minutes, bool) \
            or not math.isfinite(minutes) or minutes <= 0:
        raise CorpusError("capture minutes must be a positive finite duration")
    moment = time.time() if now is None else now
    return _write_spec(capture_spec_path(root),
                       {"enabled": True, "purposes": list(purposes), "max_items": max_items,
                        "until": moment + minutes * 60.0})


def capture_off(root: Path | str | None = None) -> dict:
    """Disable capture, keeping the previous purposes and cap for the next opt-in."""
    path = capture_spec_path(root)
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    return _write_spec(path, {"enabled": False, "purposes": previous.get("purposes", []),
                              "max_items": previous.get("max_items", 0),
                              "until": previous.get("until", 0)})


def capture_status(root: Path | str | None = None, now: float | None = None) -> dict:
    """The capture spec plus liveness: active only while enabled, unexpired, and capped.
    Pure read (plus the item count); never writes."""
    moment = time.time() if now is None else now
    try:
        spec = json.loads(capture_spec_path(root).read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        spec = {}
    if not isinstance(spec, dict):
        spec = {}
    active = (spec.get("enabled") is True and isinstance(spec.get("until"), (int, float))
              and spec["until"] > moment and isinstance(spec.get("purposes"), list)
              and bool(spec["purposes"]) and isinstance(spec.get("max_items"), int)
              and not isinstance(spec.get("max_items"), bool) and spec["max_items"] >= 1)
    return {"enabled": spec.get("enabled") is True, "active": active,
            "purposes": spec.get("purposes", []), "max_items": spec.get("max_items", 0),
            "until": spec.get("until", 0), "items": capture_items(root)}
