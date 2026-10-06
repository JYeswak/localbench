"""Decision models on Ollama 0.35's POST /v1/systemone: a strict loopback client, pinned role suites, per-question
quality and calibration metrics, live latency, and a paired comparison against a baseline arm.

System One answers typed questions about a state with candidate probabilities computed from label logits (no
sampling), so answer quality is deterministic and can be measured anytime. Latency and availability are measured
live with other models co-resident; the load a passed-in sysstats Sampler records is reported and never voids the
run. Wire shapes follow ollama v0.35.0: decision/types.go (Request, Response, NoulAnswer, ChoiceAnswer, ScoreAnswer),
decision/systemone.go (Compile, compileField, Answer: 1-64 questions, 2-26 candidates, softmax probabilities, choice =
argmax, score = sum(j * p_j)) and server/routes.go SystemOneHandler (64 KiB request body cap).

Hosted decision service (TypeSafe, same request/response shape) is a comparison arm only when the caller passes `hosted=Hosted()`;
it is never a fallback for the local arm, and its API key is read from the environment and never recorded.

A `laya:<hf repo>[@<subfolder>]` local arm (LayaShim, run_laya) is a Laya-MLX checkpoint behind the same wire shape:
localbench/shims/laya_systemone.py, run by the laya venv's python3 on a free loopback port for one run, offline from
the HF cache. No Ollama runner context applies; Laya truncates states over its max_len, which the shim reports per
answer and the receipt counts (decision.laya.truncated_items).
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import ipaddress
import json
import math
import os
import random
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from localbench import backends, stats, sysstats

from . import lifecycle

TYPES = ("choice", "noul", "score")
ROLES = {                       # role -> question types its items may ask (rank = top-1 among 2-26 candidates)
    "decision.choice": frozenset({"choice"}),
    "decision.noul": frozenset({"noul"}),
    "decision.rank": frozenset({"choice"}),
    "decision.score": frozenset({"score"}),
}
MAX_BODY_BYTES = 64 << 10       # routes.go: http.MaxBytesReader(..., 64<<10)
MIN_QUESTIONS, MAX_QUESTIONS = 1, 64
MIN_CANDIDATES, MAX_CANDIDATES = 2, 26
TOL = 1e-6                      # probability sums and score expectations
EPS = 1e-12                     # float slack when judging a delta against its noise
ECE_BINS = 10
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260930
CORPORA = Path.home() / ".localbench" / "corpora"   # captured omp items and converted suites live outside git

BETTER = {"accuracy": "higher", "macro_f1": "higher", "brier": "lower", "ece": "lower", "mae": "lower",
          "agreement": "higher", "mean_abs_dp": "lower", "agreement_mae": "lower", "error_rate": "lower"}
QUALITY = ("accuracy", "macro_f1", "brier", "ece", "mae")     # vs labels; a loss beyond noise makes compare() WORSE
LABEL_METRICS = {"choice": ("accuracy", "macro_f1", "brier", "ece"), "noul": ("accuracy", "macro_f1", "brier", "ece"),
                 "score": ("accuracy", "macro_f1", "mae", "ece")}
REFERENCE_METRICS = {"choice": ("agreement",), "noul": ("agreement", "mean_abs_dp"), "score": ("agreement_mae",)}
LATENCY_P95 = "decision.latency.warm_p95_s"
ERROR_RATE = "decision.error_rate"
COST = "decision.cost_usd"
# /v1/systemone has no options: the resident runner's context bounds every prompt (Ollama 0.35 answers HTTP 400
# "prompt 0 has 9568 tokens; expected 1–8194 (input is never truncated)").
CONTEXT_ERROR = re.compile(r"prompt \d+ has \d+ tokens; expected 1\s*[–-]\s*\d+")
# Conservative prompt-token estimate from the compiled prompt's UTF-8 bytes (prompt_bytes). Calibrated on the 99
# context 400s of the 2026-10-01 find-judgments screen (nimble:latest, tev1:latest): every reported count was
# <= bytes/2.5 + 89, and bytes/2 + 256 exceeds each by >= 886 tokens (margin for system prompt and chat template).
CONTEXT_BYTES_PER_TOKEN = 2.0
CONTEXT_OVERHEAD_TOKENS = 256
WARM_KEEP_ALIVE = "30m"
# Ollama's loaded-model cap: OLLAMA_MAX_LOADED_MODELS (envconfig/config.go:279, `Uint(..., 0)`, 0 = automatic); when
# 0 the scheduler uses defaultModelsPerGPU * max(gpu count, 1) (server/sched.go:87 `var defaultModelsPerGPU = 3`,
# :286-293), and at the cap it unloads one runner before loading a new one (:269-271, findRunnerToUnload :1679).
# Ollama v0.35.0. Apple Silicon's Metal backend reports one GPU, so the automatic cap here is 3.
DEFAULT_MODELS_PER_GPU = 3
METAL_GPUS = 1


class RequestError(ValueError):
    """A System One request this client refuses to send (it would be rejected, or it would leave the host)."""


class SuiteError(ValueError):
    """A suite that cannot be trusted: manifest malformed, label hash or source hash mismatch, item invalid."""


class DecisionError(Exception):
    """One request's ERROR outcome: kind is invalid (malformed response), context (prompt longer than the resident
    runner's context: a harness load error), http, timeout, unavailable or refused."""

    def __init__(self, kind: str, detail: str, latency_s: float | None = None):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail, self.latency_s = kind, detail, latency_s


class ContextError(RuntimeError):
    """The decision model cannot get the context the suite needs; refused before any item is sent."""


@dataclass(frozen=True)
class Hosted:
    """The hosted decision service comparison arm. Only an explicit `run_suite(..., hosted=Hosted())` sends items to it."""
    base_url: str = "https://api.typesafe.ai"
    model: str = "proj-b-latest"
    api_key_env: str = "TYPESAFE_API_KEY"
    usd_per_input_token: float = 0.042e-6


BASELINE_KINDS = ("fixed", "route")


@dataclass(frozen=True)
class Baseline:
    """A banked decision receipt (or arm) on the same pinned suite to compare the local arm against: kind `fixed` (a
    pinned configuration, e.g. an effort level) or `route` (today's route for the feature, e.g.
    ollama/qwen3.8:27b-mlx). `id` names it in the verdict and in the not-better problem."""
    kind: str
    id: str
    receipt: dict


# ---------------------------------------------------------------- request

def _nonempty_content(v) -> bool:
    """Ollama's content(): a string, object or array; here also non-empty (whitespace-only strings are refused)."""
    if isinstance(v, str):
        return bool(v.strip())
    return isinstance(v, dict | list) and bool(v)


def _real(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def question_options(q: dict) -> list[str]:
    """Candidate keys in request order, as Ollama's Answer() keys probabilities: choice -> criteria keys, noul ->
    false/true, score -> "0".."n-1". `q` must already be valid."""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]


def check_question(name, q) -> list[str]:
    """Validate one question as decision.compileField does; returns its candidate keys."""
    if not isinstance(name, str) or not name.strip():
        raise RequestError("question names must be non-empty strings")
    if not isinstance(q, dict):
        raise RequestError(f"question {name!r}: not an object")
    unknown = sorted(set(q) - {"type", "instructions", "criteria"})
    if unknown:
        raise RequestError(f"question {name!r}: unknown keys {unknown}")
    if not _nonempty_content(q.get("instructions")):
        raise RequestError(f"question {name!r}: instructions must be a non-empty string, object or array")
    kind, crit = q.get("type"), q.get("criteria")
    if kind == "choice":
        if (not isinstance(crit, dict) or any(not isinstance(k, str) or not k.strip() for k in crit)
                or any(v is not None and not isinstance(v, str) for v in crit.values())):
            raise RequestError(f"question {name!r}: choice criteria must map non-empty option keys to descriptions "
                               "or null")
    elif kind == "noul":
        # Absent criteria mean No/Yes; JSON null or keys other than true/false are rejected upstream.
        if "criteria" in q and (not isinstance(crit, dict) or set(crit) - {"true", "false"}
                                or any(not isinstance(v, str) for v in crit.values())):
            raise RequestError(f"question {name!r}: noul criteria must be an object of true/false descriptions")
    elif kind == "score":
        if not isinstance(crit, list) or any(not isinstance(v, str) for v in crit):
            raise RequestError(f"question {name!r}: score criteria must be an array of descriptions")
    else:
        raise RequestError(f"question {name!r}: type must be choice, noul or score")
    options = question_options(q)
    if not MIN_CANDIDATES <= len(options) <= MAX_CANDIDATES:
        raise RequestError(f"question {name!r}: {len(options)} candidates; System One takes "
                           f"{MIN_CANDIDATES}-{MAX_CANDIDATES}")
    return options


def encode(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build_request(model: str, state, questions: dict, keep_alive=None) -> dict:
    """A validated /v1/systemone body {model, state, questions, keep_alive?}; RequestError for anything Ollama 0.35
    would reject (or a body over 64 KiB)."""
    if not isinstance(model, str) or not model.strip():
        raise RequestError("model is required")
    if not _nonempty_content(state):
        raise RequestError("state must be a non-empty string, object or array")
    if not isinstance(questions, dict) or not MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS:
        raise RequestError(f"questions must be an object of {MIN_QUESTIONS}-{MAX_QUESTIONS} named questions")
    for name, q in questions.items():
        check_question(name, q)
    body = {"model": model, "state": state, "questions": questions}
    if keep_alive is not None:
        if not (isinstance(keep_alive, str) and keep_alive.strip()) and not _real(keep_alive):
            raise RequestError("keep_alive must be a duration string or a number of seconds")
        body["keep_alive"] = keep_alive
    size = len(encode(body))
    if size > MAX_BODY_BYTES:
        raise RequestError(f"request body is {size} bytes; System One accepts at most {MAX_BODY_BYTES}")
    return body


def endpoint(base_url: str, *, allow_remote: bool = False) -> str:
    """`<root>/v1/systemone` from a server root or its /v1 base. The local arm must be loopback; a remote (hosted)
    arm must use https so the state and the bearer key never cross the network in clear text."""
    u = urllib.parse.urlsplit(base_url.strip().rstrip("/"))
    if u.scheme not in ("http", "https") or not u.hostname:
        raise RequestError(f"{base_url!r}: not an http(s) URL")
    try:
        loopback = ipaddress.ip_address(u.hostname).is_loopback
    except ValueError:
        loopback = u.hostname.lower() == "localhost"
    if not loopback and not allow_remote:
        raise RequestError(f"{base_url!r}: the local System One arm must be a loopback address")
    if not loopback and u.scheme != "https":
        raise RequestError(f"{base_url!r}: a remote arm must use https")
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path.removesuffix("/v1") + "/v1/systemone", "", ""))


# ---------------------------------------------------------------- response

def _finite_float(s: str) -> float:
    v = float(s)
    if not math.isfinite(v):
        raise ValueError(f"non-finite number {s}")
    return v


def _reject_constant(s: str):
    raise ValueError(f"non-finite number {s}")


def _invalid(detail: str) -> DecisionError:
    return DecisionError("invalid", detail)


def _prob(v, where: str) -> float:
    if not _real(v) or not 0.0 <= v <= 1.0:
        raise _invalid(f"{where}: {v!r} is not a finite probability in [0, 1]")
    return float(v)


def validate_answer(q: dict, a, where: str) -> dict:
    """One answer checked against its question; returns the answer reduced to its typed fields. Any defect is an
    ERROR (DecisionError kind invalid), never a score."""
    if not isinstance(a, dict) or a.get("type") != q["type"]:
        raise _invalid(f"{where}: not a {q['type']} answer")
    if q["type"] == "noul":
        return {"type": "noul", "noul": _prob(a.get("noul"), f"{where}.noul")}
    options = question_options(q)
    probs = a.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(options):
        got = sorted(probs) if isinstance(probs, dict) else probs
        raise _invalid(f"{where}.probabilities: keys {got!r} are not the candidates {options}")
    ps = {k: _prob(probs[k], f"{where}.probabilities[{k}]") for k in options}
    total = sum(ps.values())
    if abs(total - 1.0) > TOL:
        raise _invalid(f"{where}.probabilities sum to {total!r}, not 1 within {TOL}")
    confidence = _prob(a.get("confidence"), f"{where}.confidence")
    if q["type"] == "choice":
        choice = a.get("choice")
        if not isinstance(choice, str) or choice not in ps:
            raise _invalid(f"{where}.choice: {choice!r} is not a candidate")
        if ps[choice] < max(ps.values()):
            raise _invalid(f"{where}.choice: {choice!r} (p={ps[choice]!r}) is not the argmax")
        return {"type": "choice", "choice": choice, "probabilities": ps, "confidence": confidence}
    score = a.get("score")
    if not _real(score):
        raise _invalid(f"{where}.score: {score!r} is not a finite number")
    expect = sum(j * ps[str(j)] for j in range(len(options)))
    if abs(score - expect) > TOL:
        raise _invalid(f"{where}.score: {score!r} != sum(j * p_j) = {expect!r}")
    legend = a.get("legend")
    if not isinstance(legend, dict) or set(legend) != set(options) or any(not isinstance(v, str)
                                                                           for v in legend.values()):
        raise _invalid(f"{where}.legend: not a description for each of {options}")
    return {"type": "score", "score": float(score), "legend": legend, "probabilities": ps, "confidence": confidence}


def _count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def validate_response(request: dict, doc) -> dict:
    """A parsed /v1/systemone response checked against its request: answer names in request order, every answer
    well-formed, usage token counts. Returns {model, answers, usage}; DecisionError(kind invalid) otherwise."""
    if not isinstance(doc, dict):
        raise _invalid("response is not an object")
    if not isinstance(doc.get("model"), str) or not doc["model"]:
        raise _invalid("response has no model")
    answers = doc.get("answers")
    if not isinstance(answers, dict):
        raise _invalid("response has no answers object")
    names = list(request["questions"])
    if list(answers) != names:
        raise _invalid(f"answer names/order {list(answers)} differ from the request's {names}")
    usage = doc.get("usage")
    if not isinstance(usage, dict) or not all(_count(usage.get(k)) for k in ("input_tokens", "output_tokens")):
        raise _invalid(f"usage {usage!r} lacks non-negative integer input_tokens/output_tokens")
    return {"model": doc["model"],
            "answers": {name: validate_answer(q, answers[name], name) for name, q in request["questions"].items()},
            "usage": {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}}


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
        doc = json.loads(raw)
        return str(doc.get("error", doc)) if isinstance(doc, dict) else str(doc)
    except (OSError, ValueError):
        return exc.reason if isinstance(exc.reason, str) else str(exc.reason)


def post(url: str, body: dict, *, api_key: str | None = None, timeout: float = 120.0) -> tuple[dict, float]:
    """POST a body, parse the reply as finite JSON. Returns (doc, seconds); DecisionError on any failure, carrying
    the seconds spent."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, encode(body), headers, method="POST")
    t0 = time.perf_counter()
    try:
        with backends.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as exc:
        detail = _error_text(exc)
        exc.close()
        # A prompt longer than the resident runner's context is the harness's load, not the model's answer.
        kind = "context" if exc.code == 400 and CONTEXT_ERROR.search(detail) else "http"
        raise DecisionError(kind, f"HTTP {exc.code}: {detail}", time.perf_counter() - t0) from None
    except urllib.error.URLError as exc:
        kind = "timeout" if isinstance(exc.reason, TimeoutError) else "unavailable"
        raise DecisionError(kind, str(exc.reason), time.perf_counter() - t0) from None
    except TimeoutError as exc:
        raise DecisionError("timeout", str(exc) or f"no reply within {timeout}s", time.perf_counter() - t0) from None
    except (OSError, http.client.HTTPException) as exc:
        raise DecisionError("unavailable", f"{type(exc).__name__}: {exc}", time.perf_counter() - t0) from None
    latency = time.perf_counter() - t0
    try:
        doc = json.loads(raw, parse_float=_finite_float, parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DecisionError("invalid", f"response is not finite JSON: {exc}", latency) from None
    return doc, latency


def ask(url: str, request: dict, *, api_key: str | None = None, timeout: float = 120.0) -> tuple[dict, float, dict]:
    """One validated System One exchange: (validate_response(...), seconds, the reply as parsed) or DecisionError. The
    parsed reply carries what validation drops, e.g. the Laya shim's per-answer truncation report."""
    doc, latency = post(url, request, api_key=api_key, timeout=timeout)
    try:
        return validate_response(request, doc), latency, doc
    except DecisionError as exc:
        exc.latency_s = latency
        raise


# ---------------------------------------------------------------- suites

@dataclass(frozen=True)
class Suite:
    name: str
    role: str
    manifest: Path
    manifest_sha256: str
    items_path: Path
    items_sha256: str
    sources: tuple[dict, ...]
    gate: dict
    items: tuple[dict, ...]
    has_reference: bool

    def pin(self) -> dict:
        return {"name": self.name, "role": self.role, "manifest": str(self.manifest),
                "manifest_sha256": self.manifest_sha256, "items_sha256": self.items_sha256,
                "sources": [dict(s) for s in self.sources], "n_items": len(self.items),
                "has_reference": self.has_reference}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(base: Path, p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else base / path


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _check_label(q: dict, label, where: str) -> None:
    kind, options = q["type"], question_options(q)
    ok = ((kind == "choice" and isinstance(label, str) and label in options)
          or (kind == "noul" and isinstance(label, bool))
          or (kind == "score" and isinstance(label, int) and not isinstance(label, bool)
              and 0 <= label < len(options)))
    if not ok:
        raise SuiteError(f"{where}: label {label!r} is not a {kind} answer for candidates {options}")


def gate_metrics(types) -> set[str]:
    """Metric names a gate may name for suites asking these question types."""
    return {ERROR_RATE} | {f"decision.{t}.{m}" for t in types for m in (*LABEL_METRICS[t], "error_rate")}


def _check_item(it, where: str, role: str) -> bool:
    """Validate one item; returns whether it carries reference answers."""
    if not isinstance(it, dict):
        raise SuiteError(f"{where}: not an object")
    unknown = sorted(set(it) - {"id", "state", "questions", "labels", "reference", "source"})
    if unknown:
        raise SuiteError(f"{where}: unknown keys {unknown}")
    if not isinstance(it.get("source"), str) or not it["source"].strip():
        raise SuiteError(f"{where}: no source")
    try:
        build_request("suite", it.get("state"), it.get("questions"))
    except RequestError as exc:
        raise SuiteError(f"{where}: {exc}") from None
    questions = it["questions"]
    wrong = sorted({q["type"] for q in questions.values()} - ROLES[role])
    if wrong:
        raise SuiteError(f"{where}: {role} items ask {sorted(ROLES[role])}, not {wrong}")
    labels = it.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(questions):
        raise SuiteError(f"{where}: labels must name every question exactly")
    for name, q in questions.items():
        _check_label(q, labels[name], f"{where} label {name!r}")
    if "reference" not in it:
        return False
    ref = it["reference"]
    if not isinstance(ref, dict) or set(ref) != set(questions):
        raise SuiteError(f"{where}: reference must answer every question exactly")
    for name, q in questions.items():
        try:
            validate_answer(q, ref[name], name)
        except DecisionError as exc:
            raise SuiteError(f"{where} reference: {exc.detail}") from None
    return True


def load_suite(path) -> Suite:
    """A suite from its manifest (or the directory holding manifest.json). The manifest pins role, the items file's
    sha256 (the label-hash check) and each source's path + sha256; any mismatch or malformed item is refused."""
    p = Path(path).expanduser()
    manifest = p / "manifest.json" if p.is_dir() else p
    if not manifest.is_file():
        raise SuiteError(f"{manifest}: no suite manifest")
    raw = manifest.read_bytes()
    try:
        m = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SuiteError(f"{manifest}: not JSON ({exc})") from None
    if not isinstance(m, dict):
        raise SuiteError(f"{manifest}: not an object")
    if not isinstance(m.get("name"), str) or not m["name"].strip():
        raise SuiteError(f"{manifest}: no name")
    role = m.get("role")
    if role not in ROLES:
        raise SuiteError(f"{manifest}: role {role!r} is not one of {sorted(ROLES)}")
    if not isinstance(m.get("items"), str) or not _is_sha(m.get("items_sha256")):
        raise SuiteError(f"{manifest}: needs items (path) and items_sha256 (64 lowercase hex)")
    sources = m.get("sources")
    if (not isinstance(sources, list) or not sources
            or any(not isinstance(s, dict) or not isinstance(s.get("path"), str) or not _is_sha(s.get("sha256"))
                   for s in sources)):
        raise SuiteError(f"{manifest}: sources must be a non-empty list of {{path, sha256}}")
    items_path = _resolve(manifest.parent, m["items"])
    if not items_path.is_file():
        raise SuiteError(f"{manifest}: items file {items_path} is missing")
    data = items_path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != m["items_sha256"]:
        raise SuiteError(f"label-hash check: {items_path} has sha256 {got}, the manifest pins {m['items_sha256']}; "
                         "items or labels changed after the suite was pinned")
    for s in sources:
        sp = _resolve(manifest.parent, s["path"])
        if not sp.is_file():
            raise SuiteError(f"{manifest}: source {sp} is missing")
        sha = _sha256(sp)
        if sha != s["sha256"]:
            raise SuiteError(f"{manifest}: source {sp} has sha256 {sha}, the manifest pins {s['sha256']}")
    items, ids, refs, types = [], set(), set(), set()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SuiteError(f"{items_path}: not UTF-8 ({exc})") from None
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"{items_path}:{lineno}"
        try:
            it = json.loads(line, parse_float=_finite_float, parse_constant=_reject_constant)
        except ValueError as exc:
            raise SuiteError(f"{where}: not finite JSON ({exc})") from None
        if not isinstance(it, dict) or not isinstance(it.get("id"), str) or not it["id"].strip():
            raise SuiteError(f"{where}: no id")
        if it["id"] in ids:
            raise SuiteError(f"{where}: duplicate id {it['id']!r}")
        ids.add(it["id"])
        refs.add(_check_item(it, f"{where} ({it['id']})", role))
        types |= {q["type"] for q in it["questions"].values()}
        items.append(it)
    if not items:
        raise SuiteError(f"{items_path}: no items")
    if len(refs) > 1:
        raise SuiteError(f"{items_path}: reference answers must be on every item or none (denominators are fixed)")
    gate = m.get("gate")
    allowed = gate_metrics(types)
    if not isinstance(gate, dict) or not gate:
        raise SuiteError(f"{manifest}: gate must predeclare at least one metric bound, e.g. "
                         f"{{\"decision.{min(types)}.accuracy\": {{\"min\": 0.8}}}}")
    for metric, bound in gate.items():
        if metric not in allowed:
            raise SuiteError(f"{manifest}: gate metric {metric!r} is not one of {sorted(allowed)}")
        if not isinstance(bound, dict) or len(bound) != 1 or not set(bound) <= {"min", "max"} \
                or not _real(next(iter(bound.values()))):
            raise SuiteError(f"{manifest}: gate {metric!r} needs exactly one finite min or max")
    return Suite(name=m["name"], role=role, manifest=manifest, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                 items_path=items_path, items_sha256=got,
                 sources=tuple({"path": s["path"], "sha256": s["sha256"]} for s in sources), gate=gate,
                 items=tuple(items), has_reference=refs == {True})


def _write_private(path: Path, text: str) -> None:
    """Write text to a file only its owner can read (0600 from creation, also when the file existed)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        os.fchmod(fh.fileno(), 0o600)
        fh.write(text)


def write_suite(directory, *, name: str, role: str, items: list[dict], sources: list, gate: dict) -> Suite:
    """Write items.jsonl + manifest.json (pinning the items' and each source's sha256) and load them back. Sources
    are referenced by absolute path, never copied: private corpora stay where they are."""
    d = Path(directory).expanduser()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)              # items hold private omp states: owner-only, also when the directory existed
    items_path = d / "items.jsonl"
    _write_private(items_path, "".join(json.dumps(it, ensure_ascii=False, separators=(",", ":")) + "\n"
                                       for it in items))
    pinned = []
    for s in sources:
        sp = Path(s).expanduser().resolve()
        if not sp.is_file():
            raise SuiteError(f"source {sp} is missing")
        pinned.append({"path": str(sp), "sha256": _sha256(sp)})
    manifest = {"name": name, "role": role, "items": items_path.name, "items_sha256": _sha256(items_path),
                "sources": pinned, "gate": gate}
    _write_private(d / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    return load_suite(d)


def resolve_suite(spec: str) -> Suite:
    """`--suite <path|name>`: a manifest or suite directory path, else ~/.localbench/corpora/<name>/."""
    p = Path(spec).expanduser()
    return load_suite(p if p.exists() else CORPORA / spec)


# ---------------------------------------------------------------- metrics

def _reduced_reference(answer: dict):
    return answer.get("choice", answer.get("noul", answer.get("score")))


def _cell(name: str, meta: dict, label, ref, answer: dict | None) -> dict:
    """One question of one item: what the metrics need, with an ERROR (answer None) scored as the worst outcome:
    wrong, full confidence, the largest Brier / absolute error the question allows."""
    kind, options = meta["type"], meta["options"]
    c = {"type": kind, "q": name, "err": answer is None}
    if kind == "noul":
        c["gold"] = "true" if label else "false"
        p = answer["noul"] if answer else None
        c["pred"] = None if p is None else ("true" if p > 0.5 else "false")
        c["conf"] = 1.0 if p is None else max(p, 1.0 - p)
        c["brier"] = 1.0 if p is None else (p - float(label)) ** 2
        if ref is not None:
            c["agreement"] = c["pred"] == ("true" if ref > 0.5 else "false")
            c["mean_abs_dp"] = max(ref, 1.0 - ref) if p is None else abs(p - ref)
    else:
        probs = answer["probabilities"] if answer else None
        c["gold"] = label if kind == "choice" else str(label)
        if probs is None:
            c["pred"], c["conf"] = None, 1.0
        else:
            c["pred"] = answer["choice"] if kind == "choice" else max(options, key=lambda k: probs[k])
            c["conf"] = max(probs.values())
        if kind == "choice":
            c["brier"] = 2.0 if probs is None else sum((probs[k] - (k == label)) ** 2 for k in options)
            if ref is not None:
                c["agreement"] = c["pred"] == ref
        else:
            top = len(options) - 1
            c["mae"] = max(label, top - label) if probs is None else abs(answer["score"] - label)
            if ref is not None:
                c["agreement_mae"] = max(ref, top - ref) if probs is None else abs(answer["score"] - ref)
    c["correct"] = c["pred"] == c["gold"]
    return c


def _macro_f1(cells: list[dict]) -> float:
    classes = {(c["q"], c["gold"]) for c in cells} | {(c["q"], c["pred"]) for c in cells if c["pred"] is not None}
    tp, fp, fn = dict.fromkeys(classes, 0), dict.fromkeys(classes, 0), dict.fromkeys(classes, 0)
    for c in cells:
        gold, pred = (c["q"], c["gold"]), (c["q"], c["pred"])
        if c["correct"]:
            tp[gold] += 1
        else:
            fn[gold] += 1
            if c["pred"] is not None:
                fp[pred] += 1
    return sum(2 * tp[k] / (2 * tp[k] + fp[k] + fn[k]) for k in classes) / len(classes)


def _ece(cells: list[dict]) -> float:
    """Expected calibration error, ECE_BINS equal-width bins on the top-candidate probability."""
    bins: list[list[dict]] = [[] for _ in range(ECE_BINS)]
    for c in cells:
        bins[min(int(c["conf"] * ECE_BINS), ECE_BINS - 1)].append(c)
    n = len(cells)
    return sum(len(b) / n * abs(sum(c["correct"] for c in b) / len(b) - sum(c["conf"] for c in b) / len(b))
               for b in bins if b)


def _aggregate(item_cells: list[list[dict]], has_reference: bool) -> dict[str, tuple[float, int]]:
    """Quality metrics over every question of every item given (the denominator is all of them, ERRORs included):
    {name: (value, n)}."""
    by_type: dict[str, list[dict]] = {}
    for cells in item_cells:
        for c in cells:
            by_type.setdefault(c["type"], []).append(c)
    out = {}
    for kind, cs in by_type.items():
        n = len(cs)
        pre = f"decision.{kind}."
        out[pre + "accuracy"] = (sum(c["correct"] for c in cs) / n, n)
        out[pre + "macro_f1"] = (_macro_f1(cs), n)
        out[pre + "ece"] = (_ece(cs), n)
        out[pre + "error_rate"] = (sum(c["err"] for c in cs) / n, n)
        for key in ("brier", "mae", "mean_abs_dp", "agreement_mae"):
            if key in cs[0]:
                out[pre + key] = (sum(c[key] for c in cs) / n, n)
        if has_reference and "agreement" in cs[0]:
            out[pre + "agreement"] = (sum(c["agreement"] for c in cs) / n, n)
    return out


def _nearest_rank(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def _item_cells(arm: dict, repeat: int) -> list[list[dict]]:
    outs = {o["id"]: o for o in arm["outcomes"] if o["repeat"] == repeat}
    cells = []
    for item in arm["items"]:
        o = outs[item["id"]]
        ref = item.get("reference") or {}
        cells.append([_cell(name, meta, item["labels"][name], ref.get(name), o["answers"][name] if o["ok"] else None)
                      for name, meta in item["questions"].items()])
    return cells


def _warm(outcomes: list[dict]) -> list[float]:
    return [o["latency_s"] for o in outcomes if o["ok"] and not o["cold"]]


def _error_fraction(outcomes: list[dict]) -> float:
    return sum(not o["ok"] for o in outcomes) / len(outcomes) if outcomes else 0.0


def arm_metrics(arm: dict) -> dict:
    """Metrics of one arm: quality per question type (repeat 0; `spread` = [min, max] over repeats when there are
    several), error rate over every request, latency cold/warm p50/p95 overall and by question count, usage tokens,
    and for a hosted arm its cost."""
    repeats = arm["repeats"]
    per_repeat = [_aggregate(_item_cells(arm, r), arm["has_reference"]) for r in range(repeats)]
    metrics: dict[str, dict] = {}
    for name, (value, n) in per_repeat[0].items():
        m = {"value": value, "better": BETTER[name.rsplit(".", 1)[1]], "n": n}
        if repeats > 1:
            vals = [q[name][0] for q in per_repeat]
            m["spread"] = [min(vals), max(vals)]
        metrics[name] = m
    outs = arm["outcomes"]
    by_repeat = [[o for o in outs if o["repeat"] == r] for r in range(repeats)]
    metrics[ERROR_RATE] = {"value": _error_fraction(outs), "better": "lower", "n": len(outs)}
    cold = [o for o in outs if o["cold"] and o["ok"]]
    metrics["decision.latency.cold_s"] = {"value": cold[0]["latency_s"] if cold else None, "better": "lower",
                                          "n": len(cold)}
    warm = _warm(outs)
    metrics["decision.latency.warm_p50_s"] = {"value": _nearest_rank(warm, 0.5), "better": "lower", "n": len(warm)}
    metrics[LATENCY_P95] = {"value": _nearest_rank(warm, 0.95), "better": "lower", "n": len(warm)}
    if repeats > 1:
        rates = [_error_fraction(rs) for rs in by_repeat]
        metrics[ERROR_RATE]["spread"] = [min(rates), max(rates)]
        p95s = [_nearest_rank(_warm(rs), 0.95) for rs in by_repeat]
        if None not in p95s:
            metrics[LATENCY_P95]["spread"] = [min(p95s), max(p95s)]
    for k in sorted({o["n_questions"] for o in outs}):
        lat = _warm([o for o in outs if o["n_questions"] == k])
        for q, label in ((0.5, "p50"), (0.95, "p95")):
            metrics[f"decision.latency.q{k}.warm_{label}_s"] = {"value": _nearest_rank(lat, q), "better": "lower",
                                                                "n": len(lat)}
    ok = [o for o in outs if o["ok"]]
    for key in ("input_tokens", "output_tokens"):
        metrics[f"decision.usage.{key}_mean"] = {
            "value": sum(o["usage"][key] for o in ok) / len(ok) if ok else None, "better": "lower", "n": len(ok)}
    total_in = sum(o["usage"]["input_tokens"] for o in ok)
    metrics["decision.usage.input_tokens_total"] = {"value": total_in, "better": "lower", "n": len(ok)}
    if arm.get("usd_per_input_token") is not None:
        metrics[COST] = {"value": total_in * arm["usd_per_input_token"], "better": "lower", "n": len(ok)}
    return metrics


# ---------------------------------------------------------------- comparison

def _arm_of(x: dict) -> dict:
    if x.get("kind") == "run":
        return x["run"]["decision"]["local"]
    if "outcomes" in x and "items" in x:
        return x
    raise ValueError("compare() takes a decision receipt or one of its arms")


def baseline_ids(x: dict) -> set[str]:
    """The ids a baseline receipt answers to: the model its local arm scored, also as `<backend>/<model>` (the omp
    route spelling, e.g. ollama/qwen3.8:27b-mlx), and its recorded `setting` (a fixed configuration such as an effort
    level) when the receipt carries one."""
    model = _arm_of(x)["model"]
    backend = ((x.get("run") or {}).get("provenance") or {}).get("pins", {}).get("backend") or "ollama"
    ids = {model, f"{backend}/{model}"}
    if isinstance(x.get("setting"), str) and x["setting"]:
        ids.add(x["setting"])
    return ids


def _ci(xs: list[float]) -> list[float]:
    s = sorted(xs)
    return [s[max(0, math.ceil(0.025 * len(s)) - 1)], s[max(0, math.ceil(0.975 * len(s)) - 1)]]


def _band(cm: dict, bm: dict) -> float | None:
    spreads = [m.get("spread") for m in (cm, bm)]
    if any(s is None for s in spreads):
        return None
    return max(hi - lo for lo, hi in spreads)


def compare(candidate: dict, baseline: dict, *, seed: int = BOOTSTRAP_SEED,
            resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Candidate vs baseline on the same suite items. Verdict:
    - WORSE: some quality metric (accuracy, macro-F1, Brier, ECE, MAE vs labels; ERRORs count as wrong) is worse
      beyond noise;
    - BETTER: no such loss AND at least one measured win: warm p95 latency or error rate better beyond noise, or
      privacy (and cost, when the baseline recorded a price) because the baseline arm is hosted and the candidate local;
    - NOT_BETTER: otherwise (a tie on everything included).
    Noise is the A/A spread over repeats when both arms ran >= 2 repeats, else a seeded paired bootstrap over items
    (95% CI of the delta, `resamples` draws)."""
    c, b = _arm_of(candidate), _arm_of(baseline)
    if c["suite"]["items_sha256"] != b["suite"]["items_sha256"] or [i["id"] for i in c["items"]] != \
            [i["id"] for i in b["items"]]:
        raise ValueError("compare() needs both arms on the same pinned suite items")
    cm, bm = c["metrics"], b["metrics"]
    aa = c["repeats"] >= 2 and b["repeats"] >= 2
    quality = [k for k in cm if k in bm and k.rsplit(".", 1)[1] in QUALITY and cm[k]["value"] is not None
               and bm[k]["value"] is not None]
    boots: dict[str, list[float]] = {k: [] for k in (*quality, LATENCY_P95, ERROR_RATE)}
    if not aa:
        cc, bc = _item_cells(c, 0), _item_cells(b, 0)
        c_lat = {i["id"]: _warm([o for o in c["outcomes"] if o["id"] == i["id"]]) for i in c["items"]}
        b_lat = {i["id"]: _warm([o for o in b["outcomes"] if o["id"] == i["id"]]) for i in b["items"]}
        c_err = {i["id"]: _error_fraction([o for o in c["outcomes"] if o["id"] == i["id"]]) for i in c["items"]}
        b_err = {i["id"]: _error_fraction([o for o in b["outcomes"] if o["id"] == i["id"]]) for i in b["items"]}
        ids = [i["id"] for i in c["items"]]
        rng = random.Random(seed)
        n = len(ids)
        for _ in range(resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            qc = _aggregate([cc[i] for i in idx], c["has_reference"])
            qb = _aggregate([bc[i] for i in idx], b["has_reference"])
            for k in quality:
                if k in qc and k in qb:
                    boots[k].append(qc[k][0] - qb[k][0])
            pc = _nearest_rank([x for i in idx for x in c_lat[ids[i]]], 0.95)
            pb = _nearest_rank([x for i in idx for x in b_lat[ids[i]]], 0.95)
            if pc is not None and pb is not None:
                boots[LATENCY_P95].append(pc - pb)
            boots[ERROR_RATE].append(sum(c_err[ids[i]] - b_err[ids[i]] for i in idx) / n)

    def judge(k: str) -> dict:
        cv, bv = cm[k]["value"], bm[k]["value"]
        better = cm[k]["better"]
        sign = 1.0 if better == "higher" else -1.0
        row = {"candidate": cv, "baseline": bv, "delta": None if cv is None or bv is None else cv - bv,
               "better": better, "judgement": "unmeasured"}
        if row["delta"] is None:
            return row
        gain = sign * row["delta"]
        if aa:
            band = _band(cm[k], bm[k])
            if band is None:
                return row
            row["band"] = band
            row["judgement"] = "gain" if gain > band + EPS else "loss" if gain < -(band + EPS) else "within_noise"
        else:
            if len(boots[k]) < resamples // 2:
                return row
            lo, hi = _ci(boots[k])
            row["ci95"] = [lo, hi]
            g_lo, g_hi = sorted((sign * lo, sign * hi))
            row["p_value"] = (sum(sign * value <= 0 for value in boots[k]) + 1) / (len(boots[k]) + 1)
            row["judgement"] = "gain" if g_lo > EPS else "loss" if g_hi < -EPS else "within_noise"
        return row

    deltas, losses, wins = {}, [], []
    for k in quality:
        deltas[k] = {"class": "quality", **judge(k)}
        if deltas[k]["judgement"] == "loss":
            losses.append(k)
    for k, label in ((LATENCY_P95, "p95_latency"), (ERROR_RATE, "error_rate")):
        if k in cm and k in bm:
            deltas[k] = {"class": "latency" if k == LATENCY_P95 else "availability", **judge(k)}
            if deltas[k]["judgement"] == "gain":
                wins.append(label)
    if b.get("arm") == "hosted" and c.get("arm") == "local":
        wins.append("privacy")
        if (bm.get(COST) or {}).get("value"):
            wins.append("cost")
    for k in sorted(set(cm) & set(bm) - set(deltas)):
        cv, bv = cm[k]["value"], bm[k]["value"]
        deltas[k] = {"class": "reported", "candidate": cv, "baseline": bv, "better": cm[k]["better"],
                     "delta": None if cv is None or bv is None else cv - bv}
    benefit_p = {label: deltas[key].get("p_value") for key, label in ((LATENCY_P95, "p95_latency"),
                                                        (ERROR_RATE, "error_rate"))
                 if label in wins and deltas.get(key, {}).get("p_value") is not None}
    if benefit_p:
        from .stats import holm_reject
        accepted = holm_reject(benefit_p)
        wins = [label for label in wins if label not in benefit_p or accepted[label]]
    verdict = "WORSE" if losses else ("BETTER" if wins else "NOT_BETTER")
    return {"verdict": verdict, "quality_losses": losses, "wins": wins,
            "noise": "aa_spread" if aa else {"bootstrap": resamples, "seed": seed, "paired_over": "items"},
            "candidate": {"arm": c.get("arm"), "model": c.get("model")},
            "baseline": {"arm": b.get("arm"), "model": b.get("model")}, "deltas": deltas}


# ---------------------------------------------------------------- paired

PAIRED_RESAMPLES = 10000
PAIRED_SEED = 20261001


class PairedError(ValueError):
    """A paired scoring refusal: the receipt and the suite on disk cannot be scored pair-by-pair."""


def _paired_correct(kind: str, options: list[str], label, answer: dict) -> bool:
    """Correctness of one validated answer, the proj-b-uhc5 rule per question type: noul over 0.5 against
    the boolean label, score clamp-round against the integer label, choice against the label (which
    validate_answer already enforces to be the argmax)."""
    if kind == "noul":
        return (answer["noul"] > 0.5) == bool(label)
    if kind == "score":
        top = len(options) - 1
        return min(top, max(0, round(answer["score"]))) == int(label)
    return answer["choice"] == label


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p on discordant pairs: b local-only wins, c hosted-only (doubled lower tail).
    Exact integer ratio (no float conversion of 2**n, which overflows past ~1023 pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, float(Fraction(2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)), 1 << n)))


def _paired_table(pairs: list[tuple[bool, bool]], alpha: float, seed: int, resamples: int) -> dict:
    """One question type's paired verdict: accuracies, discordant counts, exact McNemar p, a seeded paired
    bootstrap 95% CI of the accuracy difference, and BETTER / NOT_WORSE / WORSE at `alpha`."""
    n = len(pairs)
    local = sum(1 for correct, _ in pairs if correct)
    hosted = sum(1 for _, correct in pairs if correct)
    diff = (local - hosted) / n
    b = sum(1 for local_ok, host_ok in pairs if local_ok and not host_ok)
    c = sum(1 for local_ok, host_ok in pairs if host_ok and not local_ok)
    p = _mcnemar_exact(b, c)
    rng = random.Random(seed)
    diffs = [sum(pairs[i][0] - pairs[i][1] for i in (rng.randrange(n) for _ in range(n))) / n
             for _ in range(resamples)]
    mean = sum(diffs) / resamples
    lo, hi = _ci(diffs)
    if p < alpha and diff > 0:
        verdict = "BETTER"
    elif p < alpha and diff < 0:
        verdict = "WORSE"
    else:
        verdict = "NOT_WORSE"
    return {"n": n, "local_accuracy": local / n, "hosted_accuracy": hosted / n, "diff_pp": diff * 100,
            "b": b, "c": c, "mcnemar_p": p,
            "bootstrap": {"mean_pp": mean * 100, "lo_pp": lo * 100, "hi_pp": hi * 100,
                          "resamples": resamples, "seed": seed},
            "verdict": verdict}


def _find_suite(pin: dict, corpora: Path = CORPORA) -> Suite:
    """The one suite on disk the receipt's pin names: exactly one items_sha256 match, then the name and
    manifest pins must agree and hosted reference answers must be present."""
    matches = []
    for manifest in sorted(corpora.rglob("manifest.json")) if corpora.is_dir() else []:
        try:
            suite = load_suite(manifest)
        except (SuiteError, OSError, ValueError):
            continue
        if suite.items_sha256 == pin.get("items_sha256"):
            matches.append(suite)
    if not matches:
        raise PairedError(f"no suite under {corpora} has items_sha256 {pin.get('items_sha256')}; "
                          "the receipt's suite is not on disk")
    if len(matches) > 1:
        raise PairedError(f"items_sha256 {pin.get('items_sha256')} matches several suites: "
                          + ", ".join(str(s.manifest) for s in matches))
    suite = matches[0]
    if suite.name != pin.get("name") or suite.manifest_sha256 != pin.get("manifest_sha256"):
        raise PairedError(f"receipt suite pins {pin.get('name')} {pin.get('manifest_sha256')} do not match "
                          f"the suite on disk {suite.name} {suite.manifest_sha256}")
    if not suite.has_reference:
        raise PairedError(f"suite {suite.name} carries no hosted reference answers")
    return suite


def paired(receipt: dict, *, alpha: float = 0.05, seed: int = PAIRED_SEED,
           resamples: int = PAIRED_RESAMPLES, corpora: Path | None = None) -> dict:
    """Per-question-type paired local-vs-hosted scoring of a banked decision receipt (or arm): n, accuracies,
    the difference, discordant counts, the exact McNemar p, a seeded paired-bootstrap 95% CI, and a
    BETTER / NOT_WORSE / WORSE verdict at `alpha`. Repeat-0 outcomes only (repeats are not independent
    pairs); a not-ok outcome scores incorrect like _cell, but a missing outcome, a missing hosted reference,
    or a malformed local answer refuses the verb instead of dropping silently."""
    if not isinstance(receipt, dict):
        raise PairedError("receipt is not an object")
    try:
        arm = _arm_of(receipt)
    except (ValueError, KeyError, TypeError) as exc:
        raise PairedError(f"receipt is not a decision run or arm: {exc}") from None
    pin = arm.get("suite")
    if not isinstance(pin, dict):
        raise PairedError("arm carries no suite pin")
    suite = _find_suite(pin, CORPORA if corpora is None else corpora)
    outs: dict[str, dict] = {}
    for outcome in arm.get("outcomes", []):
        if not isinstance(outcome, dict) or not isinstance(outcome.get("id"), str):
            raise PairedError("arm has an outcome without an id")
        if outcome.get("repeat", 0) == 0 and outcome["id"] not in outs:
            outs[outcome["id"]] = outcome
    by_type: dict[str, list[tuple[bool, bool]]] = {}
    missing_items: list[str] = []
    for item in suite.items:
        outcome = outs.get(item["id"])
        if outcome is None:
            missing_items.append(item["id"])
        if outcome is not None and "ok" not in outcome:
            raise PairedError(f"item {item['id']}: local outcome has no ok flag")
        for name, question in item["questions"].items():
            label = item["labels"][name]
            ref = (item.get("reference") or {}).get(name)
            if ref is None:
                raise PairedError(f"item {item['id']} question {name}: no hosted reference")
            if outcome is None or not outcome["ok"]:
                local_ok = False
            else:
                answers = outcome.get("answers")
                if not isinstance(answers, dict) or answers.get(name) is None:
                    raise PairedError(f"item {item['id']} question {name}: no local outcome")
                try:
                    validate_answer(question, answers[name], f"{item['id']}.{name}")
                except DecisionError as exc:
                    raise PairedError(f"item {item['id']} question {name}: malformed local outcome "
                                      f"({exc.detail})") from None
                local_ok = _paired_correct(question["type"], question_options(question), label,
                                           answers[name])
            host_ok = _paired_correct(question["type"], question_options(question), label, ref)
            by_type.setdefault(question["type"], []).append((local_ok, host_ok))
    return {"suite": {"name": suite.name, "manifest": str(suite.manifest),
                      "items_sha256": suite.items_sha256, "n_items": len(suite.items)},
            "alpha": alpha, "seed": seed,
            "missing_items": missing_items,
            "availability_errors": len(missing_items),
            "types": {kind: _paired_table(by_type[kind], alpha, seed, resamples) for kind in sorted(by_type)}}

# ---------------------------------------------------------------- run

def _json_get(url: str, timeout: float = 10.0):
    try:
        with backends.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except (OSError, ValueError, http.client.HTTPException):
        return None


def _root_url(base_url: str) -> str:
    u = urllib.parse.urlsplit(endpoint(base_url))
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path.removesuffix("/v1/systemone"), "", ""))


def ollama_pins(base_url: str, model: str) -> dict:
    """Backend pins of the Ollama answering at base_url: its own /api/version (not the CLI's), the app binary's
    sha16 (backends.Ollama.BINARY) and the model digest from /api/tags."""
    root_url = _root_url(base_url)
    version = _json_get(root_url + "/api/version")
    tags = _json_get(root_url + "/api/tags")
    models = {t.get("name"): t for t in (tags or {}).get("models") or [] if isinstance(t, dict)}
    tag = models.get(model) or models.get(f"{model}:latest") or {}
    return {"backend": "ollama",
            "backend_version": version.get("version") if isinstance(version, dict) else None,
            "backend_sha": backends.sha16(backends.Ollama.BINARY), "model": model,
            "model_digest": tag.get("digest") or None, "backend_args": ""}


# ---------------------------------------------------------------- context

def _content(v) -> str:
    """decision.content(): a string as is, an object or array as compact JSON."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def state_bytes(item: dict) -> int:
    return len(_content(item["state"]).encode("utf-8"))


def prompt_bytes(item: dict) -> int:
    """UTF-8 bytes of the item's longest System One prompt as decision.Compile builds it: {context, schema} JSON
    (every question's field, with all choices) plus "Requested field: <name>". Chat template and system prompt are
    outside it (CONTEXT_OVERHEAD_TOKENS)."""
    fields = []
    for name, q in item["questions"].items():
        crit = q.get("criteria")
        if q["type"] == "noul":
            crit = crit or {}
            choices = [(False, crit.get("false", "No")), (True, crit.get("true", "Yes"))]
        elif q["type"] == "choice":
            choices = [(k, k if v is None else v) for k, v in crit.items()]
        else:
            choices = [(str(i), v) for i, v in enumerate(crit)]
        fields.append({"name": name, "description": _content(q["instructions"]),
                       "choices": [{"code": chr(ord("A") + i), "value": value, "description": desc}
                                   for i, (value, desc) in enumerate(choices)]})
    data = json.dumps({"context": _content(item["state"]), "schema": fields}, ensure_ascii=False,
                      separators=(",", ":"))
    longest = max(len(json.dumps(name, ensure_ascii=False).encode("utf-8")) for name in item["questions"])
    return len(data.encode("utf-8")) + len("\n\nRequested field: ") + longest


def required_context(suite: Suite) -> dict:
    """The context (tokens) the suite's longest prompt needs, by the conservative estimate, and the sizes behind it."""
    longest = max(suite.items, key=prompt_bytes)
    biggest = prompt_bytes(longest)
    return {"required_tokens": math.ceil(biggest / CONTEXT_BYTES_PER_TOKEN) + CONTEXT_OVERHEAD_TOKENS,
            "max_prompt_bytes": biggest, "max_prompt_item": longest["id"],
            "max_state_bytes": max(state_bytes(it) for it in suite.items),
            "estimate": {"bytes_per_token": CONTEXT_BYTES_PER_TOKEN, "overhead_tokens": CONTEXT_OVERHEAD_TOKENS}}


def _tagged(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def resident_contexts(base_url: str, timeout: float = 120.0) -> dict[str, int | None] | None:
    """Resident models (tagged names) -> context_length from Ollama's /api/ps; None when Ollama does not answer at
    all. A reply that cannot be read is a ContextError: the context must be known, not assumed."""
    url = _root_url(base_url) + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:   # /api/ps stalls while the scheduler loads
            raw = r.read()
    except urllib.error.HTTPError as exc:
        exc.close()
        raise ContextError(f"{url} answered HTTP {exc.code}; resident context is unknown") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return None
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ContextError(f"{url} is not JSON; resident context is unknown") from None
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("name"), str)
                                           for m in models):
        raise ContextError(f"{url} has no readable models list; resident context is unknown")
    return {_tagged(m["name"]): (m["context_length"] if _count(m.get("context_length")) and m["context_length"]
                                 else None) for m in models}


def default_in_use(model: str) -> list[str]:
    """Why reloading `model` now could pull it from under another client ([] = none seen): gateway requests in
    flight for it, established non-gateway client sockets to Ollama, GPU time on its runner over one second.
    Telemetry that cannot be read is itself a reason (fail closed)."""
    reasons = []
    try:
        from localbench import gateway  # lazy: the gateway module is only needed for this check
        path = gateway.database_path()
        if path.is_file():
            store = gateway.GatewayStore(path)
            n = sum(store.active_requests(m) for m in {model, _tagged(model)})
            if n:
                reasons.append(f"{n} gateway request(s) in flight for {model}")
    except Exception as exc:   # noqa: BLE001 - unknown in-flight state refuses the reload
        reasons.append(f"gateway in-flight count unreadable ({type(exc).__name__})")
    try:
        for c in sysstats.inference_clients():
            if ("ollama" in c["servers"] and c["pid"] != os.getpid()
                    and not re.search(r"\bgateway\b", c.get("cmd") or "")):
                reasons.append(f"established Ollama client pid {c['pid']} ({c.get('name')})")
    except Exception as exc:   # noqa: BLE001
        reasons.append(f"Ollama client sockets unreadable ({type(exc).__name__})")
    try:
        t0, before = time.time(), sysstats.gpu_time_by_pid()
        time.sleep(1.0)
        after = sysstats.gpu_time_by_pid()
        target = _tagged(model)
        for r in sysstats.gpu_share(before, after, time.time() - t0, min_pct=0.5):
            if target in {_tagged(n) for n in r.get("model_names") or []} or r.get("model") in (model, target):
                reasons.append(f"its runner (pid {r['pid']}) used {r['pct']}% GPU in the last second")
    except Exception as exc:   # noqa: BLE001
        reasons.append(f"GPU activity unreadable ({type(exc).__name__})")
    return reasons


NUM_CTX_PARAM = re.compile(r"^num_ctx\s+(\S+)\s*$", re.M)
WEIGHTS_FROM = re.compile(r"^FROM\s+\S*sha256[-:]([0-9a-f]{64})\s*$", re.M)


def _native(base_url: str, path: str, body: dict | None = None, timeout: float = 60.0):
    """GET (body None) or POST an Ollama native API path on loopback; the parsed JSON reply. urllib's errors and
    ValueError (not JSON) propagate: each caller says what an unreadable answer means for it."""
    url = _root_url(base_url) + path
    req = (urllib.request.Request(url) if body is None else
           urllib.request.Request(url, encode(body), {"Content-Type": "application/json"}, method="POST"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        exc.msg = f"{path}: {_error_text(exc)}"
        exc.close()
        raise


def shipped_num_ctx(show: dict) -> int | None:
    """The num_ctx in a model's params layer, from Ollama's /api/show `parameters` text (`num_ctx   2050`); None
    when the model sets none. A value that is not a whole number is a ValueError."""
    m = NUM_CTX_PARAM.search(show.get("parameters") or "")
    if m is None:
        return None
    value = float(m.group(1))
    if not value.is_integer() or value < 1:
        raise ValueError(f"num_ctx parameter {m.group(1)!r} is not a positive whole number")
    return int(value)


def weights_digest(show: dict) -> str | None:
    """sha256 of the model's weights layer: the blob its /api/show modelfile names in `FROM .../sha256-<hex>`."""
    m = WEIGHTS_FROM.search(show.get("modelfile") or "")
    return m.group(1) if m else None


def other_parameters(show: dict) -> list[str]:
    """The params layer's lines other than num_ctx, whitespace-normalized and sorted (Ollama prints them in map
    order)."""
    lines = [" ".join(ln.split()) for ln in (show.get("parameters") or "").splitlines() if ln.strip()]
    return sorted(ln for ln in lines if ln.split(" ", 1)[0] != "num_ctx")


def check_shipped_context(base_url: str, model: str, required: int) -> int | None:
    """The num_ctx `model` ships (POST /api/show). Ollama's /v1/systemone takes no options and loads its runner at
    that value whatever loaded it before (2026-10-01: a warm-load at 21507 tokens was back at tev1's shipped 2050 by
    item 0), so a shipped num_ctx below `required` is a ContextError before any load or item, naming the derive verb.
    None: no num_ctx parameter (the after-run loaded_context readback detects a shrink), or Ollama does not answer
    (left to the run's availability errors)."""
    try:
        show = _native(base_url, "/api/show", {"model": model})
        shipped = shipped_num_ctx(show if isinstance(show, dict) else {})
    except urllib.error.HTTPError as exc:
        raise ContextError(f"{model}: Ollama answered HTTP {exc.code} ({exc.msg}); its shipped num_ctx is "
                           "unknown") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return None
    except (ValueError, UnicodeDecodeError) as exc:
        raise ContextError(f"{model}: /api/show unreadable ({exc}); its shipped num_ctx is unknown") from None
    if shipped is not None and shipped < required:
        raise ContextError(f"{model} ships num_ctx {shipped} in its parameters, and /v1/systemone loads its runner at "
                           f"that context whatever loaded it before; this suite's longest prompt needs {required} "
                           f"tokens. Derive a model with the context and run that: localbench decision derive "
                           f"ollama:{model} --num-ctx {required}")
    return shipped


def _launchctl_getenv(name: str) -> str | None:
    """The launchd user-session value of `name` (what Ollama.app inherits); None when unset or unreadable."""
    try:
        out = subprocess.run(["launchctl", "getenv", name], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def loaded_model_cap() -> dict:
    """Ollama's effective loaded-model cap: {cap, source}. OLLAMA_MAX_LOADED_MODELS from the app's launchd
    environment when set and positive, else Ollama's automatic default (DEFAULT_MODELS_PER_GPU per Metal GPU). A set
    value that is not a whole number is a ContextError: the cap is unknown."""
    raw = _launchctl_getenv("OLLAMA_MAX_LOADED_MODELS")
    if raw is not None:
        if not raw.isdigit():
            raise ContextError(f"OLLAMA_MAX_LOADED_MODELS={raw!r} in launchd is not a whole number; Ollama's "
                               "loaded-model cap is unknown")
        if int(raw) > 0:
            return {"cap": int(raw), "source": "launchctl OLLAMA_MAX_LOADED_MODELS"}
    return {"cap": DEFAULT_MODELS_PER_GPU * METAL_GPUS,
            "source": f"Ollama default {DEFAULT_MODELS_PER_GPU} per GPU x {METAL_GPUS} Metal GPU"}


def ensure_context(base_url: str, model: str, required: int, *, in_use=None, keep_alive: str = WARM_KEEP_ALIVE,
                   timeout: float = 600.0, allow_evict: bool = False) -> dict:
    """Make the resident runner of `model` hold at least `required` tokens of context before a run. First the
    model's shipped num_ctx (check_shipped_context: below `required` is a ContextError before anything loads). The
    runner must then sit at the context /v1/systemone will ask for: the shipped num_ctx when the model sets one (a
    runner at any other size is reloaded by the first item), else at least `required`. Resident at that: used as is.
    Not resident: loaded with POST /api/generate {model, prompt: "", options: {num_ctx}, keep_alive}; when the
    resident set is already at Ollama's loaded-model cap the load evicts a resident, so any other resident that
    `in_use(name)` names a user of is a ContextError before the load, unless `allow_evict` (recorded). Resident at
    another size (or unknown): reloaded the same way, unless `in_use(model)` names a client using it, then
    ContextError (never reload a model under another client). The load is read back from /api/ps; a context below
    `required` is a ContextError. Ollama not answering at all is recorded (`action: unreachable`) and left to the
    run's availability errors."""
    shipped = check_shipped_context(base_url, model, required)
    want = max(required, shipped or 0)
    rec = {"required_tokens": required, "model_num_ctx": shipped, "before": None, "action": "unreachable",
           "loaded_context": None, "evicted": [], "allow_evict": allow_evict}
    before = resident_contexts(base_url)
    if before is None:
        return rec
    target = _tagged(model)
    resident, ctx = target in before, before.get(target)
    rec["before"] = ctx
    if resident and ctx is not None and (ctx == shipped if shipped is not None else ctx >= required):
        rec.update(action="resident", loaded_context=ctx)
        return rec
    if not resident:
        cap = loaded_model_cap()
        rec["loaded_model_cap"] = cap
        if len(before) >= cap["cap"]:
            # Ollama picks the runner to unload (idle first, then shortest keep-alive) and may unload more to fit
            # memory; /api/ps does not say which, so every other resident is at risk.
            at_risk = {name: users for name in sorted(before) if (users := (in_use or default_in_use)(name))}
            rec["at_risk"] = at_risk
            if at_risk and not allow_evict:
                raise ContextError(
                    f"loading {model} would exceed Ollama's loaded-model cap ({len(before)} resident, cap "
                    f"{cap['cap']}: {cap['source']}) and evict a resident, and these residents are in use: "
                    + "; ".join(f"{name} ({', '.join(users)})" for name, users in at_risk.items())
                    + ". Refusing to evict them; retry when they are idle, or pass --allow-evict")
    if resident:
        users = (in_use or default_in_use)(model)
        if users:
            raise ContextError(f"{model} is resident with a {ctx}-token context, not the {want} tokens this "
                               f"suite needs, and is in use ({'; '.join(users)}): refusing to reload it under "
                               "another client; retry when it is idle")
    body = {"model": model, "prompt": "", "options": {"num_ctx": want}, "keep_alive": keep_alive}
    url = _root_url(base_url) + "/api/generate"
    req = urllib.request.Request(url, encode(body), {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except urllib.error.HTTPError as exc:
        detail = _error_text(exc)
        exc.close()
        raise ContextError(f"loading {model} with num_ctx={want} failed: HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise ContextError(f"loading {model} with num_ctx={want} failed: {exc}") from None
    after = resident_contexts(base_url)
    loaded = (after or {}).get(target)
    if loaded is None or loaded < required:
        raise ContextError(f"after loading {model} with num_ctx={want}, /api/ps reports context {loaded}; the "
                           "suite's longest prompt would not fit (the model's maximum may be smaller)")
    rec.update(action="reloaded" if resident else "loaded", loaded_context=loaded,
               evicted=sorted(set(before) - set(after) - {target}))
    return rec


# ---------------------------------------------------------------- derived models

class DeriveError(RuntimeError):
    """A derived decision model that cannot be planned, created or read back as asked."""


def derived_name(model: str, num_ctx: int) -> str:
    """Default name of `model` with num_ctx N: nimble:latest -> nimble-ctx32768, tev1:0.8b -> tev1:0.8b-ctx4096."""
    tagged = _tagged(model)
    name, tag = tagged.rsplit(":", 1)
    return f"{name}-ctx{num_ctx}" if tag == "latest" else f"{name}:{tag}-ctx{num_ctx}"


def _derive_call(base_url: str, path: str, body: dict | None = None, timeout: float = 60.0):
    try:
        doc = _native(base_url, path, body, timeout)
    except urllib.error.HTTPError as exc:
        raise DeriveError(f"Ollama answered HTTP {exc.code} ({exc.msg})") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise DeriveError(f"Ollama at {_root_url(base_url)} did not answer {path}: {exc}") from None
    except (ValueError, UnicodeDecodeError) as exc:
        raise DeriveError(f"{path} reply is not JSON ({exc})") from None
    if not isinstance(doc, dict):
        raise DeriveError(f"{path} reply is not an object")
    return doc


def _installed(base_url: str) -> dict[str, str | None]:
    models = _derive_call(base_url, "/api/tags").get("models")
    if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("name"), str)
                                           for m in models):
        raise DeriveError("/api/tags has no readable models list")
    return {_tagged(m["name"]): m.get("digest") for m in models}


def _identity(show: dict, what: str) -> tuple[str, int | None, list[str]]:
    weights = weights_digest(show)
    if weights is None:
        raise DeriveError(f"{what}: /api/show names no sha256 weights blob (FROM line); its weights cannot be checked")
    try:
        return weights, shipped_num_ctx(show), other_parameters(show)
    except ValueError as exc:
        raise DeriveError(f"{what}: {exc}") from None


def plan_derive(base_url: str, model: str, num_ctx: int, name: str | None = None) -> dict:
    """What `decision derive` would do, read from Ollama: the /api/create request deriving `name` FROM `model` with
    parameters {num_ctx}, the base's identity (tags digest, weights blob, other parameters) the readback must match,
    and `refuse` (a reason) or `noop` (the name already is this derivation). Never writes."""
    if not isinstance(num_ctx, int) or isinstance(num_ctx, bool) or num_ctx < 1:
        raise DeriveError(f"num_ctx {num_ctx!r} is not a positive whole number")
    name = name or derived_name(model, num_ctx)
    plan = {"base": model, "name": name, "num_ctx": num_ctx, "refuse": None, "noop": None,
            "request": {"model": name, "from": model, "parameters": {"num_ctx": num_ctx}, "stream": False}}
    if _tagged(name) == _tagged(model):
        plan["refuse"] = f"--name {name} is the base model itself; a derive never touches the base"
        return plan
    installed = _installed(base_url)
    if _tagged(model) not in installed:
        plan["refuse"] = f"{model} is not installed on this Ollama; nothing to derive from"
        return plan
    weights, base_ctx, params = _identity(_derive_call(base_url, "/api/show", {"model": model}), model)
    plan.update(base_digest=installed[_tagged(model)], base_weights=weights, base_num_ctx=base_ctx,
                base_parameters=params)
    if _tagged(name) in installed:
        got = _identity(_derive_call(base_url, "/api/show", {"model": name}), name)
        if got == (weights, num_ctx, params):
            plan["noop"] = (f"{name} already derives from {model} with num_ctx {num_ctx} (weights {weights[:12]}); "
                            "nothing to do")
        else:
            plan["refuse"] = (f"{name} already exists and is not {model} with num_ctx {num_ctx} (its weights "
                              f"{got[0][:12]} vs {weights[:12]}, num_ctx {got[1]}); pick another --name")
    return plan


def derive(base_url: str, plan: dict, timeout: float = 600.0) -> dict:
    """Create the planned derived model with POST /api/create and read it back: /api/show must report num_ctx ==
    N, the base's weights blob and the base's other parameters, and the base's own tags digest must be unchanged.
    DeriveError names any mismatch (a model created but not as asked is left in place and named, never deleted)."""
    if plan.get("refuse") or plan.get("noop"):
        raise DeriveError(f"plan is not a derivation to perform: {plan.get('refuse') or plan.get('noop')}")
    reply = _derive_call(base_url, "/api/create", plan["request"], timeout)
    if reply.get("error") or reply.get("status") != "success":
        raise DeriveError(f"/api/create {plan['name']}: {reply.get('error') or reply.get('status')!r}")
    installed = _installed(base_url)
    base, name = plan["base"], plan["name"]
    wrong = []
    if installed.get(_tagged(base)) != plan["base_digest"]:
        wrong.append(f"the base {base} changed digest ({plan['base_digest']} -> {installed.get(_tagged(base))})")
    if _tagged(name) not in installed:
        wrong.append(f"{name} is not listed by /api/tags")
    weights, ctx, params = _identity(_derive_call(base_url, "/api/show", {"model": name}), name)
    if ctx != plan["num_ctx"]:
        wrong.append(f"num_ctx reads back {ctx}, not {plan['num_ctx']}")
    if weights != plan["base_weights"]:
        wrong.append(f"weights {weights[:12]} are not the base's {plan['base_weights'][:12]}")
    if params != plan["base_parameters"]:
        wrong.append(f"other parameters {params} differ from the base's {plan['base_parameters']}")
    if wrong:
        raise DeriveError(f"{name} was created but does not read back as asked: {'; '.join(wrong)}")
    return {"name": name, "digest": installed[_tagged(name)], "num_ctx": ctx, "weights": weights, "base": base,
            "base_digest": plan["base_digest"]}


# ---------------------------------------------------------------- laya

LAYA_PREFIX = "laya:"
LAYA_BACKEND = "laya-mlx"
LAYA_SHIM = Path(__file__).resolve().parent / "shims" / "laya_systemone.py"
LAYA_READY_TIMEOUT = 300.0      # bounded wait for the checkpoint load (421M, fp16, from an external volume)
LAYA_STOP_TIMEOUT = 10.0
HF_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
GIT_SHA = re.compile(r"[0-9a-f]{40}")


class ShimError(RuntimeError):
    """The Laya shim did not start, exited, or did not become ready in time: no item was sent to it."""


@dataclass(frozen=True)
class LayaSpec:
    """`laya:<hf repo>[@<subfolder>]`: a Laya-MLX checkpoint in the local Hugging Face cache."""
    repo: str
    subfolder: str | None = None

    @property
    def name(self) -> str:
        """The model name the shim serves and echoes, and the receipt's `model`."""
        return self.repo + (f"@{self.subfolder}" if self.subfolder else "")


def parse_laya_spec(spec: str) -> LayaSpec:
    """`laya:<org>/<name>[@<relative subfolder>]` -> LayaSpec; ValueError for a local path, an empty part, or a
    subfolder that is absolute or leaves the repo."""
    if not isinstance(spec, str) or not spec.startswith(LAYA_PREFIX):
        raise ValueError(f"{spec!r}: not a laya:<hf repo>[@<subfolder>] spec")
    repo, at, sub = spec[len(LAYA_PREFIX):].partition("@")
    if not HF_REPO_ID.fullmatch(repo) or ".." in repo:
        raise ValueError(f"{spec!r}: {repo!r} is not a Hugging Face repo id <org>/<name>")
    if at and (not sub or sub.startswith("/") or any(p in ("", ".", "..") for p in sub.split("/"))):
        raise ValueError(f"{spec!r}: subfolder {sub!r} must be a relative path inside the repo")
    return LayaSpec(repo, sub if at else None)


def laya_venv() -> Path:
    """The venv whose python3 runs the shim (laya_mlx, mlx, numpy installed there, never in localbench's)."""
    return Path(os.environ.get("LOCALBENCH_LAYA_VENV")
                or Path.home() / ".local" / "share" / "laya-mlx" / "venv").expanduser()


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def snapshot_sha(model_dir, repo: str) -> str | None:
    """The Hugging Face cache snapshot (commit) `model_dir` was loaded from: `.../models--<org>--<name>/snapshots/
    <sha>[/<subfolder>]` for this repo; None for any other directory (then the run is not pinned)."""
    if not isinstance(model_dir, str) or not model_dir:
        return None
    parts = Path(model_dir).parts
    for i in range(1, len(parts) - 1):
        if (parts[i] == "snapshots" and parts[i - 1] == "models--" + repo.replace("/", "--")
                and GIT_SHA.fullmatch(parts[i + 1])):
            return parts[i + 1]
    return None


def laya_commit(source) -> str | None:
    """backends.git_commit of the checkout `source` (laya_mlx/__init__.py as the shim imported it: the editable
    ~/Developer/laya-mlx) lives in, with `-dirty` when its tracked files differ from HEAD; None outside a git tree."""
    sha = backends.git_commit(source) if isinstance(source, str) else None
    if sha is None:
        return None
    try:
        p = subprocess.run(["git", "-C", str(Path(source).absolute().parent), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True, timeout=10, check=False,
                           env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")})
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None if p.returncode else sha + ("-dirty" if p.stdout.strip() else "")


class LayaShim:
    """localbench/shims/laya_systemone.py run by the laya venv's python3 for one checkpoint on a free loopback port,
    as a context manager: enter starts it and waits (bounded by ready_timeout) until GET /health names the spec;
    exit always stops it (SIGTERM, then SIGKILL after LAYA_STOP_TIMEOUT), also when the run raises. ShimError (with
    the log's tail) when the venv is missing, the shim exits first, answers for another model, or stays unready.
    The shim runs offline (HF_HUB_OFFLINE=1): a checkpoint missing from the cache is refused, never downloaded."""

    def __init__(self, spec: LayaSpec, *, venv=None, script=LAYA_SHIM, log=None,
                 ready_timeout: float = LAYA_READY_TIMEOUT):
        self.spec, self.script, self.log_path, self.ready_timeout = spec, Path(script), log, ready_timeout
        self.venv = Path(venv) if venv is not None else laya_venv()
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.health: dict | None = None
        self._log = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LayaShim:
        if not (self.venv / "bin" / "python3").is_file():
            raise ShimError(f"no python3 in the laya venv {self.venv} (set LOCALBENCH_LAYA_VENV)")
        self.port = free_loopback_port()
        argv = ["python3", str(self.script), "--model", self.spec.repo, "--port", str(self.port)]
        if self.spec.subfolder:
            argv += ["--subfolder", self.spec.subfolder]
        env = dict(os.environ)
        env["PATH"] = str(self.venv / "bin") + os.pathsep + env.get("PATH", "")   # the venv's python3 first
        env["HF_HUB_OFFLINE"] = "1"
        # a+b: the child appends; _tail reads the end back for a ShimError.
        self._log = open(self.log_path, "a+b") if self.log_path else tempfile.TemporaryFile()  # noqa: SIM115
        try:
            self.proc = lifecycle.spawn(argv, env=env, stdin=subprocess.DEVNULL, stdout=self._log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            self._wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def _tail(self, n: int = 1500) -> str:
        if self._log is None:
            return ""
        self._log.flush()
        self._log.seek(0, os.SEEK_END)
        size = self._log.tell()
        self._log.seek(max(0, size - n))
        return self._log.read().decode("utf-8", "replace").strip()

    def _wait_ready(self) -> None:
        name, deadline = self.spec.name, time.monotonic() + self.ready_timeout
        start, last = time.monotonic(), "no probe attempted"
        while True:
            rc = self.proc.poll()
            if rc is not None:
                raise ShimError(f"laya shim for {name} exited with status {rc} before it was ready: {self._tail()}")
            try:
                with backends.urlopen(self.url + "/health", timeout=2.0) as r:
                    h = json.load(r)
            except Exception as exc:  # noqa: BLE001 — diagnosis only; the loop still enforces the deadline
                last, h = f"{type(exc).__name__}: {exc}", None
            if isinstance(h, dict):
                if h.get("ready") is not True or h.get("model") != name or not _count(h.get("max_len")):
                    raise ShimError(f"{self.url}/health answers {json.dumps(h)[:300]}, not a ready shim for {name}")
                self.health = h
                return
            if time.monotonic() >= deadline:
                connect = self._connect_state()
                listener = self._listener_state()
                raise ShimError(f"laya shim for {name} not ready within {self.ready_timeout:g}s "
                                f"(up {time.monotonic() - start:.1f}s, last probe: {last}, "
                                f"connect 127.0.0.1:{self.port}: {connect}, listener: {listener}): {self._tail()}")
            time.sleep(0.2)

    def _connect_state(self) -> str:
        """One-shot TCP probe distinguishing a refused listener from a blackholed one in timeout reports."""
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=2.0):
                return "open"
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}"

    def _listener_state(self) -> str:
        """Bind-probe census for timeout reports: free means the child never bound; held plus a
        connect timeout means the platform drops loopback SYNs to a live listener."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", self.port))
            return "port free (no listener)"
        except OSError as exc:
            return f"port held ({type(exc).__name__}: {exc})"
        finally:
            probe.close()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> int | None:
        """Stop the shim (idempotent); its exit status."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(LAYA_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self._log is not None:
            self._log.close()
            self._log = None
        return None if self.proc is None else self.proc.returncode

    def pins(self) -> dict:
        """Backend pins of the running shim: laya_mlx's version and laya_commit() of the checkout it was imported
        from, the HF snapshot sha of the loaded checkpoint as model_digest, dtype and batch size as backend_args."""
        h = self.health or {}
        return {"backend": LAYA_BACKEND, "backend_version": h.get("laya_version"),
                "backend_sha": laya_commit(h.get("laya_file")), "model": self.spec.name,
                "model_digest": snapshot_sha(h.get("model_dir"), self.spec.repo),
                "backend_args": f"dtype={h.get('dtype')} batch_size={h.get('batch_size')}",
                "laya_source": h.get("laya_file"), "model_dir": h.get("model_dir"), "mlx_version": h.get("mlx_version")}


def laya_notes(request: dict, doc: dict) -> dict:
    """Question names whose answer the shim reports as state-truncated, head-truncated (instructions/options cut),
    or longer than max_len untruncated; `unreported` names answers without a well-formed report."""
    out: dict[str, list[str]] = {"truncated": [], "head_truncated": [], "over_max_len": [], "unreported": []}
    for name in request["questions"]:
        rep = (doc["answers"].get(name) or {}).get("laya")
        if not (isinstance(rep, dict) and isinstance(rep.get("truncated"), bool)
                and isinstance(rep.get("head_truncated"), bool) and _count(rep.get("tokens"))
                and _count(rep.get("max_len"))):
            out["unreported"].append(name)
            continue
        for key, hit in (("truncated", rep["truncated"]), ("head_truncated", rep["head_truncated"]),
                         ("over_max_len", rep["tokens"] > rep["max_len"])):
            if hit:
                out[key].append(name)
    return out


def laya_counts(arm: dict) -> dict:
    """Distinct suite items per truncation report, over the arm's answered requests (every repeat)."""
    ok = [o for o in arm["outcomes"] if o["ok"]]

    def items(key: str) -> int:
        return len({o["id"] for o in ok if o["laya"][key]})

    return {"items_over_max_len": items("over_max_len"), "items_truncated": items("truncated"),
            "items_head_truncated": items("head_truncated"), "items_unreported": items("unreported"),
            "items_answered": len({o["id"] for o in ok}), "items": len(arm["items"])}


def run_laya(spec: LayaSpec, suite: Suite, *, shim: LayaShim | None = None, **kw) -> dict:
    """run_suite against a Laya shim started for this run and stopped after it, also when the run raises (`shim`:
    a configured LayaShim, default LayaShim(spec)). ShimError when it cannot be started; nothing was sent then."""
    with (shim if shim is not None else LayaShim(spec)) as server:
        return run_suite(server.url, spec.name, suite, laya=server, **kw)


class _Arm:
    def __init__(self, arm: str, model: str, url: str, api_key: str | None, usd_per_input_token: float | None,
                 laya: bool = False):
        self.arm, self.model, self.url, self.api_key = arm, model, url, api_key
        self.usd_per_input_token, self.laya = usd_per_input_token, laya
        self.outcomes: list[dict] = []

    def ask(self, item: dict, repeat: int, timeout: float) -> None:
        o = {"id": item["id"], "repeat": repeat, "cold": not self.outcomes, "n_questions": len(item["questions"])}
        try:
            request = build_request(self.model, item["state"], item["questions"])
            reply, latency, doc = ask(self.url, request, api_key=self.api_key, timeout=timeout)
            # Ollama echoes the requested model; another name means the pinned digest is not what answered.
            if self.arm == "local" and reply["model"] != self.model:
                raise DecisionError("invalid", f"answered by model {reply['model']!r}, not {self.model!r}", latency)
        except RequestError as exc:
            o.update(ok=False, error="refused", detail=str(exc), latency_s=None)
        except DecisionError as exc:
            o.update(ok=False, error=exc.kind, detail=exc.detail[:500],
                     latency_s=None if exc.latency_s is None else round(exc.latency_s, 6))
        else:
            o.update(ok=True, latency_s=round(latency, 6), model=reply["model"], answers=reply["answers"],
                     usage=reply["usage"])
            if self.laya:
                o["laya"] = laya_notes(request, doc)
        self.outcomes.append(o)

    def doc(self, suite: Suite, repeats: int) -> dict:
        items = [{"id": it["id"],
                  "questions": {n: {"type": q["type"], "options": question_options(q)}
                                for n, q in it["questions"].items()},
                  "labels": it["labels"], "state_bytes": state_bytes(it), "prompt_bytes": prompt_bytes(it),
                  "reference": ({n: _reduced_reference(a) for n, a in it["reference"].items()}
                                if "reference" in it else None)}
                 for it in suite.items]
        arm = {"arm": self.arm, "model": self.model, "endpoint": self.url, "suite": suite.pin(), "repeats": repeats,
               "has_reference": suite.has_reference, "usd_per_input_token": self.usd_per_input_token,
               "items": items, "outcomes": self.outcomes}
        arm["metrics"] = arm_metrics(arm)
        return arm


def _gate(metrics: dict, gate: dict) -> dict:
    out = {}
    for metric, bound in gate.items():
        (op, limit), = bound.items()
        value = (metrics.get(metric) or {}).get("value")
        ok = value is not None and (value >= limit if op == "min" else value <= limit)
        out[f"gate:{metric}"] = {"level": "MUST", "verdict": "PASS" if ok else "FAIL", "value": value, op: limit}
    return out


def _deterministic(arm: dict) -> dict | None:
    """Do repeats give the same answers (probabilities within TOL)? None with one repeat."""
    if arm["repeats"] < 2:
        return None
    first = {o["id"]: o for o in arm["outcomes"] if o["repeat"] == 0}
    differs = []
    for o in arm["outcomes"]:
        f = first[o["id"]]
        if o["repeat"] == 0 or not (o["ok"] and f["ok"]):
            continue
        for name, a in o["answers"].items():
            b = f["answers"][name]
            pa = a.get("probabilities") or {"true": a.get("noul")}
            pb = b.get("probabilities") or {"true": b.get("noul")}
            if a.get("choice") != b.get("choice") or any(abs(pa[k] - pb[k]) > TOL for k in pa):
                differs.append(f"{o['id']}#{o['repeat']}:{name}")
    # MUST: decision quality is measured as deterministic (label log-probs, no sampling); a drift voids that premise.
    return {"level": "MUST", "verdict": "FAIL" if differs else "PASS", "differs": differs[:10],
            "n_differs": len(differs)}


def run_suite(base_url: str, model: str, suite: Suite, *, repeats: int = 1, hosted: Hosted | None = None,
              baseline: Baseline | None = None, sampler=None, timeout: float = 120.0, host: dict | None = None,
              rev: str | None = None, feature: str | None = None, omp_module_sha: str | None = None,
              setting: str | None = None, in_use=None, allow_evict: bool = False,
              laya: LayaShim | None = None) -> dict:
    """Run every suite item `repeats` times against the local System One at base_url (loopback only) and, only when
    `hosted` is given, against the hosted arm on the same items in the same invocation (interleaved per item; a
    hosted failure stays a hosted ERROR, a local failure stays a local ERROR). Returns a `kind: run` receipt that
    `localbench show`/`validate` read: pins, per-item outcomes, metrics, gate conformance, the co-resident load the
    passed-in sysstats Sampler recorded (never voids the run), and the comparison against the hosted arm or a banked
    `baseline` (at most one). `verdict` states it at top level; a comparison that is not BETTER is a problem: the
    receipt is still evidence (negative), never proof. `feature` (registry id), `omp_module_sha` and `setting` (the
    fixed configuration this run measured, which a later `Baseline("fixed", setting, receipt)` must name) are written
    at top level when given. Before any item, ensure_context() gives the resident runner the context the suite's
    longest prompt needs (`in_use(model)` -> reasons another client is using it; default_in_use by default) or raises
    ContextError; the context loaded is pinned as `loaded_context`. `allow_evict` lets a load at Ollama's loaded-model
    cap evict an in-use resident (recorded in the context; an eviction is still a problem).

    `laya` (a started LayaShim serving `model` at base_url; run_laya starts one) replaces the Ollama specifics: no
    runner context to ensure (Laya truncates the state to the checkpoint's max_len instead of refusing), so the
    context is {action: n/a, max_len, ...} with items_over_max_len from the shim's per-answer reports, the pins are
    LayaShim.pins(), and decision.laya.truncated_items (a diagnostic metric, never a gate) counts the items whose
    state was cut."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not isinstance(model, str) or not model.strip():
        raise RequestError("model is required")
    if hosted is not None and baseline is not None:
        raise ValueError("one comparison per run: the hosted arm or a banked baseline, not both")
    if baseline is not None and (baseline.kind not in BASELINE_KINDS or not baseline.id):
        raise ValueError(f"baseline kind must be one of {BASELINE_KINDS} with a non-empty id")
    if baseline is not None and baseline.id not in baseline_ids(baseline.receipt):
        raise ValueError(f"baseline id {baseline.id!r} is not what the baseline receipt measured "
                         f"({sorted(baseline_ids(baseline.receipt))})")
    # A baseline anchors the verdict: an unsound one (MUST FAIL, pins changed, unknown residency...) cannot make a
    # candidate BETTER. Bare arm docs carry no problems field and are taken as they are.
    if baseline is not None and baseline.receipt.get("kind") == "run" and baseline.receipt.get("problems") != []:
        raise ValueError(f"baseline {baseline.kind} {baseline.id!r} is not a sound receipt (problems: "
                         f"{baseline.receipt.get('problems')!r}); bank a clean baseline first")
    arms = [_Arm("local", model, endpoint(base_url), None, None, laya=laya is not None)]
    if hosted is not None:
        key = os.environ.get(hosted.api_key_env, "").strip()
        if not key:
            raise RequestError(f"hosted arm requested but {hosted.api_key_env} is not set; the hosted arm is never "
                               "run without its key nor used in place of the local arm")
        arms.append(_Arm("hosted", hosted.model, endpoint(hosted.base_url, allow_remote=True), key,
                         hosted.usd_per_input_token))
    if laya is None:
        context = required_context(suite)
        context |= ensure_context(base_url, model, context["required_tokens"], in_use=in_use,
                                  allow_evict=allow_evict)
    else:
        max_len = laya.health["max_len"]
        context = {"action": "n/a", "max_len": max_len, "head_max_len": laya.health.get("head_max_len"),
                   "loaded_context": max_len, "model_num_ctx": max_len, "required_tokens": None, "evicted": []}
    host = host if host is not None else sysstats.host()
    if rev is None:
        from localbench.__main__ import _rev  # lazy: __main__ imports this module's callers
        rev = _rev()
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    before = ollama_pins(base_url, model) if laya is None else laya.pins()
    t0 = time.perf_counter()
    early_stop: dict | None = None
    obf_gate: tuple[str, float] | None = None
    obf_points: dict[int, float] = {}
    if repeats == 1:
        # Reject-only O'Brien-Fleming interim looks (kit-v8t): at 25/50/75% of items,
        # stop a screen whose accuracy already fails beyond the nominal boundary. The
        # final look is the suite gate itself. Multi-repeat runs always complete.
        for metric in sorted(suite.gate):
            parts = metric.split(".")
            bound = suite.gate[metric]
            if len(parts) == 3 and parts[0] == "decision" and parts[2] == "accuracy" \
                    and set(bound) == {"min"}:
                obf_gate = (parts[1], float(bound["min"]))
                break
        if obf_gate is not None and len(suite.items) > 0:
            total = len(suite.items)
            obf_points = {n: f for f in sorted(stats.OBF_ALPHA) if (n := math.ceil(f * total)) < total}
    items_by_id = {it["id"]: it for it in suite.items}

    def _interim_correct() -> tuple[int, int]:
        """Correct/total gate cells, excluding infrastructure failures."""
        assert obf_gate is not None
        kind = obf_gate[0]
        correct = total = 0
        for o in arms[0].outcomes:
            if not o.get("ok") and o.get("error") in {"http", "timeout", "unavailable"}:
                continue
            it = items_by_id[o["id"]]
            ref = {n: _reduced_reference(a) for n, a in (it.get("reference") or {}).items()}
            for name, meta in it["questions"].items():
                if meta["type"] != kind:
                    continue
                total += 1
                ans = o["answers"][name] if o.get("ok") else None
                cell = _cell(name, {**meta, "options": question_options(meta)},
                             it["labels"][name], ref.get(name), ans)
                if cell["correct"]:
                    correct += 1
        return correct, total

    with sampler if sampler is not None else contextlib.nullcontext():
        for r in range(repeats):
            for item in suite.items:
                for arm in arms:
                    arm.ask(item, r, timeout)
                if (repeats == 1 and obf_gate is not None
                        and len(arms[0].outcomes) in obf_points):
                    look = obf_points[len(arms[0].outcomes)]
                    correct, total = _interim_correct()
                    if total > 0 and stats.obf_reject(correct, total, obf_gate[1], look):
                        early_stop = {"look": look, "asked": len(arms[0].outcomes),
                                      "items": len(suite.items), "correct": correct,
                                      "cells": total, "gate": f"decision.{obf_gate[0]}.accuracy",
                                      "gate_min": obf_gate[1]}
                if early_stop is not None:
                    break
            if early_stop is not None:
                break
    wall = time.perf_counter() - t0
    if early_stop is not None:
        asked = {o["id"] for o in arms[0].outcomes}
        suite = replace(suite, items=tuple(it for it in suite.items if it["id"] in asked))
    local, *rest = [a.doc(suite, repeats) for a in arms]
    hosted_doc = rest[0] if rest else None
    if laya is None:
        after = ollama_pins(base_url, model)
        try:
            context["after_run"] = (resident_contexts(base_url) or {}).get(_tagged(model))
        except ContextError:
            context["after_run"] = None
    else:
        after = laya.pins()
        context |= laya_counts(local) | {"shim_alive_after_run": laya.alive()}

    pins = {**before, "loaded_context": context["loaded_context"], "model_num_ctx": context["model_num_ctx"],
            "macos_build": host.get("macos_build"),
            "host_id": host.get("host_id"), "suite_manifest_sha256": suite.manifest_sha256,
            "suite_items_sha256": suite.items_sha256}
    pins_changed = {k: [before[k], after[k]] for k in ("backend_version", "backend_sha", "model_digest")
                    if before[k] != after[k]}
    if laya is None and context["action"] != "unreachable" and context["after_run"] != context["loaded_context"]:
        pins_changed["loaded_context"] = [context["loaded_context"], context["after_run"]]
    conformance = _gate(local["metrics"], suite.gate)
    errors = [o for o in local["outcomes"] if not o["ok"]]
    invalid = [o for o in errors if o["error"] == "invalid"]
    # MUST: a malformed answer is a model/server defect, not a score. Timeouts and HTTP failures are availability,
    # measured by decision.error_rate.
    conformance["decision.responses_valid"] = {"level": "MUST", "verdict": "FAIL" if invalid else "PASS",
                                               "invalid": len(invalid), "requests": len(local["outcomes"])}
    det = _deterministic(local)
    if det:
        conformance["decision.deterministic"] = det
    too_long = [o for o in errors if o["error"] == "context"]
    # MUST: a prompt the resident runner could not hold measured the previous client's load, not the model.
    conformance["decision.context_fits"] = {"level": "MUST", "verdict": "FAIL" if too_long else "PASS",
                                            "context_errors": len(too_long),
                                            "loaded_context": context["loaded_context"],
                                            "required_tokens": context["required_tokens"]}
    must_fail = sorted(k for k, e in conformance.items() if e["level"] == "MUST" and e["verdict"] != "PASS")
    during = sampler.summary() if sampler is not None else None
    problems = [f"MUST FAIL: {', '.join(must_fail)}"] if must_fail else []
    if before["model_digest"] is None:
        where = (f"{base_url}/api/tags" if laya is None
                 else f"{before.get('model_dir')} (not a Hugging Face cache snapshot of {laya.spec.repo})")
        problems.append(f"model digest of {model!r} unreadable from {where}: the run is not pinned")
    if laya is not None and before["backend_sha"] is None:
        problems.append(f"laya_mlx loaded from {before.get('laya_source')} is not in a readable git checkout: the "
                        "run is not pinned")
    elif laya is not None and before["backend_sha"].endswith("-dirty"):
        problems.append(f"the laya checkout of {before.get('laya_source')} has uncommitted changes: backend_sha "
                        f"{before['backend_sha']} does not pin the code that answered")
    if laya is not None and context["items_unreported"]:
        problems.append(f"the laya shim sent no truncation report for {context['items_unreported']} item(s)")
    if pins_changed:
        problems.append(f"PINS CHANGED mid-run (another generation measured part of it): {pins_changed}")
    unknown = (during or {}).get("resident_unknown_samples") or 0
    if unknown:
        problems.append(f"resident model state unknown in {unknown} sampler sample(s); latency is non-proof")
    if context["evicted"]:
        problems.append(f"loading {model} with num_ctx={context['required_tokens']} evicted resident model(s) "
                        f"{', '.join(context['evicted'])}")
    kinds: dict[str, int] = {}
    for o in errors:
        kinds[o["error"]] = kinds.get(o["error"], 0) + 1
    details = {"decision.suite": {k: v for k, v in suite.pin().items() if k != "sources"},
               "decision.context": {k: v for k, v in context.items() if k != "estimate"}}
    if early_stop is not None:
        details["decision.early_stop"] = early_stop
    if errors:
        details["decision.errors"] = {"count": len(errors), "by_kind": kinds,
                                      "first": [{"id": o["id"], "repeat": o["repeat"], "error": o["error"],
                                                 "detail": o["detail"][:200]} for o in errors[:5]]}
    metrics = dict(local["metrics"])
    if laya is not None:
        metrics["decision.laya.truncated_items"] = {"value": context["items_truncated"], "better": "lower",
                                                    "n": context["items_answered"]}
    comparison, against = None, {"kind": "none", "id": None}
    if hosted_doc is not None:
        metrics |= {f"hosted.{k}": v for k, v in hosted_doc["metrics"].items()}
        comparison, against = compare(local, hosted_doc), {"kind": "hosted", "id": hosted.model}
    elif baseline is not None:
        try:
            comparison, against = compare(local, baseline.receipt), {"kind": baseline.kind, "id": baseline.id}
        except ValueError as exc:
            if early_stop is None:
                raise
            problems.append(f"early stop at {early_stop['asked']}/{early_stop['items']} items: "
                            f"baseline comparison skipped ({exc})")
    if comparison is not None:
        details["decision.compare"] = {"verdict": comparison["verdict"], "wins": comparison["wins"],
                                       "quality_losses": comparison["quality_losses"], "baseline": against}
        if comparison["verdict"] != "BETTER":
            problems.append(f"not better than {against['kind']} {against['id']}: {comparison['verdict']}")
    run = {
        "label": "decision",
        "provenance": {"pins": pins, "fingerprint": {**before, "endpoint": local["endpoint"]},
                       "localbench_rev": rev, "created": created, "tiers": [suite.role], "repeats": repeats,
                       "label": "decision", "suite": suite.pin(),
                       "hosted": None if hosted is None else {"base_url": hosted.base_url, "model": hosted.model,
                                                              "api_key_env": hosted.api_key_env,
                                                              "usd_per_input_token": hosted.usd_per_input_token}},
        # Co-resident load is the measurement condition for decision latency: recorded, never `contended`.
        "verdicts": {"contended": False, "must_fail": must_fail, "preflight_problems": [], "allow_busy": False,
                     "pins_changed": pins_changed},
        "metrics": metrics,
        "conformance": conformance,
        "run_dir": None,
        "details": details,
        "system": {"during": during, "contention": list(getattr(sampler, "contention", []) or []),
                   "load_spikes": list(getattr(sampler, "load_spikes", []) or []), "host": host},
        "decision": {"wall_s": round(wall, 3), "context": context, "local": local, "hosted": hosted_doc,
                     "compare": comparison},
    }
    receipt = {"kind": "run", "problems": problems,
               "verdict": {"compare": comparison["verdict"] if comparison else "NONE", "baseline": against},
               "run": run}
    if feature is not None:
        receipt["feature"] = feature
    if omp_module_sha is not None:
        receipt["omp_module_sha"] = omp_module_sha
    if setting is not None:
        receipt["setting"] = setting
    return receipt
