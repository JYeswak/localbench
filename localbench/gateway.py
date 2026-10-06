"""Loopback-only OMP Ollama gateway with durable, finite residency leases."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import http.server
import json
import math
import os
import plistlib
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import venv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from . import corpus, sysstats
from .proxy import purpose as chat_purpose

HOST = "127.0.0.1"
PORT = sysstats.OLLAMA_GATEWAY_PORT
OLLAMA_ROOT = "http://127.0.0.1:11434"
LABEL = "com.localbench.ollama-gateway"
IDLE_SECONDS = 300.0
REQUEST_BODY_TIMEOUT_SECONDS = 30.0
GATEWAY_EXPORTS = "gateway-exports"
GATEWAY_EXPORT_MARKER = ".localbench-gateway-export"
RETRY_SECONDS = 15.0
READINESS_TIMEOUT = 20.0
MAX_BODY_BYTES = 128 * 1024 * 1024
AUX_CONCURRENCY_ENV = "LOCALBENCH_GATEWAY_AUX_CONCURRENCY"
MAX_GO_DURATION_NS = 2**63 - 1
# proj-b's Clef-Flash decision server (kit-jtq2): proj-b owns serve.py on :8010; the gateway only fronts it, so its
# requests are attributed (profile proj-b, feature proj-b-<route>) and yield to localbench's park windows. Never leased:
# localbench does not load, unload or stop it.
CLEF_ROOT = "http://127.0.0.1:8010"
CLEF_PROFILE = "proj-b"
CLEF_MODEL = "clef-flash"
CLEF_CONCURRENCY = 2
CLEF_BUSY_RETRY_S = 2
CLEF_FENCED_RETRY_S = 60
EXTERNAL_PROFILES = frozenset({CLEF_PROFILE})
_CLEF_ROUTE = re.compile(r"[a-z0-9-]{1,64}")
# Per-request rows (kit-8gh): bounded by age and count, so the contention analysis never grows the store.
REQUESTS_RETENTION_S = 7 * 24 * 3600
REQUESTS_MAX_ROWS = 200_000
VACUUM_INTERVAL_S = 30 * 24 * 3600
_DURATION_PART = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|μs|ms|s|m|h)")
_DURATION_UNIT_NS = {
    "ns": 1, "us": 1_000, "µs": 1_000, "μs": 1_000,
    "ms": 1_000_000, "s": 1_000_000_000,
    "m": 60_000_000_000, "h": 3_600_000_000_000,
}


class GatewayError(RuntimeError):
    """Gateway installation, storage, or lifecycle failure."""


class RouteError(GatewayError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def parse_finite_duration(value: str) -> float:
    """Parse finite Go-style durations as seconds; unbounded, negative, and overflowing values fail."""
    if not isinstance(value, str) or not value:
        raise ValueError("duration must be finite (for example 5m, 30m, or 2h)")
    if value == "0":
        return 0.0
    if value[0] in "+-":
        if value[0] == "-":
            raise ValueError("duration must not be negative")
        value = value[1:]
    if not value:
        raise ValueError("duration must be finite")
    total_ns = Decimal(0)
    offset = 0
    for match in _DURATION_PART.finditer(value):
        if match.start() != offset:
            raise ValueError(f"invalid finite duration: {value!r}")
        try:
            total_ns += Decimal(match.group(1)) * _DURATION_UNIT_NS[match.group(2)]
        except InvalidOperation as exc:
            raise ValueError(f"invalid finite duration: {value!r}") from exc
        if total_ns > MAX_GO_DURATION_NS:
            raise ValueError("duration exceeds the maximum supported finite lease")
        offset = match.end()
    if offset != len(value) or offset == 0:
        raise ValueError(f"invalid finite duration: {value!r}")
    return float(total_ns / Decimal(1_000_000_000))


def state_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".localbench" / "ollama-gateway"


def database_path(home: Path | None = None) -> Path:
    return state_dir(home) / "leases.sqlite"


def lock_path(home: Path | None = None) -> Path:
    return state_dir(home) / "service.lock"


def plist_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def launch_target() -> str:
    return f"gui/{os.getuid()}/{LABEL}"


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _pid_alive(pid: int) -> bool | None:
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


@dataclass(frozen=True)
class Route:
    kind: str
    profile: str | None = None
    upstream_path: str | None = None
    upstream: str = "ollama"
    feature: str = ""


def route_request(method: str, path: str, profiles: set[str] | dict[str, str]) -> Route:
    """Resolve allowed metadata and profile-prefixed inference paths; all other paths fail closed."""
    parsed = urlsplit(path)
    pathname = parsed.path
    if method == "GET" and pathname == "/healthz":
        return Route("health")
    if method == "GET" and pathname == "/api/tags":
        return Route("discovery", upstream_path="/api/tags")
    if method == "POST" and pathname == "/api/show":
        return Route("discovery", upstream_path="/api/show")
    if method == "POST" and pathname == "/v1/systemone":
        # Unprofiled loopback route for local judges and the proj-b CLI: same lease,
        # fence and in-flight accounting as profiled chat, attributed to profile "".
        return Route("inference", profile=None, upstream_path="/v1/systemone")
    if pathname == "/proj-b/clef-flash" or pathname.startswith("/proj-b/clef-flash/"):
        return _clef_route(method, pathname.removeprefix("/proj-b/clef-flash").removeprefix("/"))
    prefix = "/omp-profile/"
    if not pathname.startswith(prefix):
        raise RouteError(404, "route is not managed by the Ollama residency gateway")
    pieces = pathname[len(prefix):].split("/")
    if len(pieces) < 2 or not pieces[0] or not all(pieces[1:]):
        raise RouteError(404, "invalid OMP profile route")
    try:
        profile = unquote(pieces[0], errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise RouteError(400, "invalid profile route encoding") from exc
    if profile in {".", ".."} or "/" in profile or "\\" in profile:
        raise RouteError(400, "invalid OMP profile name")
    registered = set(profiles)
    if profile not in registered:
        raise RouteError(404, "OMP profile is not registered with the gateway")
    if method != "POST":
        raise RouteError(405, "OMP inference endpoints require POST")
    endpoint = "/".join(pieces[1:])
    if endpoint.startswith("v1/"):
        endpoint = endpoint[3:]
    if endpoint not in {"responses", "chat/completions", "completions", "embeddings", "systemone"}:
        raise RouteError(404, "unsupported OMP inference endpoint")
    return Route("inference", profile=profile, upstream_path=f"/v1/{endpoint}")


def _clef_route(method: str, tail: str) -> Route:
    """proj-b's Clef-Flash routes: GET /proj-b/clef-flash/ is serve.py's identity (unaccounted, like discovery);
    POST /proj-b/clef-flash/<route> is one decision call, attributed to profile proj-b as feature proj-b-<route>, sent
    to serve.py as /v1/systemone (serve.py answers any POST path; the name says what it serves). A trailing
    /v1/systemone after <route> is accepted and stripped: omp's typesafe provider appends it to its baseUrl."""
    if method == "GET":
        if tail:
            raise RouteError(404, "clef-flash GET serves only /proj-b/clef-flash/")
        return Route("discovery", upstream_path="/", upstream="clef")
    if method != "POST":
        raise RouteError(405, "clef-flash decision calls require POST")
    route = tail.removesuffix("/v1/systemone")
    if not _CLEF_ROUTE.fullmatch(route):
        raise RouteError(400, "clef-flash route must be 1-64 characters of [a-z0-9-]")
    return Route("inference", profile=CLEF_PROFILE, upstream_path="/v1/systemone", upstream="clef",
                 feature=f"proj-b-{route}")


def decision_purpose(payload: dict) -> str:
    """Purpose key for a System One decision request: 'decision' plus the question names it carries.
    On the wire questions is an object keyed by question id (ollama decision/types.go; omp
    judgment/typesafe.ts iterates `for (const id in request.questions)`), so the names are its
    keys; a list of names (or name fields) is accepted as a fallback. Only names are kept
    (capped in count and length); unknown shapes still count as 'decision'."""
    names = []
    questions = payload.get("questions")
    if isinstance(questions, dict):
        candidates = list(questions.keys())
    elif isinstance(questions, list):
        candidates = questions
    else:
        candidates = []
    for question in candidates[:32]:
        if isinstance(question, str):
            name = question
        elif isinstance(question, dict):
            name = question.get("name")
        else:
            continue
        if isinstance(name, str) and name.strip():
            names.append(name.strip()[:128])
    joined = ",".join(names)
    if len(joined) > 184:
        # Per-name caps alone cannot bound the join (32 x 128 chars exceeds the store's 256
        # purpose limit and would 503 a valid request); truncate with a stable hash suffix.
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
        joined = joined[:167] + "#" + digest
    return "decision" + (":" + joined if names else "")


def auxiliary_purpose(purpose: str) -> bool:
    """Return whether a forwarded inference purpose is auxiliary rather than measured work."""
    return purpose != "main" and not purpose.startswith("decision")


def request_purpose(route: Route, payload: dict) -> str:
    """Attribution key stored with the lease: decision questions for System One,
    the bench proxy's request-shape classification for chat endpoints."""
    if route.upstream_path == "/v1/systemone":
        return decision_purpose(payload)
    return chat_purpose(payload)


CAPTURE_RESPONSE_MAX_BYTES = 1024 * 1024

_capture_cache: dict[str, tuple[float, dict | None]] = {}


def _capture_active(spec) -> bool:
    """An opt-in spec only counts while enabled, unexpired, purposeful and capped."""
    return (isinstance(spec, dict) and spec.get("enabled") is True
            and isinstance(spec.get("until"), (int, float)) and spec["until"] > time.time()
            and isinstance(spec.get("purposes"), list) and bool(spec["purposes"])
            and isinstance(spec.get("max_items"), int) and not isinstance(spec.get("max_items"), bool)
            and spec["max_items"] >= 1)


def read_capture_spec() -> dict | None:
    """The active opt-in capture plan, or None. The spec file is re-read only when its mtime
    moved, so the per-request cost is one stat call when capture is off."""
    path = corpus.capture_spec_path()
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _capture_cache.pop(key, None)
        return None
    cached = _capture_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        spec = None
    plan = spec if _capture_active(spec) else None
    _capture_cache[key] = (mtime, plan)
    return plan


def capture_matches(plan: dict, purpose: str) -> bool:
    """A purpose is captured when named exactly or under a named family ('decision' covers
    'decision:effort,find')."""
    return any(isinstance(entry, str) and (purpose == entry or purpose.startswith(entry + ":"))
               for entry in plan.get("purposes", []))


_OMP_TOKEN = re.compile(r"omp/([^\s+]+)(?:\+([0-9a-fA-F]{7,64}))?")
_OMP_SHA = re.compile(r"(?:^|[\s(@])sha[:=]?([0-9a-fA-F]{7,64})(?:$|[\s).,;])")


def _omp_identity(user_agent: str | None) -> dict:
    """Best-effort omp version/sha from the User-Agent header, else nulls with a note.
    Only an explicit omp token counts; anything else is reported, never guessed."""
    if not user_agent:
        return {"omp_version": None, "omp_sha": None,
                "omp_note": "no User-Agent header to read an omp version from"}
    version, sha, match = None, None, _OMP_TOKEN.search(user_agent)
    if match is not None:
        version, sha = match.group(1) or None, match.group(2)
    if sha is None:
        loose = _OMP_SHA.search(user_agent)
        sha = loose.group(1) if loose is not None else None
    if version is None:
        return {"omp_version": None, "omp_sha": sha,
                "omp_note": "User-Agent carries no omp version token"}
    return {"omp_version": version, "omp_sha": sha, "omp_note": None}



def _capture_document(meta: dict, kind: str, body: bytes) -> bytes:
    try:
        content = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        content = {"base64": base64.b64encode(body).decode("ascii")}
    return json.dumps({"meta": meta, kind: content},
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _ollama_version(upstream_root: str) -> str | None:
    """Best-effort Ollama version for capture metadata, read from the same upstream the
    request goes to; None when the lookup fails."""
    try:
        with urlopen(upstream_root.rstrip("/") + "/api/version", timeout=2) as response:
            doc = json.load(response)
        version = doc.get("version") if isinstance(doc, dict) else None
        return version if isinstance(version, str) and version.strip() else None
    except Exception:  # noqa: BLE001 - capture metadata must never break forwarding
        return None


def capture_body(*, purpose: str, meta: dict, body: bytes, kind: str) -> Path | None:
    """Write one content-addressed capture file, or None when capture is off, capped, refused
    or fails. Never raises: a capture failure must not break forwarding."""
    try:
        return _capture_body(purpose=purpose, meta=meta, body=body, kind=kind)
    except Exception:  # noqa: BLE001 - see above
        return None


def _capture_body(*, purpose: str, meta: dict, body: bytes, kind: str) -> Path | None:
    plan = read_capture_spec()
    if plan is None or not capture_matches(plan, purpose):
        return None
    if not corpus.valid_capture_purpose(purpose):
        return None
    captured = corpus.capture_spec_path().parent / "captured"
    corpus.refuse_repo_path(captured)
    # The budget counts request documents (pairs); a response is admitted with its request, so a
    # max_items=1 budget still captures the response.
    if kind == "request" and corpus.capture_purpose_items(None, purpose) >= plan["max_items"]:
        return None
    digest = hashlib.sha256(body).hexdigest()
    leaf = f"{digest}.json" if kind == "request" else f"{digest}.response.json"
    target = captured / purpose / leaf
    if target.is_file():
        return target
    corpus.secure_dir(target.parent)
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, _capture_document(meta, kind, body))
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def _evict_requests(db, now: float) -> None:
    """Bound the requests table by age then count; the caller holds the write transaction."""
    db.execute("DELETE FROM requests WHERE ended_at < ?", (now - REQUESTS_RETENTION_S,))
    over = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] - REQUESTS_MAX_ROWS
    if over > 0:
        db.execute("""DELETE FROM requests WHERE request_id IN
                   (SELECT request_id FROM requests ORDER BY ended_at, request_id LIMIT ?)""", (over,))


class GatewayStore:
    """SQLite lease ledger; stores model/profile/timing metadata, never request or response content."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else database_path()
        _secure_dir(self.path.parent)
        self._initialize()
        self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5.0, factory=_ClosingConnection)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    name TEXT PRIMARY KEY,
                    route TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gateway_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO gateway_state(key,value) VALUES('accepting','1');

                CREATE TABLE IF NOT EXISTS leases (
                    model TEXT PRIMARY KEY,
                    last_completed_at REAL,
                    idle_expires_at REAL,
                    manual_expires_at REAL,
                    last_outcome TEXT,
                    last_error TEXT,
                    next_retry_at REAL
                );
                CREATE TABLE IF NOT EXISTS lease_profiles (
                    model TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    last_seen_at REAL NOT NULL,
                    PRIMARY KEY (model, profile)
                );
                CREATE TABLE IF NOT EXISTS active_requests (
                    request_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    owner_pid INTEGER NOT NULL DEFAULT 0,
                    instance_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS active_requests_model ON active_requests(model);
                CREATE TABLE IF NOT EXISTS park_fences (
                    model TEXT PRIMARY KEY,
                    fence_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS park_fences_id ON park_fences(fence_id);
                CREATE TABLE IF NOT EXISTS purpose_stats (
                    profile TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    bucket_start REAL NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    busy_s REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (profile, purpose, bucket_start)
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    profile TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    model TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    first_byte_at REAL,
                    ended_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    feature TEXT NOT NULL DEFAULT '',
                    abandon_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS requests_ended ON requests(ended_at);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(active_requests)")}
            if "purpose" not in columns:
                db.execute("ALTER TABLE active_requests ADD COLUMN purpose TEXT NOT NULL DEFAULT ''")
            if "first_byte_at" not in columns:
                db.execute("ALTER TABLE active_requests ADD COLUMN first_byte_at REAL")
            request_columns = {row["name"] for row in db.execute("PRAGMA table_info(requests)")}
            if "feature" not in request_columns:
                db.execute("ALTER TABLE requests ADD COLUMN feature TEXT NOT NULL DEFAULT ''")
            if "abandon_reason" not in request_columns:
                db.execute("ALTER TABLE requests ADD COLUMN abandon_reason TEXT")
            active_columns = {row["name"] for row in db.execute("PRAGMA table_info(active_requests)")}
            if "feature" not in active_columns:
                db.execute("ALTER TABLE active_requests ADD COLUMN feature TEXT NOT NULL DEFAULT ''")
            if "owner_pid" not in active_columns:
                db.execute("ALTER TABLE active_requests ADD COLUMN owner_pid INTEGER NOT NULL DEFAULT 0")

    def profile_paths(self) -> dict[str, str]:
        with self._connect() as db:
            return {row["name"]: row["route"] for row in db.execute("SELECT name, route FROM profiles ORDER BY name")}

    def set_accepting(self, accepting: bool) -> None:
        with self._connect() as db:
            db.execute("UPDATE gateway_state SET value=? WHERE key='accepting'", ("1" if accepting else "0",))

    def accepting(self) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT value FROM gateway_state WHERE key='accepting'").fetchone()
            return row is not None and row["value"] == "1"

    def park_fence(self, model: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT fence_id FROM park_fences WHERE model=?", (model,)).fetchone()
            return None if row is None else str(row["fence_id"])

    def fences(self) -> list[dict]:
        """Every active admission fence: {model, fence_id, created_at}, oldest first."""
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT model, fence_id, created_at FROM park_fences ORDER BY created_at, model")]

    def acquire_park_fence(self, models: list[str], fence_id: str, now: float | None = None) -> str | None:
        """Atomically fence every model unless one has an admitted request or another park fence."""
        names = sorted(set(models))
        if not names:
            return None
        if not fence_id or "\x00" in fence_id:
            raise GatewayError("park fence id is missing or invalid")
        if any(not model or len(model) > 512 or "\x00" in model for model in names):
            raise GatewayError("park fence model name is missing or invalid")
        now = time.time() if now is None else now
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for model in names:
                if db.execute("SELECT 1 FROM active_requests WHERE model=? LIMIT 1", (model,)).fetchone():
                    db.rollback()
                    return f"gateway request is in flight for this model: {model}"
                if db.execute("SELECT 1 FROM park_fences WHERE model=? LIMIT 1", (model,)).fetchone():
                    db.rollback()
                    return f"model already has an active park fence: {model}"
            db.executemany("INSERT INTO park_fences(model,fence_id,created_at) VALUES(?,?,?)",
                           ((model, fence_id, now) for model in names))
            db.commit()
        return None

    def release_park_fence(self, fence_id: str) -> None:
        if not fence_id:
            return
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM park_fences WHERE fence_id=?", (fence_id,))
            db.commit()

    def set_profiles(self, profiles: dict[str, str]) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM profiles")
            db.executemany("INSERT INTO profiles(name,route) VALUES(?,?)", sorted(profiles.items()))
            db.commit()

    def start_request(self, profile: str, model: str, instance_id: str, now: float | None = None,
                      purpose: str = "", feature: str = "") -> str:
        if not model or len(model) > 512 or "\x00" in model:
            raise GatewayError("request model name is missing or invalid")
        if "\x00" in purpose or len(purpose) > 256:
            raise GatewayError("request purpose is invalid")
        if "\x00" in feature or len(feature) > 256:
            raise GatewayError("request feature is invalid")
        now = time.time() if now is None else now
        request_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            accepting = db.execute("SELECT value FROM gateway_state WHERE key='accepting'").fetchone()
            if accepting is None or accepting["value"] != "1":
                db.rollback()
                raise GatewayError("gateway is draining and will not accept inference requests")
            external = profile in EXTERNAL_PROFILES
            # An external server (proj-b's Clef-Flash) shares the GPU with every model: any park fence holds it.
            fenced = (db.execute("SELECT 1 FROM park_fences LIMIT 1") if external else
                      db.execute("SELECT 1 FROM park_fences WHERE model=? LIMIT 1", (model,))).fetchone()
            if fenced is not None:
                db.rollback()
                raise GatewayError("model is fenced for park; retry after localbench unpark")

            pending = db.execute("SELECT last_outcome FROM leases WHERE model=?", (model,)).fetchone()
            if pending is not None and pending["last_outcome"] == "unload_pending":
                db.rollback()
                raise GatewayError("model unload is in progress; retry the inference request")
            # The bare loopback inference routes (e.g. POST /v1/systemone) carry no profile;
            # they share the lease, fence and in-flight accounting under profile "".
            if profile != "" and not external and db.execute(
                    "SELECT 1 FROM profiles WHERE name=?", (profile,)).fetchone() is None:
                db.rollback()
                raise GatewayError("request profile is not registered")
            if not external:
                # External servers are never leased, so expiry never tries to unload them.
                db.execute("""INSERT INTO leases(model) VALUES(?) ON CONFLICT(model) DO UPDATE SET
                           last_outcome=NULL, last_error=NULL, next_retry_at=NULL""", (model,))
                db.execute("""INSERT INTO lease_profiles(model,profile,last_seen_at) VALUES(?,?,?)
                           ON CONFLICT(model,profile) DO UPDATE SET last_seen_at=excluded.last_seen_at""",
                           (model, profile, now))
            db.execute("""INSERT INTO active_requests(request_id,model,profile,started_at,instance_id,purpose,feature,owner_pid)
                       VALUES(?,?,?,?,?,?,?,?)""",
                       (request_id, model, profile, now, instance_id, purpose, feature, os.getpid()))
            db.commit()
        return request_id

    def finish_request(
        self,
        request_id: str,
        completed: bool,
        outcome: str,
        now: float | None = None,
        abandon_reason: str | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._connect() as db:
            last = db.execute("SELECT value FROM gateway_state WHERE key='requests_vacuum_at'").fetchone()
            if now - (float(last[0]) if last is not None else 0.0) >= VACUUM_INTERVAL_S:
                try:
                    db.execute("VACUUM")
                except sqlite3.OperationalError:
                    pass    # another writer holds the lock; the next finish retries next month
                else:
                    db.execute("""INSERT INTO gateway_state(key,value) VALUES('requests_vacuum_at',?)
                               ON CONFLICT(key) DO UPDATE SET value=excluded.value""", (str(now),))
                db.commit()
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT model, profile, purpose, feature, started_at, first_byte_at FROM active_requests
                             WHERE request_id=?""", (request_id,)).fetchone()
            if row is None:
                db.commit()
                return
            model = row["model"]
            db.execute("DELETE FROM active_requests WHERE request_id=?", (request_id,))
            if outcome != "abandoned":
                # Reconciliation time is not the unknown request end time.
                bucket = math.floor(row["started_at"] / 3600) * 3600
                busy = max(0.0, now - row["started_at"])
                db.execute("""INSERT INTO purpose_stats(profile,purpose,bucket_start,requests,busy_s)
                           VALUES(?,?,?,1,?) ON CONFLICT(profile,purpose,bucket_start) DO UPDATE SET
                           requests=purpose_stats.requests+1, busy_s=purpose_stats.busy_s+excluded.busy_s""",
                           (row["profile"], row["purpose"] or "unspecified", bucket, busy))
            db.execute("""INSERT INTO requests(request_id,profile,purpose,model,started_at,first_byte_at,
                       ended_at,status,feature,abandon_reason) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (request_id, row["profile"], row["purpose"] or "unspecified", model, row["started_at"],
                        row["first_byte_at"], now, outcome, row["feature"] or "", abandon_reason))
            _evict_requests(db, now)
            if db.execute("SELECT 1 FROM active_requests WHERE model=? LIMIT 1", (model,)).fetchone() is None:
                db.execute("""UPDATE leases SET last_completed_at=CASE WHEN ? THEN ? ELSE last_completed_at END,
                           idle_expires_at=?,last_outcome=?,last_error=?,next_retry_at=NULL WHERE model=?""",
                           (int(completed), now, now + IDLE_SECONDS, outcome, abandon_reason, model))
            db.commit()

    def reconcile_abandoned_requests(
        self, established_connections: int | None, now: float | None = None
    ) -> int:
        """Finish requests whose gateway owner is gone or whose client socket no longer exists."""
        if established_connections is not None and established_connections < 0:
            raise GatewayError("gateway connection count cannot be negative")
        now = time.time() if now is None else now
        with self._connect() as db:
            rows = db.execute(
                "SELECT request_id, owner_pid FROM active_requests ORDER BY started_at, request_id"
            ).fetchall()

        abandoned: list[tuple[str, str]] = []
        for row in rows:
            owner_pid = int(row["owner_pid"] or 0)
            if owner_pid > 0 and _pid_alive(owner_pid) is False:
                reason = f"owner gateway pid {owner_pid} is gone"
            elif established_connections == 0:
                reason = "no established gateway client connections"
            else:
                continue
            abandoned.append((str(row["request_id"]), reason))

        for request_id, reason in abandoned:
            self.finish_request(request_id, completed=False, outcome="abandoned", now=now,
                                abandon_reason=reason)
        return len(abandoned)

    def mark_first_byte(self, request_id: str, now: float | None = None) -> None:
        """The upstream-forwarding timestamp: time from admission to response headers (gateway queue plus
        upstream time-to-first-byte). Missing rows stay missing; never raises for an unknown id."""
        now = time.time() if now is None else now
        with self._connect() as db:
            db.execute("UPDATE active_requests SET first_byte_at=? WHERE request_id=?", (now, request_id))
            db.commit()

    def refuse_request(self, profile: str, model: str, purpose: str = "", reason: str = "",
                       now: float | None = None) -> str:
        """Record an admission refusal (draining, fenced, unregistered, unload pending): no active row ever
        existed, so this inserts directly with a NULL first byte. The future aux-cap A/B counts these."""
        now = time.time() if now is None else now
        status = "draining" if "draining" in (reason or "") else "refused"
        request_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""INSERT INTO requests(request_id,profile,purpose,model,started_at,first_byte_at,
                       ended_at,status,feature) VALUES(?,?,?,?,?,?,?,?,?)""",
                       (request_id, profile, purpose or "unspecified", model, now, None, now, status, ""))
            _evict_requests(db, now)
            db.commit()
        return request_id

    def requests_report(self, since: float, until: float, *, purpose: str | None = None,
                        profile: str | None = None, limit: int = 200) -> list[dict]:
        """Per-request rows started in [since, until): status, wait, and abandonment reason. Busy duration is
        unknown for abandoned rows. Newest first, bounded; request and response content is not stored."""
        query = ("SELECT request_id, profile, purpose, model, started_at, first_byte_at, ended_at, status, feature, "
                 "abandon_reason FROM requests WHERE started_at >= ? AND started_at < ?")
        args: list = [since, until]
        if purpose is not None:
            query += " AND purpose=?"
            args.append(purpose)
        if profile is not None:
            query += " AND profile=?"
            args.append(profile)
        query += " ORDER BY started_at DESC, request_id LIMIT ?"
        args.append(max(1, min(int(limit), 5000)))
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [{"request_id": r[0], "profile": r[1], "purpose": r[2], "model": r[3], "started_at": r[4],
                 "first_byte_at": r[5], "ended_at": r[6], "status": r[7],
                 **({"feature": r[8]} if r[8] else {}),
                 **({"abandon_reason": r[9]} if r[9] else {}),
                 "queue_wait_s": (r[5] - r[4] if r[5] is not None else None),
                 "busy_s": None if r[7] == "abandoned" else r[6] - r[4]} for r in rows]

    def set_manual_lease(self, model: str, expires_at: float, now: float | None = None) -> None:
        current = time.time() if now is None else now
        if not model or not math.isfinite(expires_at) or not math.isfinite(current) or expires_at <= current:
            raise GatewayError("manual lease expiry must be finite and in the future")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT last_outcome FROM leases WHERE model=?", (model,)).fetchone()
            if row is not None and row["last_outcome"] == "unload_pending":
                db.rollback()
                raise GatewayError("model unload is in progress; finite keep was refused")
            db.execute("""INSERT INTO leases(model,manual_expires_at,last_outcome,last_error,next_retry_at)
                       VALUES(?,?,NULL,NULL,NULL) ON CONFLICT(model) DO UPDATE SET
                       manual_expires_at=excluded.manual_expires_at,last_outcome=NULL,last_error=NULL,next_retry_at=NULL""",
                       (model, expires_at))
            db.commit()

    def clear_manual_lease(self, model: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE leases SET manual_expires_at=NULL WHERE model=?", (model,))

    def active_requests(self, model: str | None = None) -> int:
        with self._connect() as db:
            if model is None:
                return int(db.execute("SELECT COUNT(*) FROM active_requests").fetchone()[0])
            return int(db.execute("SELECT COUNT(*) FROM active_requests WHERE model=?", (model,)).fetchone()[0])

    def claim_due(self, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        claimed = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("""SELECT l.model FROM leases l
                WHERE (l.idle_expires_at IS NOT NULL OR l.manual_expires_at IS NOT NULL)
                  AND MAX(COALESCE(l.idle_expires_at,0),COALESCE(l.manual_expires_at,0)) <= ?
                  AND (l.next_retry_at IS NULL OR l.next_retry_at <= ?)
                  AND COALESCE(l.last_outcome,'') NOT IN
                    ('unload_pending','unloaded_confirmed','not_resident_confirmed','unmanaged_on_remove')
                  AND NOT EXISTS (SELECT 1 FROM active_requests a WHERE a.model=l.model)
                ORDER BY l.model""", (now, now)).fetchall()
            for row in rows:
                model = row["model"]
                updated = db.execute("""UPDATE leases SET last_outcome='unload_pending',last_error=NULL
                    WHERE model=? AND COALESCE(last_outcome,'') NOT IN
                    ('unload_pending','unloaded_confirmed','not_resident_confirmed','unmanaged_on_remove')""", (model,))
                if updated.rowcount:
                    claimed.append({"model": model})
            db.commit()
        return claimed

    def mark_result(self, model: str, outcome: str, detail: str | None = None,
                    now: float | None = None, retry_seconds: float = RETRY_SECONDS) -> None:
        now = time.time() if now is None else now
        safe_detail = (detail or "")[:500]
        next_retry = now + retry_seconds if outcome in {"deferred", "unload_failed", "verification_failed"} else None
        with self._connect() as db:
            db.execute("""UPDATE leases SET last_outcome=?,last_error=?,next_retry_at=? WHERE model=?""",
                       (outcome, safe_detail or None, next_retry, model))

    def mark_unmanaged(self) -> None:
        """Stop retiring leases after OMP profile routing is removed."""
        with self._connect() as db:
            db.execute("""UPDATE leases SET last_outcome='unmanaged_on_remove',last_error=NULL,
                       next_retry_at=NULL""")

    def lease(self, model: str) -> dict | None:
        now = time.time()
        with self._connect() as db:
            row = db.execute("SELECT * FROM leases WHERE model=?", (model,)).fetchone()
            if row is None:
                return None
            profiles = [r[0] for r in db.execute(
                "SELECT profile FROM lease_profiles WHERE model=? ORDER BY profile", (model,))]
            active = db.execute("SELECT COUNT(*) FROM active_requests WHERE model=?", (model,)).fetchone()[0]
            result = dict(row)
            result["profiles"] = profiles
            result["active_requests"] = int(active)
            expires_at = max((value for value in (row["idle_expires_at"], row["manual_expires_at"])
                              if value is not None), default=None)
            expired = expires_at is not None and expires_at <= now
            if active == 0 and expired and result["last_outcome"] in (None, "active"):
                result["last_outcome"] = "expired"
            return result

    def leases(self) -> list[dict]:
        with self._connect() as db:
            models = [r[0] for r in db.execute("SELECT model FROM leases ORDER BY model")]
        return [lease for model in models if (lease := self.lease(model)) is not None]

    def purpose_report(self, since: float, until: float) -> list[dict]:
        """Per-profile, per-purpose request counts and busy seconds for hourly buckets
        overlapping [since, until). Read-only: buckets are summed whole, so window edges
        are approximate to the hour. No request or response content is stored or returned."""
        with self._connect() as db:
            rows = db.execute("""SELECT profile, purpose, SUM(requests), SUM(busy_s) FROM purpose_stats
                              WHERE bucket_start < ? AND bucket_start + 3600 > ?
                              GROUP BY profile, purpose ORDER BY profile, purpose""",
                              (until, since)).fetchall()
        return [{"profile": row[0], "purpose": row[1],
                 "requests": int(row[2] or 0), "busy_s": float(row[3] or 0.0)} for row in rows]


    def recover_instance(self, instance_id: str, previous_dead: bool, now: float | None = None) -> int:
        if not previous_dead:
            raise GatewayError("cannot reclaim in-flight requests until prior process death is proven")
        now = time.time() if now is None else now
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            models = [r[0] for r in db.execute(
                "SELECT DISTINCT model FROM active_requests WHERE instance_id=?", (instance_id,))]
            db.execute("DELETE FROM active_requests WHERE instance_id=?", (instance_id,))
            for model in models:
                if db.execute("SELECT 1 FROM active_requests WHERE model=? LIMIT 1", (model,)).fetchone() is None:
                    db.execute("""UPDATE leases SET idle_expires_at=?,last_outcome='interrupted_on_restart',
                               last_error='prior gateway process ended with request in flight',next_retry_at=NULL
                               WHERE model=?""", (now + IDLE_SECONDS, model))
            db.commit()
        return len(models)

    def recover_stale(self, current_instance: str, now: float | None = None) -> int:
        """Clear old leases only after this process holds the exclusive service lock."""
        now = time.time() if now is None else now
        with self._connect() as db:
            old_ids = [r[0] for r in db.execute(
                "SELECT DISTINCT instance_id FROM active_requests WHERE instance_id<>?", (current_instance,))]
        recovered = sum(self.recover_instance(old, previous_dead=True, now=now) for old in old_ids)
        with self._connect() as db:
            db.execute("""UPDATE leases SET last_outcome='deferred',
                       last_error='gateway restarted during unload; resident state will be rechecked',
                       next_retry_at=? WHERE last_outcome='unload_pending'""", (now,))
        return recovered


class InstanceLock:
    """An advisory exclusive lock: acquiring it proves an earlier lock-holding process has exited."""

    def __init__(self, path: Path | None = None):
        self.path = path or lock_path()
        self.fd: int | None = None
        self.instance_id = uuid.uuid4().hex

    def acquire(self) -> str:
        _secure_dir(self.path.parent)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise GatewayError("Ollama residency gateway is already running") from exc
        os.ftruncate(fd, 0)
        os.write(fd, self.instance_id.encode())
        os.fsync(fd)
        self.fd = fd
        return self.instance_id

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


class OllamaClient:
    """Strict native Ollama control calls; absence or malformed /api/ps is never treated as empty."""

    def __init__(self, root: str = OLLAMA_ROOT, timeout: float = 10.0):
        parsed = urlsplit(root)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise GatewayError("Ollama upstream must be a loopback HTTP endpoint")
        self.root = root.rstrip("/")
        self.timeout = timeout

    def resident_models(self) -> set[str]:
        with urlopen(self.root + "/api/ps", timeout=self.timeout) as response:
            payload = json.load(response)
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise GatewayError("Ollama /api/ps returned an unknown resident-model shape")
        names = set()
        for row in models:
            name = row.get("name") if isinstance(row, dict) else None
            if not isinstance(name, str) or not name:
                raise GatewayError("Ollama /api/ps returned a model without a name")
            names.add(name)
        return names

    def unload(self, model: str) -> None:
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = Request(self.root + "/api/generate", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=self.timeout) as response:
            response.read()


class GatewayPolicy:
    def __init__(self, store: GatewayStore, ollama: Any,
                 external_check: Callable[[str], tuple[bool | None, str | None]],
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.ollama = ollama
        self.external_check = external_check
        self.clock = clock
        self._last_maintenance = 0.0

    def expire_due(self, now: float | None = None) -> list[dict]:
        now = self.clock() if now is None else now
        outcomes = []
        for lease in self.store.claim_due(now=now):
            model = lease["model"]
            try:
                safe, reason = self.external_check(model)
            except Exception:
                safe, reason = None, "external-client activity probe failed"
            if safe is not True:
                detail = reason or "external-client activity is unknown"
                self.store.mark_result(model, "deferred", detail, now=now)
                outcomes.append({"model": model, "outcome": "deferred", "reason": detail})
                continue
            try:
                residents = self.ollama.resident_models()
            except Exception:
                detail = "Ollama resident state is unavailable; unload deferred"
                self.store.mark_result(model, "deferred", detail, now=now)
                outcomes.append({"model": model, "outcome": "deferred", "reason": detail})
                continue
            if model not in residents:
                self.store.mark_result(model, "not_resident_confirmed", None, now=now)
                outcomes.append({"model": model, "outcome": "not_resident_confirmed"})
                continue
            try:
                self.ollama.unload(model)
            except Exception:
                detail = "Ollama rejected or did not complete the unload request"
                self.store.mark_result(model, "unload_failed", detail, now=now)
                outcomes.append({"model": model, "outcome": "unload_failed", "reason": detail})
                continue
            try:
                after = self.ollama.resident_models()
            except Exception:
                detail = "Ollama /api/ps readback failed after unload; residency is unknown"
                self.store.mark_result(model, "verification_failed", detail, now=now)
                outcomes.append({"model": model, "outcome": "verification_failed", "reason": detail})
                continue
            if model in after:
                detail = "Ollama still reports the model resident after unload"
                self.store.mark_result(model, "verification_failed", detail, now=now)
                outcomes.append({"model": model, "outcome": "verification_failed", "reason": detail})
            else:
                self.store.mark_result(model, "unloaded_confirmed", None, now=now)
                outcomes.append({"model": model, "outcome": "unloaded_confirmed"})
        return outcomes


@dataclass(frozen=True)
class ClientConnection:
    pid: int
    name: str
    port: int


class ExternalActivityProbe:
    """Defer unloads for established external clients and observed Ollama GPU activity."""

    def __init__(self, run=subprocess.run, clock: Callable[[], float] = time.monotonic,
                 gpu_reader: Callable[[], dict] = sysstats.gpu_time_by_pid):
        self.runner = run
        self.clock = clock
        self.gpu_reader = gpu_reader
        try:
            self.last_gpu: dict | None = gpu_reader()
        except Exception:
            self.last_gpu = None
        self.last_at = clock()

    @staticmethod
    def _port(endpoint: str) -> tuple[str, int] | None:
        endpoint = endpoint.strip()
        if endpoint.startswith("["):
            match = re.fullmatch(r"\[([^]]+)\]:(\d+)", endpoint)
            return (match.group(1), int(match.group(2))) if match else None
        host, sep, port = endpoint.rpartition(":")
        if not sep or not port.isdigit():
            return None
        return host, int(port)

    def _clients(self, gateway_pid: int) -> list[ClientConnection]:
        binary = shutil.which("lsof")
        if not binary:
            raise GatewayError("lsof is unavailable; external Ollama clients cannot be ruled out")
        result = self.runner([binary, "-nP", "-iTCP:11434", "-sTCP:ESTABLISHED", "-Fpcn"],
                          capture_output=True, text=True, timeout=5, check=False)
        if result.returncode not in (0, 1):
            raise GatewayError("lsof could not establish the Ollama client set")
        if result.stderr.strip():
            raise GatewayError("lsof reported incomplete visibility into Ollama clients")
        clients = []
        current_pid = None
        current_name = ""
        for line in result.stdout.splitlines():
            if line.startswith("p"):
                try:
                    current_pid = int(line[1:])
                except ValueError:
                    current_pid = None
                current_name = ""
            elif line.startswith("c"):
                current_name = line[1:]
            elif line.startswith("n") and current_pid is not None and current_pid != gateway_pid:
                endpoints = line[1:].split("->", 1)
                if len(endpoints) != 2:
                    continue
                local, remote = self._port(endpoints[0]), self._port(endpoints[1])
                if local is None or remote is None:
                    raise GatewayError("lsof returned an unrecognized Ollama socket")
                if remote[1] == 11434 and remote[0] in {"127.0.0.1", "localhost", "::1"}:
                    clients.append(ClientConnection(current_pid, current_name, local[1]))
        return clients

    def check(self, model: str, gateway_pid: int | None = None) -> tuple[bool | None, str | None]:
        gateway_pid = os.getpid() if gateway_pid is None else gateway_pid
        now = self.clock()
        try:
            clients = self._clients(gateway_pid)
        except Exception:
            return None, "external-client activity telemetry is unavailable"
        if clients:
            try:
                self.last_gpu = self.gpu_reader()
            except Exception:
                self.last_gpu = None
            self.last_at = now
            return False, "established non-gateway Ollama client connection remains"
        try:
            after_gpu = self.gpu_reader()
        except Exception:
            self.last_at = now
            return None, "GPU activity telemetry is unavailable"
        if self.last_gpu is None:
            self.last_gpu, self.last_at = after_gpu, now
            return None, "GPU activity telemetry has no baseline"
        elapsed = max(now - self.last_at, 0.001)
        try:
            gpu_rows = sysstats.gpu_share(self.last_gpu, after_gpu, elapsed, min_pct=0.5)
        except Exception:
            self.last_gpu, self.last_at = after_gpu, now
            return None, "GPU activity telemetry is unavailable"
        active_models = [row for row in gpu_rows if row.get("model") == model]
        active_ollama = [row for row in gpu_rows if row.get("name") in {"ollama", "llama-server"}]
        self.last_gpu, self.last_at = after_gpu, now
        if active_models or active_ollama:
            return False, "Ollama GPU activity was observed during the safety window"
        return True, None


def _aux_concurrency() -> int:
    raw = os.environ.get(AUX_CONCURRENCY_ENV, "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise GatewayError(f"{AUX_CONCURRENCY_ENV} must be a non-negative integer") from exc
    if value < 0:
        raise GatewayError(f"{AUX_CONCURRENCY_ENV} must be a non-negative integer")
    return value


class GatewayHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address, handler, policy: GatewayPolicy, store: GatewayStore,
                 upstream_root: str, instance_id: str, aux_concurrency: int | None = None,
                 clef_root: str = CLEF_ROOT):
        if address[0] != HOST:
            raise GatewayError("the Ollama residency gateway may bind only to 127.0.0.1")
        self.policy = policy
        self.store = store
        self.upstream_root = upstream_root.rstrip("/")
        self.instance_id = instance_id
        self.aux_concurrency = _aux_concurrency() if aux_concurrency is None else aux_concurrency
        if self.aux_concurrency < 0:
            raise GatewayError("auxiliary concurrency cap must be non-negative")
        self._aux_slots = threading.BoundedSemaphore(self.aux_concurrency) if self.aux_concurrency else None
        self.clef_root = clef_root.rstrip("/")
        self._clef_slots = threading.BoundedSemaphore(CLEF_CONCURRENCY)
        super().__init__(address, handler)
        self.timeout = 0.5

    def acquire_aux(self) -> bool:
        return self._aux_slots is None or self._aux_slots.acquire(blocking=False)

    def release_aux(self) -> None:
        if self._aux_slots is not None:
            self._aux_slots.release()

    def acquire_clef(self) -> bool:
        return self._clef_slots.acquire(blocking=False)

    def release_clef(self) -> None:
        self._clef_slots.release()

    def maintenance(self) -> None:
        now = time.monotonic()
        if now - self.policy._last_maintenance < 1.0:
            return
        self.policy._last_maintenance = now
        self.policy.expire_due()


class GatewayRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server: Any

    def log_message(self, format: str, *args) -> None:
        # Request paths, headers and bodies are not written to LaunchAgent logs.
        return

    def _json_error(self, status: int, message: str, code: str | None = None, retry_after: int | None = None,
                    request_id: str | None = None) -> None:
        body = json.dumps({"error": message, **({"code": code} if code else {})}, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        if request_id is not None:
            self.send_header("X-Localbench-Request-Id", request_id)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise RouteError(411, "Content-Length is required")
        try:
            length = int(raw)
        except ValueError as exc:
            raise RouteError(400, "invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise RouteError(413, "request body exceeds the gateway limit")
        deadline = time.monotonic() + REQUEST_BODY_TIMEOUT_SECONDS
        original_timeout = self.connection.gettimeout()
        body = bytearray()
        try:
            while len(body) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RouteError(408, "request body read timed out")
                self.connection.settimeout(remaining)
                try:
                    chunk = self.rfile.read1(min(64 * 1024, length - len(body)))
                except TimeoutError as exc:
                    raise RouteError(408, "request body read timed out") from exc
                if not chunk:
                    raise RouteError(400, "request body ended before Content-Length")
                body.extend(chunk)
        finally:
            self.connection.settimeout(original_timeout)
        return bytes(body)

    def _capture_meta(self, route: Route, purpose: str, model: str) -> dict | None:
        """Capture metadata for an admitted request, or None when nothing would be stored.
        The Ollama version lookup runs only for requests a live spec would capture."""
        plan = read_capture_spec()
        if plan is None or not capture_matches(plan, purpose):
            return None
        return {"profile": route.profile or "", "purpose": purpose, "model": model,
                "user_agent": self.headers.get("User-Agent"),
                **_omp_identity(self.headers.get("User-Agent")),
                "ollama_version": _ollama_version(self.server.upstream_root), "t": time.time()}


    def _send_upstream(self, route: Route, body: bytes = b"", model: str | None = None) -> None:
        request_id = None
        upstream = None
        completed = False
        outcome = "interrupted"
        capture_meta: dict | None = None
        aux_slot = False
        clef = route.upstream == "clef"
        clef_slot = False
        if route.kind == "inference":
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json_error(400, "clef-flash request body must be JSON" if clef else
                                 "inference request must contain a JSON model field")
                return
            if clef:
                model, purpose, feature = CLEF_MODEL, route.feature, route.feature
                clef_slot = self.server.acquire_clef()
                if not clef_slot:
                    try:
                        self.server.store.refuse_request(CLEF_PROFILE, model, purpose,
                                                         "clef-flash concurrency cap reached")
                    except Exception:  # noqa: BLE001 - a lost refusal row must not break the 503
                        pass
                    self._json_error(503, f"clef-flash already has {CLEF_CONCURRENCY} calls in flight",
                                     code="busy", retry_after=CLEF_BUSY_RETRY_S)
                    return
            else:
                model = payload.get("model") if isinstance(payload, dict) else None
                if isinstance(model, str) and model.startswith("ollama/"):
                    model = model[len("ollama/"):]
                if not isinstance(model, str) or not model.strip():
                    self._json_error(400, "inference request must contain a non-empty model field")
                    return
                purpose = request_purpose(route, payload if isinstance(payload, dict) else {})
                aux_slot = auxiliary_purpose(purpose) and self.server.acquire_aux()
                if auxiliary_purpose(purpose) and not aux_slot:
                    try:
                        self.server.store.refuse_request(route.profile or "", model, purpose,
                                                         "auxiliary concurrency cap reached")
                    except Exception:
                        pass
                    self._json_error(503, "auxiliary concurrency cap reached")
                    return
                feature = self.headers.get("X-Localbench-Feature", "")
            try:
                request_id = self.server.store.start_request(
                    route.profile or "", model, self.server.instance_id, purpose=purpose, feature=feature,
                )
            except GatewayError as exc:
                reason = str(exc)
                message = ("gateway is draining and will not accept new requests" if "draining" in reason else
                           "model is fenced at the gateway (park or an agreed window); retry after it is released"
                           if "fenced" in reason else
                           "OMP profile is not registered with the residency gateway" if "not registered" in reason
                           else reason)
                try:
                    self.server.store.refuse_request(route.profile or "", model, purpose, reason)
                except Exception:  # noqa: BLE001 - a lost refusal row must not break the 503
                    pass
                if clef_slot:
                    self.server.release_clef()
                if clef:
                    self._json_error(503, message, code="fenced" if "fenced" in reason else "unavailable",
                                     retry_after=CLEF_FENCED_RETRY_S)
                else:
                    self._json_error(503, message)
                return
            capture_meta = None if clef else self._capture_meta(route, purpose, model)
            if capture_meta is not None:
                capture_body(purpose=purpose, meta=capture_meta, body=body, kind="request")
        headers = {}
        excluded = {"host", "content-length", "connection", "transfer-encoding", "expect", "accept-encoding"}
        for key, value in self.headers.items():
            if key.lower() not in excluded:
                headers[key] = value
        if body:
            headers["Content-Length"] = str(len(body))
        request = Request((self.server.clef_root if clef else self.server.upstream_root) + (route.upstream_path or "/"),
                          data=body if self.command in {"POST", "PUT", "PATCH"} else None,
                          headers=headers, method=self.command)
        terminal_tail = b""
        stream = False
        capture_response = (capture_meta is not None and route.kind == "inference"
                            and route.upstream_path == "/v1/systemone")
        response_parts: list[bytes] | None = [] if capture_response else None
        response_size = 0
        try:
            try:
                upstream = urlopen(request, timeout=3600)
            except HTTPError as response_error:
                upstream = response_error
            status = int(getattr(upstream, "status", None) or upstream.getcode() or 502)
            response_headers = upstream.headers
            stream = "text/event-stream" in response_headers.get("Content-Type", "").lower()
            if clef and status >= 500:
                status = 502    # serve.py down or failing: proj-b records NOT_RUN
            self.send_response(status)
            if request_id is not None:
                self.send_header("X-Localbench-Request-Id", request_id)
            hop_headers = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                           "te", "trailers", "transfer-encoding", "upgrade", "x-localbench-request-id"}
            for key, value in response_headers.items():
                if key.lower() not in hop_headers:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if request_id is not None:
                try:
                    self.server.store.mark_first_byte(request_id)
                except Exception:  # noqa: BLE001 - timing metadata must never break forwarding
                    pass
            read_chunk = getattr(upstream, "read1", upstream.read)
            while True:
                chunk = read_chunk(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if response_parts is not None:
                    response_size += len(chunk)
                    if response_size > CAPTURE_RESPONSE_MAX_BYTES:
                        response_parts = None
                    else:
                        response_parts.append(chunk)
                if stream:
                    sample = terminal_tail + chunk
                    if (b"data: [DONE]" in sample or b"response.completed" in sample
                            or b"response.failed" in sample or b"response.incomplete" in sample):
                        completed = True
                    terminal_tail = sample[-128:]
            if not stream:
                completed = True
            outcome = "completed" if completed else "stream_ended_without_terminal_event"
            if response_parts and completed and capture_meta is not None:
                capture_body(purpose=purpose,
                             meta={**capture_meta,
                                   "request_sha": hashlib.sha256(body).hexdigest()},
                             body=b"".join(response_parts), kind="response")
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError, URLError):
            outcome = "client_or_upstream_disconnected"
            if not getattr(self, "_headers_buffer", None):
                try:
                    self._json_error(502, "clef-flash upstream request failed" if clef else
                                     "Ollama upstream request failed", code="upstream" if clef else None,
                                     request_id=request_id if clef else None)
                except OSError:
                    pass
        finally:
            try:
                if upstream is not None:
                    upstream.close()
            finally:
                try:
                    if aux_slot:
                        self.server.release_aux()
                    if clef_slot:
                        self.server.release_clef()
                finally:
                    if request_id is not None:
                        self.server.store.finish_request(request_id, completed, outcome)

    def _dispatch(self) -> None:
        # Health and discovery routes resolve without the lease store, so a
        # held SQLite write (e.g. an unpark catalog refresh) or a slow sweep
        # never stalls /healthz. Only /omp-profile/ routes need the registry.
        try:
            route = route_request(self.command, self.path, ())
        except RouteError:
            try:
                profiles = self.server.store.profile_paths()
                route = route_request(self.command, self.path, profiles)
            except RouteError as exc:
                self._json_error(exc.status, str(exc))
                return
        if route.kind == "health":
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            body = self._body() if self.command == "POST" else b""
        except RouteError as exc:
            try:
                self._json_error(exc.status, str(exc))
            except OSError:
                pass
            return
        self._send_upstream(route, body)

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()


def create_server(address: tuple[str, int], store: GatewayStore, policy: GatewayPolicy,
                  upstream_root: str = OLLAMA_ROOT, instance_id: str = "test-instance",
                  clef_root: str = CLEF_ROOT) -> GatewayHTTPServer:
    return GatewayHTTPServer(address, GatewayRequestHandler, policy, store, upstream_root, instance_id,
                             clef_root=clef_root)


def _probe_health(timeout: float = 1.0) -> dict:
    """One timed /healthz probe: verdict, latency in seconds, and a reason.
    Both status verbs share this, so a slow probe shows its latency and cause
    instead of a bare UNAVAILABLE on one verb and healthy on the other."""
    start = time.monotonic()
    try:
        with urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=timeout) as response:
            healthy = json.load(response) == {"ok": True}
        reason = None if healthy else "gateway answered /healthz with an unexpected payload"
    except (OSError, ValueError) as exc:
        healthy, reason = False, str(exc) or type(exc).__name__
        if isinstance(exc, TimeoutError) or "timed out" in reason:
            reason = f"health probe timed out after {timeout} s: {reason}"
    return {"ok": healthy, "latency_s": time.monotonic() - start, "reason": reason}


def _health(timeout: float = 1.0) -> bool:
    return _probe_health(timeout)["ok"]


def _load_launchctl(run=subprocess.run) -> tuple[bool, str]:
    try:
        result = run(["launchctl", "print", launch_target()], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False, "launchctl status unavailable"
    return result.returncode == 0, result.stdout or result.stderr


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_sha(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in package.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _export_manifest(root: Path) -> dict | None:
    try:
        manifest = json.loads((root / GATEWAY_EXPORT_MARKER).read_text())
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _valid_export(root: Path, package_sha: str, gateway_sha: str) -> bool:
    manifest = _export_manifest(root)
    interpreter = root / "venv" / "bin" / "python3"
    package = root / "localbench"
    return bool(
        manifest
        and manifest.get("package_sha") == package_sha
        and manifest.get("gateway_sha") == gateway_sha
        and manifest.get("python") == "venv/bin/python3"
        and interpreter.is_file()
        and os.access(interpreter, os.X_OK)
        and package.is_dir()
        and _package_sha(package) == package_sha
        and _file_sha(package / "gateway.py") == gateway_sha
    )


def create_pinned_export(home: Path | None = None, source_root: Path | None = None) -> tuple[Path, str]:
    """Copy the package into a content-addressed runtime export with a local venv entry point."""
    home = home or Path.home()
    source_root = source_root or Path(__file__).resolve().parent.parent
    source_package = source_root / "localbench"
    gateway_sha = _file_sha(source_package / "gateway.py")
    package_sha = _package_sha(source_package)
    parent = home / ".localbench" / GATEWAY_EXPORTS
    target = parent / package_sha
    if target.exists():
        if not _valid_export(target, package_sha, gateway_sha):
            raise GatewayError(f"pinned gateway export is corrupt: {target}")
        return target, gateway_sha

    _secure_dir(parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{package_sha}.", dir=parent))
    try:
        shutil.copytree(source_package, temporary / "localbench",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        venv.EnvBuilder(with_pip=False, symlinks=True).create(temporary / "venv")
        manifest = {"package_sha": package_sha, "gateway_sha": gateway_sha, "python": "venv/bin/python3"}
        (temporary / GATEWAY_EXPORT_MARKER).write_text(json.dumps(manifest, sort_keys=True) + "\n")
        if not _valid_export(temporary, package_sha, gateway_sha):
            raise GatewayError(f"created gateway export failed verification: {temporary}")
        try:
            os.replace(temporary, target)
        except OSError:
            if not target.exists() or not _valid_export(target, package_sha, gateway_sha):
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target, gateway_sha


def launch_agent_code_status(home: Path | None = None) -> dict:
    """Compare the installed LaunchAgent pin with both the export and this loaded gateway module."""
    current_sha = _file_sha(Path(__file__).resolve())
    try:
        document = _read_managed_launch_agent(home)
        if document is None:
            raise GatewayError("gateway LaunchAgent is not installed")
        args = document["ProgramArguments"]
        workdir = Path(document.get("WorkingDirectory", ""))
        interpreter = Path(args[0])
        env = document.get("EnvironmentVariables", {})
        promoted_sha = env.get("LOCALBENCH_GATEWAY_SHA") if isinstance(env, dict) else None
        manifest = _export_manifest(workdir)
        package = workdir / "localbench"
        package_sha = manifest.get("package_sha") if manifest else None
        expected_interpreter = workdir / "venv" / "bin" / "python3"
        valid = bool(
            isinstance(promoted_sha, str)
            and len(promoted_sha) == 64
            and current_sha == promoted_sha
            and manifest
            and manifest.get("gateway_sha") == promoted_sha
            and manifest.get("python") == "venv/bin/python3"
            and interpreter.absolute() == expected_interpreter.absolute()
            and interpreter.is_file()
            and os.access(interpreter, os.X_OK)
            and workdir.is_dir()
            and package.is_dir()
            and package_sha == workdir.name
            and _package_sha(package) == package_sha
            and _file_sha(package / "gateway.py") == promoted_sha
        )
        return {"ok": valid, "code_sha": current_sha, "promoted_sha": promoted_sha,
                "export": str(workdir), "interpreter": str(interpreter),
                "reason": None if valid else "gateway code SHA or pinned export does not match"}
    except (GatewayError, OSError, ValueError, TypeError) as exc:
        return {"ok": False, "code_sha": current_sha, "promoted_sha": None,
                "reason": str(exc)}


def launchd_plist(home: Path | None = None, python: str | None = None, port: int = PORT) -> dict:
    home = home or Path.home()
    base = state_dir(home)
    export, gateway_sha = create_pinned_export(home)
    interpreter = export / "venv" / "bin" / "python3"
    if python is not None and Path(python).absolute() != interpreter.absolute():
        raise GatewayError("gateway LaunchAgent interpreter must come from its pinned export")
    return {
        "Label": LABEL,
        "ProgramArguments": [str(interpreter), "-m", "localbench", "gateway", "serve",
                             "--host", HOST, "--port", str(port)],
        "WorkingDirectory": str(export),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(base / "service.log"),
        "StandardErrorPath": str(base / "service.log"),
        "EnvironmentVariables": {"HOME": str(home), "LOCALBENCH_HOME": str(Path(__file__).resolve().parent.parent),
                                "LOCALBENCH_GATEWAY_SHA": gateway_sha},
    }


def _read_managed_launch_agent(home: Path | None = None) -> dict | None:
    path = plist_path(home)
    if path.is_symlink():
        raise GatewayError("gateway LaunchAgent path is a symlink")
    if not path.exists():
        return None
    try:
        document = plistlib.loads(path.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        raise GatewayError("gateway LaunchAgent plist is unreadable") from exc
    args = document.get("ProgramArguments") if isinstance(document, dict) else None
    if (document.get("Label") != LABEL or not isinstance(args, list)
            or len(args) < 7 or args[1:5] != ["-m", "localbench", "gateway", "serve"]):
        raise GatewayError("gateway LaunchAgent path is not owned by localbench")
    try:
        host_index = args.index("--host")
    except ValueError as exc:
        raise GatewayError("managed gateway LaunchAgent has no explicit loopback host") from exc
    if args[host_index + 1:host_index + 2] != [HOST]:
        raise GatewayError("managed gateway LaunchAgent does not bind to loopback")
    return document

def write_launch_agent(home: Path | None = None, python: str | None = None, port: int = PORT) -> Path:
    path = plist_path(home)
    _secure_dir(path.parent)
    _secure_dir(state_dir(home))
    existing = _read_managed_launch_agent(home)
    if existing is not None:
        if existing != launchd_plist(home, python, port):
            raise GatewayError("an existing Ollama gateway LaunchAgent differs; remove it before reinstalling")
        return path
    payload = plistlib.dumps(launchd_plist(home, python, port), fmt=plistlib.FMT_XML, sort_keys=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
        raise
    return path


def _wait_health(timeout: float = READINESS_TIMEOUT, clock=time.monotonic) -> bool:
    deadline = clock() + timeout
    while clock() < deadline:
        if _health(timeout=0.5):
            return True
        time.sleep(0.2)
    return _health(timeout=0.5)


def install_launch_agent(home: Path | None = None, port: int = PORT, run=subprocess.run,
                         wait_health: Callable[[float], bool] = _wait_health) -> Path:
    path = plist_path(home)
    was_installed = path.exists()
    path = write_launch_agent(home, port=port)
    loaded, _ = _load_launchctl(run)
    if loaded:
        if _health():
            return path
        raise GatewayError("gateway LaunchAgent is loaded but its loopback health check failed")
    bootstrapped = False
    try:
        run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
            capture_output=True, text=True, timeout=10, check=True)
        bootstrapped = True
        if not wait_health(READINESS_TIMEOUT):
            raise GatewayError("gateway LaunchAgent did not pass its loopback health check")
    except BaseException:
        if bootstrapped:
            run(["launchctl", "bootout", launch_target()],
                capture_output=True, text=True, timeout=10, check=False)
        if not was_installed:
            path.unlink(missing_ok=True)
        raise
    return path


def start_launch_agent(home: Path | None = None, run=subprocess.run,
                       wait_health: Callable[[float], bool] = _wait_health) -> bool:
    path = plist_path(home)
    if _read_managed_launch_agent(home) is None:
        raise GatewayError("gateway LaunchAgent is not installed")
    loaded, _ = _load_launchctl(run)
    if not loaded:
        run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
            capture_output=True, text=True, timeout=10, check=True)
    if not wait_health(READINESS_TIMEOUT):
        raise GatewayError("gateway LaunchAgent did not pass its loopback health check")
    return True


STOP_UNLOAD_TIMEOUT = 10.0
STOP_UNLOAD_POLL = 0.25
STOP_PROBE_TIMEOUT = 0.5


def _await_gateway_down(run, timeout: float = STOP_UNLOAD_TIMEOUT) -> None:
    """Return once launchd no longer lists the job and /healthz stops answering.
    A stop that returns while either still holds lets an immediate start read a stale
    healthy and no-op while the gateway is dying. GatewayError names the stuck side."""
    deadline = time.monotonic() + timeout
    while True:
        loaded, _ = _load_launchctl(run)
        if loaded:
            if time.monotonic() >= deadline:
                raise GatewayError(
                    f"gateway LaunchAgent still loaded {timeout:g} s after bootout; "
                    "not stopped")
            time.sleep(STOP_UNLOAD_POLL)
            continue
        if _probe_health(timeout=STOP_PROBE_TIMEOUT)["ok"]:
            if time.monotonic() >= deadline:
                raise GatewayError(
                    f"gateway still answers /healthz {timeout:g} s after launchd unload; "
                    "not stopped")
            time.sleep(STOP_UNLOAD_POLL)
            continue
        return


def stop_launch_agent(home: Path | None = None, run=subprocess.run,
                      unload_timeout: float = STOP_UNLOAD_TIMEOUT) -> bool:
    if _read_managed_launch_agent(home) is None:
        return False
    loaded, _ = _load_launchctl(run)
    if loaded:
        run(["launchctl", "bootout", launch_target()], capture_output=True, text=True, timeout=10, check=True)
    _await_gateway_down(run, unload_timeout)
    return True


def remove_launch_agent(home: Path | None = None, run=subprocess.run) -> bool:
    path = plist_path(home)
    if _read_managed_launch_agent(home) is None:
        return False
    stop_launch_agent(home, run)
    path.unlink()
    return True


def service_status(home: Path | None = None, run=subprocess.run) -> dict:
    loaded, output = _load_launchctl(run)
    plist = plist_path(home)
    plist_error = None
    try:
        managed_plist = _read_managed_launch_agent(home) is not None
    except GatewayError as exc:
        managed_plist = False
        plist_error = str(exc)
    probe = _probe_health()
    return {
        "label": LABEL,
        "plist_installed": plist.exists() or plist.is_symlink(),
        "plist_managed": managed_plist,
        "plist_error": plist_error,
        "launchd_loaded": loaded,
        "launchd_state": "running" if re.search(r"^\s*state = running\s*$", output, re.MULTILINE)
        else "loaded" if loaded else "not_loaded",
        "health": probe["ok"],
        "health_latency_s": probe["latency_s"],
        "health_reason": probe["reason"],
        "bind": f"{HOST}:{PORT}",
    }

def _existing_store(home: Path | None = None) -> GatewayStore | None:
    path = database_path(home)
    return GatewayStore(path) if path.is_file() else None


def status(home: Path | None = None, resident_state=... ) -> dict:
    store = _existing_store(home)
    actual = sysstats.ollama_residents() if resident_state is ... else resident_state
    resident_names = None if actual is None else [name for name, _expiry in actual]
    leases = store.leases() if store is not None else []
    owned = {row["model"] for row in leases
             if row["last_outcome"] not in {"unloaded_confirmed", "not_resident_confirmed", "unmanaged_on_remove"}}
    unowned = None if resident_names is None else sorted(set(resident_names) - owned)
    return {
        "service": service_status(home),
        "profiles": store.profile_paths() if store is not None else {},
        "leases": leases,
        "active_requests": store.active_requests() if store is not None else 0,
        "accepting_requests": store.accepting() if store is not None else None,
        "unowned_residents": unowned,
        "ollama_state": "unknown" if actual is None else "available",
        "persistence": "sqlite; request bodies, headers, prompts, and completions are not stored",
    }




def maintenance_loop(server: GatewayHTTPServer, stop_event: threading.Event, interval: float = 1.0) -> None:
    """Run due-lease sweeps off the accept thread: each sweep can block for
    seconds (store write lock, lsof, Ollama unload round-trips), and while it
    runs the accept loop must keep answering /healthz."""
    while not stop_event.wait(interval):
        try:
            server.maintenance()
        except Exception as exc:
            print(f"ollama-gateway maintenance failed: {exc}", file=sys.stderr, flush=True)


def _server(home: Path | None = None, host: str = HOST, port: int = PORT) -> None:
    if host != HOST:
        raise GatewayError("the Ollama residency gateway may bind only to 127.0.0.1")
    store = GatewayStore(database_path(home))
    lock = InstanceLock(lock_path(home))
    instance_id = lock.acquire()
    server = None
    prior = None
    try:
        store.recover_stale(instance_id)
        try:
            probe = ExternalActivityProbe()
        except Exception:
            probe = None
        def external_check(model: str) -> tuple[bool | None, str | None]:
            return ((None, "external-client activity telemetry is unavailable") if probe is None
                    else probe.check(model, os.getpid()))
        policy = GatewayPolicy(store, OllamaClient(), external_check)
        server = create_server((host, port), store, policy, instance_id=instance_id)
        store.set_accepting(True)
        stop_event = threading.Event()
        prior = signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
        sweeper = threading.Thread(target=maintenance_loop, args=(server, stop_event), daemon=True)
        sweeper.start()
        while not stop_event.is_set():
            server.handle_request()
    finally:
        if prior is not None:
            signal.signal(signal.SIGTERM, prior)
        if server is not None:
            server.server_close()
        lock.release()


def serve(host: str = HOST, port: int = PORT, home: Path | None = None) -> None:
    _server(home=home, host=host, port=port)


def ensure_port_free(host: str = HOST, port: int = PORT) -> None:
    if host != HOST:
        raise GatewayError("gateway must bind to 127.0.0.1")
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise GatewayError(f"gateway port {port} is already in use") from exc


def keep_state(model: str, duration: str | float, home: Path | None = None,
               now: float | None = None) -> float:
    seconds = parse_finite_duration(duration) if isinstance(duration, str) else float(duration)
    if not math.isfinite(seconds) or seconds <= 0:
        raise GatewayError("finite keep lease must be greater than zero")
    now = time.time() if now is None else now
    expires_at = now + seconds
    if not math.isfinite(expires_at):
        raise GatewayError("finite keep lease expiry is out of range")
    GatewayStore(database_path(home)).set_manual_lease(model, expires_at, now=now)
    return expires_at


def safe_to_unload(
    model: str, home: Path | None = None, *, park_fence_id: str | None = None
) -> tuple[bool | None, str | None]:

    store = _existing_store(home)
    if store is not None:
        if store.active_requests(model):
            return False, "gateway request is in flight for this model"
        fence_id = store.park_fence(model)
        if fence_id is not None and fence_id != park_fence_id:
            return False, "model has an active park fence"
        lease = store.lease(model)
        if lease and lease["last_outcome"] == "unload_pending":
            return False, "gateway unload is already in progress for this model"
    probe = ExternalActivityProbe()
    return probe.check(model)


def acquire_park_fence(models: list[str], fence_id: str, home: Path | None = None) -> tuple[bool, str | None]:
    """Persist a per-model admission fence, then retain the external-client fail-closed check."""
    names = sorted(set(models))
    if not names:
        return True, None
    store = GatewayStore(database_path(home))
    if refusal := store.acquire_park_fence(names, fence_id):
        return False, refusal
    try:
        for model in names:
            safe, reason = safe_to_unload(model, home, park_fence_id=fence_id)
            if safe is not True:
                store.release_park_fence(fence_id)
                return False, reason or "external Ollama client activity is unknown"
    except Exception:
        store.release_park_fence(fence_id)
        raise
    return True, None

def release_park_fence(fence_id: str, home: Path | None = None) -> None:
    GatewayStore(database_path(home)).release_park_fence(fence_id)



def active_request_count(home: Path | None = None) -> int:
    store = _existing_store(home)
    return store.active_requests() if store is not None else 0


def clear_manual_lease(model: str, home: Path | None = None) -> None:
    store = _existing_store(home)
    if store is not None:
        store.clear_manual_lease(model)


def mark_unloaded(model: str, home: Path | None = None) -> None:
    store = _existing_store(home)
    if store is not None:
        store.clear_manual_lease(model)
        store.mark_result(model, "unloaded_confirmed")


def plan_remove(home: Path | None = None) -> list[str]:
    from . import omp_profiles

    manager = omp_profiles.ProfileManager.current(state_dir(home), home)
    manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    manager.plan_revert(manifest)
    return sorted(manager.dirs)


def install(home: Path | None = None, port: int = PORT, run=subprocess.run,
            wait_health: Callable[[float], bool] = _wait_health) -> dict:
    from . import omp_profiles

    manager = omp_profiles.ProfileManager.current(state_dir(home), home)
    existing_plist = _read_managed_launch_agent(home)
    if existing_plist is not None and existing_plist != launchd_plist(home, port=port):
        raise GatewayError("an existing Ollama gateway LaunchAgent differs; remove it before reinstalling")
    manager.plan(port)
    manifest_existed = manager.manifest_path.exists()
    service_before = service_status(home, run)
    if service_before["launchd_loaded"] and not service_before["health"]:
        raise GatewayError("gateway LaunchAgent is loaded but unhealthy; reconcile it before install")
    if service_before["health"] and not service_before["launchd_loaded"]:
        raise GatewayError("an unmanaged process already occupies the gateway port")
    plist_was_installed = service_before["plist_installed"]
    store = GatewayStore(database_path(home))
    prior_profiles = store.profile_paths()
    prior_accepting = store.accepting()
    if not service_before["health"]:
        ensure_port_free(port=port)
    manifest = manager.install(port)
    routes = {name: omp_profiles.provider_url(name, port) for name in manager.dirs}
    try:
        store.set_profiles(routes)
        install_launch_agent(home=home, port=port, run=run, wait_health=wait_health)
        readback = omp_profiles.verify_omp_profiles(sorted(manager.dirs))
    except BaseException as exc:
        if manifest_existed:
            store.set_profiles(routes)
            store.set_accepting(True)
            raise GatewayError("existing OMP routes and gateway were retained; profile readback failed") from exc
        store.set_accepting(False)
        if store.active_requests():
            store.set_profiles(routes)
            store.set_accepting(True)
            raise GatewayError("OMP readback failed while gateway requests were active; routes were retained") from exc
        if not service_before["health"]:
            try:
                stop_launch_agent(home=home, run=run)
            except BaseException:
                store.set_profiles(routes)
                store.set_accepting(True)
                raise GatewayError("OMP readback failed and the gateway could not be stopped; routes were retained") from exc
        try:
            manager.revert(manifest)
        except BaseException as rollback_error:
            store.set_profiles(routes)
            store.set_accepting(True)
            if not service_before["health"] and plist_path(home).exists():
                start_launch_agent(home=home, run=run)
            raise GatewayError("OMP readback failed and profile rollback was blocked; routes were retained") from rollback_error
        store.set_profiles(prior_profiles)
        store.set_accepting(prior_accepting)
        if not service_before["health"] and not plist_was_installed:
            remove_launch_agent(home=home, run=run)
        raise
    return {"profiles": sorted(manager.dirs), "port": port, "manifest": manifest, "readback": readback}


def _gateway_established_connections(run=subprocess.run) -> int | None:
    binary = shutil.which("lsof")
    if not binary:
        return None
    try:
        result = run([binary, "-nP", f"-iTCP:{PORT}", "-sTCP:ESTABLISHED", "-Fpcn"],
                     capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in (0, 1) or result.stderr.strip():
        return None

    connections = 0
    for line in result.stdout.splitlines():
        if not line.startswith("n"):
            continue
        endpoints = line[1:].split("->", 1)
        if len(endpoints) != 2:
            return None
        local = ExternalActivityProbe._port(endpoints[0])
        remote = ExternalActivityProbe._port(endpoints[1])
        if local is None or remote is None:
            return None
        if local[1] == PORT:
            connections += 1
    return connections


def stop_preflight(run=subprocess.run) -> str | None:
    """Read-only stop guard; zero gateway client sockets makes stored rows reclaimable."""
    connections = _gateway_established_connections(run=run)
    if connections is None:
        return "gateway client connection state is unknown; stop refused"
    if connections:
        return "gateway has established client connections; stop refused"
    return None


def start(home: Path | None = None, run=subprocess.run,
          wait_health: Callable[[float], bool] = _wait_health) -> bool:
    return start_launch_agent(home=home, run=run, wait_health=wait_health)


def stop(home: Path | None = None, run=subprocess.run) -> bool:
    path = plist_path(home)
    if not path.exists():
        return False
    store = _existing_store(home)
    if store is not None:
        store.set_accepting(False)
    try:
        connections = _gateway_established_connections(run=run)
        if store is not None:
            store.reconcile_abandoned_requests(connections)
        if connections is None:
            raise GatewayError("gateway client connection state is unknown; stop refused")
        if connections:
            raise GatewayError("gateway has established client connections; stop refused")
        if store is not None and store.active_requests():
            raise GatewayError("gateway has in-flight requests after stale reconciliation; stop refused")
        return stop_launch_agent(home=home, run=run)
    except BaseException:
        if store is not None:
            store.set_accepting(True)
        raise


def remove(home: Path | None = None, run=subprocess.run) -> list[str]:
    from . import omp_profiles

    manager = omp_profiles.ProfileManager.current(state_dir(home), home)
    manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    manager.plan_revert(manifest)
    store = _existing_store(home)
    if store is not None:
        store.set_accepting(False)
        if store.active_requests():
            store.set_accepting(True)
            raise GatewayError("gateway has in-flight requests; remove refused")
    try:
        stop_launch_agent(home=home, run=run)
    except BaseException:
        if store is not None:
            store.set_accepting(True)
        raise
    try:
        reverted = manager.revert(manifest)
    except BaseException:
        if store is not None:
            store.set_accepting(True)
        if plist_path(home).exists():
            start_launch_agent(home=home, run=run)
        raise
    if store is not None:
        store.set_profiles({})
        store.mark_unmanaged()
    try:
        remove_launch_agent(home=home, run=run)
    except BaseException as exc:
        raise GatewayError("OMP profiles were reverted, but the stopped LaunchAgent plist remains") from exc
    return reverted
