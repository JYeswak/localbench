"""Release watch: new local models and runtimes worth a stage-1 screen, filed as candidate beads and queued for the
next standing live window. Discovery and screening only: nothing is pulled, installed, routed or adopted here.

Four streams, bounded by the tables below:
1. decision models on the Ollama library (/v1/systemone judges: nimble, tev1, their tags and successors);
2. extraction/chat families for memory and smol (qwen3.x) on the Ollama library and the HF publishers models.py
   already watches (models.WATCH_PUBLISHERS plus the orgs installed MLX models came from);
3. embedding families (bge, gte, nomic-embed; memory recall today is local/fast-bge-base-en-v1.5) on the Ollama
   library and their HF publishers;
4. runtimes: GitHub releases of ollama/ollama, ddalcu/mlx-serve and jundot/omlx (stable tags only).

Bounds: a model must belong to a FAMILIES row and its weights must fit beside qwen3.8 (MAX_WEIGHT_BYTES); at most
MAX_QUEUED screens wait in the queue for one standing window; a candidate past the cap is deferred (not marked seen)
and comes back on a later run once the window has taken the queue.

State is ~/.localbench/watch/seen.json (items already handled and the fetch units that have a baseline) and
queue.json (screens for the next window); both refuse any path inside a git work tree. The first successful fetch
of a unit (one HF query, one runtime repo, one Ollama library name) records what it lists as a baseline without
filing anything, so a fresh install or a new table row does not flood beads. A name that appears on the Ollama
library after the library itself was baselined is a new family: its tags are candidates. A unit whose fetch fails
changes nothing and is reported; other units still run.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import models

REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL = "com.localbench.release-watch"
OLLAMA_LIBRARY = "https://ollama.com/library"
GITHUB_API = "https://api.github.com/repos"
GITHUB_ACCEPT = {"Accept": "application/vnd.github+json"}
# A candidate must fit in memory beside the incumbent qwen3.8 27B: weight bytes (decimal GB), not download size.
MAX_WEIGHT_BYTES = 40 * 10**9
# Screens waiting for one standing live window.
MAX_QUEUED = 2
# Daily, at a quiet hour; launchd runs a missed slot at the next wake.
SCHEDULE = {"Hour": 6, "Minute": 17}
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".onnx", ".npz")
UNSTABLE_TAG = re.compile(r"(?i)(rc|alpha|beta|pre|dev)\.?\d*$")


@dataclass(frozen=True)
class Family:
    """A role-relevant model family. `include` must match the whole lowercase model name (Ollama library name or the
    HF repo name after the org); a hit of `exclude` anywhere in it puts the model out of scope."""
    role: str
    include: str
    exclude: str = ""


# First matching row wins, so embedding comes before the chat families whose names it could share.
FAMILIES = (
    Family("embedding", r"(bge|gte|nomic-embed)[\w.\-]*", exclude=r"vl\b|rerank"),
    Family("decision", r"(nimble|tev)\d*([.\-][\w.\-]*)?"),
    Family("extraction", r"qwen3(\.\d+)?([.\-][\w.\-]*)?", exclude=r"embed|coder|vl\b|vision"),
)
# Ollama library names checked even if the listing page stops showing them.
OLLAMA_PINNED = ("nimble", "tev1")
# HF chat-family search used against every watched publisher.
HF_EXTRACTION_SEARCH = "qwen3"
# HF publishers of the embedding families (fastembed's ONNX builds come from Qdrant).
HF_EMBEDDING_QUERIES = (("Qdrant", "bge"), ("BAAI", "bge"), ("nomic-ai", "nomic-embed"), ("thenlper", "gte"),
                        ("Alibaba-NLP", "gte"))
RUNTIMES = ("ollama/ollama", "ddalcu/mlx-serve", "jundot/omlx")

# Stage-1 screens (AGENTS.md: the screen can drop or hold a candidate and never adopts one).
DECISION_SUITE = "banking77-10"
DECISION_FEATURE = "auto-thinking"
SMOL_INCUMBENT = "ollama:qwen3.8:27b-mlx"
MLX_INCUMBENT = Path(".mlx-serve") / "models" / "ddalcu" / "Qwen3.6-35B-A3B-MLX-Serve-4bit"
SCREEN_AB = ("--tiers", "think,sess", "--repeats", "1", "--pairs", "1")
# Upstream-controlled strings that end up in queued argv and paths: anything else is refused, not filed.
SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SAFE_NAME = {"ollama": re.compile(r"[a-z0-9][a-z0-9._-]*"),
             "hf": re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"),
             "github": re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")}
SAFE_URL = re.compile(r"https://([a-z0-9.-]+)(/[A-Za-z0-9._~:%+/-]*)")
URL_HOST = {"ollama": "ollama.com", "hf": "huggingface.co", "github": "github.com"}
SHA40 = re.compile(r"[0-9a-f]{40}")
MLX_SERVE_ASSET = "mlx-serve-bin-macos-arm64.tar.gz"
NO_SCREEN = {
    "embedding": "no stage-1 suite proves recall-embeddings yet (registries/features.tsv proof_suite is -)",
    "decision-hf": "decision suites run on Ollama /v1/systemone; this needs an Ollama build of the repo first",
    "ollama/ollama": "Ollama.app supervises the one ollama serve; screening a release means upgrading it (adoption)",
    "jundot/omlx": "oMLX resolves from PATH for both ab legs; there is no side-by-side B-leg override yet",
}


class WatchError(Exception):
    pass


class FetchError(Exception):
    pass


Fetch = Callable[..., tuple[int, bytes]]
Br = Callable[[list[str]], tuple[int, str, str]]


# --- pure helpers ---------------------------------------------------------------------------------------------------

def role_of(name: str) -> str | None:
    """The FAMILIES role of a model name (`qwen3.8`, `Qwen3.8-27B-4bit`, `bge-m3`), else None (out of scope)."""
    n = name.split("/")[-1].lower()
    for f in FAMILIES:
        if re.fullmatch(f.include, n):
            return None if f.exclude and re.search(f.exclude, n) else f.role
    return None


def library_names(html: str) -> list[str]:
    """Model names linked from an ollama.com/library page, in page order."""
    return list(dict.fromkeys(re.findall(r'href="/library/([a-z0-9][a-z0-9._\-]*)"', html)))


def library_tags(name: str, html: str) -> dict[str, list[str]]:
    """digest12 -> tags of `name` on its ollama.com tags page. Alias tags share one manifest, so they are one
    candidate. Rows without a manifest digest (cloud tags) are not local models and are left out."""
    links = list(re.finditer(r'href="/library/' + re.escape(name) + r':([^"]+)"', html))
    out: dict[str, list[str]] = {}
    for i, m in enumerate(links):
        row = html[m.end():links[i + 1].start() if i + 1 < len(links) else len(html)]
        d = re.search(r'font-mono[^>]*>\s*([0-9a-f]{12})\s*<', row)
        if d and m.group(1) not in out.setdefault(d.group(1), []):
            out[d.group(1)].append(m.group(1))
    return {k: v for k, v in out.items() if v}


def stable_releases(releases: list[dict]) -> list[dict]:
    """Published, non-draft, non-prerelease releases whose tag is not an rc/alpha/beta/pre/dev build."""
    return [r for r in releases if not r.get("draft") and not r.get("prerelease")
            and not UNSTABLE_TAG.search(r.get("tag_name") or "rc")]


def refusal(item: dict) -> str | None:
    """Why an upstream item cannot be filed (its name, tags or URLs fall outside the strict patterns, or a URL is not
    https on the source's own host and path), else None. Checked before anything is fetched, filed or queued."""
    src, name = item["source"], item["name"]
    if not SAFE_NAME[src].fullmatch(name) or ".." in name:
        return f"unsafe name {name!r}"
    for tag in dict.fromkeys([*([item["tag"]] if "tag" in item else []), *item.get("tags", [])]):
        if not isinstance(tag, str) or not SAFE_TAG.fullmatch(tag) or ".." in tag:
            return f"unsafe tag {tag!r}"
    prefixes = {"ollama": f"/library/{name}:", "hf": f"/{name}", "github": f"/{name}/releases/tag/"}
    urls = [("url", item["url"], prefixes[src])]
    if item.get("asset_url") is not None:
        urls.append(("asset url", item["asset_url"], f"/{name}/releases/download/{item.get('tag')}/"))
    for what, url, prefix in urls:
        m = SAFE_URL.fullmatch(url) if isinstance(url, str) else None
        if not m or m.group(1) != URL_HOST[src] or not m.group(2).startswith(prefix) or ".." in url:
            return f"unsafe {what} {url!r}"
    return None


def commands(item: dict, home: Path) -> tuple[list[list[str]] | None, list[str] | None, str | None]:
    """(pull argvs, screen argv, no-screen reason) for one refusal()-clean candidate. Queued as argv lists for the
    standing window, never shell strings; this module never runs them."""
    src, role = item["source"], item["role"]
    if src == "ollama":
        spec = f"ollama:{item['name']}:{item['tag']}"
        pull = [["localbench", "pull", spec]]
        if role == "decision":
            return pull, ["localbench", "decision", "run", spec, "--suite", DECISION_SUITE,
                          "--feature", DECISION_FEATURE], None
        if role == "extraction":
            return pull, ["localbench", "ab", SMOL_INCUMBENT, spec, *SCREEN_AB], None
        return pull, None, NO_SCREEN["embedding"]
    if src == "hf":
        target = home / ".mlx-serve" / "models" / item["name"]
        pull = [["localbench", "pull", f"hf:{item['name']}", "--to", str(target)]]
        if role == "extraction":
            return pull, ["localbench", "ab", SMOL_INCUMBENT, f"mlx-serve:{target}", *SCREEN_AB], None
        return pull, None, NO_SCREEN["embedding" if role == "embedding" else "decision-hf"]
    repo = item["name"]
    if repo == "ddalcu/mlx-serve" and item.get("asset_url"):
        d = home / ".localbench" / f"mlx-serve-{item['tag'].lstrip('v')}"
        archive = str(d / MLX_SERVE_ASSET)
        pull = [["mkdir", "-p", str(d)], ["curl", "-fL", "-o", archive, item["asset_url"]],
                ["tar", "-xzf", archive, "-C", str(d)]]
        incumbent = f"mlx-serve:{home / MLX_INCUMBENT}"
        return pull, ["localbench", "ab", incumbent, incumbent, *SCREEN_AB,
                      "--b-mlx-serve", str(d / "mlx-serve-macos-arm64" / "mlx-serve")], None
    return None, None, NO_SCREEN.get(repo, "no side-by-side screen for this runtime")


def shown(argvs: list[list[str]] | None) -> str:
    """Human-readable form of queued argvs, each shlex-quoted."""
    return " && ".join(shlex.join(a) for a in argvs) if argvs else "n/a"


def _gb(size: int | None) -> str:
    return "n/a" if size is None else f"{size / 1e9:.2f} GB ({size} bytes)"


def bead_argv(item: dict, pull: list[list[str]] | None, screen: list[str] | None, reason: str | None) -> list[str]:
    """`br create` arguments for one candidate: role, source URL, size, digest/commit and the stage-1 screen."""
    what = {"ollama": f"{item['name']}:{item.get('tag')}", "github": f"{item['name']} {item.get('tag')}"}.get(
        item["source"], item["name"])
    title = f"Release watch: {item['role']} candidate {what}"
    ident = item.get("digest") or item.get("commit") or "n/a"
    lines = ["Release watch candidate (kit-release-watch-y5n). Discovery only: never adopt without a stage-2 win.",
             "",
             f"- role: {item['role']}",
             f"- source: {item['url']}",
             f"- size: {_gb(item.get('size'))}",
             f"- {'commit' if item.get('commit') else 'digest'}: {ident}"]
    if item.get("tags"):
        lines.append(f"- tags: {', '.join(item['tags'])}")
    lines += ["", f"Pull (standing window): {shown(pull)}",
              f"Stage-1 screen: {shown([screen])}" if screen else f"Stage-1 screen: none ({reason})"]
    return ["create", "--silent", "--type", "task", "--priority", "3", "--labels", "side-model,release-watch",
            "--external-ref", item["url"], "--title", title, "--description", "\n".join(lines)]


# --- discovery ------------------------------------------------------------------------------------------------------

def _get(fetch: Fetch, url: str, headers: dict | None = None) -> bytes:
    status, body = fetch(url, headers) if headers else fetch(url)
    if status != 200:
        raise FetchError(f"{url}: HTTP {status or 'unreachable'}")
    return body


def _json(fetch: Fetch, url: str, headers: dict | None = None):
    body = _get(fetch, url, headers)
    try:
        return json.loads(body)
    except ValueError as exc:
        raise FetchError(f"{url}: not JSON ({exc})") from exc


@dataclass
class Unit:
    """One fetch unit: its key, the candidates it lists, whether it is a newly appeared Ollama family, and the
    error that stopped it (then nothing it lists is used)."""
    key: str
    items: list[dict]
    new_family: bool = False
    error: str | None = None


def _ollama_units(fetch: Fetch, state: dict) -> Iterator[Unit]:
    try:
        listed = library_names(_get(fetch, f"{OLLAMA_LIBRARY}?sort=newest").decode("utf-8", "replace"))
    except FetchError as exc:
        yield Unit("ollama:library", [], error=str(exc))
        return
    yield Unit("ollama:library", [{"names": listed}])
    baselined = "ollama:library" in state["units"]
    known = set(state.get("library", []))
    for name in dict.fromkeys([*listed, *OLLAMA_PINNED]):
        role = role_of(name)
        if role is None:
            continue
        key = f"ollama:{name}"
        try:
            page = _get(fetch, f"{OLLAMA_LIBRARY}/{name}/tags").decode("utf-8", "replace")
        except FetchError as exc:
            yield Unit(key, [], error=str(exc))
            continue
        items = []
        for digest12, tags in library_tags(name, page).items():
            tag = "latest" if "latest" in tags else tags[0]
            items.append({"id": f"ollama:{name}@{digest12}", "source": "ollama", "role": role, "name": name,
                          "tag": tag, "tags": tags, "digest12": digest12,
                          "url": f"{OLLAMA_LIBRARY}/{name}:{tag}"})
        yield Unit(key, items, new_family=baselined and name not in known)


def _hf_units(fetch: Fetch, publishers: tuple[str, ...]) -> Iterator[Unit]:
    queries = [(p, HF_EXTRACTION_SEARCH) for p in publishers] + list(HF_EMBEDDING_QUERIES)
    for author, search in queries:
        key = f"hf:{author}:{search}"
        q = urllib.parse.urlencode({"author": author, "search": search, "sort": "lastModified", "direction": -1,
                                    "limit": 30, "expand[]": ["sha", "lastModified", "pipeline_tag"]}, doseq=True)
        try:
            rows = _json(fetch, f"{models.HF_API}?{q}")
            if not isinstance(rows, list):
                raise FetchError(f"{key}: listing is not a list")
            items = []
            for m in rows:
                role = role_of(m["id"])
                # Image generators are in no serving path (same rule as models.releases).
                if role and m.get("pipeline_tag") != "text-to-image":
                    items.append({"id": f"hf:{m['id']}", "source": "hf", "role": role, "name": m["id"],
                                  "url": f"https://huggingface.co/{m['id']}"})
        except (FetchError, KeyError, TypeError) as exc:
            yield Unit(key, [], error=str(exc))
            continue
        yield Unit(key, items)


def _runtime_units(fetch: Fetch) -> Iterator[Unit]:
    for repo in RUNTIMES:
        key = f"github:{repo}"
        try:
            rows = _json(fetch, f"{GITHUB_API}/{repo}/releases?per_page=10", GITHUB_ACCEPT)
            if not isinstance(rows, list):
                raise FetchError(f"{key}: releases is not a list")
            items = []
            for r in stable_releases(rows):
                asset = next((a["browser_download_url"] for a in r.get("assets") or []
                              if a.get("name") == MLX_SERVE_ASSET), None)
                items.append({"id": f"github:{repo}@{r['tag_name']}", "source": "github", "role": "runtime",
                              "name": repo, "tag": r["tag_name"], "url": r["html_url"], "asset_url": asset})
        except (FetchError, KeyError, TypeError) as exc:
            yield Unit(key, [], error=str(exc))
            continue
        yield Unit(key, items)


def detail(item: dict, fetch: Fetch) -> dict:
    """The candidate with its weight size and digest (models) or commit (runtimes) from the authoritative source."""
    if item["source"] == "ollama":
        url = f"{models.REGISTRY}/library/{item['name']}/manifests/{item['tag']}"
        body = _get(fetch, url, {"Accept": models.MANIFEST})
        full = hashlib.sha256(body).hexdigest()
        if not full.startswith(item["digest12"]):
            raise FetchError(f"{url}: manifest sha256 {full[:12]} is not the library page's {item['digest12']}")
        try:
            size = sum(int(layer["size"]) for layer in json.loads(body)["layers"])
        except (ValueError, KeyError, TypeError) as exc:
            raise FetchError(f"{url}: unreadable manifest ({exc})") from exc
        return {**item, "digest": f"sha256:{full}", "size": size}
    if item["source"] == "hf":
        url = f"{models.HF_API}/{item['name']}?blobs=true"
        meta = _json(fetch, url)
        try:
            weights = [s for s in meta["siblings"] if s["rfilename"].endswith(WEIGHT_SUFFIXES)]
            size = sum(int(s["size"]) for s in weights)
            commit = meta["sha"]
            if not isinstance(commit, str) or not SHA40.fullmatch(commit):
                raise KeyError("sha")
        except (KeyError, TypeError, ValueError) as exc:
            raise FetchError(f"{url}: unreadable model info ({exc})") from exc
        if not weights:
            raise FetchError(f"{url}: no weight files listed")
        return {**item, "commit": commit, "size": size}
    url = f"{GITHUB_API}/{item['name']}/commits/{urllib.parse.quote(item['tag'])}"
    meta = _json(fetch, url, GITHUB_ACCEPT)
    if not isinstance(meta, dict) or not isinstance(meta.get("sha"), str) or not SHA40.fullmatch(meta["sha"]):
        raise FetchError(f"{url}: no commit sha")
    return {**item, "commit": meta["sha"], "size": None}


# --- state ----------------------------------------------------------------------------------------------------------

def refuse_repo_path(target: Path) -> Path:
    """Watch state names private model choices: refuse this repo and any git work tree."""
    t = Path(target).expanduser().resolve()
    if t == REPO_ROOT or REPO_ROOT in t.parents:
        raise WatchError(f"watch state {t} is inside {REPO_ROOT}; refusing")
    for p in (t, *t.parents):
        if (p / ".git").exists():
            raise WatchError(f"watch state {t} is inside the git work tree {p}; refusing")
    return t


def watch_dir(home: Path | None = None) -> Path:
    return refuse_repo_path((home or Path.home()) / ".localbench" / "watch")


def _read(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except ValueError as exc:
        raise WatchError(f"{path} is not JSON ({exc}); fix or move it") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise WatchError(f"{path} is not a version-1 watch file; fix or move it")
    return data


def _write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def _locked(d: Path):
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(d / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WatchError("another release-watch run holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def hf_publishers() -> tuple[str, ...]:
    """models.WATCH_PUBLISHERS plus the orgs installed MLX/Splash models came from (as models.releases watches)."""
    orgs = {d.name for root in (models.MLX_MODELS, models.SPLASH_MODELS) if root.is_dir()
            for d in root.iterdir() if d.is_dir()}
    return tuple(sorted({*models.WATCH_PUBLISHERS, *orgs}))


def _br(argv: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(["br", *argv], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return p.returncode, p.stdout, p.stderr


def _bead_id(stdout: str) -> str | None:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines and re.fullmatch(r"[A-Za-z0-9][\w.\-]*", lines[-1]) else None


# --- the run --------------------------------------------------------------------------------------------------------

def run_once(fetch: Fetch = models._fetch, br: Br = _br, now: datetime | None = None, home: Path | None = None,
             max_queued: int = MAX_QUEUED, max_bytes: int = MAX_WEIGHT_BYTES,
             publishers: tuple[str, ...] | None = None) -> dict:
    """One watch pass (the LaunchAgent entry). Returns a report: filed (bead + maybe queue entry), skipped (out of
    bounds, marked seen), deferred (queue full; retried next run), baselined count, errors. Never raises for a
    network or br failure; state is written only for what succeeded."""
    at = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    home = home or Path.home()
    report: dict = {"at": at, "filed": [], "skipped": [], "deferred": [], "baselined": 0, "errors": []}
    try:
        d = watch_dir(home)
        with _locked(d):
            _watch_pass(d, fetch, br, at, home, max_queued, max_bytes,
                        hf_publishers() if publishers is None else publishers, report)
    except WatchError as exc:
        report["errors"].append(str(exc))
    report["ok"] = not report["errors"]
    return report


def _watch_pass(d: Path, fetch: Fetch, br: Br, at: str, home: Path, max_queued: int, max_bytes: int,
                publishers: tuple[str, ...], report: dict) -> None:
    seen_path, queue_path = d / "seen.json", d / "queue.json"
    state = _read(seen_path, {"version": 1, "units": {}, "library": [], "items": {}})
    queue = _read(queue_path, {"version": 1, "entries": []})
    handled = set(state["items"]) | {e["id"] for e in queue["entries"]}
    dirty = False

    def save_state():
        _write(seen_path, state)

    units = [*_ollama_units(fetch, state), *_hf_units(fetch, publishers), *_runtime_units(fetch)]
    for unit in units:
        if unit.error:
            report["errors"].append(unit.error)
            continue
        if unit.key == "ollama:library":
            continue
        silent = unit.key not in state["units"] and not unit.new_family
        for item in unit.items:
            if item["id"] in handled:
                continue
            if silent:
                state["items"][item["id"]] = {"at": at, "outcome": "baseline"}
                handled.add(item["id"])
                report["baselined"] += 1
                dirty = True
                continue
            why = refusal(item)
            if why:
                state["items"][item["id"]] = {"at": at, "outcome": "refused", "reason": why}
                handled.add(item["id"])
                report["skipped"].append({"id": item["id"], "reason": f"refused: {why}"})
                dirty = True
                continue
            try:
                full = detail(item, fetch)
            except FetchError as exc:
                report["errors"].append(str(exc))
                continue
            if full.get("size") is not None and full["size"] > max_bytes:
                state["items"][item["id"]] = {"at": at, "outcome": "oversize", "size": full["size"]}
                handled.add(item["id"])
                report["skipped"].append({"id": item["id"], "reason": f"oversize: {_gb(full['size'])}"})
                dirty = True
                continue
            pull, screen, reason = commands(full, home)
            if screen and sum(e.get("status") == "queued" for e in queue["entries"]) >= max_queued:
                report["deferred"].append({"id": item["id"], "reason": f"queue full ({max_queued} screens waiting)"})
                continue
            code, out, err = br(bead_argv(full, pull, screen, reason))
            bead = _bead_id(out) if code == 0 else None
            if bead is None:
                why = (err or out).strip()[-300:]
                report["errors"].append(f"br create for {item['id']} failed (exit {code}): {why}")
                continue
            entry = {"id": item["id"], "bead": bead, "role": full["role"], "source": full["url"],
                     "size": full.get("size"), "digest": full.get("digest"), "commit": full.get("commit"),
                     "pull": pull, "screen": screen, "queued_at": at, "status": "queued"}
            if screen:
                # Queue first: an id in queue.json counts as handled even if the seen write is lost.
                queue["entries"].append(entry)
                _write(queue_path, queue)
            state["items"][item["id"]] = {"at": at, "outcome": "queued" if screen else "filed", "bead": bead,
                                          **({} if screen else {"no_screen": reason})}
            handled.add(item["id"])
            save_state()
            report["filed"].append({**entry, **({} if screen else {"status": "filed", "no_screen": reason})})
        if unit.key not in state["units"]:
            state["units"][unit.key] = at
            dirty = True
    library = next((u for u in units if u.key == "ollama:library" and not u.error), None)
    if library is not None:
        failed = {u.key.split(":", 1)[1] for u in units if u.key.startswith("ollama:") and u.error}
        names = sorted(set(state["library"]) | {n for n in library.items[0]["names"] if n not in failed})
        if names != state["library"] or "ollama:library" not in state["units"]:
            state["library"] = names
            state["units"].setdefault("ollama:library", at)
            dirty = True
    if dirty:
        save_state()


# --- status, LaunchAgent --------------------------------------------------------------------------------------------

def status(home: Path | None = None) -> dict:
    """Read-only: queued screens and counts of handled items by outcome."""
    d = watch_dir(home)
    state = _read(d / "seen.json", {"version": 1, "units": {}, "library": [], "items": {}})
    queue = _read(d / "queue.json", {"version": 1, "entries": []})
    outcomes: dict[str, int] = {}
    for v in state["items"].values():
        outcomes[v["outcome"]] = outcomes.get(v["outcome"], 0) + 1
    return {"dir": str(d), "queued": [e for e in queue["entries"] if e.get("status") == "queued"],
            "seen": outcomes, "units": len(state["units"])}


def report_lines(report: dict) -> list[str]:
    out = [f"release watch {report['at']}: {len(report['filed'])} filed, {len(report['skipped'])} skipped, "
           f"{len(report['deferred'])} deferred, {report['baselined']} baselined, {len(report['errors'])} errors"]
    out += [f"  filed {e['bead']}: {e['id']} ({e['role']}) " + (f"queued: {shown([e['screen']])}" if e.get('screen')
                                                                 else f"no screen: {e.get('no_screen')}")
            for e in report["filed"]]
    out += [f"  skipped {e['id']}: {e['reason']}" for e in report["skipped"]]
    out += [f"  deferred {e['id']}: {e['reason']}" for e in report["deferred"]]
    out += [f"  error: {e}" for e in report["errors"]]
    return out


def plist_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def launchd_plist(home: Path | None = None, python: str | None = None) -> dict:
    """A daily LaunchAgent running `localbench watch-releases --once` from this checkout; br must be on its PATH."""
    home = home or Path.home()
    log = str(home / ".localbench" / "watch" / "launchd.log")
    br = shutil.which("br")
    path = [str(Path(br).parent)] if br else []
    path += [str(home / ".local" / "bin"), "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    return {
        "Label": LABEL,
        "ProgramArguments": [python or sys.executable, "-m", "localbench", "watch-releases", "--once"],
        "WorkingDirectory": str(REPO_ROOT),
        "StartCalendarInterval": dict(SCHEDULE),
        "RunAtLoad": False,
        "ProcessType": "Background",
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": {"HOME": str(home), "PATH": ":".join(dict.fromkeys(path))},
    }


def install_agent(home: Path | None = None, python: str | None = None, run=subprocess.run) -> Path:
    """Write the plist (refusing a symlink or a foreign file at its path) and (re)load it with launchctl."""
    path = plist_path(home)
    watch_dir(home).mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise WatchError(f"{path} is a symlink; refusing")
    if path.exists():
        try:
            label = plistlib.loads(path.read_bytes()).get("Label")
        except (OSError, ValueError, plistlib.InvalidFileException) as exc:
            raise WatchError(f"{path} is unreadable; remove it first") from exc
        if label != LABEL:
            raise WatchError(f"{path} is not localbench's release watch; refusing")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(plistlib.dumps(launchd_plist(home, python), fmt=plistlib.FMT_XML, sort_keys=True))
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    domain = f"gui/{os.getuid()}"
    run(["launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True, text=True, check=False)
    p = run(["launchctl", "bootstrap", domain, str(path)], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise WatchError(f"launchctl bootstrap {path} failed: {(p.stderr or p.stdout).strip()}")
    return path


def cli(once: bool = False, install: bool = False, as_json: bool = False) -> int:
    """`localbench watch-releases [--once] [--install-agent] [--json]`; no flag prints the queue (read-only)."""
    if install:
        try:
            path = install_agent()
        except WatchError as exc:
            print(f"watch-releases: {exc}", file=sys.stderr)
            return 1
        print(f"installed {path} (daily {SCHEDULE['Hour']:02d}:{SCHEDULE['Minute']:02d}); "
              f"remove: launchctl bootout gui/{os.getuid()}/{LABEL} && rm {path}")
        if not once:
            return 0
    if once:
        report = run_once()
        print(json.dumps(report, indent=2, sort_keys=True) if as_json else "\n".join(report_lines(report)))
        return 0 if report["ok"] else 1
    try:
        s = status()
    except WatchError as exc:
        print(f"watch-releases: {exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(s, indent=2, sort_keys=True))
    else:
        print(f"{s['dir']}: {len(s['queued'])} queued screens; seen {s['seen'] or 'nothing yet'}")
        for e in s["queued"]:
            print(f"  {e['bead']} {e['id']} ({e['role']}): {shown([e['screen']])}")
    return 0
