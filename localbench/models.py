"""Which local models this machine has, whether each is still the newest build of its source, which omp features route
to it, and what appeared upstream in the families and publishers it came from. Besides the GPU servers (ollama,
mlx-serve, Splash) this covers the models omp runs itself on the CPU: its tiny models and mnemopi's embedding model.

Freshness is exact where the source allows: an ollama model's ID is the first 12 hex of sha256(registry manifest),
so fetching the tag's manifest and hashing it says whether the installed build is the one the registry serves now
(checked 2026-09-23: qwen3.6:35b-mlx → e92a3e94bbca both ways). A Hugging Face directory is current only when a
recorded commit for that repo matches the API's commit. That record does not verify the bytes on disk; without it,
a digest of small configuration files identifies the local artifact but cannot establish upstream freshness.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import park
from .workloads import omp_bin, omp_env

MLX_MODELS = Path.home() / ".mlx-serve" / "models"
SPLASH_MODELS = Path.home() / "Library" / "Application Support" / "Splash" / "models"
OMP_TINY_CACHE = Path.home() / ".omp" / "agent" / "cache" / "tiny-models"
FASTEMBED_CACHE = Path.home() / ".omp" / "cache" / "fastembed"
OLLAMA = "http://127.0.0.1:11434"
REGISTRY = "https://registry.ollama.ai/v2"
HF_API = "https://huggingface.co/api/models"
MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
# Publishers besides the ones installed models came from, searched for the families in use.
WATCH_PUBLISHERS = ("mlx-community",)


def _fetch(url: str, headers: dict | None = None, timeout: float = 15) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "localbench", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b""


def _family(name: str) -> str:
    """Model family used to search upstream: `qwen3.6:35b-mlx` → qwen3.6, `Qwen3.6-35B-A3B-MLX-Serve-4bit` → Qwen3.6."""
    return re.split(r"[:\-]", name.split("/")[-1], maxsplit=1)[0]


def ollama_models() -> list[dict]:
    status, body = _fetch(f"{OLLAMA}/api/tags", timeout=5)
    if status != 200:
        return []
    parked = {p["parked_as"]: p["name"] for p in (json.loads(park.STATE.read_text()) if park.STATE.exists() else [])}
    out = []
    for m in json.loads(body).get("models", []):
        name, digest = m["name"], m["digest"][:12]
        source = parked.get(name, name)
        row = {"server": "ollama", "name": name, "digest": digest, "gb": round(m.get("size", 0) / 1e9, 1),
               "source": source, "parked": name in parked, "upstream_modified": None,
               "upstream_date_source": "unavailable"}
        if source.endswith(":cloud") or m.get("remote_host"):
            row["freshness"] = "cloud model (runs remotely)"
        else:
            repo, _, tag = source.partition(":")
            path = repo if "/" in repo else f"library/{repo}"
            code, manifest = _fetch(f"{REGISTRY}/{path}/manifests/{tag or 'latest'}", {"Accept": MANIFEST})
            if code == 200:
                upstream = hashlib.sha256(manifest).hexdigest()[:12]
                row["upstream_digest"] = upstream
                row["freshness"] = "current" if upstream == digest else f"update available (registry {upstream})"
            else:
                row["freshness"] = f"not in the ollama registry (HTTP {code or 'unreachable'})"
        out.append(row)
    return out


def _hf_artifact(d: Path, repo: str) -> tuple[str | None, str, str | None, str | None]:
    """Return display identity, its provenance, comparable full commit, and reason if comparison is unsound."""
    metadata = (".hf_commit", "refs/main", ".cache/huggingface/download/.gitattributes.metadata")
    records: list[tuple[str, str]] = []
    invalid = False
    for name in metadata:
        path = d / name
        if path.is_file():
            try:
                value = path.read_text().split()[0]
            except (OSError, UnicodeError, IndexError):
                invalid = True
                continue
            if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
                invalid = True
                continue
            records.append((name, value.lower()))

    # A naked commit in an unrelated directory is not evidence that it belongs to the requested repo.
    linked = "/".join(d.parts[-2:]) == repo
    config = d / "config.json"
    if config.is_file():
        try:
            configured_repo = json.loads(config.read_text()).get("_name_or_path")
            if configured_repo:
                linked = configured_repo == repo
        except (OSError, UnicodeError, ValueError, AttributeError):
            pass

    if records and linked and not invalid and len({value for _, value in records}) == 1:
        name, commit = records[0]
        return commit[:12], f"{name} (recorded HF commit; bytes not verified)", commit, None

    # No weights are read here. This is an artifact identifier, not a proof of upstream revision.
    h = hashlib.sha256()
    names = ("config.json", "model.safetensors.index.json", "tokenizer_config.json")
    found = []
    for name in names:
        path = d / name
        if path.is_file():
            h.update(path.read_bytes())
            found.append(name)
    identity = "files:" + h.hexdigest()[:12] if found else None
    source = "config/index/tokenizer SHA-256 (partial file digest; weights not hashed)" if found else "no small artifact files"
    if invalid:
        reason = "invalid recorded HF commit"
    elif len({value for _, value in records}) > 1:
        reason = "conflicting recorded HF commits"
    elif records and not linked:
        reason = f"recorded commit repo cannot be linked to {repo}"
    else:
        reason = "no recorded HF commit; file digest cannot be compared with upstream commit"
    return identity, source, None, reason


def _hf_row(d: Path, server: str, name: str, repo: str) -> dict:
    """Inventory an HF directory without treating timestamps or a partial file digest as commit evidence."""
    files = [f for f in d.rglob("*") if f.is_file() and not any(p.startswith(".") for p in f.relative_to(d).parts)]
    local = max((f.stat().st_mtime for f in files), default=d.stat().st_mtime)
    identity, identity_source, commit, reason = _hf_artifact(d, repo)
    row = {"server": server, "name": name, "repo": repo, "gb": round(sum(f.stat().st_size for f in files) / 1e9, 1),
           "local_copy": time.strftime("%Y-%m-%d %H:%M", time.localtime(local)),
           "installed_artifact": identity, "installed_artifact_source": identity_source,
           "upstream_modified": None, "upstream_date_source": "unavailable"}
    code, body = _fetch(f"{HF_API}/{repo}")
    if code != 200:
        row["freshness"] = f"unavailable (Hugging Face API HTTP {code or 'unreachable'})"
        return row
    try:
        info = json.loads(body)
        sha = info.get("sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            raise ValueError("missing or invalid commit SHA")
        sha = sha.lower()
    except (ValueError, TypeError, AttributeError, UnicodeError):
        row["freshness"] = "unknown (invalid Hugging Face response: missing or invalid commit SHA)"
        return row
    row["upstream_sha"] = sha[:12]
    for date_field in ("lastModified", "createdAt"):
        value = info.get(date_field)
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
            except ValueError:
                continue
            row.update(upstream_modified=value[:16], upstream_date_source=date_field)
            break
    if info.get("id") and info["id"] != repo:
        row["freshness"] = f"unknown (Hugging Face response repo {info['id']} differs from {repo})"
    elif commit is not None:
        row["freshness"] = ("current (recorded HF commit matches; bytes not verified)" if commit == sha
                            else f"update available (upstream commit {sha[:12]} differs from recorded commit)")
    else:
        row["freshness"] = f"unknown ({reason})"
    return row


def mlx_models() -> list[dict]:
    """mlx-serve model dirs (<org>/<name>/config.json) and Inco Splash packages (<org>/<name>/target|draft)."""
    dirs = [(c.parent, "mlx-serve") for c in sorted(MLX_MODELS.glob("*/*/config.json"))] if MLX_MODELS.is_dir() else []
    if SPLASH_MODELS.is_dir():
        dirs += [(d, "splash") for d in sorted(SPLASH_MODELS.glob("*/*")) if (d / "target").is_dir()]
    return [_hf_row(d, server, f"{d.parent.name}/{d.name}", f"{d.parent.name}/{d.name}") for d, server in dirs]


def omp_cpu_models() -> list[dict]:
    """Models omp runs itself, outside ollama/mlx-serve, so the GPU samplers never attribute them:
    - tiny models (`omp tiny-models`): ONNX on the CPU unless providers.tinyModelDevice is `mlx`; serve features a
      profile routes to `local/<key>` (titles, and memory/judge through the tiny fallback). Keyed by the spec's repo.
    - mnemopi's embedding model: fastembed ONNX in a worker subprocess of every memory-on omp process, used by recall
      and retain unless mnemopi.noEmbeddings. The weights are fastembed's ONNX export; freshness is checked against the
      Hugging Face repo its config names (the repo mnemopi fetches the sidecars from)."""
    out = []
    raw = subprocess.run([omp_bin(), "tiny-models", "list", "--json"], capture_output=True, text=True, timeout=60,
                         env=omp_env(), check=False).stdout
    for spec in json.loads(raw or "{}").get("models", []):
        d = OMP_TINY_CACHE / spec["repo"]
        if d.is_dir():
            out.append({**_hf_row(d, "omp-tiny", spec["key"], spec["repo"]), "source": spec["key"]})
    for d in sorted(p for p in FASTEMBED_CACHE.glob("*") if p.is_dir()) if FASTEMBED_CACHE.is_dir() else []:
        cfg = d / "config.json"
        repo = json.loads(cfg.read_text()).get("_name_or_path", d.name) if cfg.is_file() else d.name
        out.append({**_hf_row(d, "fastembed", d.name, repo), "source": d.name})
    return out


def profiles() -> list[str]:
    """omp profiles on this host: default, then every ~/.omp/profiles/<name> with an agent config."""
    return ["default", *sorted(p.parent.parent.name for p in
                              (Path.home() / ".omp" / "profiles").glob("*/agent/config.yml"))]


DISABLED = "disabled (task.disabledAgents)"


def agent_config(profile: str) -> Path:
    """The config.yml omp reads for `profile`: ~/.omp/agent for default, else ~/.omp/profiles/<profile>/agent."""
    home = Path.home() / ".omp"
    return (home / "agent" if profile == "default" else home / "profiles" / profile / "agent") / "config.yml"


def disabled_agents(profile: str) -> set[str]:
    """The bundled subagents `profile` disables (`task.disabledAgents` in its config.yml; block or flow list). Read
    from the file, not `omp config list`, so listing routes costs no second omp call per profile."""
    cfg = agent_config(profile)
    lines = cfg.read_text(encoding="utf-8").splitlines() if cfg.is_file() else []
    task = next((i for i, line in enumerate(lines) if re.fullmatch(r"task:\s*(#.*)?", line)), None)
    if task is None:
        return set()
    out: set[str] = set()
    key_indent = None
    for line in lines[task + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break
        if key_indent is None:
            m = re.fullmatch(r"(\s+)['\"]?disabledAgents['\"]?:\s*(.*?)\s*(#.*)?", line)
            if not m:
                continue
            flow = m.group(2)
            if flow.startswith("["):
                return {a.strip().strip("'\"") for a in flow.strip("[]").split(",") if a.strip()}
            key_indent = len(m.group(1))
        elif line.lstrip().startswith("-") and indent >= key_indent:
            out.add(line.lstrip()[1:].split("#", 1)[0].strip().strip("'\""))
        else:
            break
    return out


def _agent(feature: str) -> str | None:
    """The bundled subagent a park.local_routes feature is (`scout subagents (...)` -> scout), else None."""
    m = re.match(r"(\S+) subagents\b", feature)
    return m.group(1) if m else None


def local_routes(profile: str = "default") -> dict[str, str]:
    """park.local_routes for `profile`, with every subagent the profile disables mapped to DISABLED instead of the
    model it would run on: since 2026-10-01 01:43Z scout is in task.disabledAgents of every local-smol profile, so
    no scout turn reaches ollama/qwen3.8 there."""
    off = disabled_agents(profile)
    return {feature: DISABLED if _agent(feature) in off else target
            for feature, target in park.local_routes(profile).items()}


def routes_by_model(names: list[str], disabled: dict[str, list[str]] | None = None) -> dict[str, dict[str, list[str]]]:
    """model name → {feature: [profiles]} over the profiles `names` (omp's own resolved settings per profile). A
    disabled subagent is not a route: it is left out, and recorded as feature → [profiles] in `disabled` when given."""
    uses: dict[str, dict[str, list[str]]] = {}
    for profile in names:
        for feature, target in local_routes(profile).items():
            if target == DISABLED:
                if disabled is not None:
                    disabled.setdefault(feature, []).append(profile)
                continue
            model = target.split("/", 1)[1]
            base, _, last = model.rpartition(":")
            if base and last in park.THINKING:
                model = base
            uses.setdefault(model, {}).setdefault(feature, []).append(profile)
    return uses


def releases(days: int, installed: list[dict]) -> list[dict]:
    """Hugging Face models published or updated in the last `days` by the publishers installed MLX models came from
    (all their models) and by WATCH_PUBLISHERS (only the families in use)."""
    since = datetime.now(UTC) - timedelta(days=days)
    families = sorted({_family(m["source" if m["server"] == "ollama" else "name"]) for m in installed
                       if not m.get("freshness", "").startswith("cloud")})
    publishers = sorted({m["name"].split("/")[0] for m in installed if m["server"] in ("mlx-serve", "splash")})
    queries = [{"author": p} for p in publishers] + [{"author": p, "search": f} for p in WATCH_PUBLISHERS
                                                      for f in families]
    seen: dict[str, dict] = {}
    for q in queries:
        url = f"{HF_API}?{urllib.parse.urlencode({**q, 'sort': 'lastModified', 'direction': -1, 'limit': 30})}"
        code, body = _fetch(url)
        if code != 200:
            continue
        for m in json.loads(body):
            when = m.get("lastModified") or m.get("createdAt")
            # Image generators are not in any serving path here; everything else (text, vision-text, untagged) is.
            if when and m.get("pipeline_tag") != "text-to-image" and datetime.fromisoformat(when) >= since:
                seen[m["id"]] = {"id": m["id"], "modified": when[:16], "created": (m.get("createdAt") or "")[:10],
                                 "task": m.get("pipeline_tag"), "via": q}
    return sorted(seen.values(), key=lambda r: r["modified"], reverse=True)
