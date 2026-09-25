"""Which local models this machine has, whether each is still the newest build of its source, which omp features route
to it, and what appeared upstream in the families and publishers it came from. Besides the GPU servers (ollama,
mlx-serve, Splash) this covers the models omp runs itself on the CPU: its tiny models and mnemopi's embedding model.

Freshness is exact where the source allows: an ollama model's ID is the first 12 hex of sha256(registry manifest),
so fetching the tag's manifest and hashing it says whether the installed build is the one the registry serves now
(checked 2026-09-23: qwen3.6:35b-mlx → e92a3e94bbca both ways). A Hugging Face repo has no per-file digest in its
summary, so an MLX directory is compared by time: upstream `lastModified` after the local copy's newest file means
the upstream changed since download. Network reads only (registry, HF API); no inference leaves the machine.
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
               "source": source, "parked": name in parked}
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


def _hf_row(d: Path, server: str, name: str, repo: str) -> dict:
    """A model directory downloaded from Hugging Face repo `repo`: size, newest local file, and whether the repo changed
    after that (HF summaries carry no per-file digest, so this compares times)."""
    files = [f for f in d.rglob("*") if f.is_file() and not any(p.startswith(".") for p in f.relative_to(d).parts)]
    local = max((f.stat().st_mtime for f in files), default=d.stat().st_mtime)
    row = {"server": server, "name": name, "repo": repo, "gb": round(sum(f.stat().st_size for f in files) / 1e9, 1),
           "local_copy": time.strftime("%Y-%m-%d %H:%M", time.localtime(local))}
    code, body = _fetch(f"{HF_API}/{repo}")
    if code == 200:
        info = json.loads(body)
        upstream = datetime.fromisoformat(info["lastModified"]).timestamp()
        row.update(upstream_sha=info.get("sha", "")[:12], upstream_modified=info["lastModified"][:16])
        row["freshness"] = ("upstream changed after the local copy" if upstream > local + 60
                            else "current (local copy is newer than the last upstream change)")
    else:
        row["freshness"] = f"no Hugging Face repo {repo} (HTTP {code or 'unreachable'})"
    return row


def mlx_models() -> list[dict]:
    """mlx-serve model dirs (<org>/<name>/config.json) and Inco Splash packages (<org>/<name>/target|draft): both are
    Hugging Face repos, compared by time."""
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


def routes_by_model(names: list[str]) -> dict[str, dict[str, list[str]]]:
    """model name → {feature: [profiles]} over the profiles `names` (omp's own resolved settings per profile)."""
    uses: dict[str, dict[str, list[str]]] = {}
    for profile in names:
        for feature, target in park.local_routes(profile).items():
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
