"""Backend adapters. Each one can (a) pin its generation (version, binary hash, model digest),
(b) guarantee the model under test is the ONLY model it holds, and (c) reset to a cold KV/prefix
state on demand (`isolate`), so "first" means the same thing on every backend.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from functools import cache
from pathlib import Path
from typing import Self


def _get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _first_line(*cmd: str) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except OSError:
        return ""
    lines = [ln for ln in (out.stdout + out.stderr).splitlines() if ln and not ln.startswith("[mem]")]
    return lines[0] if lines else ""


def sha16(path: str) -> str | None:
    """First 16 hex of sha256 — the pin format used in docs/evidence/incumbents.md. Cached per (resolved path,
    mtime, size): omp was upgraded in place mid-campaign on 2026-09-23 (18.2.10 -> 18.2.11 at 05:28:44Z), and a
    path-keyed cache would have kept reporting the old sha for the rest of the process."""
    p = Path(path).resolve()
    if not p.is_file():
        return None
    st = p.stat()
    return _sha16(str(p), st.st_mtime_ns, st.st_size)


@cache
def _sha16(path: str, _mtime_ns: int, _size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]

def splash_pin() -> dict:
    """Splash binary version and sha16. None when `splash` is not on PATH. This pins the binary; it does not
    measure the model Splash serves."""
    binary = shutil.which("splash")
    if not binary:
        return {"splash_version": None, "splash_sha": None}
    version = _first_line("splash", "--version").removeprefix("Splash ").strip() or None
    return {"splash_version": version, "splash_sha": sha16(binary)}


def _warm(base_url: str, model: str) -> None:
    _post(base_url + "/chat/completions",
          {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})


class Ollama:
    name = "ollama"
    BINARY = "/Applications/Ollama.app/Contents/Resources/ollama"

    def __init__(self, root: str = "http://127.0.0.1:11434"):
        self.root = root
        self.base_url = root + "/v1"

    def loaded(self) -> list[dict]:
        # /api/ps stalls while the scheduler loads a model; the adapter must wait it out, not crash the run.
        return _get(self.root + "/api/ps", timeout=120).get("models", [])

    def isolate(self, model: str, warm: bool = True) -> list[str]:
        """Unload everything (the target too, so its prefix cache starts cold), then load `model`
        through the same /v1 path omp uses, so it gets the context length omp gets. warm=False leaves the model
        unloaded: the next real request loads it and meets a completely empty prefix cache."""
        evicted = []
        for m in self.loaded():
            _post(self.root + "/api/generate", {"model": m["name"], "keep_alive": 0})
            if m["name"] != model:
                evicted.append(m["name"])
        if warm:
            _warm(self.base_url, model)
        return evicted

    def pins(self, model: str) -> dict:
        tags = {t["name"]: t for t in _get(self.root + "/api/tags").get("models", [])}
        return {
            "backend": self.name,
            "backend_version": _first_line("ollama", "--version").removeprefix("ollama version is ").strip(),
            "backend_sha": sha16(self.BINARY),
            "model": model,
            "model_digest": (tags.get(model, {}).get("digest") or "")[:12] or None,
            "backend_args": "",
        }

    def fingerprint(self, model: str) -> dict:
        show = _post(self.root + "/api/show", {"model": model})
        running = {m["name"]: m for m in self.loaded()}
        details = show.get("details", {})
        return {
            **self.pins(model),
            "architecture": show.get("model_info", {}).get("general.architecture"),
            "quantization": details.get("quantization_level"),
            "parameters": details.get("parameter_size"),
            "loaded_context": running.get(model, {}).get("context_length"),
        }


def mlx_serve_bin() -> str:
    """The mlx-serve every launch and pin uses: LOCALBENCH_MLX_SERVE (e.g. a side-by-side release extracted outside
    Homebrew, whose formula lagged three releases behind on 2026-09-25), else PATH's. Read at every call, so a per-leg
    override (ab --b-mlx-serve) reaches the leg's launches and its pins alike."""
    return os.environ.get("LOCALBENCH_MLX_SERVE") or shutil.which("mlx-serve") or "mlx-serve"


class MlxServe:
    """mlx-serve pinned to one model directory. The process IS the isolation boundary: `isolate`
    restarts it, which drops the in-process prefix cache (the SSD tier stays off: no
    --prefix-cache-disk), so cold means cold."""

    name = "mlx-serve"

    def __init__(self, model_dir: str | Path, extra_args: tuple[str, ...] = (), host: str = "127.0.0.1",
                 port: int = 11234):
        self.model_dir = Path(model_dir).expanduser()
        self.extra_args = tuple(a for a in extra_args if not a.startswith("--prefix-cache-disk"))
        self.host, self.port = host, port
        self.root = f"http://{host}:{port}"
        self.base_url = self.root + "/v1"
        self.log = Path("/tmp") / f"{self.model_dir.name}.mlx-serve.log"
        self._proc: subprocess.Popen | None = None

    def up(self) -> bool:
        try:
            _get(self.base_url + "/models", timeout=2)
            return True
        except OSError:
            return False

    def models(self) -> list[dict]:
        return _get(self.base_url + "/models").get("data", [])

    def _spawn(self, log) -> subprocess.Popen:
        return subprocess.Popen(
            [mlx_serve_bin(), "--model", str(self.model_dir), "--serve", "--host", self.host,
             "--port", str(self.port), "--metrics", *self.extra_args],
            stdout=log, stderr=subprocess.STDOUT)

    def port_taken(self) -> bool:
        with socket.socket() as s:
            s.settimeout(1)
            return s.connect_ex((self.host, self.port)) == 0

    def start(self, ready_timeout: float = 600) -> None:
        if self.up():
            raise RuntimeError(f"{self.root} already serving {[m['id'] for m in self.models()]}; stop it first "
                               "(one model at a time)")
        if self.port_taken():
            # 2026-09-24: port 8765 answered 401 from the Agent Mail MCP server; a harness that took that for its
            # own server would measure nothing it started.
            raise RuntimeError(f"{self.host}:{self.port} is in use by another program; {self.name} cannot bind it")
        with self.log.open("a") as fh:
            self._proc = self._spawn(fh)
        deadline = time.time() + ready_timeout
        while not self.up():
            if self._proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited rc={self._proc.returncode}; see {self.log}")
            if time.time() > deadline:
                raise TimeoutError(f"{self.name} not ready after {ready_timeout}s; see {self.log}")
            time.sleep(1)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def model_id(self) -> str:
        ids = [m["id"] for m in self.models()]
        if len(ids) != 1:
            raise RuntimeError(f"{self.name} must serve exactly one model, serves {ids}")
        return ids[0]

    def isolate(self, model: str, warm: bool = True) -> list[str]:
        self.stop()
        self.start()
        if self.model_id() != model:
            raise RuntimeError(f"{self.name} restarted with {self.model_id()}, expected {model}")
        if warm:
            _warm(self.base_url, model)
        return []

    def model_digest(self) -> str:
        """HF commit if the pull recorded one, else a hash of the files that define the model."""
        for name in (".hf_commit", "refs/main", ".cache/huggingface/download/.gitattributes.metadata"):
            p = self.model_dir / name
            if p.is_file():
                return p.read_text().split()[0][:12]
        h = hashlib.sha256()
        for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
            p = self.model_dir / name
            if p.is_file():
                h.update(p.read_bytes())
        return "files:" + h.hexdigest()[:12]

    def pins(self, model: str) -> dict:
        binary = mlx_serve_bin()
        return {
            "backend": self.name,
            "backend_version": _first_line(binary, "--version").removeprefix("mlx-serve ").strip(),
            "backend_sha": sha16(binary) if Path(binary).is_file() else None,
            "model": model,
            "model_digest": self.model_digest(),
            # Server flags change what is measured (--mtp, --drafter, --ctx-size): part of the generation.
            "backend_args": " ".join(self.extra_args),
        }

    def fingerprint(self, model: str) -> dict:
        row = next((m for m in self.models() if m["id"] == model), {})
        meta = row.get("meta") or {}
        return {
            **self.pins(model),
            "architecture": meta.get("architecture"),
            "quantization": meta.get("quantization"),
            "mtp_loaded": meta.get("mtp_loaded"),
            "loaded_context": row.get("context_length") or meta.get("context_length"),
            "model_dir": str(self.model_dir),
        }


class OMLX(MlxServe):
    """oMLX (github.com/jundot/omlx) pinned to one model directory, started and stopped by the harness like mlx-serve.

    oMLX serves every model under --model-dir, so each start gets a fresh temporary directory holding one symlink to
    the model under test, and its own paged SSD cache directory: prefix caching (the partial-block caching oMLX
    advertises) works within a run, and a restart is cold, as with mlx-serve. --no-hf-cache keeps it from also serving
    the Hugging Face cache. Port 11236: oMLX's default 8000 is Splash's, and 8765 is the Agent Mail MCP server. With no
    API key configured, loopback inference needs none (checked 2026-09-24, oMLX 0.7.0rc1)."""

    name = "omlx"

    def __init__(self, model_dir: str | Path, extra_args: tuple[str, ...] = (), host: str = "127.0.0.1",
                 port: int = 11236):
        super().__init__(model_dir, (), host, port)
        self.extra_args = tuple(extra_args)
        self.log = Path("/tmp") / f"{self.model_dir.name}.omlx.log"
        self._tmp: Path | None = None

    def _spawn(self, log) -> subprocess.Popen:
        self._tmp = Path(tempfile.mkdtemp(prefix="localbench-omlx-"))
        (self._tmp / "models").mkdir()
        (self._tmp / "cache").mkdir()
        (self._tmp / "models" / self.model_dir.name).symlink_to(self.model_dir.resolve())
        return subprocess.Popen(
            ["omlx", "serve", "--model-dir", str(self._tmp / "models"), "--host", self.host, "--port", str(self.port),
             "--paged-ssd-cache-dir", str(self._tmp / "cache"), "--no-hf-cache", *self.extra_args],
            stdout=log, stderr=subprocess.STDOUT)

    def stop(self) -> None:
        super().stop()
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def status(self) -> dict:
        return _get(self.root + "/api/status")

    def pins(self, model: str) -> dict:
        return {
            "backend": self.name,
            "backend_version": self.status().get("version"),
            "backend_sha": omlx_sha(),
            "model": model,
            "model_digest": self.model_digest(),
            "backend_args": " ".join(self.extra_args),
        }

    def fingerprint(self, model: str) -> dict:
        row = next((m for m in self.models() if m["id"] == model), {})
        cfg = {}
        with contextlib.suppress(OSError, ValueError):
            cfg = json.loads((self.model_dir / "config.json").read_text())
        text = cfg.get("text_config") or cfg
        quant = cfg.get("quantization") or {}
        return {
            **self.pins(model),
            "architecture": text.get("model_type") or cfg.get("model_type"),
            "quantization": f"{quant['bits']}-bit" if quant.get("bits") else None,
            "loaded_context": row.get("max_model_len"),
            "model_dir": str(self.model_dir),
        }


def omlx_sha() -> str | None:
    """sha16 of the installed oMLX package's RECORD (every file's hash, so any reinstall or upgrade moves it). The
    `omlx` executable is a small launcher whose bytes do not change between versions."""
    exe = shutil.which("omlx")
    if not exe:
        return None
    env = Path(exe).resolve().parent.parent
    records = sorted(env.glob("lib/python*/site-packages/omlx-*.dist-info/RECORD"))
    return sha16(str(records[-1])) if records else None
