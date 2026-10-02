"""Transactional, marker-owned Ollama provider routing in OMP profile models.yml files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from .smol import profile_dirs

BEGIN = "  # >>> localbench ollama residency (managed)"
END = "  # <<< localbench ollama residency (managed)"
MANIFEST_NAME = "profiles.json"


class ProfileConflict(RuntimeError):
    """A profile contains an unowned Ollama route or a managed block changed outside localbench."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provider_url(profile: str, port: int) -> str:
    if not profile or "/" in profile or "\\" in profile or profile in {".", ".."}:
        raise ValueError(f"invalid OMP profile name: {profile!r}")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("gateway port must be between 1 and 65535")
    return f"http://127.0.0.1:{port}/omp-profile/{quote(profile, safe='-._~')}"


def provider_block(profile: str, port: int) -> str:
    return "\n".join((
        BEGIN,
        "  ollama:",
        f"    baseUrl: {provider_url(profile, port)}",
        "    api: openai-responses",
        "    auth: none",
        "    discovery:",
        "      type: ollama",
        END,
        "",
    ))


def _provider_region(text: str) -> tuple[int, int] | None:
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.fullmatch(r"providers:[ \t]*(?:#.*)?(?:\r?\n)?", line)]
    inline = [i for i, line in enumerate(lines) if re.match(r"^providers:[ \t]+\S", line)]
    if inline:
        raise ProfileConflict("inline providers mapping cannot be edited safely")
    if len(starts) > 1:
        raise ProfileConflict("multiple top-level providers mappings")
    if not starts:
        return None
    start = starts[0]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line[0].isspace():
            end = i
            break
    child_indents = []
    for line in lines[start + 1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        prefix = line[:len(line) - len(line.lstrip())]
        if "\t" in prefix:
            raise ProfileConflict("tab-indented providers mapping cannot be edited safely")
        child_indents.append(len(prefix))
    if child_indents and min(child_indents) != 2:
        raise ProfileConflict("providers mapping does not use the supported two-space child indentation")
    return start, end


def _managed_block(text: str) -> str | None:
    starts = [m.start() for m in re.finditer(rf"^{re.escape(BEGIN)}\r?$", text, re.MULTILINE)]
    ends = [m.start() for m in re.finditer(rf"^{re.escape(END)}\r?$", text, re.MULTILINE)]
    if not starts and not ends:
        return None
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ProfileConflict("managed Ollama provider markers are missing, duplicated, or reversed")
    end_match = re.search(rf"^{re.escape(END)}\r?\n?", text[ends[0]:], re.MULTILINE)
    if end_match is None:
        raise ProfileConflict("managed Ollama provider end marker is malformed")
    end = ends[0] + end_match.end()
    return text[starts[0]:end]


def _validate_managed_location(text: str, block: str) -> None:
    region = _provider_region(text)
    if region is None:
        raise ProfileConflict("managed Ollama provider is outside a top-level providers mapping")
    lines = text.splitlines(keepends=True)
    start, end = region
    begin = sum(len(line) for line in lines[:start])
    finish = sum(len(line) for line in lines[:end])
    position = text.find(block)
    if position < begin or position + len(block) > finish:
        raise ProfileConflict("managed Ollama provider is outside a top-level providers mapping")

def _install_text(original: str | None, profile: str, port: int) -> str:
    text = original or ""
    block = provider_block(profile, port)
    existing = _managed_block(text)
    if existing is not None:
        if existing.rstrip("\r\n") != block.rstrip("\r\n"):
            raise ProfileConflict(f"{profile}: managed Ollama provider differs from the requested route")
        _validate_managed_location(text, existing)
        return text
    region = _provider_region(text)
    if region is None:
        if re.search(r"^providers:", text, re.MULTILINE):
            raise ProfileConflict(f"{profile}: unsupported providers mapping")
        prefix = text
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        return prefix + "providers:\n" + block
    start, end = region
    lines = text.splitlines(keepends=True)
    for line in lines[start + 1:end]:
        if re.match(r"^  (?:ollama|[\"']ollama[\"'])[ \t]*:", line):
            raise ProfileConflict(f"{profile}: an explicit, unmanaged Ollama provider already exists")
    lines.insert(end, block)
    return "".join(lines)


def _strip_block(text: str, expected: str) -> str:
    existing = _managed_block(text)
    if existing is None or existing.rstrip("\r\n") != expected.rstrip("\r\n"):
        raise ProfileConflict("managed Ollama provider changed since installation")
    _validate_managed_location(text, existing)
    return text.replace(existing, "", 1)


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
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


def _secure_dir(path: Path) -> None:
    if path.is_symlink():
        raise ProfileConflict(f"refusing symlink state directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ProfileConflict(f"state directory became a symlink: {path}")
    path.chmod(0o700)


class ProfileManager:
    """Install/revert the one managed provider block, preserving all unrelated profile content."""

    def __init__(self, dirs: dict[str, Path], state_dir: Path, rollback_root: Path | None = None):
        self.dirs = {name: Path(path) for name, path in dirs.items()}
        self.state_dir = Path(state_dir)
        self.rollback_root = Path(rollback_root) if rollback_root is not None else self.state_dir / "rollback"
        self.manifest_path = self.state_dir / MANIFEST_NAME

    @classmethod
    def current(cls, state_dir: Path, home: Path | None = None) -> "ProfileManager":
        user_home = home or Path.home()
        rollback_root = user_home / ".localbench" / "rollback" / "omp-residency"
        return cls(profile_dirs(home), state_dir, rollback_root)

    def plan(self, port: int) -> dict[str, str]:
        if not self.dirs:
            raise ProfileConflict("no OMP profiles with config.yml were found")
        planned = {}
        for name, agent_dir in sorted(self.dirs.items()):
            path = agent_dir / "models.yml"
            if path.is_symlink():
                raise ProfileConflict(f"{name}: models.yml is a symlink; refusing to replace it")
            try:
                original = path.read_text(encoding="utf-8") if path.exists() else None
            except (OSError, UnicodeError) as exc:
                raise ProfileConflict(f"{name}: models.yml cannot be read") from exc
            planned[name] = _install_text(original, name, port)
        return planned

    def install(self, port: int) -> dict:
        planned = self.plan(port)
        if self.manifest_path.exists():
            try:
                prior = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ProfileConflict("existing profile manifest is unreadable") from exc
            if prior.get("port") == port and set(prior.get("profiles", {})) == set(planned):
                if all((self.dirs[name] / "models.yml").read_text(encoding="utf-8") == planned[name]
                       for name in planned):
                    return prior
            raise ProfileConflict("an existing OMP residency installation must be removed before changing it")

        _secure_dir(self.state_dir)
        _secure_dir(self.rollback_root)
        backup_dir = self.rollback_root / f"omp-profiles-{time.time_ns()}"
        _secure_dir(backup_dir)
        records: dict[str, dict] = {}
        originals: dict[str, tuple[bool, bytes, int]] = {}
        installed: list[str] = []
        try:
            for name, agent_dir in sorted(self.dirs.items()):
                path = agent_dir / "models.yml"
                if path.is_symlink():
                    raise ProfileConflict(f"{name}: models.yml is a symlink; refusing to replace it")
                existed = path.exists()
                original = path.read_bytes() if existed else b""
                original_text = original.decode("utf-8") if existed else None
                if _install_text(original_text, name, port) != planned[name]:
                    raise ProfileConflict(f"{name}: models.yml changed during install preflight")
                mode = path.stat().st_mode & 0o777 if existed else 0o600
                originals[name] = (existed, original, mode)

            backups = {}
            for name, (existed, original, _mode) in originals.items():
                backup = None
                if existed:
                    backup_path = backup_dir / f"{quote(name, safe='-._~')}.models.yml"
                    _atomic_write(backup_path, original, 0o600)
                    backup = str(backup_path)
                backups[name] = backup

            for name, agent_dir in sorted(self.dirs.items()):
                path = agent_dir / "models.yml"
                existed, original, mode = originals[name]
                current_exists = path.exists()
                current = path.read_bytes() if current_exists else b""
                if path.is_symlink() or current_exists != existed or current != original:
                    raise ProfileConflict(f"{name}: models.yml changed during install; no profile was overwritten")
                output = planned[name].encode("utf-8")
                _atomic_write(path, output, mode)
                installed.append(name)
                if path.is_symlink() or not path.is_file() or path.read_bytes() != output:
                    raise ProfileConflict(f"{name}: models.yml changed during gateway provider installation")
                records[name] = {
                    "path": str(path), "original_exists": existed, "original_sha256": _sha(original),
                    "installed_sha256": _sha(output), "backup": backups[name],
                    "block": provider_block(name, port),
                }
            manifest = {"version": 1, "port": port, "backup_dir": str(backup_dir), "profiles": records}
            encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            _atomic_write(self.manifest_path, encoded, 0o600)
            return manifest
        except BaseException:
            for name in reversed(installed):
                path = self.dirs[name] / "models.yml"
                existed, original, mode = originals[name]
                current = path.read_bytes() if path.exists() else b""
                expected = planned[name].encode("utf-8")
                if not path.is_symlink() and current == expected:
                    if existed:
                        _atomic_write(path, original, mode)
                    else:
                        path.unlink(missing_ok=True)
            shutil.rmtree(backup_dir, ignore_errors=True)
            raise

    def _load_manifest(self, manifest: dict | None) -> dict:
        if manifest is None:
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ProfileConflict("no readable OMP residency profile manifest exists") from exc
        if not isinstance(manifest, dict):
            raise ProfileConflict("OMP residency profile manifest has an invalid shape")
        return manifest

    def plan_revert(self, manifest: dict | None = None) -> dict[str, str | None]:
        """Validate every managed block before any profile is changed."""
        manifest = self._load_manifest(manifest)
        records = manifest.get("profiles")
        if not isinstance(records, dict) or set(records) != set(self.dirs):
            raise ProfileConflict("profile manifest does not match the current OMP profile set")
        planned: dict[str, str | None] = {}
        for name, agent_dir in sorted(self.dirs.items()):
            record = records[name]
            path = agent_dir / "models.yml"
            if path.is_symlink() or record.get("path") != str(path) or not path.is_file():
                raise ProfileConflict(f"{name}: managed models.yml is missing or moved")
            try:
                after = _strip_block(path.read_text(encoding="utf-8"), record["block"])
            except (UnicodeError, KeyError) as exc:
                raise ProfileConflict(f"{name}: managed provider cannot be verified") from exc
            planned[name] = None if not record.get("original_exists") and after.strip() in {"", "providers:"} else after
        return planned

    def revert(self, manifest: dict | None = None) -> list[str]:
        manifest = self._load_manifest(manifest)
        outputs = self.plan_revert(manifest)
        records = manifest["profiles"]
        planned: dict[str, tuple[Path, bool, bytes, int]] = {}
        snapshots: dict[str, tuple[bytes, int]] = {}
        for name, output in outputs.items():
            path = self.dirs[name] / "models.yml"
            before = path.read_bytes()
            record = records[name]
            try:
                after_text = _strip_block(before.decode("utf-8"), record["block"])
            except (UnicodeError, KeyError) as exc:
                raise ProfileConflict(f"{name}: managed provider changed during revert preflight") from exc
            after = None if not record.get("original_exists") and after_text.strip() in {"", "providers:"} else after_text
            if after != output:
                raise ProfileConflict(f"{name}: models.yml changed during revert preflight")
            mode = path.stat().st_mode & 0o777
            snapshots[name] = (before, mode)
            planned[name] = (path, output is None, (output or "").encode("utf-8"), mode)

        changed: list[str] = []
        try:
            for name, (path, delete, data, mode) in planned.items():
                before, _ = snapshots[name]
                if path.is_symlink() or not path.is_file() or path.read_bytes() != before:
                    raise ProfileConflict(f"{name}: models.yml changed during revert; no edit was applied")
                if delete:
                    path.unlink()
                else:
                    _atomic_write(path, data, mode)
                changed.append(name)
                if (path.exists() if delete else path.read_bytes() != data):
                    raise ProfileConflict(f"{name}: models.yml changed during gateway provider reversion")
        except BaseException:
            for name in reversed(changed):
                path, delete, data, _mode = planned[name]
                previous, mode = snapshots[name]
                ours_remain = (not path.exists() if delete else
                               not path.is_symlink() and path.is_file() and path.read_bytes() == data)
                if ours_remain:
                    _atomic_write(path, previous, mode)
            raise

        self.manifest_path.unlink(missing_ok=True)
        backup_dir = Path(manifest.get("backup_dir", ""))
        allowed_roots = {self.rollback_root.resolve(), (self.state_dir / "rollback").resolve()}
        if backup_dir.resolve().parent in allowed_roots and backup_dir.exists():
            shutil.rmtree(backup_dir)
        return list(sorted(self.dirs))
def verify_omp_profiles(profiles: list[str], run=None) -> dict[str, int]:
    """Read each installed Ollama model catalog through OMP; this resolves metadata only, never a model request."""
    from .workloads import omp_bin, omp_env

    runner = run or subprocess.run
    counts = {}
    for profile in sorted(profiles):
        args = [omp_bin(), *([] if profile == "default" else ["--profile", profile]),
                "models", "ollama", "--json"]
        try:
            result = runner(args, capture_output=True, text=True, timeout=120, env=omp_env(), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProfileConflict(f"{profile}: OMP Ollama catalog readback failed") from exc
        if result.returncode != 0:
            raise ProfileConflict(f"{profile}: OMP Ollama catalog readback exited {result.returncode}")
        try:
            document = json.loads(result.stdout or "{}")
        except ValueError as exc:
            raise ProfileConflict(f"{profile}: OMP Ollama catalog readback was not JSON") from exc
        models = document.get("models") if isinstance(document, dict) else None
        if not isinstance(models, list):
            raise ProfileConflict(f"{profile}: OMP Ollama catalog readback has no model list")
        counts[profile] = len(models)
    return counts
