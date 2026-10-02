"""Frozen omp snapshots (`localbench omp freeze`, `run|ab --omp-frozen`).

uca updates the global omp install (~/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent, entry ~/.bun/bin/omp,
a symlink to the package's dist/cli.js run by `#!/usr/bin/env bun`) about every 3 h. On 2026-10-02 it replaced 18.4.9
with 18.4.10 between the legs of a 50-min memory A/B and voided it (arm A legs ran different omp_version). A snapshot is
a private copy of everything that omp needs at runtime, so every leg of a study can run one omp:

  <FROZEN_ROOT>/<version>-<sha16 of dist/cli.js>/
    node_modules/<the package and its runtime dependency closure, same relative layout as the global install>
    bin/bun          a copy of the bun that runs the live omp (the shebang's `bun` on PATH), version pinned in the marker
    omp              -> node_modules/@oh-my-pi/pi-coding-agent/.localbench-frozen/omp, the entry script:
                        exec <snapshot>/bin/bun <snapshot>/node_modules/.../dist/cli.js "$@"
    .localbench-omp-freeze   marker + manifest (JSON); retention only ever touches directories holding one

The closure is the package's dependencies, optionalDependencies and peerDependencies resolved the way node and bun
resolve them (nested node_modules first, then each ancestor's), transitively, never above the install root; files are
copied (symlinks followed), so nothing in a snapshot points back into the live install. `omp` is a symlink to a script
two levels inside the frozen package, so park.omp_package(<snapshot>/omp) finds the frozen package (its src/ resolver),
and backends.sha16(<snapshot>/omp) hashes that script, whose text names the snapshot id: one omp_sha per snapshot.

Build is atomic (copy into a hidden temp dir beside the target, then rename) and idempotent (an existing snapshot with
the same id is reused). Snapshots are read-only after the copy. Retention keeps the newest KEEP snapshots that carry the
marker; a snapshot a run holds (hold(), a shared flock under .locks/) is never removed."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import park
from .backends import sha16
from .workloads import omp_bin

FROZEN_ROOT = Path.home() / ".localbench" / "omp-frozen"
MARKER = ".localbench-omp-freeze"
ENTRY_DIR = ".localbench-frozen"
KEEP = 3
DEP_KINDS = ("dependencies", "optionalDependencies", "peerDependencies")


def root() -> Path:
    """The snapshot directory, resolved at call time from FROZEN_ROOT (tests point it at a temporary dir)."""
    return FROZEN_ROOT


@dataclass(frozen=True)
class Plan:
    snapshot_id: str
    target: Path          # <root>/<snapshot_id>
    install: Path         # the live node_modules root the package sits in
    package: Path         # the live package dir
    entry: Path           # the live package's omp entry (dist/cli.js)
    version: str
    packages: tuple[Path, ...]   # package dirs relative to `install`, the package itself first
    size_bytes: int
    bun: Path             # the live bun, symlinks resolved
    exists: bool          # a snapshot with this id is already there (reuse it)
    refuse: str | None    # why this target cannot be written or reused


def entry_path(target: Path) -> Path:
    return target / "omp"


def _resolve(install: Path, frm: Path, name: str) -> Path | None:
    """Where `name` required from package dir `frm` resolves: <ancestor>/node_modules/<name> for frm and each ancestor,
    nearest first, never above the install root."""
    for anc in (frm, *frm.parents):
        if anc.name != "node_modules":
            cand = anc / "node_modules" / name
            if (cand / "package.json").is_file():
                return cand
        if anc == install.parent:
            return None
    return None


def closure(install: Path, package: Path) -> list[Path]:
    """The package and every package it can load at runtime, as dirs relative to `install`. A declared dependency that
    resolves nowhere is an error (the live omp could not load it either); a missing optional or peer one is skipped."""
    seen = {package: None}
    stack = [package]
    while stack:
        pkg = stack.pop()
        meta = json.loads((pkg / "package.json").read_text())
        for kind in DEP_KINDS:
            for name in meta.get(kind) or {}:
                found = _resolve(install, pkg, name)
                if found is None:
                    if kind == "dependencies":
                        raise RuntimeError(f"{pkg / 'package.json'}: dependency {name} resolves nowhere under {install}")
                    continue
                if found not in seen:
                    seen[found] = None
                    stack.append(found)
    return [p.relative_to(install) for p in seen]


def _tree_bytes(path: Path, skip_top_node_modules: bool) -> int:
    total = 0
    for dirpath, dirs, files in os.walk(path):
        if skip_top_node_modules and dirpath == str(path) and "node_modules" in dirs:
            dirs.remove("node_modules")
        for f in files:
            with contextlib.suppress(OSError):
                total += os.stat(os.path.join(dirpath, f)).st_size
    return total


def _install_root(package: Path, name: str) -> Path:
    install = package.parent.parent if "/" in name else package.parent
    if install.name != "node_modules":
        raise RuntimeError(f"omp package {package} ({name}) is not inside a node_modules install; nothing to freeze")
    return install


def _bun() -> Path:
    found = shutil.which("bun")
    if not found:
        raise RuntimeError("bun is not on PATH: the live omp (#!/usr/bin/env bun) could not run either")
    return Path(os.path.realpath(found))


def _manifest(target: Path) -> dict | None:
    try:
        doc = json.loads((target / MARKER).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("snapshot_id") == target.name else None


def plan(binary: str | None = None) -> Plan:
    """What freezing the omp at `binary` (default omp_bin()) would do; reads only."""
    package = park.omp_package(binary or omp_bin())
    meta = json.loads((package / "package.json").read_text())
    bins = meta.get("bin")
    rel = bins.get("omp") if isinstance(bins, dict) else bins
    if not isinstance(rel, str):
        raise RuntimeError(f"{package / 'package.json'} has no bin.omp entry")
    entry = package / rel
    version, digest = meta.get("version"), sha16(str(entry))
    if not version or not digest:
        raise RuntimeError(f"omp package {package}: no version in package.json or no entry file {entry}")
    install = _install_root(package, meta.get("name") or "")
    packages = closure(install, package)
    snapshot_id = f"{version}-{digest}"
    target = root() / snapshot_id
    refuse = None
    exists = target.exists() or target.is_symlink()
    if exists and (target.is_symlink() or not target.is_dir() or _manifest(target) is None):
        refuse = f"{target} exists but is not a snapshot localbench made (no valid {MARKER}); refusing to touch it"
    # A reuse copies nothing, so it does not walk the ~800 MB tree (run|ab --omp-frozen plan on every invocation).
    size = 0 if exists else sum(_tree_bytes(install / p, True) for p in packages)
    return Plan(snapshot_id, target, install, package, entry, version, tuple(packages), size, _bun(), exists, refuse)


def _ignore_top_node_modules(top: Path):
    def ignore(dirpath: str, names: list[str]) -> set[str]:
        return {"node_modules"} if Path(dirpath) == top and "node_modules" in names else set()
    return ignore


def _read_only(path: Path) -> None:
    for dirpath, dirs, files in os.walk(path, topdown=False):
        for f in files:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        os.chmod(dirpath, os.stat(dirpath).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _remove(path: Path) -> None:
    """rmtree a read-only snapshot: give the owner write access back on the way down."""
    for dirpath, _dirs, _files in os.walk(path):
        with contextlib.suppress(OSError):
            os.chmod(dirpath, os.stat(dirpath).st_mode | stat.S_IWUSR)
    shutil.rmtree(path)


def _first_output_line(out: subprocess.CompletedProcess) -> str:
    lines = [ln for ln in (out.stdout + out.stderr).splitlines() if ln.strip()]
    return lines[0].strip() if lines else ""


def _path_first(directory: Path) -> dict:
    """The environment with `directory` first on PATH, so a bare program name runs the frozen copy in it."""
    return {**os.environ, "PATH": f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"}


def bun_version(bun_dir: Path) -> str:
    """`bun --version` of the bun in `bun_dir` (the snapshot's copy), "" when it does not run."""
    try:
        out = subprocess.run(["bun", "--version"], env=_path_first(bun_dir), capture_output=True, text=True,
                             timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return _first_output_line(out)


def entry_version(target: Path) -> str:
    """`omp --version` through the snapshot's own entry (`<target>/omp` first on PATH), without the `omp/` prefix."""
    try:
        out = subprocess.run(["omp", "--version"], env=_path_first(target), capture_output=True, text=True,
                             timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return _first_output_line(out).removeprefix("omp/").strip()


def _entry_script(target: Path, entry_rel: Path, p: Plan) -> str:
    return ("#!/bin/sh\n"
            f"# localbench omp freeze {p.snapshot_id}: omp {p.version} frozen from {p.package}\n"
            f"exec \"{target / 'bin' / 'bun'}\" \"{target / 'node_modules' / entry_rel}\" \"$@\"\n")


def freeze(p: Plan) -> dict:
    """Build the snapshot `p` names (or reuse it) and return its manifest plus {created}. Raises RuntimeError when
    `p.refuse` is set or the built entry does not report the frozen version."""
    if p.refuse:
        raise RuntimeError(p.refuse)
    if p.exists:
        return {**(_manifest(p.target) or {}), "created": False}
    base = root()
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f".tmp-{p.snapshot_id}-{os.getpid()}"
    if tmp.exists():
        _remove(tmp)
    try:
        for rel in p.packages:
            src = p.install / rel
            shutil.copytree(src, tmp / "node_modules" / rel, symlinks=False, ignore_dangling_symlinks=True,
                            ignore=_ignore_top_node_modules(src))
        (tmp / "bin").mkdir()
        shutil.copy2(p.bun, tmp / "bin" / "bun")
        bun = bun_version(tmp / "bin")
        package_rel = p.package.relative_to(p.install)
        entry_rel = p.entry.relative_to(p.install)
        copied = tmp / "node_modules" / entry_rel
        copied_version = json.loads((tmp / "node_modules" / package_rel / "package.json").read_text()).get("version")
        if f"{copied_version}-{sha16(str(copied))}" != p.snapshot_id:
            raise RuntimeError(f"omp at {p.package} changed during the copy (an update landed): copied "
                               f"{copied_version}-{sha16(str(copied))}, planned {p.snapshot_id}; run the freeze again")
        script = tmp / "node_modules" / package_rel / ENTRY_DIR / "omp"
        script.parent.mkdir()
        script.write_text(_entry_script(p.target, entry_rel, p))
        script.chmod(0o755)
        (tmp / "omp").symlink_to(Path("node_modules") / package_rel / ENTRY_DIR / "omp")
        manifest = {"snapshot_id": p.snapshot_id, "omp_version": p.version, "entry_sha16": sha16(str(copied)),
                    "source_package": str(p.package), "source_install": str(p.install),
                    "packages": [str(r) for r in p.packages], "bun_source": str(p.bun),
                    "bun_version": bun, "bun_sha16": sha16(str(tmp / "bin" / "bun")),
                    "size_bytes": _tree_bytes(tmp, False), "made_at": time.time()}
        (tmp / MARKER).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _read_only(tmp)
        try:
            os.rename(tmp, p.target)
        except OSError:
            if _manifest(p.target) is not None:   # a concurrent freeze of the same omp won the rename
                return {**(_manifest(p.target) or {}), "created": False}
            raise
    finally:
        if tmp.exists():
            _remove(tmp)
    got = entry_version(p.target)
    if got != p.version:
        _remove(p.target)
        raise RuntimeError(f"frozen omp {entry_path(p.target)} --version printed {got!r}, not {p.version}; "
                           "snapshot removed")
    return {**manifest, "created": True}


def _lock_path(snapshot_id: str) -> Path:
    return root() / ".locks" / f"{snapshot_id}.lock"


@contextlib.contextmanager
def hold(snapshot_id: str):
    """Hold a snapshot for a run: prune() skips it while any holder lives (shared flock, released on exit or death)."""
    path = _lock_path(snapshot_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        try:
            if _manifest(root() / snapshot_id) is None:
                raise RuntimeError(f"omp snapshot {root() / snapshot_id} is gone (pruned by another freeze); "
                                   "run `localbench omp freeze` again")
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def snapshots() -> list[tuple[Path, dict]]:
    """Snapshots this verb made (a valid marker naming the directory), newest first."""
    base = root()
    if not base.is_dir():
        return []
    out = []
    for d in base.iterdir():
        if d.name.startswith(".") or d.is_symlink() or not d.is_dir():
            continue
        m = _manifest(d)
        if m is not None:
            out.append((d, m))
    return sorted(out, key=lambda dm: float(dm[1].get("made_at") or 0), reverse=True)


def prune_candidates(keep_id: str, keep: int = KEEP) -> list[Path]:
    """Snapshots retention would remove: all but `keep_id` and the newest others, `keep` in total."""
    others = [d for d, _ in snapshots() if d.name != keep_id]
    return others[max(keep - 1, 0):]


def prune(keep_id: str, keep: int = KEEP) -> list[Path]:
    """Remove prune_candidates() not held by a run; return the removed ones."""
    removed = []
    for d in prune_candidates(keep_id, keep):
        path = _lock_path(d.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                _remove(d)
                removed.append(d)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        path.unlink(missing_ok=True)
    return removed
