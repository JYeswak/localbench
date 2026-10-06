#!/usr/bin/env python3
"""Run a localbench proof plan from an immutable main-tree export.

Execution keeps Beads writes in the main repo and persists new receipts and runs before normal export cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify_beads_root(root: Path, env: dict[str, str]) -> None:
    expected = (root.resolve() / ".beads").resolve()
    try:
        result = subprocess.run(["br", "where", "--json"], cwd=root, env=env,
                                capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"run_plan: cannot verify Beads workspace: {exc}") from exc
    if result.returncode:
        raise SystemExit(f"run_plan: br where --json failed: {result.stderr.strip()}")
    try:
        actual = Path(json.loads(result.stdout)["path"]).expanduser().resolve()
    except (ValueError, KeyError, TypeError):
        raise SystemExit("run_plan: br where --json returned no usable Beads path") from None
    if actual != expected:
        raise SystemExit(f"run_plan: br resolves Beads to {actual}; expected {expected}")


def persist_artifacts(export: Path, root: Path, prior_receipts: dict[str, bytes]) -> None:
    source_receipts = export / "docs" / "evidence" / "receipts"
    target_receipts = root / "docs" / "evidence" / "receipts"
    if source_receipts.is_dir():
        for source in source_receipts.iterdir():
            if not source.is_file() or source.name.startswith("."):
                continue
            if prior_receipts.get(source.name) == source.read_bytes():
                continue
            target_receipts.mkdir(parents=True, exist_ok=True)
            target = target_receipts / source.name
            if target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise RuntimeError(f"run_plan: refusing to overwrite receipt {target}")
                continue
            shutil.copy2(source, target)

    source_runs = export / "runs"
    if source_runs.is_dir():
        for source in source_runs.iterdir():
            if not source.is_dir() or source.is_symlink():
                continue
            target = root / "runs" / source.name
            if target.exists():
                raise RuntimeError(f"run_plan: refusing to overwrite run directory {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec")
    ap.add_argument("--omp-frozen", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true", help="execute proof workloads; admission gates still apply")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    work = ROOT / "var" / "agent-tmp" / f"run-plan-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=False)
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    preserve_work = False
    try:
        (work / ".owner").write_text(
            f"pid={os.getpid()} label=run-plan repo={ROOT.resolve()} created={datetime.now(timezone.utc).isoformat()}\n"
        )
        # Pin once, then archive that exact object. A concurrent main advance must not
        # make the archive's bytes disagree with the export's recorded HEAD.
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, env=git_env,
                              capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        archive = subprocess.run(["git", "archive", head], cwd=ROOT, env=git_env,
                                 capture_output=True, timeout=120, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(work)], input=archive, capture_output=True,
                       timeout=120, check=True)
        archived_receipts = work / "docs" / "evidence" / "receipts"
        prior_receipts = ({item.name: item.read_bytes() for item in archived_receipts.iterdir()
                           if item.is_file()} if archived_receipts.is_dir() else {})
        # Give the export isolated git metadata backed only by immutable objects from HEAD.
        subprocess.run(["git", "init", "--quiet"], cwd=work, env=git_env, timeout=30, check=True)
        objects = subprocess.run(["git", "rev-parse", "--git-path", "objects"], cwd=ROOT,
                                 env=git_env, capture_output=True, text=True, timeout=30,
                                 check=True).stdout.strip()
        object_dir = (ROOT / objects).resolve() if not Path(objects).is_absolute() else Path(objects)
        alternates = work / ".git" / "objects" / "info" / "alternates"
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(f"{object_dir}\n")
        subprocess.run(["git", "update-ref", "refs/heads/main", head], cwd=work,
                       env=git_env, timeout=30, check=True)
        subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=work,
                       env=git_env, timeout=30, check=True)
        subprocess.run(["git", "read-tree", head], cwd=work, env=git_env, timeout=30, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=work, env=git_env, capture_output=True, text=True,
                                timeout=30, check=True).stdout.strip()
        if status:
            raise SystemExit(f"run_plan: export content differs from pinned HEAD {head[:12]}: {status}")
        export_root = work.resolve()
        target = (work / args.spec).resolve()
        try:
            spec_rel = target.relative_to(export_root).as_posix()
        except ValueError:
            raise SystemExit(f"run_plan: spec path escapes HEAD export: {args.spec}") from None
        if not target.is_file():
            raise SystemExit(f"run_plan: spec not in HEAD export: {args.spec}")
        env = git_env.copy()
        env["PYTHONPATH"] = str(work)
        # The proof code and data stay pinned to the export. Beads writes target the main repo.
        env["LOCALBENCH_HOME"] = str(work.resolve())
        env["LOCALBENCH_BEADS_ROOT"] = str(ROOT.resolve())
        if args.execute:
            verify_beads_root(ROOT, env)
        command = [
            "uv", "run", "--quiet", "--project", str(export_root),
            "python", "-m", "localbench", "prove", spec_rel, "--json",
        ]
        if not args.execute:
            command.append("--dry-run")
        if args.omp_frozen:
            command.append("--omp-frozen")
        result = subprocess.run(command, cwd=work, env=env, text=True, timeout=600)
        if args.execute:
            try:
                persist_artifacts(work, ROOT, prior_receipts)
            except Exception:
                preserve_work = True
                print(f"run_plan: artifact persistence failed; export retained at {work}", file=sys.stderr)
                raise
        return result.returncode
    finally:
        if not preserve_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
