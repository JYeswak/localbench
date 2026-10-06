#!/usr/bin/env python3
"""Run a resumable matrix of localbench CLI commands and rank exit cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localbench.heavyslot import acquire  # noqa: E402

SCHEMA = "localbench.cli-profile.v1"
_COMMAND_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class ProfileError(ValueError):
    """A profile or its saved results cannot safely be used."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _validate_profile(profile: object, *, source: str) -> dict:
    if not isinstance(profile, dict) or profile.get("schema_version") != SCHEMA:
        raise ProfileError(f"unsupported or invalid CLI profile: {source}")
    if not isinstance(profile.get("name"), str) or not profile["name"]:
        raise ProfileError("profile name must be a non-empty string")
    for key, minimum in (("samples", 1), ("warmups", 0)):
        value = profile.get(key)
        if type(value) is not int or value < minimum:
            raise ProfileError(f"profile {key} must be an integer >= {minimum}")
    commands = profile.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ProfileError("profile needs at least one command")
    seen = set()
    for command in commands:
        if not isinstance(command, dict):
            raise ProfileError("each profile command must be an object")
        command_id, argv = command.get("id"), command.get("argv")
        if not isinstance(command_id, str) or not _COMMAND_ID.fullmatch(command_id) or command_id in seen:
            raise ProfileError(f"invalid or duplicate command id: {command_id!r}")
        if (not isinstance(argv, list) or not argv or not isinstance(argv[0], str) or not argv[0]
                or any(not isinstance(arg, str) for arg in argv)):
            raise ProfileError(f"command {command_id!r} argv must be a non-empty string list")
        seen.add(command_id)
    return profile


def _load_profile(path: Path) -> tuple[dict, str]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read profile {path}: {exc}") from exc
    profile = _validate_profile(profile, source=str(path))
    return profile, _canonical_sha256(profile)


def _run_sample(argv: list[str]) -> dict:
    started = time.perf_counter()
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return {
        "elapsed_s": time.perf_counter() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def _new_manifest(profile: dict, digest: str) -> dict:
    return {"schema_version": SCHEMA, "profile": profile, "profile_sha256": digest, "status": "running"}


def _manifest(run_dir: Path) -> tuple[dict, str]:
    path = run_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read profile run manifest {path}: {exc}") from exc
    if (not isinstance(data, dict) or data.get("schema_version") != SCHEMA
            or not isinstance(data.get("profile_sha256"), str)):
        raise ProfileError(f"invalid profile run manifest: {path}")
    profile = _validate_profile(data.get("profile"), source=f"saved manifest {path}")
    digest = data["profile_sha256"]
    if _canonical_sha256(profile) != digest:
        raise ProfileError(f"invalid profile run manifest: {path}")
    return data, digest


def _read_command_result(path: Path, *, profile_sha256: str, command: dict, samples: int) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read command result {path}: {exc}") from exc
    rows = result.get("samples") if isinstance(result, dict) else None
    if (not isinstance(result, dict) or result.get("schema_version") != SCHEMA
            or result.get("profile_sha256") != profile_sha256 or result.get("command") != command
            or not isinstance(rows, list) or len(rows) != samples):
        raise ProfileError(f"invalid or incomplete command result: {path}")
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("elapsed_s")) not in (int, float)
                or not math.isfinite(row["elapsed_s"]) or row["elapsed_s"] < 0
                or not isinstance(row.get("stdout"), str) or not isinstance(row.get("stderr"), str)
                or type(row.get("returncode")) is not int
                or row.get("exit_class") != ("success" if row["returncode"] == 0 else "failure")):
            raise ProfileError(f"invalid sample in command result: {path}")
    return result


def run_profile(profile_path: Path, run_dir: Path, *, resume: bool = False, runner=None) -> dict:
    """Run only commands without a complete atomic sample record; completed command records are immutable."""
    profile, digest = _load_profile(Path(profile_path))
    run_dir = Path(run_dir)
    if resume:
        manifest, saved_digest = _manifest(run_dir)
        if saved_digest != digest or manifest["profile"] != profile:
            raise ProfileError("profile identity changed; refusing to resume")
    else:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise ProfileError(f"profile run already exists: {run_dir}; use --resume") from exc
        _atomic_json(run_dir / "manifest.json", _new_manifest(profile, digest))
        (run_dir / "commands").mkdir()
    run_sample = runner or _run_sample
    commands_dir = run_dir / "commands"
    for command in profile["commands"]:
        result_path = commands_dir / f"{command['id']}.json"
        if result_path.exists():
            _read_command_result(result_path, profile_sha256=digest, command=command,
                                 samples=profile["samples"])
            continue
        for _ in range(profile["warmups"]):
            run_sample(command["argv"])
        rows = []
        for _ in range(profile["samples"]):
            row = run_sample(command["argv"])
            code = row.get("returncode")
            if type(code) is not int:
                raise ProfileError(f"command {command['id']!r} runner returned no integer exit code")
            rows.append({"elapsed_s": row["elapsed_s"], "stdout": row["stdout"], "stderr": row["stderr"],
                         "returncode": code, "exit_class": "success" if code == 0 else "failure"})
        _atomic_json(result_path, {"schema_version": SCHEMA, "profile_sha256": digest,
                                   "command": command, "samples": rows})
    report = build_report(run_dir)
    manifest, _ = _manifest(run_dir)
    manifest["status"] = report["status"].lower()
    _atomic_json(run_dir / "manifest.json", manifest)
    return report


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def build_report(run_dir: Path) -> dict:
    manifest, digest = _manifest(Path(run_dir))
    profile = manifest["profile"]
    cohorts = {"success": [], "failure": []}
    missing = []
    for command in profile["commands"]:
        path = Path(run_dir) / "commands" / f"{command['id']}.json"
        if not path.is_file():
            missing.append(command["id"])
            continue
        result = _read_command_result(path, profile_sha256=digest, command=command, samples=profile["samples"])
        for cohort, predicate in (("success", lambda row: row["returncode"] == 0),
                                  ("failure", lambda row: row["returncode"] != 0)):
            times = [float(row["elapsed_s"]) for row in result["samples"] if predicate(row)]
            if times:
                cohorts[cohort].append({"command": command["id"], "argv": command["argv"], "n": len(times),
                                        "p50_s": median(times), "p95_s": _percentile(times, 0.95),
                                        "mean_s": mean(times), "min_s": min(times)})
    for rows in cohorts.values():
        rows.sort(key=lambda row: (row["p50_s"], row["command"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
    return {"profile": profile["name"], "status": "COMPLETE" if not missing else "INCOMPLETE",
            "completed_commands": len(profile["commands"]) - len(missing),
            "total_commands": len(profile["commands"]), "missing_commands": missing, "cohorts": cohorts}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="run a profile; completed commands are saved atomically")
    run.add_argument("spec", type=Path)
    run.add_argument("--output", type=Path, required=True, help="new run directory, or existing directory with --resume")
    run.add_argument("--resume", action="store_true", help="skip commands with complete saved sample records")
    run.add_argument("--wait-slot", type=float, default=0, metavar="SECONDS",
                     help="queue for the heavy-job slot up to SECONDS")
    report = commands.add_parser("report", help="rank saved successful and failed command cohorts")
    report.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.action == "report":
        result = build_report(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "COMPLETE" else 1



    result: dict = {}
    try:
        with acquire("cli-profile", wait_s=max(0.0, args.wait_slot), needs_gpu=False):
            result = run_profile(args.spec, args.output, resume=args.resume)
    except KeyboardInterrupt:
        print("profile interrupted; completed commands are saved; resume with --resume", file=sys.stderr)
        return 130
    except (ProfileError, OSError) as exc:
        print(f"cli profile: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
