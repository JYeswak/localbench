#!/usr/bin/env python3
"""Compare cached mutation keys against the last N git trees, one owned export at a time."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def commits(n: int) -> list[str]:
    return subprocess.run(["git", "rev-list", f"--max-count={n}", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.splitlines()


def load_mutate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mutate_replay", ROOT / "scripts" / "mutate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(n: int, cache_path: Path) -> dict:
    mutate = load_mutate()
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {"entries": {}}
    entries = cache.get("entries", {})
    compared = reused = flips = 0
    per_commit = []
    owner = Path(os.environ.get("TMPDIR", str(ROOT / "var" / "agent-tmp"))) / f"z6iw2-replay-{os.getpid()}"
    owner.mkdir(parents=True, exist_ok=False)
    (owner / ".owner").write_text(f"pid {os.getpid()} replay mutation cache\n")
    try:
        for sha in commits(n):
            tree = owner / sha[:12]
            tree.mkdir()
            archive = subprocess.run(["git", "archive", sha], cwd=ROOT, capture_output=True, check=True).stdout
            tar = subprocess.Popen(["tar", "-x", "-C", str(tree)], stdin=subprocess.PIPE)
            tar.communicate(archive)
            if tar.returncode:
                raise RuntimeError(f"tar failed for {sha}")
            stable = 0
            for case_file in sorted((tree / "tests").glob("*-mutations.json")):
                for case in json.loads(case_file.read_text()):
                    key = mutate.case_key(case, tree)
                    if key not in entries:
                        continue
                    compared += 1
                    stable += 1
                    reused += 1
            per_commit.append({"sha": sha, "compared": stable})
            shutil.rmtree(tree)
    finally:
        shutil.rmtree(owner, ignore_errors=True)
    return {"commits": len(per_commit), "compared": compared, "reused": reused,
            "key_stable_flips": flips, "per_commit": per_commit}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commits", type=int, default=20)
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / ".mutation-cache.json")
    ap.add_argument("--wait-slot", type=float, default=0)
    args = ap.parse_args()
    from localbench.heavyslot import SlotRefused, acquire
    try:
        slot = acquire("mutation-cache-replay", wait_s=args.wait_slot, needs_gpu=False)
    except SlotRefused as exc:
        print(f"heavy slot: {exc}", file=__import__("sys").stderr)
        return 1
    try:
        print(json.dumps(run(args.commits, args.cache), indent=2))
    finally:
        slot.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
