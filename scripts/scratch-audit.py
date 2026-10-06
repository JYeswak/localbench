#!/usr/bin/env python3
"""Audit localbench scratch ownership and nested repository hazards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def scan(root: Path) -> dict:
    """Return every direct scratch subtree, owner metadata, and nested .git hazard."""
    base = root / "var" / "agent-tmp"
    if not base.is_dir():
        return {"root": str(base), "subtrees": 0, "ownerless": [], "nested_git": [], "foreign": []}
    ownerless: list[str] = []
    nested_git: list[str] = []
    foreign: list[str] = []
    subtrees = [p for p in sorted(base.iterdir()) if p.is_dir() and not p.is_symlink()]
    for subtree in subtrees:
        if not (subtree / ".owner").is_file():
            ownerless.append(str(subtree.relative_to(root)))
        for git in subtree.rglob(".git"):
            if git.is_dir() or git.is_file():
                nested_git.append(str(git.relative_to(root)))
        owner = subtree / ".owner"
        if owner.is_file():
            text = owner.read_text(encoding="utf-8", errors="replace")
            declared = None
            for line in text.splitlines():
                if line.startswith("repo="):
                    declared = line.split("=", 1)[1].strip()
            if declared and Path(declared).resolve() != root.resolve():
                foreign.append(f"{subtree.relative_to(root)}: owner repo {declared}")
    return {"root": str(base), "subtrees": len(subtrees), "ownerless": ownerless,
            "nested_git": nested_git, "foreign": foreign}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = scan(Path(args.root).expanduser().resolve())
    bad = result["ownerless"] or result["nested_git"] or result["foreign"]
    if args.json:
        print(json.dumps({**result, "status": "FAIL" if bad else "PASS"}, sort_keys=True))
    else:
        print(f"scratch {result['status'] if 'status' in result else ('FAIL' if bad else 'PASS')}: "
              f"{result['subtrees']} subtrees")
        for key in ("ownerless", "nested_git", "foreign"):
            for item in result[key]:
                print(f"{key}: {item}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
