#!/usr/bin/env python3
"""Validate the generated localbench parser contract matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MUTATING = {"memory", "watch-releases", "prove", "memory-verdict", "omp", "run", "eval", "aa", "ab",
            "bank", "record", "park", "unpark", "keep", "gateway", "pull", "create", "smol", "quiet",
            "ollama-app"}


def check(doc: dict) -> list[str]:
    rows = doc.get("parsers")
    if not isinstance(rows, list) or not rows:
        return ["matrix has no parser rows"]
    problems = []
    for row in rows:
        path = row.get("path", "")
        if row.get("rc") != 0:
            problems.append(f"{path}: --help rc={row.get('rc')}")
        if not row.get("help_bytes"):
            problems.append(f"{path}: empty help")
        top = path.split()[0] if path else ""
        if row.get("contract_required") and not row.get("has_dry_run"):
            problems.append(f"{path}: missing --dry-run")
        if row.get("contract_required") and not row.get("has_explain"):
            problems.append(f"{path}: missing --explain")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matrix", type=Path)
    args = ap.parse_args()
    problems = check(json.loads(args.matrix.read_text()))
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print(f"CLI-MATRIX-PASS {len(json.loads(args.matrix.read_text())['parsers'])} parsers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
