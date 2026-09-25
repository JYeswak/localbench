#!/usr/bin/env python3
"""List the local models this machine keeps, why each is kept, and delete the ones named on the command line.

Anti-ceremony (A12):
- Consumer: the owner, reclaiming disk from models no omp feature, golden or running server uses (2026-09-24: a rejected
  22 GB Nemotron and a 19 GB model only ever loaded as omp's fallback).
- Gate: nothing is deleted unless named exactly with --delete and unkept; the listing is the default.
- Defect class: deleting a model a golden pins (its gate goes UNAVAILABLE), one an omp profile routes a feature to (every
  session's titles/memory/scouts break), the resident one, or a parked copy while PARKED.json still promises to restore it.
- Delete when: `localbench models` grows a prune verb with the same keep rules.

    uv run python scripts/prune_models.py                       # list: kept (with reasons) and candidates
    uv run python scripts/prune_models.py --delete NAME [NAME]   # delete exactly these candidates

Kept: resident now; pinned by a golden (by model digest, so a parked name counts too); routed by an omp profile
(omp's own resolved settings, via the original name for a parked copy); named in another tool's config or a skill
(CONFIGS: Codex's socraticode/skill-search embed with nomic-embed-text, the splash skill requires its Splash model;
neither goes through omp, and the first dry run offered both); cloud stubs (no local weights). Everything
else is a candidate, shown with when observe.db last saw it resident. omp's own CPU models (tiny, fastembed) are out
of scope: omp manages them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from localbench import golden, models, observe, park, sysstats
from localbench.workloads import ROOT

LOG = ROOT / "runs" / "prune-log.jsonl"
RUN_PATTERN = "localbench (aa|run|ab|record)"
HOME = Path.home()
CONFIGS = (HOME / ".codex" / "config.toml", HOME / ".claude.json", HOME / ".claude" / "settings.json",
           HOME / ".omp" / "agent" / "mcp.json", HOME / ".omp" / "agent" / "config.yml")


def config_texts(paths=CONFIGS, skills: Path = HOME / ".agents" / "skills") -> dict[str, str]:
    """label -> text of each tool config and skill doc that exists: where a model outside omp's routes is named."""
    files = [p for p in paths if p.is_file()]
    files += sorted(skills.glob("*/SKILL.md")) if skills.is_dir() else []
    out = {}
    for p in files:
        try:
            out[str(p).replace(str(HOME), "~", 1)] = p.read_text(errors="replace")
        except OSError:
            continue
    return out


def names_of(row: dict) -> set[str]:
    """How a config would name this model: an ollama tag's repo (`nomic-embed-text` for nomic-embed-text:latest) and
    full tag, via the original name for a parked copy; a model directory's `org/name`."""
    if row["server"] == "ollama":
        source = row.get("source") or row["name"]
        return {source, source.split(":")[0]}
    return {row["name"]}


def named_in(row: dict, configs: dict[str, str]) -> list[str]:
    """Config labels that name this model as a whole token: `qwen3.8` is not named by `qwen3.8-uncensored`."""
    pats = [re.compile(r"(?<![\w.-])" + re.escape(n) + r"(?![\w-])") for n in names_of(row)]
    return [label for label, text in configs.items() if any(p.search(text) for p in pats)]


def golden_pins(root: Path = golden.GOLDENS) -> list[dict]:
    """(backend, model, model_digest) of every banked golden on this checkout."""
    out = []
    for path in sorted(root.glob("*/*.json")):
        pins = json.loads(path.read_text()).get("pins") or {}
        out.append({"backend": pins.get("backend"), "model": pins.get("model"),
                    "digest": (pins.get("model_digest") or "")[:12], "golden": path.name})
    return out


def last_resident(db: Path = observe.DB) -> dict[tuple[str, str], float]:
    """(server, model) -> last time `localbench watch` saw it loaded."""
    if not db.exists():
        return {}
    con = sqlite3.connect(db)
    try:
        return {(s, m): t for s, m, t in con.execute("select server, model, max(t) from resident group by 1, 2")}
    finally:
        con.close()


def plan(rows: list[dict], goldens: list[dict], routes: dict, resident: dict, last_seen: dict,
         configs: dict[str, str] | None = None) -> list[dict]:
    """Each inventory row plus `keep` (reasons; empty means a delete candidate) and `last_seen` (epoch or None).
    A row is matched to goldens and residents by its last path segment, so mlx-serve's `org/name` meets a pin's `name`."""
    out = []
    for r in rows:
        server, name = r["server"], r["name"]
        short = name.split("/")[-1]
        keep = []
        if r.get("freshness", "").startswith("cloud model"):
            keep.append("cloud model (no local weights)")
        if short in (resident.get(server) or []):
            keep.append("resident now")
        backend = "ollama" if server == "ollama" else "mlx-serve"
        for g in goldens:
            if g["backend"] == backend and (g["model"] == short or (r.get("digest") and g["digest"] == r["digest"])):
                keep.append(f"golden {g['golden']}")
        used = routes.get(r.get("source") or name) or {}
        if used:
            keep.append("omp routes " + "; ".join(f"{f} ({', '.join(p)})" for f, p in sorted(used.items())))
        where = named_in(r, configs or {})
        if where:
            keep.append("named in " + ", ".join(where[:3]) + (f" (+{len(where) - 3} more)" if len(where) > 3 else ""))
        seen = [t for (s, m), t in last_seen.items() if s == server and m.split("/")[-1] == short]
        if r.get("parked"):
            seen += [t for (s, m), t in last_seen.items() if s == server and m == r.get("source")]
        out.append({**r, "keep": keep, "last_seen": max(seen) if seen else None})
    return out


def check_delete(planned: list[dict], names: list[str]) -> tuple[list[dict], list[str]]:
    """The rows `names` select, and one refusal per name that is unknown or kept."""
    by_name = {r["name"]: r for r in planned}
    targets, refusals = [], []
    for n in names:
        r = by_name.get(n)
        if r is None:
            refusals.append(f"{n}: not installed (names are as listed, e.g. qwen3.6:35b-mlx or org/name)")
        elif r["keep"]:
            refusals.append(f"{n}: kept ({'; '.join(r['keep'])})")
        else:
            targets.append(r)
    return targets, refusals


def delete(row: dict, state: Path = park.STATE) -> None:
    """Remove one candidate. An ollama tag goes through ollama's API (blobs no other tag uses are freed); a parked copy
    also leaves PARKED.json, so `localbench unpark` does not try to restore it. A model directory is removed whole."""
    if row["server"] == "ollama":
        park._delete(row["name"])
        if row.get("parked") and state.exists():
            left = [p for p in json.loads(state.read_text()) if p["parked_as"] != row["name"]]
            if left:
                state.write_text(json.dumps(left, indent=2) + "\n")
            else:
                state.unlink()
        return
    base = models.SPLASH_MODELS if row["server"] == "splash" else models.MLX_MODELS
    path = (base / row["name"]).resolve()
    if base.resolve() not in path.parents:
        raise RuntimeError(f"{path} is outside {base}; refusing to remove it")
    shutil.rmtree(path)


def run_alive() -> bool:
    return subprocess.run(["pgrep", "-f", RUN_PATTERN], capture_output=True, check=False).returncode == 0


def gather() -> list[dict]:
    rows = models.ollama_models() + models.mlx_models()
    return plan(rows, golden_pins(), models.routes_by_model(models.profiles()), sysstats.resident_models(),
                last_resident(), config_texts())


def show(planned: list[dict]) -> None:
    def when(t):
        return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else "never (in observe.db)"
    for label, rows in (("KEPT", [r for r in planned if r["keep"]]), ("CANDIDATES", [r for r in planned if not r["keep"]])):
        print(f"{label} ({sum(r['gb'] for r in rows):.1f} GB):")
        for r in sorted(rows, key=lambda r: -r["gb"]):
            extra = "; ".join(r["keep"]) if r["keep"] else f"last resident {when(r['last_seen'])}"
            parked = f" [parked copy of {r['source']}]" if r.get("parked") else ""
            print(f"  {r['gb']:6.1f} GB  {r['server']:<9} {r['name']}{parked}  — {extra}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="List local models, why each is kept, and delete named candidates.")
    ap.add_argument("--delete", nargs="+", metavar="NAME", default=[])
    a = ap.parse_args(argv)
    planned = gather()
    if not a.delete:
        show(planned)
        return 0
    if run_alive():
        print("a localbench run is alive; delete after it ends")
        return 1
    targets, refusals = check_delete(planned, a.delete)
    for msg in refusals:
        print("refused:", msg)
    if refusals:
        return 1
    for r in targets:
        delete(r)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as fh:
            fh.write(json.dumps({"t": time.time(), "server": r["server"], "name": r["name"], "gb": r["gb"],
                                 "digest": r.get("digest"), "source": r.get("source")}) + "\n")
        print(f"deleted {r['server']} {r['name']} ({r['gb']:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
