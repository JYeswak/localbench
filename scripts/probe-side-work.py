"""Where should omp's side work run? Same-invocation comparison of four routings for one tool-using `omp -p` turn,
with every call timed by the proxy and every row's GPU time split by process.

Consumer: the 2026-09-23 routing decision (user: "measure first") and its ledger row. The main model is the MoE
under test on ollama; side work goes to the user's real smol model (qwen3.8 27B dense), reached under its PARKED
name through the proxy, so only this probe's side calls can load it while other omp sessions' smol calls fail
(`localbench park` must be active). Arms:
  current  default profile as configured: auto-thinking classifier + mnemopi memory on the dense model
  tiny     modelRoles.tiny: local/lfm2.5-230m (in-process ONNX, CPU): classifier and memory move off the GPU
  pinned   --thinking low: no classifier; memory still on the dense model
  floor    --thinking low + memory off: no side work
One warm-up round (discarded), then N rounds with the arm order rotated. Each arm keeps its own cwd (mnemopi's
per-project bank grows only with its own turns). Rows where a non-ollama process used >25% GPU are flagged, and an
arm's wall/pre-main ranking is only printed when its cv% is <= 10 (noise cannot land a verdict).
Since 2026-09-23 children run in localbench's own agent dir (workloads.child_env): the `current` arm no longer sees
the live default profile's roles or memory settings, only fixtures/omp/agent/config.yml plus the arm's overlay.
Results banked before that date measured the live profile.

Run with localbench parked and nothing else on the GPU:
  ~/.local/share/uv/tools/localbench/bin/python scripts/probe-side-work.py 6
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from localbench import park, sysstats
from localbench.__main__ import _busy_check
from localbench.backends import Ollama
from localbench.proxy import Proxy
from localbench.workloads import LEAN_FLAGS, _busy_s, _omp_final, child_env, ensure_localbench_model, omp_bin

MOE = "qwen3.6:35b-mlx"
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
parked = [p for p in (json.loads(park.STATE.read_text()) if park.STATE.exists() else [])
          if p["name"] == "qwen3.8:27b-mlx"]
if not parked:
    sys.exit("qwen3.8:27b-mlx is not parked: other omp sessions' side calls would share the dense model (run "
             "`localbench park`)")
DENSE = parked[0]["parked_as"]
CONTENT = "4817"
PROMPT = "Read the file answer.txt in the current directory and reply with only its contents."
ARMS = {
    "current": ([], None),
    "tiny": ([], "modelRoles:\n  tiny: local/lfm2.5-230m\n"),
    "pinned": (["--thinking", "low"], None),
    "floor": (["--thinking", "low"], "memory:\n  backend: off\n"),
}

summ, cpu, problems = _busy_check()
if problems:
    sys.exit("preflight refused: " + "; ".join(problems))

root = Path(f"/tmp/lbside-{time.strftime('%Y%m%dT%H%M%S')}")
cfg_dir = Path(tempfile.mkdtemp(prefix="lbside-overlays-"))
arms = {}
for name, (extra, yml) in ARMS.items():
    flags = ["--model", f"localbench/{MOE}", "--smol", f"localbench/{DENSE}", "--mode", "json", "--no-session",
             *LEAN_FLAGS, *extra]
    if yml:
        (cfg_dir / f"{name}.yml").write_text(yml)
        flags += ["--config", str(cfg_dir / f"{name}.yml")]
    cwd = root / name
    cwd.mkdir(parents=True)
    (cwd / "answer.txt").write_text(CONTENT + "\n")
    arms[name] = (flags, cwd)

be = Ollama()
be.isolate(MOE)
ctx_len = be.fingerprint(MOE)["loaded_context"]
calls = root / "calls.jsonl"
rows: list[dict] = []
names = list(arms)
commands: dict[int, str] = {}


def attempt(name: str, rnd: int) -> dict:
    flags, cwd = arms[name]
    g0, t0, started = sysstats.gpu_time_by_pid(), time.perf_counter(), time.time()
    proc = subprocess.run([omp_bin(), "-p", PROMPT, *flags], cwd=cwd, capture_output=True, text=True, timeout=900,
                          env=child_env(), stdin=subprocess.DEVNULL, check=False)
    wall, ended = time.perf_counter() - t0, time.time()
    gpu = sysstats.gpu_share(g0, sysstats.gpu_time_by_pid(), wall, commands)
    text, usage = _omp_final(proc.stdout)
    got = [r for r in map(json.loads, calls.read_text().splitlines()) if r["t_start"] >= started and r["t"] <= ended] \
        if calls.exists() else []
    main = sorted((r for r in got if r.get("tools")), key=lambda r: r["t_start"])
    side = [r for r in got if not r.get("tools")]
    by_purpose = {}
    for p in sorted({r.get("purpose", "aux") for r in side}):
        sel = [r for r in side if r.get("purpose", "aux") == p]
        by_purpose[p] = {"calls": len(sel), "busy_s": _busy_s(sel), "aborted": sum(bool(r["aborted"]) for r in sel),
                         "completion_tokens": sum(r.get("completion_tokens") or 0 for r in sel),
                         "models": sorted({r["model"] for r in sel})}
    return {
        "round": rnd, "arm": name, "rc": proc.returncode, "wall_s": round(wall, 3),
        "correct": proc.returncode == 0 and text.strip() == CONTENT, "answer": text[:80],
        "pre_main_s": round(main[0]["t_start"] - started, 3) if main else None,
        "main_calls": len(main), "main_s": _busy_s(main), "side": by_purpose,
        "gpu": {"moe": sum(r["pct"] for r in gpu if r.get("model") == MOE),
                "dense": sum(r["pct"] for r in gpu if r.get("model") == DENSE),
                "other": [(r["name"], r["pct"]) for r in gpu if r["name"] != "ollama"]},
        "foreign_gpu": [(r["name"], r["pid"], r["pct"]) for r in gpu if r["name"] != "ollama" and r["pct"] > 25],
        "input": usage["input"],
    }


with Proxy(be.base_url, calls, label="side"):
    ensure_localbench_model(MOE, ctx_len, extra=(DENSE,))
    for name in names:  # warm-up: loads the dense model and the tiny worker; discarded
        print(json.dumps({"warmup": attempt(name, -1)}), flush=True)
    for r in range(rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            row = attempt(name, r)
            rows.append(row)
            print(json.dumps(row), flush=True)


def stat(vals: list[float]) -> dict | None:
    if not vals:
        return None
    mean = statistics.fmean(vals)
    return {"median": round(statistics.median(vals), 3),
            "cv_pct": round(100 * statistics.pstdev(vals) / mean, 1) if mean else None}


summary = {}
for name in names:
    sel = [r for r in rows if r["arm"] == name]
    clean = [r for r in sel if not r["foreign_gpu"]]
    summary[name] = {
        "n": len(sel), "clean": len(clean), "correct": sum(r["correct"] for r in sel),
        "wall_s": stat([r["wall_s"] for r in clean]),
        "pre_main_s": stat([r["pre_main_s"] for r in clean if r["pre_main_s"] is not None]),
        "main_s": stat([r["main_s"] for r in clean if r["main_s"] is not None]),
        "side_busy_s": stat([sum(v["busy_s"] or 0 for v in r["side"].values()) for r in clean]),
        "gpu_dense_pct": stat([r["gpu"]["dense"] for r in clean]),
    }
    walls = summary[name]["wall_s"]
    summary[name]["rankable"] = bool(walls and walls["cv_pct"] is not None and walls["cv_pct"] <= 10)
print(json.dumps({"summary": summary, "moe": MOE, "dense": DENSE, "rounds": rounds, "root": str(root)}))
