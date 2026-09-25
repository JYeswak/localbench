"""Same-invocation comparison of omp memory configurations: what does each cost an `omp -p` launch, and does it
change the answer?

Consumer: the 2026-09-23 ledger rows on memory. Every variant runs the same prompt against one warm backend through
the localbench proxy (main model = the model under test), rounds interleaved with a rotating variant order so drift
spreads across variants. Each variant gets its own fresh cwd, so mnemopi's per-project bank starts empty and grows
only with that variant's own turns (as real use would). One JSON line per attempt, then a summary per variant
(median, cv%, correct count, whether any foreign model appeared on ollama).

Two axes are separated, because both hit the GPU: the smol role (omp calls it on every launch, e.g. the auto
thinking-level classifier, in parallel with the main turn) and memory. All variants except the two `user*` ones route
smol to the model under test (`--smol localbench/<model>`), so they differ only in memory.

Variants (overlays are written to a temp dir and passed with --config):
  off          --smol moe, memory.backend: off                   (what localbench's children use)
  smol-moe     --smol moe, memory as configured (mnemopi, llmMode smol -> the model under test)
  llm-none     --smol moe, mnemopi.llmMode: none                 (recall + retain, no LLM extraction)
  lexical      --smol moe, llmMode none + noEmbeddings: true     (FTS only, no embedding model)
  tiny         --smol moe, modelRoles.memory: local/lfm2.5-230m  (in-process tiny model; `omp tiny-models download`)
  idle         --smol moe, mnemopi.autoRecall/autoRetain: false  (backend on, no automatic work)
  user         no overlay, no --smol: the default profile as configured (smol + memory -> ollama qwen3.8 dense)
  user-memoff  no --smol, memory.backend: off                    (smol still -> dense; isolates the smol cost)

Since 2026-09-23 children run in localbench's own agent dir (workloads.child_env): the `user*` variants no longer see
the live profile's roles and memory settings, only fixtures/omp/agent/config.yml plus the overlay. Results banked
before that date measured the live profile.

Run (uv tool env of localbench; backend idle, model loadable):
  ~/.local/share/uv/tools/localbench/bin/python scripts/probe-memory-overhead.py qwen3.6:35b-mlx 6
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from localbench.backends import Ollama
from localbench.proxy import Proxy
from localbench.workloads import LEAN_FLAGS, _attempt_calls, _omp_final, child_env, ensure_localbench_model, omp_bin

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6:35b-mlx"
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
prompt = "Reply with exactly: OK"
stamp = time.strftime("%Y%m%dT%H%M%S")
root = Path(f"/tmp/lbmem-{stamp}")
root.mkdir(parents=True)
overlays = {
    "off": "memory:\n  backend: off\n",
    "smol-moe": None,
    "llm-none": "mnemopi:\n  llmMode: none\n",
    "lexical": "mnemopi:\n  llmMode: none\n  noEmbeddings: true\n",
    "tiny": "modelRoles:\n  memory: local/lfm2.5-230m\n",
    "idle": "mnemopi:\n  autoRecall: false\n  autoRetain: false\n",
    "user": None,
    "user-memoff": "memory:\n  backend: off\n",
}
cfg_dir = Path(tempfile.mkdtemp(prefix="lbmem-overlays-"))
variants = {}
for name, yml in overlays.items():
    flags = ["--model", f"localbench/{model}", "--mode", "json", "--no-session", *LEAN_FLAGS]
    if yml is not None:
        path = cfg_dir / f"{name}.yml"
        path.write_text(yml)
        flags += ["--config", str(path)]
    if not name.startswith("user"):
        flags += ["--smol", f"localbench/{model}"]
    cwd = root / name
    cwd.mkdir()
    variants[name] = (flags, cwd)

be = Ollama()
be.isolate(model)
ctx_len = be.fingerprint(model)["loaded_context"]
calls = root / "calls.jsonl"
rows = []
names = list(variants)
with Proxy(be.base_url, calls, label="memprobe"):
    ensure_localbench_model(model, ctx_len)
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            flags, cwd = variants[name]
            t0, started = time.perf_counter(), time.time()
            proc = subprocess.run([omp_bin(), "-p", prompt, *flags], cwd=cwd, capture_output=True, text=True,
                                  timeout=900, env=child_env(), stdin=subprocess.DEVNULL, check=False)
            wall = time.perf_counter() - t0
            split = _attempt_calls(calls, started, time.time())
            text, usage = _omp_final(proc.stdout)
            resident = sorted(m["name"] for m in be.loaded())
            row = {"round": r, "variant": name, "rc": proc.returncode, "wall_s": round(wall, 3),
                   "correct": proc.returncode == 0 and text.strip().rstrip(".") == "OK", "answer": text[:60],
                   "input": usage["input"], "llm_calls": usage["llm_calls"], "resident": resident, **split}
            rows.append(row)
            print(json.dumps(row), flush=True)


def med(vals: list[float]) -> float | None:
    return round(statistics.median(vals), 3) if vals else None


summary = {}
for name in names:
    sel = [r for r in rows if r["variant"] == name]
    walls = [r["wall_s"] for r in sel]
    summary[name] = {
        "n": len(sel), "correct": sum(r["correct"] for r in sel),
        "wall_s": med(walls),
        "wall_cv_pct": round(100 * statistics.pstdev(walls) / statistics.fmean(walls), 1) if len(walls) > 1 else None,
        "startup_s": med([r["startup_s"] for r in sel if r["startup_s"] is not None]),
        "llm_s": med([r["llm_s"] for r in sel if r["llm_s"] is not None]),
        "input_tokens": med([r["input"] for r in sel]),
        "foreign_resident": sorted({m for r in sel for m in r["resident"] if m != model}),
    }
print(json.dumps({"summary": summary, "cwd_root": str(root), "rounds": rounds, "prompt": prompt}))
