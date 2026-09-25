"""Tiers of work, each case producing metrics and/or a conformance verdict.

micro   raw engine speed: decode, cold prefill 1k/8k/32k, prefix-cache hit, A,B,A eviction probe
conf    MUST/SHOULD clauses omp depends on: tool calls, usage, no truncation, streaming
replay  recorded omp request bodies (fixtures/omp/<label>.json + .meta.json) replayed cold / warm / turn-2
e2e     real `omp -p` turns with the lean launcher flags, answers checked, every "first" run cold
rel     opt-in: each e2e task REL_ATTEMPTS times, warm; wrong-answer rate with a Wilson 95% interval
relcold opt-in: the "ok" task REL_ATTEMPTS times, backend re-isolated (cold KV) before every attempt
relfresh opt-in: relcold without the one-token warm-up request after each isolate

A metric is {"value", "better", "spread": [min, max], "n"}; a metric that cannot be trusted carries
"void": "<reason>" instead of being averaged in. Tolerance bands are NOT set here: goldens derive them
from an A/A null (see golden.py).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import random
import re
import shutil
import statistics
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .client import Sample, chat_stream
from .render import clip

# The data root: a clone's fixtures/, goldens/, runs/ and docs/evidence/. By default the checkout this module was loaded
# from (`uv tool install -e .`); LOCALBENCH_HOME points any other install at a clone. __main__ refuses a root without
# fixtures/omp, because a wheel install resolves this to site-packages and would answer from empty state.
ROOT = Path(os.environ.get("LOCALBENCH_HOME") or Path(__file__).resolve().parent.parent).expanduser().resolve()
FIXTURES = ROOT / "fixtures"
# The mem tier's omp overlay. A variant (FTS-only recall, memory LLM elsewhere) is passed per run with --mem-config, or
# for ab's B leg only with --b-mem-config; the run pins hash whichever one ran (omp_mem_config).
MEM_CONFIG = FIXTURES / "omp" / "child-config-mem.yml"
MEM_ROUNDS = 3

# Rough chars/token for the synthetic corpus; the server-reported prompt_tokens is what gets recorded.
CHARS_PER_TOKEN = 3.4


def recorded_answers(texts: list[str]) -> list[str]:
    """Answer text a correctness verdict keeps, on pass and on fail. Short text stays whole; longer text
    names how many characters were cut. Omitting this is how c677837's receipt lost the failed answer."""
    return [clip(t) for t in texts]


@dataclass
class Ctx:
    backend: object
    model: str
    repeats: int
    run_dir: Path
    emit: Callable[[dict], None]
    pins: dict = field(default_factory=dict)
    loaded_context: int | None = None
    mem_config: Path = MEM_CONFIG
    mem_rounds: int = MEM_ROUNDS
    samples: list[dict] = field(default_factory=list)

    def chat(self, label: str, body: dict) -> Sample:
        body = {"model": self.model, "temperature": 0, **body}
        s = chat_stream(self.backend.base_url, body, backend=self.backend.name, label=label)
        row = s.to_dict()
        row["text"] = row["text"][:400]
        row["reasoning"] = row["reasoning"][:400]
        self.samples.append(row)
        self.emit({"event": "sample", **{k: row[k] for k in (
            "label", "prompt_tokens", "cached_tokens", "completion_tokens", "ttft_s", "prefill_tps",
            "decode_tps", "total_s", "error")}})
        return s


@dataclass
class Result:
    case: str
    tier: str
    level: str  # "perf" | "MUST" | "SHOULD"
    metrics: dict = field(default_factory=dict)  # name -> metric dict (see module docstring)
    verdict: str | None = None  # conformance: PASS | FAIL | VOID
    detail: dict = field(default_factory=dict)


def metric(values: list[float | None], better: str) -> dict:
    vals = [v for v in values if v]
    if not vals:
        return {"value": None, "better": better, "void": "no valid samples"}
    return {"value": round(statistics.median(vals), 4), "better": better,
            "spread": [round(min(vals), 4), round(max(vals), 4)], "n": len(vals)}


def void(better: str, reason: str) -> dict:
    return {"value": None, "better": better, "void": reason}


def corpus(approx_tokens: int, seed: int = 7) -> str:
    """Deterministic code-shaped filler: identical bytes every run, so token counts are stable."""
    rng = random.Random(seed)
    words = ["self", "value", "index", "result", "config", "buffer", "stream", "token", "cache", "items",
             "length", "offset", "handler", "request", "response", "error", "state", "window", "model", "batch"]
    lines, size, n = [], 0, 0
    target = int(approx_tokens * CHARS_PER_TOKEN)
    while size < target:
        n += 1
        a, b, c = rng.sample(words, 3)
        line = rng.choice([
            f"def {a}_{b}_{n}({c}, {a}=None):",
            f"    {a} = {b}.get('{c}', {rng.randint(0, 999)})",
            f"    if {a} is not None and len({b}) > {rng.randint(1, 64)}:",
            f"        return {c}[{a}:{a} + {rng.randint(1, 32)}]",
            f"    # {a} {b} {c}: keep {rng.randint(2, 9)} entries",
            f"class {a.title()}{b.title()}{n}:",
        ])
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def _nonce() -> str:
    return f"[run {uuid.uuid4().hex}]\n"


# ---------------------------------------------------------------- micro

def micro(ctx: Ctx) -> list[Result]:
    out = []
    dec = [ctx.chat("micro.decode", {
        "messages": [{"role": "user", "content": _nonce() + "Explain, in detail, how a hash map resolves collisions."}],
        "max_tokens": 256}) for _ in range(ctx.repeats)]
    out.append(Result("micro.decode", "micro", "perf", {
        "decode_tps": metric([s.decode_tps for s in dec], "higher"),
        "ttft_s": metric([s.ttft_s for s in dec], "lower")}))

    for label, toks in (("1k", 1000), ("8k", 8000), ("32k", 32000)):
        body = corpus(toks) + "\n\nIn one sentence, what does this code do?"
        runs = [ctx.chat(f"micro.prefill_{label}", {
            "messages": [{"role": "user", "content": _nonce() + body}], "max_tokens": 16})
            for _ in range(ctx.repeats)]
        out.append(Result(f"micro.prefill_{label}", "micro", "perf", {
            "prefill_tps": metric([s.prefill_tps for s in runs], "higher"),
            "ttft_s": metric([s.ttft_s for s in runs], "lower")},
            detail={"prompt_tokens": runs[-1].prompt_tokens}))

    # Prefix cache: identical 8k prompt twice; the second TTFT is what turn 2 of a chat feels like.
    a_body = [{"role": "user", "content": _nonce() + corpus(8000) + "\n\nName one function above."}]
    cold = ctx.chat("micro.cache_cold_8k", {"messages": a_body, "max_tokens": 16})
    warm = ctx.chat("micro.cache_warm_8k", {"messages": a_body, "max_tokens": 16})
    out.append(Result("micro.cache_hit_8k", "micro", "perf", {
        "warm_ttft_s": metric([warm.ttft_s], "lower"),
        "speedup_x": metric([cold.ttft_s / warm.ttft_s if warm.ttft_s else None], "higher")},
        detail={"cold_ttft_s": cold.ttft_s, "cached_tokens": warm.cached_tokens}))

    # Eviction probe: A (cached above), then a different 8k prompt B, then A again.
    # A-again fast => the backend kept A's prefix while serving B (multi-entry); slow => single slot.
    b_body = [{"role": "user", "content": _nonce() + corpus(8000, seed=11) + "\n\nName one class above."}]
    ctx.chat("micro.evict_b_8k", {"messages": b_body, "max_tokens": 16})
    again = ctx.chat("micro.evict_a_again_8k", {"messages": a_body, "max_tokens": 16})
    ratio = again.ttft_s / cold.ttft_s if cold.ttft_s else None
    out.append(Result("micro.eviction_probe_8k", "micro", "perf", {
        "a_again_ttft_s": metric([again.ttft_s], "lower")},
        detail={"a_cold_ttft_s": cold.ttft_s, "a_again_over_cold": round(ratio, 3) if ratio else None,
                "verdict": None if ratio is None else ("evicted" if ratio > 0.5 else "retained")}))
    return out


# ---------------------------------------------------------------- conformance

TOOL = {"type": "function", "function": {
    "name": "read_file", "description": "Read a UTF-8 text file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}


def conformance(ctx: Ctx) -> list[Result]:
    out = []

    s = ctx.chat("conf.tool_call", {
        "messages": [{"role": "user", "content": "Use the read_file tool to read /etc/hosts. Do not answer from memory."}],
        "tools": [TOOL], "max_tokens": 512})
    shape, ok = None, False
    if s.tool_calls:
        call = s.tool_calls[0]
        try:
            args = json.loads(call["arguments"] or "{}")
            shape = {"name": call["name"], "arg_keys": sorted(args)}
            ok = call["name"] == "read_file" and args.get("path") == "/etc/hosts"
        except json.JSONDecodeError as exc:
            shape = {"name": call["name"], "arguments_error": str(exc)}
    out.append(Result("conf.tool_call", "conf", "MUST", verdict="PASS" if ok else "FAIL",
                      detail={"shape": shape, "finish_reason": s.finish_reason, "text": s.text[:200]}))

    out.append(Result("conf.usage_reported", "conf", "MUST",
                      verdict="PASS" if s.prompt_tokens > 0 and s.completion_tokens > 0 else "FAIL",
                      detail={"prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens}))

    small = ctx.chat("conf.ctx_8k", {"messages": [{"role": "user", "content": _nonce() + corpus(8000) + "\nOK?"}],
                                     "max_tokens": 4})
    big = ctx.chat("conf.ctx_64k", {"messages": [{"role": "user", "content": _nonce() + corpus(64000) + "\nOK?"}],
                                    "max_tokens": 4})
    ratio = big.prompt_tokens / small.prompt_tokens if small.prompt_tokens else 0
    out.append(Result("conf.no_truncation_64k", "conf", "MUST", verdict="PASS" if ratio >= 7.5 else "FAIL",
                      detail={"tokens_8k": small.prompt_tokens, "tokens_64k": big.prompt_tokens,
                              "ratio": round(ratio, 2), "error": big.error}))

    long = ctx.chat("conf.streams", {"messages": [{"role": "user", "content": "Count from 1 to 60, comma separated."}],
                                     "max_tokens": 200})
    out.append(Result("conf.streams_incrementally", "conf", "MUST",
                      verdict="PASS" if long.decode_s > 0 and long.completion_tokens > 20 else "FAIL",
                      detail={"decode_s": long.decode_s, "completion_tokens": long.completion_tokens}))

    msgs = [{"role": "user", "content": "List three prime numbers greater than 100, comma separated, nothing else."}]
    a = ctx.chat("conf.greedy_a", {"messages": msgs, "max_tokens": 400})
    b = ctx.chat("conf.greedy_b", {"messages": msgs, "max_tokens": 400})
    # Reasoning models can spend the whole budget thinking; determinism is about every generated token, so the
    # comparison covers reasoning + content, and an empty generation is a FAIL, never a vacuous PASS.
    gen_a, gen_b = a.reasoning + "\n--content--\n" + a.text, b.reasoning + "\n--content--\n" + b.text
    out.append(Result("conf.greedy_deterministic", "conf", "SHOULD",
                      verdict="PASS" if (a.reasoning or a.text) and gen_a == gen_b else "FAIL",
                      detail={"a": gen_a[-160:], "b": gen_b[-160:], "finish": [a.finish_reason, b.finish_reason]}))
    return out


# ---------------------------------------------------------------- replay

def _with_nonce(msgs: list[dict]) -> list[dict]:
    m = json.loads(json.dumps(msgs))
    first = m[0]
    if isinstance(first.get("content"), str):
        first["content"] = _nonce() + first["content"]
    else:
        first["content"] = [{"type": "text", "text": _nonce()}, *(first.get("content") or [])]
    return m


def fixtures_sha() -> str:
    """Identity of the request bodies replay sends: every fixtures/omp/<label>.json (not the sidecars)."""
    h = hashlib.sha256()
    for p in sorted((FIXTURES / "omp").glob("*.json")):
        if not p.name.endswith(".meta.json"):
            h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


TOKEN_DRIFT = 0.02


def prompt_token_verdict(name: str, replayed: int, recorded: int,
                          recorded_by: str | None, run_backend: str) -> Result:
    """Compare prompt tokens only against the tokenizer that recorded them.

    The sidecar count is one backend's. A different backend is VOID, not FAIL (kit-l7l: lean.meta.json recorded
    11433 through ollama; mlx-serve's count is a different tokenizer). A sidecar with no recording backend still
    compares, so a planted wrong count cannot hide behind a missing pin. Same-backend drift past 2% is FAIL."""
    detail = {"replayed": replayed, "recorded": recorded, "recorded_by": recorded_by, "run_backend": run_backend}
    if recorded_by and recorded_by != run_backend:
        detail["reason"] = f"recorded by {recorded_by}; this run is {run_backend}"
        return Result(f"replay.{name}.prompt_tokens", "replay", "SHOULD", verdict="VOID", detail=detail)
    drift = abs(replayed - recorded) / recorded if recorded else 1.0
    detail["drift"] = round(drift, 4)
    return Result(f"replay.{name}.prompt_tokens", "replay", "SHOULD",
                  verdict="PASS" if drift <= TOKEN_DRIFT else "FAIL", detail=detail)


def replay(ctx: Ctx) -> list[Result]:
    """Replay recorded omp request bodies (see `localbench record`) against the model under test.

    The bodies are fixed data, so replay measures the backend on the request an omp generation sent (the sidecar
    names it) whatever omp is installed today; goldens bind replay rows to fixtures_sha. Whether the fixture still
    matches the running omp is recorded per row (`fixture_fresh`) and reported by `localbench status` — omp ships
    most days, and voiding replay on each release left it unmeasured. A backend loaded with less context than the
    fixture's prompt is VOID. The prompt-token count belongs to the backend that recorded the sidecar. A different
    backend is VOID, not FAIL: its tokenizer is not the one that produced the count."""
    out = []
    fixtures = sorted(p for p in (FIXTURES / "omp").glob("*.json") if not p.name.endswith(".meta.json"))
    for path in fixtures:
        name = path.stem
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            out.append(Result(f"replay.{name}", "replay", "SHOULD", verdict="FAIL",
                              detail={"reason": f"missing sidecar {meta_path.name}; re-run `localbench record`"}))
            continue
        meta = json.loads(meta_path.read_text())
        names = ("cold_ttft_s", "warm_ttft_s", "turn2_ttft_s")
        fixture_gen = (meta.get("omp_version"), meta.get("omp_sha"), (meta.get("child_config") or {}).get("sha16"))
        running_gen = (ctx.pins.get("omp_version"), ctx.pins.get("omp_sha"), ctx.pins.get("omp_child_config"))
        want = meta["prompt_tokens"]
        if ctx.loaded_context is not None and ctx.loaded_context < want:
            reason = f"loaded context {ctx.loaded_context} < fixture prompt {want}"
            out.append(Result(f"replay.{name}", "replay", "perf", {n: void("lower", reason) for n in names}))
            continue
        rec = json.loads(path.read_text())
        body = {k: v for k, v in rec.items() if k not in ("model", "stream", "stream_options")}
        msgs = body.pop("messages")
        body["max_tokens"] = 64
        cold_msgs = _with_nonce(msgs)
        cold = ctx.chat(f"replay.{name}.cold", {"messages": cold_msgs, **body})
        warm = ctx.chat(f"replay.{name}.warm", {"messages": cold_msgs, **body})
        turn2 = cold_msgs + [{"role": "assistant", "content": cold.text or "OK"},
                             {"role": "user", "content": "Now reply with exactly: DONE"}]
        t2 = ctx.chat(f"replay.{name}.turn2", {"messages": turn2, **body})
        recorded_by = (meta.get("backend_pins") or {}).get("backend")
        out.append(Result(f"replay.{name}", "replay", "perf", {
            "cold_ttft_s": metric([cold.ttft_s], "lower"),
            "warm_ttft_s": metric([warm.ttft_s], "lower"),
            "turn2_ttft_s": metric([t2.ttft_s], "lower")},
            detail={"prompt_tokens": cold.prompt_tokens, "fixture_prompt_tokens": want,
                    "fixture_generation": fixture_gen, "fixture_fresh": fixture_gen == running_gen,
                    "tools": len(body.get("tools") or []), "cached_warm": warm.cached_tokens,
                    "cached_turn2": t2.cached_tokens}))
        out.append(prompt_token_verdict(name, cold.prompt_tokens, want, recorded_by, ctx.backend.name))
    if not fixtures:
        out.append(Result("replay", "replay", "SHOULD", verdict="FAIL",
                          detail={"reason": "no fixtures/omp/*.json; run `localbench record` first"}))
    return out


# ---------------------------------------------------------------- e2e

LEAN_FLAGS = ["--no-skills", "--no-rules", "--no-lsp", "--no-title",
              "--tools=read,bash,edit,write,grep,glob,todo"]


CHILD_CONFIG = FIXTURES / "omp" / "child-config.yml"


def child_flags(model: str, config: Path = CHILD_CONFIG, mode: str = "json", tools: bool = True) -> list[str]:
    """Flags every measured `omp -p` child gets besides the prompt.
    - `--config fixtures/omp/child-config.yml` turns memory off: with the default profile's mnemopi recall/retain,
      each child's prompt carried earlier benchmark turns (including wrong answers) and wrote its own turn back
      (2026-09-23, ledger row), so prompts drifted run to run and errors fed on themselves. The `mem` tier passes
      its own overlay (memory on) instead.
    - `--smol` points omp's remaining auxiliary calls at the model under test via the proxy: with the configured
      smol parked, omp -p fell back to `qwen3.8-uncensored:latest` and loaded it mid-run (2026-09-22, ledger row).
    - `tools=False` swaps the tool list for `--no-tools`, which removes omp's built-in tools only: with bash/grep a
      mem control turn searched the machine and read other agents' databases. MCP servers from the user's config
      still load and are enabled once connected (omp 18.2.11 session-tools.ts #applyMCPToolRefresh; no CLI flag
      turns them off), and the --no-tools controls then called the user's Agent Mail, 76 side-effecting calls in
      one A/B (2026-09-23). Neither setting isolates a child; see the ledger row on mem.no_leak.
    """
    lean = LEAN_FLAGS if tools else [f for f in LEAN_FLAGS if not f.startswith("--tools")] + ["--no-tools"]
    return ["--model", f"localbench/{model}", "--smol", f"localbench/{model}", "--config", str(config),
            "--mode", mode, "--no-session", *lean]


E2E_TASKS = [
    ("tool_read", "Read the file answer.txt in the current directory and reply with only the number it contains.",
     lambda text: "4817" in text),
    ("ok", "Reply with exactly: OK", lambda text: text.strip().rstrip(".") == "OK"),
]


def omp_env() -> dict:
    """Environment for omp calls that read the USER's setup (`omp config list`, `omp tiny-models list`): the default
    profile, with a caller's own profile variables (OMP_PROFILE / PI_PROFILE / PI_CODING_AGENT_DIR) stripped. Measured
    children use child_env() instead."""
    return {k: v for k, v in os.environ.items() if k not in ("OMP_PROFILE", "PI_PROFILE", "PI_CODING_AGENT_DIR")}


AGENT_CONFIG = FIXTURES / "omp" / "agent" / "config.yml"
AGENT_DIR = ROOT / "runs" / "omp-agent"
AGENT_BANKS = AGENT_DIR / "memories" / "mnemopi" / "banks"


def child_env() -> dict:
    """Environment for every measured omp child: PI_CODING_AGENT_DIR is localbench's own agent dir (AGENT_DIR, under
    the gitignored runs/), holding only fixtures/omp/agent/config.yml and the localbench provider
    (ensure_localbench_model). Nothing of the user's agent dir reaches a child: no MCP servers (~/.omp/agent/mcp.json),
    plugins, settings or memory banks (mnemopi keeps them under the agent dir). 2026-09-23, with the user's dir:
    tool-enabled mem controls read other agents' databases under ~/.claude, and --no-tools ones called the user's
    Agent Mail MCP server (76 side-effecting calls in one A/B). CLAUDE_CONFIG_DIR is dropped too: it force-enables
    omp's `claude` discovery provider (MCP servers from ~/.claude.json); cursor/codex user configs are opt-in
    (enabledProviders) and the isolated config opts into none."""
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = AGENT_DIR / "config.yml"
    if not cfg.is_file() or cfg.read_bytes() != AGENT_CONFIG.read_bytes():
        cfg.write_bytes(AGENT_CONFIG.read_bytes())
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_PROFILE", "PI_PROFILE", "PI_CODING_AGENT_DIR", "CLAUDE_CONFIG_DIR")}
    env["PI_CODING_AGENT_DIR"] = str(AGENT_DIR)
    return env


def omp_bin() -> str:
    """The omp every child run and every pin uses. Two omp 18.2.10 installs exist on this host with different
    binaries (~/.bun/bin/omp, which the user's panes run, and ~/.local/bin/omp); which one bare `omp` finds depends
    on the caller's PATH order. LOCALBENCH_OMP pins it explicitly; either way the resolved path and its sha are
    recorded in the run pins, and fixtures/goldens bound to another sha are GENERATION-MISMATCH."""
    path = os.environ.get("LOCALBENCH_OMP") or shutil.which("omp")
    if not path:
        raise FileNotFoundError("omp not found on PATH; set LOCALBENCH_OMP")
    return path


def ensure_localbench_model(model_id: str, context_window: int, extra: tuple[str, ...] = ()) -> None:
    """Write the children's models.yml (AGENT_DIR, see child_env): the localbench provider with static entries for the
    model under test plus any `extra` models a probe routes omp's side work to (same context window). Static, and
    without `apiKey:`: with `apiKey:` set this provider fails `omp -p --model localbench/<id>` resolution (ledger
    UNKNOWN row, 2026-09-22); the static entry also pins the context window omp budgets against. Until 2026-09-23 this
    rewrote a marked block in the user's ~/.omp/agent/models.yml; that block, if still there, is inert."""
    entries = "".join(
        f"      - id: {json.dumps(mid)}\n"
        f"        contextWindow: {int(context_window)}\n"
        "        maxTokens: 32768\n"
        "        reasoning: true\n"
        "        input: [text]\n"
        for mid in (model_id, *extra))
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    (AGENT_DIR / "models.yml").write_text(
        "providers:\n"
        "  localbench:\n"
        "    baseUrl: http://127.0.0.1:11299/v1\n"
        "    api: openai-completions\n"
        "    auth: none\n"
        "    compat:\n"
        "      supportsDeveloperRole: false\n"
        "      supportsReasoningEffort: true\n"
        "      maxTokensField: max_tokens\n"
        "      thinkingFormat: qwen\n"
        "    models:\n"
        f"{entries}"
    )


def _busy_s(rows: list[dict]) -> float | None:
    """Seconds at least one of `rows` was in flight (union of [t_start, t]); parallel calls are not double-counted."""
    if not rows:
        return None
    busy, cursor = 0.0, 0.0
    for r in sorted(rows, key=lambda r: r["t_start"]):
        lo, hi = max(r["t_start"], cursor), r["t"]
        if hi > lo:
            busy += hi - lo
            cursor = hi
    return round(busy, 3)


def _attempt_calls(calls_log: Path, started: float, ended: float) -> dict:
    """What the proxy saw during one `omp -p` attempt. Main-turn calls carry omp's tools; side calls (tools=0,
    routed here by --smol) are the same model doing omp's side work and are reported apart, per purpose (proxy
    markers: `auto-thinking` is the per-turn effort classifier, which omp runs BEFORE the main call with a 4 s cap;
    2026-09-23 receipt). Backends treat side calls differently (2026-09-22: 1 completion token on mlx-serve,
    64-73 on ollama). startup_s is the time omp spent before its first request of any kind."""
    rows = []
    if calls_log.exists():
        rows = [r for r in map(json.loads, calls_log.read_text().splitlines())
                if r.get("t_start", 0) >= started and r["t"] <= ended]
    rows.sort(key=lambda r: r["t_start"])
    main = [r for r in rows if r.get("tools")]
    aux = [r for r in rows if not r.get("tools")]
    side = {}
    for name in sorted({r.get("purpose", "aux") for r in aux}):
        sel = [r for r in aux if r.get("purpose", "aux") == name]
        side[name] = {"calls": len(sel), "busy_s": _busy_s(sel), "aborted": sum(1 for r in sel if r.get("aborted")),
                      "completion_tokens": sum(r.get("completion_tokens") or 0 for r in sel)}
    return {"calls": len(rows), "main_calls": len(main), "aux_calls": len(aux),
            "aborted": sum(1 for r in rows if r.get("aborted")),
            "first_call_cached_tokens": main[0].get("cached_tokens") if main else None,
            "startup_s": round(rows[0]["t_start"] - started, 3) if rows else None,
            "llm_s": _busy_s(main), "aux_s": _busy_s(aux), "side": side}


def e2e(ctx: Ctx) -> list[Result]:
    """Real `omp -p` turns routed omp -> localbench proxy -> backend, so every LLM call omp makes is timed.
    The backend is re-isolated before every task's `first` run, so `first` is cold KV on every backend; the
    first proxied call must report no cached tokens, otherwise the metric is VOID."""
    from .proxy import Proxy

    if not ctx.loaded_context:
        return [Result("e2e", "e2e", "MUST", verdict="FAIL",
                       detail={"reason": "backend did not report its loaded context; refusing to guess one for omp"})]
    out = []
    work = Path("/tmp/localbench-e2e")
    work.mkdir(exist_ok=True)
    (work / "answer.txt").write_text("4817\n")
    calls_log = ctx.run_dir / "omp_calls.jsonl"
    with Proxy(ctx.backend.base_url, calls_log, save_dir=ctx.run_dir / "bodies", label="e2e"):
        ensure_localbench_model(ctx.model, ctx.loaded_context)
        for task, prompt, check in E2E_TASKS:
            walls, llm, starts, ok_all, usage, first_split, answers = [], [], [], True, {}, {}, []
            for attempt in ("first", "repeat"):
                if attempt == "first":
                    ctx.backend.isolate(ctx.model)
                    ctx.emit({"event": "isolated", "reason": f"e2e.{task} first run must be cold"})
                t0 = time.perf_counter()
                started = time.time()
                proc = subprocess.run([omp_bin(), "-p", prompt, *child_flags(ctx.model)], cwd=work,
                                      capture_output=True, text=True, timeout=1800, env=child_env(),
                                      stdin=subprocess.DEVNULL, check=False)
                wall = time.perf_counter() - t0
                (ctx.run_dir / f"e2e.{task}.{attempt}.omp.jsonl").write_text(proc.stdout)
                split = _attempt_calls(calls_log, started, time.time())
                if attempt == "first":
                    first_split = split
                text, usage = _omp_final(proc.stdout)
                answers.append(text)
                ok = proc.returncode == 0 and check(text)
                ok_all &= ok
                walls.append(wall)
                llm.append(split["llm_s"])
                starts.append(split["startup_s"])
                ctx.emit({"event": "e2e", "task": task, "attempt": attempt, "wall_s": round(wall, 2),
                          "ok": ok, "rc": proc.returncode, "answer": text[:80], **usage, **split,
                          **({"stderr": proc.stderr[-300:]} if proc.returncode else {})})
            cached = first_split.get("first_call_cached_tokens")
            not_cold = f"first call reported {cached} cached tokens; not cold"
            out.append(Result(f"e2e.{task}", "e2e", "perf", {
                "first_wall_s": void("lower", not_cold) if cached else metric([walls[0]], "lower"),
                "first_llm_s": void("lower", not_cold) if cached else metric([llm[0]], "lower"),
                "repeat_wall_s": metric([walls[1]], "lower"),
                "repeat_llm_s": metric([llm[1]], "lower"),
                # omp launch to its first proxied request: omp's own cost, blind to the backend's KV state, so a cached
                # first call does not void it. 18.3.0 -> 18.3.1 moved it 0.59 -> 3.2-5.4 s with no row to catch it.
                "first_startup_s": metric([s for s in starts[:1] if s is not None], "lower"),
                "repeat_startup_s": metric([s for s in starts[1:] if s is not None], "lower")},
                detail={"answer_ok": ok_all, "answers": recorded_answers(answers), "first": first_split,
                        **usage}))
            out.append(Result(f"e2e.{task}.correct", "e2e", "MUST", verdict="PASS" if ok_all else "FAIL",
                              detail={"answers": recorded_answers(answers)}))
    return out


def _omp_final(stdout: str) -> tuple[str, dict]:
    """Last assistant text and summed usage from omp --mode json event lines."""
    text, usage = "", {"input": 0, "output": 0, "llm_calls": 0}
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") if isinstance(ev, dict) else None
        if ev.get("type") == "message_end" and isinstance(msg, dict) and msg.get("role") == "assistant":
            parts = [c.get("text", "") for c in msg.get("content") or [] if c.get("type") == "text"]
            if parts:
                text = "".join(parts)
            u = msg.get("usage") or {}
            usage["input"] += u.get("input") or 0
            usage["output"] += u.get("output") or 0
            usage["llm_calls"] += 1
    return text, usage


REL_ATTEMPTS = 20


def _wilson(passed: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a pass rate (stays inside [0, 1] at 0/n and n/n)."""
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _rel(ctx: Ctx, *, cold: bool | str) -> list[Result]:
    """cold=False: warm attempts; True: isolate + one-token warm-up before each; "fresh": isolate, no warm-up."""
    from .proxy import Proxy

    tier = "relfresh" if cold == "fresh" else "relcold" if cold else "rel"
    if not ctx.loaded_context:
        return [Result(tier, tier, "MUST", verdict="FAIL",
                       detail={"reason": "backend did not report its loaded context; refusing to guess one for omp"})]
    out = []
    work = Path("/tmp/localbench-e2e")
    work.mkdir(exist_ok=True)
    (work / "answer.txt").write_text("4817\n")
    tasks = [t for t in E2E_TASKS if t[0] == "ok"] if cold else E2E_TASKS
    calls_log = ctx.run_dir / f"{tier}_calls.jsonl"
    with Proxy(ctx.backend.base_url, calls_log, save_dir=ctx.run_dir / "bodies", label=tier):
        ensure_localbench_model(ctx.model, ctx.loaded_context)
        for task, prompt, check in tasks:
            passed, wrong, walls, answers = 0, [], [], []
            for i in range(REL_ATTEMPTS):
                if cold:
                    ctx.backend.isolate(ctx.model, warm=cold != "fresh")
                t0, started = time.perf_counter(), time.time()
                proc = subprocess.run([omp_bin(), "-p", prompt, *child_flags(ctx.model)], cwd=work,
                                      capture_output=True, text=True, timeout=1800, env=child_env(),
                                      stdin=subprocess.DEVNULL, check=False)
                walls.append(time.perf_counter() - t0)
                text, _ = _omp_final(proc.stdout)
                answers.append(text)
                if proc.returncode == 0 and check(text):
                    passed += 1
                else:
                    bodies = [r.get("body") for r in map(json.loads, calls_log.read_text().splitlines())
                              if r.get("t_start", 0) >= started] if calls_log.exists() else []
                    wrong.append({"attempt": i, "rc": proc.returncode, "answer": recorded_answers([text])[0],
                                  "bodies": bodies})
                    (ctx.run_dir / f"{tier}.{task}.{i}.omp.jsonl").write_text(proc.stdout)
            rate = round(passed / REL_ATTEMPTS, 4)
            ctx.emit({"event": tier, "task": task, "passed": passed, "attempts": REL_ATTEMPTS, "wrong": wrong})
            out.append(Result(f"{tier}.{task}", tier, "perf", {
                "pass_rate": {"value": rate, "better": "higher", "spread": [rate, rate], "n": REL_ATTEMPTS},
                "wall_s": metric(walls, "lower")},
                detail={"passed": passed, "attempts": REL_ATTEMPTS, "ci95": _wilson(passed, REL_ATTEMPTS),
                        "answers": recorded_answers(answers), "wrong": wrong}))
    return out


def rel(ctx: Ctx) -> list[Result]:
    """Answer reliability: every e2e task REL_ATTEMPTS times through the same omp path and flags as e2e, warm.
    The e2e MUST is a two-attempt spot check; this tier estimates the wrong-answer rate a user would see. Every
    attempt's answer is kept, clipped, on pass and fail. Opt-in (`--tiers rel`): not part of the default tiers or of any golden."""
    return _rel(ctx, cold=False)


def relcold(ctx: Ctx) -> list[Result]:
    """Like `rel` for the "ok" task, but the backend is re-isolated (cold KV / prefix cache) before every attempt.
    Motivation: with memory off, warm attempts were 40/40 correct per config while cold first attempts in the e2e
    tier answered "Hi, how can I help you today?" and "OK\\nخوب" (2026-09-23 A/B). Opt-in (`--tiers relcold`)."""
    return _rel(ctx, cold=True)


def relfresh(ctx: Ctx) -> list[Result]:
    """relcold without the one-token "hi" warm-up after each isolate: the omp request meets an empty prefix cache
    (ollama loads the model on that request). Splits "first request after a load" from "partial prefix reuse
    against the warm-up's cache entry" as the cause of cold-start wrong answers. Opt-in (`--tiers relfresh`)."""
    return _rel(ctx, cold="fresh")


# ---------------------------------------------------------------- mem

def _mem_facts(rng: random.Random) -> list[tuple[str, str, str, str]]:
    """(name, plant prompt, question, expected answer): made-up project facts drawn fresh per round, so no model can
    know them and no earlier run's bank can supply them."""
    code = f"ZEBRA-{rng.randrange(1000, 10000)}"
    port = str(rng.randrange(20000, 60000))
    owner = rng.choice(["Okafor", "Lindqvist", "Takahashi", "Moreau", "Castellanos", "Adeyemi"])
    note = "Note for later in this project: {}. Reply with exactly: NOTED"
    return [
        ("code", note.format(f"the deploy code for project Falcon is {code}"),
         "What is the deploy code for project Falcon? Reply with only the code.", code),
        ("port", note.format(f"the staging database for project Falcon listens on port {port}"),
         "Which port does the staging database for project Falcon listen on? Reply with only the number.", port),
        ("owner", note.format(f"the on-call owner of project Falcon this week is {owner}"),
         "Who is the on-call owner of project Falcon this week? Reply with only the surname.", owner),
    ]


def _leaked(answer: str, planted: list[str]) -> list[str]:
    """The values planted so far in this run that a control answer (a fresh project, nothing planted) contains. Any of
    them is a leak, not just this round's: a control that answered the previous round's code (2026-09-23, ZEBRA-5567)
    passed the old same-round check."""
    low = answer.lower()
    return [v for v in dict.fromkeys(planted) if v.lower() in low]



_SIDE_PURPOSES = frozenset({"auto-thinking", "judge", "memory-extract", "memory-consolidate"})


def answer_call(rows: list[dict]) -> dict | None:
    """The turn's answer call, not the classifier or a memory side call.

    With tools, that call's purpose is `main`. The mem tier runs `--no-tools`, so every call has tools=0 and the
    answer is `aux`. Selecting on tools voids `pre_main_s` (ab-mem-fts-rounds12-20260924: 36 recall turns,
    no samples)."""
    for row in sorted(rows, key=lambda r: r.get("t_start") or 0):
        if row.get("purpose") not in _SIDE_PURPOSES:
            return row
    return None



def mem(ctx: Ctx) -> list[Result]:
    """Does omp's memory work across sessions, and what does it cost? Per fact and round, four `omp -p` processes
    with memory on (ctx.mem_config): PLANT the fact in a fresh project dir (retained when the process exits); RECALL it
    from a new process in the same dir; CONTROL — the same question from a fresh dir with no plant, whose answer must
    contain no value planted in this run (a hit is leakage between projects, or a guess); DERAIL — "Reply with
    exactly: OK" in the
    planted dir, which recalled memory must not break (earlier memory-on runs answered it with recalled text, ledger
    2026-09-23). Measured: recall hit rate (Wilson 95%), time to the main call and wall per recall turn, prompt tokens
    recall injects (recall minus control main prompt), plant wall (includes retention at exit), derail rate. The
    tier's own banks and dirs are removed afterwards (memory.remove_banks). Turns run with `--no-tools` (no built-in
    tools); MCP tools from the user's config still load, so a turn can still reach outside recall through them. Opt-in (`--tiers mem`)."""
    from . import memory
    from .proxy import Proxy

    if not ctx.loaded_context:
        return [Result("mem", "mem", "MUST", verdict="FAIL",
                       detail={"reason": "backend did not report its loaded context; refusing to guess one for omp"})]
    run_id = uuid.uuid4().hex[:8]
    prefix = f"localbench-mem-{run_id}-"
    calls_log = ctx.run_dir / "mem_calls.jsonl"
    rng = random.Random()
    flags = child_flags(ctx.model, ctx.mem_config, tools=False)

    def turn(prompt: str, cwd: Path) -> dict:
        t0, started = time.perf_counter(), time.time()
        proc = subprocess.run([omp_bin(), "-p", prompt, *flags], cwd=cwd, capture_output=True, text=True,
                              timeout=1800, env=child_env(), stdin=subprocess.DEVNULL, check=False)
        wall, ended = time.perf_counter() - t0, time.time()
        rows = [r for r in map(json.loads, calls_log.read_text().splitlines())
                if r.get("t_start", 0) >= started and r["t"] <= ended] if calls_log.exists() else []
        ans = answer_call(rows)
        text, _ = _omp_final(proc.stdout)
        return {"rc": proc.returncode, "text": text, "wall": wall,
                "pre_main": ans["t_start"] - started if ans else None,
                "prompt_tokens": ans.get("prompt_tokens") if ans else None}

    hits, attempts, leaks, derails, planted = 0, [], [], [], []
    plant_walls, recall_walls, pre_main, injected = [], [], [], []
    dirs = []
    try:
        with Proxy(ctx.backend.base_url, calls_log, save_dir=ctx.run_dir / "bodies", label="mem"):
            ensure_localbench_model(ctx.model, ctx.loaded_context)
            for rnd in range(ctx.mem_rounds):
                for name, plant, question, expected in _mem_facts(rng):
                    home, control = Path(f"/tmp/{prefix}{name}-{rnd}"), Path(f"/tmp/{prefix}control-{name}-{rnd}")
                    for d in (home, control):
                        d.mkdir()
                        dirs.append(d)
                    p = turn(plant, home)
                    r = turn(question, home)
                    c = turn(question, control)
                    o = turn("Reply with exactly: OK", home)
                    hit = r["rc"] == 0 and expected.lower() in r["text"].lower()
                    hits += hit
                    plant_walls.append(p["wall"])
                    recall_walls.append(r["wall"])
                    if r["pre_main"] is not None:
                        pre_main.append(r["pre_main"])
                    if r["prompt_tokens"] and c["prompt_tokens"]:
                        injected.append(r["prompt_tokens"] - c["prompt_tokens"])
                    planted.append(expected)
                    found = _leaked(c["text"], planted)
                    if found:
                        leaks.append({"fact": name, "round": rnd, "values": found, "answer": c["text"][:160]})
                    if not (o["rc"] == 0 and o["text"].strip().rstrip(".") == "OK"):
                        derails.append({"fact": name, "round": rnd, "answer": o["text"][:160]})
                    attempt = {"fact": name, "round": rnd, "expected": expected, "hit": hit,
                               "recall_answer": r["text"][:160], "plant_answer": p["text"][:80],
                               "control_answer": c["text"][:80], "rcs": [p["rc"], r["rc"], c["rc"], o["rc"]]}
                    attempts.append(attempt)
                    ctx.emit({"event": "mem", **attempt, "recall_pre_main_s": r["pre_main"]})
    finally:
        memory.remove_banks([b["bank"] for b in memory.banks(AGENT_BANKS) if b["bank"].startswith(prefix)], AGENT_BANKS)
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
    n = len(attempts)
    rate = round(hits / n, 4) if n else 0.0
    derail_ok = round((n - len(derails)) / n, 4) if n else 0.0
    return [
        Result("mem.recall", "mem", "perf", {
            "hit_rate": {"value": rate, "better": "higher", "spread": [rate, rate], "n": n},
            "pre_main_s": metric(pre_main, "lower"),
            "wall_s": metric(recall_walls, "lower")},
            detail={"hits": hits, "attempts": n, "ci95": _wilson(hits, n) if n else None, "rounds": attempts,
                    "config": str(ctx.mem_config),
                    # Recall-turn main prompt minus the control turn's: what recall added. 0 is a real answer, so it
                    # is not a golden metric (a relative band on 0 is meaningless).
                    "injected_tokens": {"median": statistics.median(injected), "min": min(injected),
                                        "max": max(injected)} if injected else None}),
        Result("mem.plant", "mem", "perf", {"wall_s": metric(plant_walls, "lower")}),
        Result("mem.derail", "mem", "perf",
               {"ok_rate": {"value": derail_ok, "better": "higher", "spread": [derail_ok, derail_ok], "n": n}},
               detail={"derails": derails}),
        Result("mem.no_leak", "mem", "MUST", verdict="PASS" if not leaks else "FAIL", detail={"leaks": leaks}),
    ]


# ---------------------------------------------------------------- sess

SESS_TURNS = 12
# omp 18.2.11's default mnemopi.retainEveryNTurns; the mem overlay does not set it. Retention (memory-LLM extraction,
# fire-and-forget on agent_end) therefore starts after turns 4, 8 and 12, and turns 5 and 9 are sent while it runs.
SESS_RETAIN_EVERY = 4
SESS_TURN_TIMEOUT_S = 600
SESS_SUBJECTS = ("deploy code", "staging port", "on-call owner", "release branch", "feature flag", "cache TTL",
                 "primary region", "queue name", "build number", "schema version", "alert channel", "canary percent")


def _merged(rows: list[dict]) -> list[list[float]]:
    out: list[list[float]] = []
    for r in sorted(rows, key=lambda r: r["t_start"]):
        if out and r["t_start"] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], r["t"])
        else:
            out.append([r["t_start"], r["t"]])
    return out


def _overlap_s(a: list[dict], b: list[dict]) -> float:
    """Seconds some call in `a` and some call in `b` were in flight at once."""
    return round(sum(max(0.0, min(h1, h2) - max(l1, l2)) for l1, h1 in _merged(a) for l2, h2 in _merged(b)), 3)


class _Rpc:
    """One `omp --mode rpc` child: JSON-line commands on stdin, events on stdout (protocol v1: the `ready` frame, then
    `{"id", "type": "prompt", "message"}` per turn; events stream until `agent_end`)."""

    def __init__(self, argv: list[str], cwd: Path, stderr_path: Path):
        self._stderr = stderr_path.open("w")
        self.proc = subprocess.Popen(argv, cwd=cwd, env=child_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._stderr, text=True, bufsize=1)
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def send(self, command: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(command) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise EOFError(f"omp closed its stdin (rc={self.proc.poll()})") from None

    def until(self, accept: Callable[[dict], bool], timeout: float) -> list[str]:
        """Output lines up to and including the first event `accept` takes. TimeoutError when none arrives in time,
        EOFError when omp exits first."""
        deadline, lines = time.monotonic() + timeout, []
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"no matching event within {timeout:.0f} s") from None
            if line is None:
                raise EOFError(f"omp exited rc={self.proc.wait()}")
            lines.append(line)
            with contextlib.suppress(json.JSONDecodeError):
                ev = json.loads(line)
                if isinstance(ev, dict) and accept(ev):
                    return lines

    def close(self, timeout: float) -> tuple[float, int]:
        """Close stdin (omp disposes the session, retention included) and wait. Returns (seconds to exit, rc)."""
        t0 = time.perf_counter()
        with contextlib.suppress(BrokenPipeError):
            self.proc.stdin.close()
        try:
            rc = self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            rc = self.proc.wait()
        self._stderr.close()
        return round(time.perf_counter() - t0, 3), rc


def _turn_end(ev: dict) -> bool:
    return (ev.get("type") == "agent_end" and ev.get("willContinue") is not True) or (
        ev.get("type") == "response" and ev.get("success") is False)


def sess(ctx: Ctx) -> list[Result]:
    """Can omp's memory LLM and the main model share the machine inside one interactive session? One-shot `omp -p`
    never runs the memory LLM on its path (ledger 2026-09-23); an interactive session does: mnemopi retains on
    agent_end every SESS_RETAIN_EVERY user turns, fire-and-forget, so the extraction call runs while the next turn is
    sent. Per session (ctx.repeats sessions), one `omp --mode rpc` child with memory on (ctx.mem_config) and smol on
    the model under test (one model on the machine) takes SESS_TURNS turns of one shape — state a made-up project fact,
    reply NOTED — sent back to back. Post-retain turns (5, 9) are compared with regular turns (turn 1 is cold and left
    out) and with their own neighbours, which cancels the growing-context trend. Memory-LLM calls are the proxy's
    `memory-*` purposes; overlap is the seconds they were in flight together with a main call. Opt-in (`--tiers sess`)."""
    from . import memory
    from .proxy import Proxy

    if not ctx.loaded_context:
        return [Result("sess", "sess", "MUST", verdict="FAIL",
                       detail={"reason": "backend did not report its loaded context; refusing to guess one for omp"})]
    prefix = f"localbench-sess-{uuid.uuid4().hex[:8]}-"
    calls_log = ctx.run_dir / "sess_calls.jsonl"
    argv = [omp_bin(), *child_flags(ctx.model, ctx.mem_config, mode="rpc"), "--max-time", "30m"]
    rng = random.Random()
    turns, sessions, dirs = [], [], []
    try:
        with Proxy(ctx.backend.base_url, calls_log, label="sess"):
            ensure_localbench_model(ctx.model, ctx.loaded_context)
            for s in range(ctx.repeats):
                cwd = Path(f"/tmp/{prefix}{s}")
                cwd.mkdir()
                dirs.append(cwd)
                started, failure, n_done = time.time(), None, 0
                rpc = _Rpc(argv, cwd, ctx.run_dir / f"sess.{s}.stderr.log")
                try:
                    rpc.until(lambda ev: ev.get("type") == "ready", 120)
                    for i, subject in enumerate(SESS_SUBJECTS[:SESS_TURNS], 1):
                        value = f"{rng.choice('ABCDEFGH')}{rng.randrange(100, 1000)}"
                        prompt = f"Note for this project: the {subject} of project Falcon is {value}. Reply with exactly: NOTED"
                        t_send, t0 = time.time(), time.perf_counter()
                        rpc.send({"id": f"s{s}t{i}", "type": "prompt", "message": prompt})
                        lines = rpc.until(_turn_end, SESS_TURN_TIMEOUT_S)
                        wall, t_end = time.perf_counter() - t0, time.time()
                        last = json.loads(lines[-1])
                        if last.get("type") == "response":
                            raise EOFError(f"prompt rejected: {str(last.get('error'))[:200]}")
                        text, _ = _omp_final("".join(lines))
                        ok = text.strip().rstrip(".") == "NOTED"
                        turns.append({"session": s, "turn": i, "wall": wall, "t_send": t_send, "t_end": t_end, "ok": ok,
                                      "post_retain": i > 1 and (i - 1) % SESS_RETAIN_EVERY == 0,
                                      **({} if ok else {"reply": text[:160]})})
                        n_done = i
                except (TimeoutError, EOFError) as exc:
                    failure = f"session {s} turn {n_done + 1}: {type(exc).__name__}: {exc}"
                exit_s, rc = rpc.close(120)
                sessions.append({"session": s, "started": started, "ended": time.time(), "turns": n_done,
                                 "failure": failure, "exit_s": exit_s, "rc": rc})
                ctx.emit({"event": "sess", **{k: v for k, v in sessions[-1].items() if k not in ("started", "ended")}})
    finally:
        memory.remove_banks([b["bank"] for b in memory.banks(AGENT_BANKS) if b["bank"].startswith(prefix)], AGENT_BANKS)
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)

    rows = [json.loads(line) for line in calls_log.read_text().splitlines()] if calls_log.exists() else []
    main = [r for r in rows if r.get("purpose") == "main"]
    mem_rows = [r for r in rows if str(r.get("purpose", "")).startswith("memory-")]
    for t in turns:
        mine = [r for r in main if t["t_send"] <= r["t_start"] <= t["t_end"]]
        t["pre_main"] = min(r["t_start"] for r in mine) - t["t_send"] if mine else None
        t["overlap_s"] = _overlap_s(mine, mem_rows)
    by_turn = {(t["session"], t["turn"]): t for t in turns}
    regular = [t for t in turns if t["turn"] > 1 and not t["post_retain"]]
    post = [t for t in turns if t["post_retain"]]
    deltas = [t["wall"] - (by_turn[(t["session"], t["turn"] - 1)]["wall"] + nxt["wall"]) / 2
              for t in post if (nxt := by_turn.get((t["session"], t["turn"] + 1)))]
    by_purpose = {p: {"calls": len(sel), "busy_s": _busy_s(sel),
                      "not_ok": sum(1 for r in sel if r.get("status") != 200 or r.get("aborted")),
                      "completion_tokens": sum(r.get("completion_tokens") or 0 for r in sel)}
                  for p in sorted({r["purpose"] for r in mem_rows})
                  for sel in [[r for r in mem_rows if r["purpose"] == p]]}
    acks = sum(t["ok"] for t in turns)
    ack_rate = round(acks / len(turns), 4) if turns else 0.0
    failures = [x["failure"] for x in sessions if x["failure"]]
    bad_mem = [{k: r.get(k) for k in ("purpose", "status", "aborted", "total_s")} for r in mem_rows
               if r.get("status") != 200 or r.get("aborted")]
    return [
        Result("sess.turn", "sess", "perf", {
            "wall_s": metric([t["wall"] for t in regular], "lower"),
            "post_retain_wall_s": metric([t["wall"] for t in post], "lower"),
            "pre_main_s": metric([t["pre_main"] for t in regular], "lower"),
            "post_retain_pre_main_s": metric([t["pre_main"] for t in post], "lower"),
            "ack_rate": {"value": ack_rate, "better": "higher", "spread": [ack_rate, ack_rate], "n": len(turns)}},
            # Signed and near 0 when retention costs nothing, so not a golden metric (a relative band on ~0 is
            # meaningless): post-retain turn wall minus the mean of its two neighbours.
            detail={"neighbour_delta_s": {"median": round(statistics.median(deltas), 3), "min": round(min(deltas), 3),
                                          "max": round(max(deltas), 3), "n": len(deltas)} if deltas else None,
                    "config": str(ctx.mem_config),
                    "turns": [{k: (round(v, 3) if isinstance(v, float) and k not in ("t_send", "t_end") else v)
                               for k, v in t.items()} for t in turns]}),
        Result("sess.memory", "sess", "perf", {
            "extract_s": metric([r["total_s"] for r in mem_rows if r["purpose"] == "memory-extract"], "lower")},
            detail={"by_purpose": by_purpose, "overlap_with_main_s": _overlap_s(main, mem_rows),
                    "exit_s": [x["exit_s"] for x in sessions], "sessions": sessions}),
        Result("sess.turns_complete", "sess", "MUST", verdict="PASS" if not failures and turns else "FAIL",
               detail={"failures": failures, "turns": len(turns), "expected": ctx.repeats * SESS_TURNS}),
        Result("sess.memory_calls_ok", "sess", "SHOULD", verdict="PASS" if not bad_mem else "FAIL",
               detail={"memory_calls": len(mem_rows), "not_ok": bad_mem}),
    ]


# ---------------------------------------------------------------- think

# Checkable questions that make a reasoning model think, answerable in well under the budget (qwen3.6 MoE spent all of a
# 4096-token budget on "divisible by 3 or 5 but not both" and never answered, 2026-09-24, so that one is not here).
THINK_TASKS = (
    ("minutes", "A train leaves at 14:35 and arrives at 17:12 the same day. How long is the trip, in minutes?", "157"),
    ("arith", "Compute 37 * 43 - 19.", "1572"),
    ("pens", "Pens are sold in packs of 3 for $2. How many dollars do 27 pens cost?", "18"),
    ("order", ("Alice is older than Bob. Carol is younger than Bob. Dave is older than Alice. Who is the second "
               "youngest? Answer with the name."), "bob"),
    ("code", "What does this Python program print?\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x) + sum(y))", "14"),
    ("digits", "What is the sum of the decimal digits of 2**15?", "26"),
)
THINK_SUFFIX = "\n\nEnd your reply with a final line of the form ANSWER: <answer>."
THINK_MAX_TOKENS = 8192
_ANSWER = re.compile(r"ANSWER\s*:\s*\**\s*\$?([A-Za-z0-9.\-]+)", re.IGNORECASE)


def final_answer(text: str) -> str | None:
    """The value on the reply's last `ANSWER:` line, lowercased, trailing period dropped; None if there is none (a
    reply cut off at the token budget usually has none)."""
    found = _ANSWER.findall(text or "")
    return found[-1].lower().rstrip(".") if found else None


def think(ctx: Ctx) -> list[Result]:
    """How much a model thinks, how long that takes, and whether it is right, on THINK_TASKS with thinking on
    (ollama's /v1 ignores enable_thinking; the model's own default thinks). One round asks every task once; per round:
    reasoning characters streamed (same-tokenizer models compare like for like), completion tokens, wall, and
    correct answers. A reply cut off at THINK_MAX_TOKENS counts as wrong. A model that streamed no reasoning at all
    voids the reasoning metric: fewer thinking tokens from not thinking is not the claim. Opt-in (`--tiers think`)."""
    rounds, chars, tokens, walls, hits = [], [], [], [], []
    for rnd in range(ctx.repeats):
        per = []
        for name, question, expected in THINK_TASKS:
            s = ctx.chat(f"think.{name}", {"messages": [{"role": "user", "content": question + THINK_SUFFIX}],
                                           "max_tokens": THINK_MAX_TOKENS})
            got = final_answer(s.text)
            # The budget is the claim's boundary: an answer the model reached only by running out of tokens is wrong.
            hit = got == expected and s.finish_reason != "length"
            per.append({"task": name, "round": rnd, "expected": expected, "answer": got, "hit": hit,
                        "reasoning_chars": len(s.reasoning), "completion_tokens": s.completion_tokens,
                        "wall_s": round(s.total_s, 2), "finish": s.finish_reason, "error": s.error})
        rounds += per
        chars.append(sum(r["reasoning_chars"] for r in per))
        tokens.append(sum(r["completion_tokens"] for r in per))
        walls.append(sum(r["wall_s"] for r in per))
        hits.append(sum(r["hit"] for r in per))
    n = len(THINK_TASKS) * len(hits)
    acc = round(sum(hits) / n, 4) if n else 0.0
    return [Result("think", "think", "perf", {
        "reasoning_chars": metric(chars, "lower") if any(chars) else void("lower", "no reasoning streamed"),
        "completion_tokens": metric(tokens, "lower"),
        "wall_s": metric(walls, "lower"),
        "accuracy": {"value": acc, "better": "higher", "spread": [min(hits) / len(THINK_TASKS),
                                                                 max(hits) / len(THINK_TASKS)] if hits else [0, 0],
                     "n": n}},
        detail={"hits": sum(hits), "attempts": n, "ci95": _wilson(sum(hits), n) if n else None,
                "cut_off": sum(1 for r in rounds if r["finish"] == "length"), "rounds": rounds})]


TIERS = {"conf": conformance, "micro": micro, "replay": replay, "e2e": e2e, "rel": rel, "relcold": relcold,
         "relfresh": relfresh, "mem": mem, "sess": sess, "think": think}
