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
import functools
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
# mnemopi.llmMode when an overlay does not set it: omp's own default (pi-coding-agent src/config/settings.ts,
# cfgMnemopiLlmMode), which the children inherit because fixtures/omp/agent/config.yml does not set it either.
MEM_LLM_MODE_DEFAULT = "smol"
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
    tok_identity: str | None = None
    mem_config: Path = MEM_CONFIG
    mem_rounds: int = MEM_ROUNDS
    # The memory (smol-role) model of the mem and sess tiers' omp children, on the same backend as `model`. None: the
    # model under test does both (one artifact). The run pins record it as smol_model/smol_digest (smol_pins).
    smol_model: str | None = None
    e2e_case: str | None = None
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

    # Prefix cache: identical 8k prompt twice; warm TTFT is a cache probe, not a measured chat turn.
    a_body = [{"role": "user", "content": _nonce() + corpus(8000) + "\n\nName one function above."}]
    cold = ctx.chat("micro.cache_cold_8k", {"messages": a_body, "max_tokens": 16})
    warm = ctx.chat("micro.cache_warm_8k", {"messages": a_body, "max_tokens": 16})
    out.append(Result("micro.cache_hit_8k", "micro", "perf", {
        "warm_ttft_s": metric([warm.ttft_s], "lower"),
        "speedup_x": metric([cold.ttft_s / warm.ttft_s if warm.ttft_s else None], "higher")},
        detail={"cold_ttft_s": cold.ttft_s, "cached_tokens": warm.cached_tokens}))

    # Eviction probe: A (cached above), then a different 8k prompt B, then A again.
    # A-again TTFT suggests reuse or a miss; timing alone cannot establish cache topology.
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
                         recorded_by: str | None, run_backend: str, *,
                         recorded_tok: str | None = None,
                         run_tok: str | None = None) -> Result:
    """Compare counts using tokenizer identity when available; absent pins retain backend-only legacy behavior."""
    detail = {"replayed": replayed, "recorded": recorded, "recorded_by": recorded_by, "run_backend": run_backend,
              "recorded_tokenizer_identity": recorded_tok, "run_tokenizer_identity": run_tok}
    if recorded_by and recorded_by != run_backend:
        detail["reason"] = f"recorded by {recorded_by}; this run is {run_backend}"
        return Result(f"replay.{name}.prompt_tokens", "replay", "SHOULD", verdict="VOID", detail=detail)
    if recorded_tok and run_tok and recorded_tok != run_tok:
        detail["reason"] = "different tokenizer identity"
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
    fixture's prompt is VOID. A different backend remains VOID. When both sidecar and backend report a tokenizer
    identity, a mismatch is VOID; when either pin is absent, the legacy backend-only comparison remains."""
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
        backend_pins = meta.get("backend_pins") or {}
        recorded_by = backend_pins.get("backend")
        recorded_tok = backend_pins.get("tokenizer_identity")
        out.append(Result(f"replay.{name}", "replay", "perf", {
            "cold_ttft_s": metric([cold.ttft_s], "lower"),
            "warm_ttft_s": metric([warm.ttft_s], "lower"),
            "turn2_ttft_s": metric([t2.ttft_s], "lower")},
            detail={"prompt_tokens": cold.prompt_tokens, "fixture_prompt_tokens": want,
                    "fixture_generation": fixture_gen, "fixture_fresh": fixture_gen == running_gen,
                    "tools": len(body.get("tools") or []), "cached_warm": warm.cached_tokens,
                    "cached_turn2": t2.cached_tokens}))
        out.append(prompt_token_verdict(name, cold.prompt_tokens, want, recorded_by, ctx.backend.name,
                                        recorded_tok=recorded_tok,
                                        run_tok=ctx.tok_identity))
    if not fixtures:
        out.append(Result("replay", "replay", "SHOULD", verdict="FAIL",
                          detail={"reason": "no fixtures/omp/*.json; run `localbench record` first"}))
    return out


# ---------------------------------------------------------------- e2e

LEAN_FLAGS = ["--no-skills", "--no-rules", "--no-lsp", "--no-title",
              "--tools=read,bash,edit,write,grep,glob,todo"]
# The mem tier's whole tool list: omp's mnemopi memory tools (omp 18.3.1 memory-backend/tool-names.ts) minus `learn`,
# which exists only with autolearn on and also writes managed skills to disk. Pinned per run (sorted) as omp_mem_tools.
MEM_TOOLS = ("memory_edit", "recall", "reflect", "retain")


CHILD_CONFIG = FIXTURES / "omp" / "child-config.yml"


def child_flags(model: str, config: Path = CHILD_CONFIG, mode: str = "json", tools: str = "lean",
                smol: str | None = None) -> list[str]:
    """Flags every measured `omp -p` child gets besides the prompt.
    - `--config fixtures/omp/child-config.yml` turns memory off: with the default profile's mnemopi recall/retain,
      each child's prompt carried earlier benchmark turns (including wrong answers) and wrote its own turn back
      (2026-09-23, ledger row), so prompts drifted run to run and errors fed on themselves. The `mem` tier passes
      its own overlay (memory on) instead.
    - `--smol` points omp's remaining auxiliary calls at the model under test via the proxy: with the configured
      smol parked, omp -p fell back to `qwen3.8-uncensored:latest` and loaded it mid-run (2026-09-22, ledger row).
      `smol` names a separate memory model instead (Ctx.smol_model; omp's memory role resolves through smol), served
      through the same proxy and backend.
    - `tools="memory"` (the mem tier) swaps the lean tool list for `--tools=<MEM_TOOLS>`: the memory tools a session
      has, and nothing that reaches the disk, a shell, the network or another process (with bash/grep a mem control
      turn searched the machine and read other agents' databases, 2026-09-23). Named in --tools they stay plain
      top-level tools: omp 18.3.1 never mounts an explicitly requested tool as an xd:// device (sdk.ts), and with
      neither read nor write granted there is no xd:// transport. The tier used to run `--no-tools`, a condition no
      session has: the model still called the recall/retain it read about in omp's prompt (and agent/proc/ssh/mcp);
      ollama turned those into calls omp answered "Tool X not found", mlx-serve 26.9.5 returned them as the answer,
      and a 2-bit model looped until the 1800 s timeout (ledger, 2026-09-27 correction). MCP servers never reach a
      child: child_env() gives it an agent dir with none.
    """
    if tools == "lean":
        lean = LEAN_FLAGS
    elif tools == "memory":
        lean = [f for f in LEAN_FLAGS if not f.startswith("--tools")] + ["--tools=" + ",".join(sorted(MEM_TOOLS))]
    else:
        raise ValueError(f"unknown child tool set {tools!r}: lean or memory")
    return ["--model", f"localbench/{model}", "--smol", f"localbench/{smol or model}", "--config", str(config),
            "--mode", mode, "--no-session", *lean]


def _tool_read_ok(text: str) -> bool:
    return text.strip() == "4817"


def _reply_ok(text: str) -> bool:
    return text.strip().rstrip(".") == "OK"


E2E_TASKS = [
    ("tool_read", "Read the file answer.txt in the current directory and reply with only the number it contains.",
     _tool_read_ok),
    ("ok", "Reply with exactly: OK", _reply_ok),
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


def smol_pins(backend, smol_model: str | None) -> dict:
    """Run pins for a declared memory model (Ctx.smol_model): its name and digest on the run's backend. {} without
    one, so a one-artifact run's pins are unchanged. Only Ollama serves a second model from the same base URL by the
    request's `model` field; a one-model server (mlx-serve, mlxfast) answers whatever model a request names with the
    model it loaded, so smol calls would silently measure the model under test: refused, not recorded."""
    if smol_model is None:
        return {}
    if getattr(backend, "name", None) != "ollama":
        raise ValueError(f"a separate smol model needs the ollama backend (one server, both models); "
                         f"{getattr(backend, 'name', backend)!r} would serve {smol_model!r} with its own model")
    return {"smol_model": smol_model, "smol_digest": backend.pins(smol_model)["model_digest"]}


def _smol_extra(ctx: Ctx) -> tuple[str, ...]:
    """models.yml entries besides the model under test: the declared smol model, so `--smol localbench/<id>`
    resolves."""
    return (ctx.smol_model,) if ctx.smol_model and ctx.smol_model != ctx.model else ()


def _smol_refusal(ctx: Ctx, tier: str) -> Result | None:
    """A declared smol model the run did not pin (absent on the backend, or a caller that skipped smol_pins) would
    leave the memory model's identity out of the receipt: the tier refuses to run rather than measure it unlabelled."""
    if ctx.smol_model and not ctx.pins.get("smol_digest"):
        return Result(tier, tier, "MUST", verdict="FAIL", detail={
            "reason": f"declared smol model {ctx.smol_model!r} has no smol_digest in the run pins "
                      "(not on this backend, or the run did not pin it)"})
    return None


MEM_LLM_MODES = ("none", "smol", "remote")
# What mnemopi does under llmMode none (fixtures/omp/child-config-mem-nollm.yml; omp 18.4.6, read from the installed
# sources). That overlay's bytes are pinned (omp_mem_config = sha16 of the file) by the 2026-09-23 receipts, so this is
# documented here rather than in its header.
# - pi-coding-agent src/mnemopi/backend.ts resolveMnemopiProviderOptions returns `llm: false` for none, so pi-mnemopi
#   (src/core/memory.ts) runs with llm {enabled: false}: no configured completion, no host or remote LLM.
# - Retention is unchanged: src/mnemopi/state.ts maybeRetainOnAgentEnd still stores the unretained transcript every
#   retainEveryNTurns (4) user turns (working_memory, source coding-agent-transcript, with its embedding text) and
#   still asks for extraction over the user-authored turns.
# - Extraction (pi-mnemopi src/core/extraction.ts extractFactCategories): llmAvailable() is false, so it falls to
#   heuristicExtractFacts: first-person regexes only ("my name is", "i am", "i work at", "i live in", "i use",
#   "i prefer", "i dislike", "i/you always|never"), at most 5 facts, in-process; the proxy sees no memory-* call. The
#   mem/sess prompts ("the <subject> of project Falcon is <value>") match none of them, so recall can only come from
#   the stored transcript (full-text and embedding search), which only mem's fresh-process recall proves.
# Scoring: sess emits sess.memory_calls_none (zero memory-LLM calls; any call FAILs) instead of the extraction oracle
# sess.memory_calls_ok, and memory_verdict reads it in that one's place.
NO_LLM_EXTRACTION = "mnemopi heuristicExtractFacts over the user turns: regex only, no LLM call"


def mem_overlay(config: Path) -> dict:
    """The settings an omp `--config` overlay sets, read from the two-level `section:` / `  key: value` YAML the
    fixtures use (comments and blank lines skipped). Anything else (deeper nesting, lists, flow or block scalars, tabs,
    duplicate keys) is refused, not guessed: a misread llmMode would score an extraction-free run with the extraction
    oracle, or the reverse."""
    out: dict = {}
    section: dict | None = None
    for n, raw in enumerate(Path(config).read_text().splitlines(), 1):
        line = raw.split(" #", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        value = value.strip()
        target = out if indent == 0 else section
        if ("\t" in line or not sep or not key or indent not in (0, 2) or target is None or key in target
                or key.startswith("-") or value.startswith(("[", "{", "|", ">", "-", "&", "*"))):
            raise ValueError(f"{config}:{n}: unsupported overlay line {raw!r}")
        if indent == 0 and not value:
            section = out[key] = {}
            continue
        target[key] = value.strip("'\"")
        if indent == 0:
            section = None
    return out


def mem_llm_mode(config: Path) -> str:
    """mnemopi.llmMode a memory overlay runs omp's children with (MEM_LLM_MODE_DEFAULT when it does not set one)."""
    section = mem_overlay(config).get("mnemopi", {})
    mode = section.get("llmMode", MEM_LLM_MODE_DEFAULT) if isinstance(section, dict) else None
    if mode not in MEM_LLM_MODES:
        raise ValueError(f"{config}: mnemopi.llmMode {mode!r} is not one of {', '.join(MEM_LLM_MODES)}")
    return mode


# Where omp's children load mnemopi's fastembed model: pi-utils getFastembedCacheDir() = <config root>/cache/fastembed,
# the config root being ~/.omp (PI_CONFIG_DIR, which child_env does not set; PI_CODING_AGENT_DIR does not move it).
# models.FASTEMBED_CACHE names the same directory for `localbench models`.
FASTEMBED_CACHE = Path.home() / ".omp" / "cache" / "fastembed"


def fastembed_digest(name: str, cache: Path | None = None) -> str | None:
    """Identity of an installed fastembed model `name` (`fast-bge-base-en-v1.5`, a `local/<name>` route's model): the
    first 12 hex of sha256 over every file under <cache>/<name> (relative path, NUL, bytes, NUL; sorted by path),
    skipping files mnemopi quarantined (`*.corrupt-*`). None when the model is not on disk. Cached per (path, mtime,
    size) of every file, so a re-download re-hashes."""
    d = (cache or FASTEMBED_CACHE) / name
    files = sorted(p for p in d.rglob("*") if p.is_file() and ".corrupt-" not in p.name) if d.is_dir() else []
    if not files:
        return None
    return _tree_digest(tuple((p.relative_to(d).as_posix(), str(p), st.st_mtime_ns, st.st_size)
                              for p in files for st in [p.stat()]))


@functools.cache
def _tree_digest(entries: tuple[tuple[str, str, int, int], ...]) -> str:
    h = hashlib.sha256()
    for rel, path, _mtime_ns, _size in entries:
        h.update(rel.encode() + b"\0")
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()[:12]


def embedding_pins(config: Path, cache: Path | None = None) -> dict:
    """Run pins for the embedding model a memory overlay's omp children run: embedder `local/<fastembed name>`
    (park.fastembed_name of the overlay's mnemopi settings: mnemopi.embeddingModel / embeddingVariant, else omp's
    default) and embedder_digest (fastembed_digest), when the overlay turns mnemopi on with embeddings (noEmbeddings
    not true). An overlay without embeddings (memory off, FTS-only) pins both None; one this reader cannot parse pins
    embedder "unknown"."""
    from .park import fastembed_name  # park imports this module
    try:
        flat = {f"{sec}.{k}": v for sec, keys in mem_overlay(config).items() if isinstance(keys, dict)
                for k, v in keys.items()}
    except (OSError, ValueError):
        return {"embedder": "unknown", "embedder_digest": None}
    if flat.get("memory.backend") != "mnemopi" or flat.get("mnemopi.noEmbeddings", "false") == "true":
        return {"embedder": None, "embedder_digest": None}
    name = fastembed_name(flat)
    return {"embedder": f"local/{name}", "embedder_digest": fastembed_digest(name, cache)}



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

BODY_ACCOUNTING_VOID = "main call body missing or unreadable"


def _body_tool_calls(body_dir: Path, row: dict) -> int | None:
    """Count assistant tool calls in a saved request body, or return None when it cannot be read."""
    body = row.get("body")
    if not isinstance(body, str):
        return None
    try:
        request = json.loads((body_dir / body).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        return None
    total = 0
    for message in request["messages"]:
        if not isinstance(message, dict):
            return None
        calls = message.get("tool_calls")
        if message.get("role") == "assistant":
            if calls is not None and not isinstance(calls, list):
                return None
            total += len(calls or [])
    return total


def _turn_tool_call_total(rows: list[dict], body_dir: Path) -> int | None:
    main_rows = [row for row in rows if row.get("purpose") == "main"]
    counts = [_body_tool_calls(body_dir, row) for row in main_rows]
    if any(count is None for count in counts):
        return None
    return max((count for count in counts if count is not None), default=0)


def _turn_tool_calls(rows: list[dict], body_dir: Path, previous_total: int = 0) -> int | None:
    total = _turn_tool_call_total(rows, body_dir)
    return None if total is None else max(0, total - previous_total)


def _turn_metrics(turns: list[dict]) -> dict:
    """Metrics for the stage-2 gate: the worst turn and the timeout count."""
    if not turns:
        return {"max_tool_calls": void("lower", "no turns recorded"),
                "timeouts": void("lower", "no turns recorded")}
    counts = [int(t["tool_calls"]) for t in turns if t.get("tool_calls") is not None]
    body_error = next((t.get("tool_calls_error") for t in turns if t.get("tool_calls") is None), None)
    max_tool_calls = (void("lower", body_error) if body_error else
                      {"value": max(counts), "better": "lower",
                       "spread": [min(counts), max(counts)], "n": len(counts)}
                      if counts else void("lower", "no valid samples"))
    timeouts = sum(bool(t.get("timeout")) for t in turns)
    return {
        "max_tool_calls": max_tool_calls,
        "timeouts": {"value": timeouts, "better": "lower",
                      "spread": [timeouts, timeouts], "n": len(turns)},
    }


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
        tasks = [row for row in E2E_TASKS if ctx.e2e_case is None or row[0] == ctx.e2e_case]
        if ctx.e2e_case is not None and not tasks:
            raise ValueError(f"unknown e2e campaign case {ctx.e2e_case!r}")
        for task, prompt, _check in tasks:
            walls, llm, starts, usage, first_split, answers, attempts = [], [], [], {}, {}, [], []
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
                trace = ctx.run_dir / f"e2e.{task}.{attempt}.omp.jsonl"
                trace.write_text(proc.stdout)
                (ctx.run_dir / f"e2e.{task}.{attempt}.result.json").write_text(
                    json.dumps({"task": task, "attempt": attempt, "returncode": proc.returncode,
                                "stderr": proc.stderr}, sort_keys=True) + "\n")
                split = _attempt_calls(calls_log, started, time.time())
                if attempt == "first":
                    first_split = split
                text, usage = _omp_final(proc.stdout)
                answers.append(text)
                scored_attempt = score_e2e_attempt(task, text, proc.returncode, proc.stdout)
                attempts.append({"stdout": proc.stdout, "returncode": proc.returncode})
                ok = scored_attempt["ok"]
                walls.append(wall)
                llm.append(split["llm_s"])
                starts.append(split["startup_s"])
                ctx.emit({"event": "e2e", "task": task, "attempt": attempt, "wall_s": round(wall, 2),
                          "ok": ok, "rc": proc.returncode, "answer": text[:80], **usage, **split,
                          **({"stderr": proc.stderr[-300:]} if proc.returncode else {})})
            cached = first_split.get("first_call_cached_tokens")
            score = score_e2e_case(task, attempts)
            ok_all = score["status"] == "PASS"
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
            out.append(Result(f"e2e.{task}.correct", "e2e", "MUST", verdict=score["status"],
                              detail={"answers": recorded_answers(answers), "attempts": score["attempts"]}))
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

def score_e2e_attempt(task: str, answer: str, returncode: int, stdout: str) -> dict:
    check = next((check for name, _, check in E2E_TASKS if name == task), None)
    if check is None:
        raise ValueError(f"unknown e2e task {task!r}")
    read_ok = task != "tool_read"
    if not read_ok:
        read_calls = set()
        expected_path = Path("/tmp/localbench-e2e/answer.txt").resolve()
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("toolName") != "read":
                continue
            call_id = event.get("toolCallId")
            if not isinstance(call_id, str):
                continue
            if event.get("type") == "tool_execution_start":
                args = event.get("args")
                path = args.get("path") if isinstance(args, dict) else None
                if isinstance(path, str):
                    file_path = Path(path)
                    if not file_path.is_absolute():
                        file_path = Path("/tmp/localbench-e2e") / file_path
                    if file_path.resolve() == expected_path:
                        read_calls.add(call_id)
            elif event.get("type") == "tool_execution_end" and call_id in read_calls:
                result = event.get("result")
                details = result.get("details") if isinstance(result, dict) else None
                display = details.get("displayContent") if isinstance(details, dict) else None
                if (event.get("isError") is False and isinstance(display, dict)
                        and display.get("text") == "4817"):
                    read_ok = True
                    break
    return {"answer": answer, "returncode": returncode, "ok": returncode == 0 and check(answer) and read_ok}


def score_e2e_case(task: str, attempts: list[dict]) -> dict:
    if len(attempts) != 2:
        raise ValueError(f"e2e case {task!r} requires first and repeat attempts, got {len(attempts)}")
    scored = []
    for attempt in attempts:
        if not isinstance(attempt.get("stdout"), str) or not isinstance(attempt.get("returncode"), int):
            raise ValueError(f"e2e case {task!r} has an invalid attempt trace")
        answer, _ = _omp_final(attempt["stdout"])
        scored.append(score_e2e_attempt(task, answer, attempt["returncode"], attempt["stdout"]))
    passed = sum(row["ok"] for row in scored)
    return {"case": f"e2e.{task}.correct", "status": "PASS" if passed == 2 else "FAIL",
            "passed": passed, "expected": 2, "attempts": scored}


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


def answer_call(rows: list[dict]) -> dict | None:
    """The turn's answer call: the first call that carries omp's tools (proxy purpose `main`), not the effort
    classifier (with or without its `submit_judgment` retry) or a memory-LLM call. The mem tier declares the memory
    tools (child_flags tools="memory"), so every answer request carries them. Under the old `--no-tools` every call
    had tools=0 and the answer was an unmarked `aux` call; a turn with no `main` call now has no answer call, and
    pre_main_s shows the missing sample (n) instead of timing a side call."""
    return next((r for r in sorted(rows, key=lambda r: r.get("t_start") or 0) if r.get("purpose") == "main"), None)


def _completed_call(row: dict | None) -> bool:
    """Only a successful, fully observed proxy response can establish a call happened."""
    return (row is not None and row.get("status") == 200 and row.get("aborted") is False
            and row.get("response_complete") is True)


def mem(ctx: Ctx) -> list[Result]:
    """Does omp's memory work across sessions, and what does it cost? Per fact and round, four `omp -p` processes
    with memory on (ctx.mem_config): PLANT the fact in a fresh project dir (retained when the process exits); RECALL it
    from a new process in the same dir; CONTROL — the same question from a fresh dir with no plant, whose answer must
    contain no value planted in this run (a hit is leakage between projects, or a guess); DERAIL — "Reply with
    exactly: OK" in the
    planted dir, which recalled memory must not break (ledger 2026-09-23). A recall hit requires a completed plant
    and main call, correct new-process recall, and a completed clean fresh control. Print-mode children persist their
    transcripts without extraction: a matching fresh-process recall proves that path. An extraction call, if emitted,
    must complete. Measured: recall hit rate (Wilson 95%), time to the main call and wall per recall turn, prompt
    tokens recall injects (recall minus control main prompt), plant wall, derail rate. The tier's own banks and dirs
    are removed afterwards
    (memory.remove_banks). Turns get omp's memory tools and no others
    (child_flags tools="memory"): memory works the way it does in a session, and a turn cannot search the machine
    for an answer. Opt-in (`--tiers mem`)."""
    from . import memory
    from .proxy import Proxy

    if not ctx.loaded_context:
        return [Result("mem", "mem", "MUST", verdict="FAIL",
                       detail={"reason": "backend did not report its loaded context; refusing to guess one for omp"})]
    refused = _smol_refusal(ctx, "mem")
    if refused:
        return [refused]
    llm_mode = mem_llm_mode(ctx.mem_config)
    run_id = uuid.uuid4().hex[:8]
    prefix = f"localbench-mem-{run_id}-"
    calls_log = ctx.run_dir / "mem_calls.jsonl"
    body_dir = ctx.run_dir / "bodies"
    rng = random.Random()
    flags = child_flags(ctx.model, ctx.mem_config, tools="memory", smol=ctx.smol_model)
    turn_details = []

    def turn(prompt: str, cwd: Path) -> dict:
        t0, started = time.perf_counter(), time.time()
        timed_out = False
        try:
            proc = subprocess.run([omp_bin(), "-p", prompt, *flags], cwd=cwd, capture_output=True, text=True,
                                  timeout=1800, env=child_env(), stdin=subprocess.DEVNULL, check=False)
            rc, stdout = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired:
            # A turn that never ends is a finding about the model or server, not a reason to lose the other legs'
            # data (2026-09-26: one Bonsai 2 turn on mlxfast ran 30 min and the traceback ended the whole A/B).
            rc, stdout, timed_out = "timeout", "", True
        wall, ended = time.perf_counter() - t0, time.time()
        # Proxy rounds the start to milliseconds; the unrounded end still bounds this child's call.
        rows = [r for r in map(json.loads, calls_log.read_text().splitlines())
                if started <= r.get("t", 0) <= ended
                and r.get("t_start", 0) >= started - 0.001] if calls_log.exists() else []
        ans = answer_call(rows)
        text, _ = _omp_final(stdout)
        tool_calls = _turn_tool_calls(rows, body_dir)

        def trace(row: dict) -> dict:
            return {k: row.get(k) for k in ("purpose", "status", "aborted", "response_complete")}
        completed_main = next((r for r in rows if r.get("purpose") == "main" and _completed_call(r)), None)
        return {"rc": rc, "text": text, "wall": wall, "timeout": timed_out,
                "tool_calls": tool_calls,
                **({"tool_calls_error": BODY_ACCOUNTING_VOID} if tool_calls is None else {}),
                "pre_main": ans["t_start"] - started if ans else None,
                "prompt_tokens": ans.get("prompt_tokens") if ans else None,
                "main_trace": trace(completed_main or ans) if completed_main or ans else None,
                "retention_trace": [trace(row) for row in rows if row.get("purpose") == "memory-extract"]}

    hits, attempts, leaks, invalid, derails, attempted_values = 0, [], [], [], [], []
    plant_walls, recall_walls, pre_main, injected = [], [], [], []
    dirs = []
    try:
        with Proxy(ctx.backend.base_url, calls_log, save_dir=ctx.run_dir / "bodies", label="mem"):
            ensure_localbench_model(ctx.model, ctx.loaded_context, extra=_smol_extra(ctx))
            for rnd in range(ctx.mem_rounds):
                for name, plant, question, expected in _mem_facts(rng):
                    home, control = Path(f"/tmp/{prefix}{name}-{rnd}"), Path(f"/tmp/{prefix}control-{name}-{rnd}")
                    for d in (home, control):
                        d.mkdir()
                        dirs.append(d)
                    attempted_values.append(expected)
                    p = turn(plant, home)
                    r = turn(question, home)
                    c = turn(question, control)
                    o = turn("Reply with exactly: OK", home)
                    plant_ok = p["rc"] == 0 and p["text"].strip().rstrip(".") == "NOTED"
                    plant_main_ok = _completed_call(p["main_trace"])
                    recall_main_ok = _completed_call(r["main_trace"])
                    retention = p["retention_trace"]
                    retention_ok = all(_completed_call(call) for call in retention)
                    control_ok = c["rc"] == 0 and bool(c["text"].strip()) and _completed_call(c["main_trace"])
                    found = _leaked(c["text"], attempted_values) if control_ok else []
                    invalid_reasons = []
                    if not plant_ok:
                        invalid_reasons.append("plant failed or did not acknowledge fact")
                    if not plant_main_ok:
                        invalid_reasons.append("plant main call missing or failed")
                    if not retention_ok:
                        invalid_reasons.append("plant extraction failed or aborted")
                    if llm_mode == "none" and retention:
                        invalid_reasons.append("memory LLM call under llmMode none: the overlay did not take effect")
                    if not control_ok:
                        invalid_reasons.append("control failed, blank, or main call missing")
                    issues = invalid_reasons.copy()
                    if r["rc"] != 0 or not recall_main_ok:
                        issues.append("recall failed or main call missing")
                    if found:
                        issues.append("control leaked planted fact")
                    if r["text"].strip().rstrip(".").casefold() != expected.casefold():
                        issues.append("recall answer did not match planted fact")
                    reason = "; ".join(issues) or None
                    hit = reason is None
                    hits += hit
                    plant_walls.append(p["wall"])
                    recall_walls.append(r["wall"])
                    if r["pre_main"] is not None:
                        pre_main.append(r["pre_main"])
                    if r["prompt_tokens"] and c["prompt_tokens"]:
                        injected.append(r["prompt_tokens"] - c["prompt_tokens"])
                    if invalid_reasons:
                        invalid.append({"fact": name, "round": rnd, "reasons": invalid_reasons})
                    if found:
                        leaks.append({"fact": name, "round": rnd, "values": found, "answer": c["text"][:160]})
                    if not (o["rc"] == 0 and o["text"].strip().rstrip(".") == "OK"):
                        derails.append({"fact": name, "round": rnd, "answer": o["text"][:160]})
                    attempt = {"fact": name, "round": rnd, "expected": expected, "hit": hit,
                               "reason": reason, "reasons": issues,
                               "retention_provenance": ("extraction" if retention else "fresh-process-recall")
                               if hit else None,
                               "recall_answer": r["text"][:160], "plant_answer": p["text"][:80],
                               "control_answer": c["text"][:80], "rcs": [p["rc"], r["rc"], c["rc"], o["rc"]],
                               "plant_trace": {"main": p["main_trace"], "retention": retention},
                               "recall_trace": r["main_trace"], "control_trace": c["main_trace"]}
                    for kind, turn_result in (("plant", p), ("recall", r), ("control", c), ("derail", o)):
                        turn_details.append({
                            "round": rnd, "fact": name, "kind": kind,
                            "tool_calls": turn_result["tool_calls"], "timeout": turn_result["timeout"],
                            **({"tool_calls_error": turn_result["tool_calls_error"]}
                               if "tool_calls_error" in turn_result else {}),
                            "rc": turn_result["rc"], "wall_s": round(turn_result["wall"], 3),
                        })
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
        Result("mem.turn", "mem", "perf", _turn_metrics(turn_details),
               detail={"turns": turn_details}),
        Result("mem.recall", "mem", "perf", {
            "hit_rate": {"value": rate, "better": "higher", "spread": [rate, rate], "n": n},
            "pre_main_s": metric(pre_main, "lower"),
            "wall_s": metric(recall_walls, "lower")},
            detail={"hits": hits, "attempts": n, "ci95": _wilson(hits, n) if n else None, "rounds": attempts,
                    "config": str(ctx.mem_config), "llm_mode": llm_mode, "smol_model": ctx.smol_model,
                    # Recall-turn main prompt minus the control turn's: what recall added. 0 is a real answer, so it
                    # is not a golden metric (a relative band on 0 is meaningless).
                    "injected_tokens": {"median": statistics.median(injected), "min": min(injected),
                                        "max": max(injected)} if injected else None}),
        Result("mem.plant", "mem", "perf", {"wall_s": metric(plant_walls, "lower")}),
        Result("mem.derail", "mem", "perf",
               {"ok_rate": {"value": derail_ok, "better": "higher", "spread": [derail_ok, derail_ok], "n": n}},
               detail={"derails": derails}),
        Result("mem.no_leak", "mem", "MUST",
               verdict="FAIL" if leaks else "VOID" if invalid or not n else "PASS",
               detail={"leaks": leaks, "invalid": invalid,
                       "reason": "planted value in fresh control" if leaks else
                                 "plant or fresh control unverified" if invalid else
                                 "no facts attempted" if not n else None}),
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
        try:
            threading.Thread(target=self._pump, daemon=True).start()
        except BaseException:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait()
            finally:
                self.proc.stdin.close()
                self.proc.stdout.close()
                self._stderr.close()
            raise

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        finally:
            self.proc.stdout.close()  # EOF follows the child's exit or kill; unclosed, the GC warned into stderr
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
        """Close stdin and reap the child, killing it on timeout or cancellation."""
        t0 = time.perf_counter()
        try:
            with contextlib.suppress(BrokenPipeError):
                self.proc.stdin.close()
            try:
                rc = self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                rc = self.proc.wait()
        except BaseException:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            raise
        finally:
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
    the model under test (one model on the machine) or on ctx.smol_model takes SESS_TURNS turns of one shape — state a
    made-up project fact, reply NOTED — sent back to back. Post-retain turns (5, 9) are compared with regular turns
    (turn 1 is cold and left out) and with their own neighbours, which cancels the growing-context trend. Memory-LLM
    calls are the proxy's `memory-*` purposes; overlap is the seconds they were in flight together with a main call.
    With a declared smol model every memory call must name it. Under mnemopi.llmMode none retention still runs at the
    same boundaries but extracts by regex in-process (NO_LLM_EXTRACTION): there is no extraction call to time or
    find, so `sess.memory_calls_ok` (extraction at each boundary) is not emitted and `sess.memory_calls_none` asserts
    the configuration ran as declared, zero memory-LLM calls; whether retention still recalls is the mem tier's
    fresh-process recall, which this tier does not test. Opt-in (`--tiers sess`)."""
    from . import memory
    from .proxy import Proxy

    if not ctx.loaded_context:
        return [Result("sess", "sess", "MUST", verdict="FAIL",
                       detail={"reason": "backend did not report its loaded context; refusing to guess one for omp"})]
    refused = _smol_refusal(ctx, "sess")
    if refused:
        return [refused]
    llm_mode = mem_llm_mode(ctx.mem_config)
    prefix = f"localbench-sess-{uuid.uuid4().hex[:8]}-"
    calls_log = ctx.run_dir / "sess_calls.jsonl"
    body_dir = ctx.run_dir / "bodies"
    argv = [omp_bin(), *child_flags(ctx.model, ctx.mem_config, mode="rpc", smol=ctx.smol_model), "--max-time", "30m"]
    rng = random.Random()
    turns, sessions, dirs = [], [], []
    try:
        with Proxy(ctx.backend.base_url, calls_log, save_dir=body_dir, label="sess"):
            ensure_localbench_model(ctx.model, ctx.loaded_context, extra=_smol_extra(ctx))
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
                        try:
                            rpc.send({"id": f"s{s}t{i}", "type": "prompt", "message": prompt})
                            lines = rpc.until(_turn_end, SESS_TURN_TIMEOUT_S)
                            wall, t_end = time.perf_counter() - t0, time.time()
                            last = json.loads(lines[-1])
                            if last.get("type") == "response":
                                raise EOFError(f"prompt rejected: {str(last.get('error'))[:200]}")
                            text, _ = _omp_final("".join(lines))
                            ok = text.strip().rstrip(".") == "NOTED"
                            turns.append({"session": s, "turn": i, "wall": wall, "t_send": t_send, "t_end": t_end,
                                          "ok": ok, "timeout": False,
                                          "post_retain": i > 1 and (i - 1) % SESS_RETAIN_EVERY == 0,
                                          **({} if ok else {"reply": text[:160]})})
                            n_done = i
                        except (TimeoutError, EOFError) as exc:
                            t_end = time.time()
                            turns.append({"session": s, "turn": i, "wall": time.perf_counter() - t0,
                                          "t_send": t_send, "t_end": t_end, "ok": False,
                                          "timeout": isinstance(exc, TimeoutError),
                                          "post_retain": i > 1 and (i - 1) % SESS_RETAIN_EVERY == 0,
                                          "error": f"{type(exc).__name__}: {exc}"})
                            failure = f"session {s} turn {i}: {type(exc).__name__}: {exc}"
                            break
                except (TimeoutError, EOFError) as exc:
                    failure = f"session {s} turn {n_done + 1}: {type(exc).__name__}: {exc}"
                finally:
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
    previous_totals = {}
    by_key = {(t["session"], t["turn"]): t for t in turns}
    ended = {x["session"]: x["ended"] for x in sessions}

    def first_main_start(t: dict) -> float:
        """Earliest main-call start within turn `t`'s [t_send, t_end]; t's own t_end when it started none."""
        return min((r["t_start"] for r in main if t["t_send"] <= r["t_start"] <= t["t_end"]), default=t["t_end"])

    for t in turns:
        # A main call belongs to the turn in whose [t_send, t_end] it STARTED. Its completion row trails omp's
        # agent_end by up to tens of ms (proxy bookkeeping: +8 to +39 ms in r3 legs a2/b2/a3, 2026-10-01, while sess
        # sends the next turn 0.4-0.6 ms after agent_end), so it must complete before the NEXT turn's first main call
        # starts (that turn's t_end when it has none), or before the session's end for the last turn; a timed-out turn
        # keeps any call it started.
        nxt = by_key.get((t["session"], t["turn"] + 1))
        until = first_main_start(nxt) if nxt else ended.get(t["session"], t["t_end"])
        mine = [r for r in main if t["t_send"] <= r["t_start"] <= t["t_end"]
                and (t["timeout"] or r["t"] <= until)]
        previous = previous_totals.get(t["session"], 0)
        total = _turn_tool_call_total(mine, body_dir)
        if total is None:
            t["tool_calls"] = None
            t["tool_calls_error"] = BODY_ACCOUNTING_VOID
        else:
            t["tool_calls"] = _turn_tool_calls(mine, body_dir, previous)
            previous_totals[t["session"]] = max(previous, total)
        t["pre_main"] = min(r["t_start"] for r in mine) - t["t_send"] if mine else None
        t["overlap_s"] = _overlap_s(mine, mem_rows)
        t["main_call_ok"] = any(_completed_call(r) for r in mine)
    by_turn = {(t["session"], t["turn"]): t for t in turns}
    regular = [t for t in turns if t["turn"] > 1 and not t["post_retain"]]
    post = [t for t in turns if t["post_retain"]]
    deltas = [t["wall"] - (by_turn[(t["session"], t["turn"] - 1)]["wall"] + nxt["wall"]) / 2
              for t in post if (nxt := by_turn.get((t["session"], t["turn"] + 1)))]
    by_purpose = {p: {"calls": len(sel), "busy_s": _busy_s(sel),
                      "not_ok": sum(not _completed_call(r) for r in sel),
                      "completion_tokens": sum(r.get("completion_tokens") or 0 for r in sel),
                      "models": sorted({str(r.get("model")) for r in sel})}
                  for p in sorted({r["purpose"] for r in mem_rows})
                  for sel in [[r for r in mem_rows if r["purpose"] == p]]}
    acks = sum(t["ok"] for t in turns)
    ack_rate = round(acks / len(turns), 4) if turns else 0.0
    failures = [x["failure"] for x in sessions if x["failure"]]
    bad_mem = [{k: r.get(k) for k in ("purpose", "status", "aborted", "response_complete", "total_s")}
               for r in mem_rows if not _completed_call(r)]
    extracts = [r for r in mem_rows if r["purpose"] == "memory-extract" and _completed_call(r)]
    observed_main = [r for r in main if _completed_call(r)]
    # Retain after turns 4, 8 and 12. The first two extractions may overlap the next main call;
    # the final one must finish by close(), not necessarily overlap another turn.
    missing_extract = []
    sessions_without_overlap = []
    for session in sessions:
        s = session["session"]
        session_extract = [r for r in extracts if session["started"] - 0.001 <= r["t_start"] <= session["ended"]
                           and r["t"] <= session["ended"]]
        session_main = [r for r in observed_main if session["started"] - 0.001 <= r["t_start"] <= session["ended"]]
        if session_extract and session_main and _overlap_s(session_main, session_extract) <= 0:
            sessions_without_overlap.append(s)
        for boundary in range(SESS_RETAIN_EVERY, SESS_TURNS + 1, SESS_RETAIN_EVERY):
            current = by_turn.get((s, boundary))
            nxt = by_turn.get((s, boundary + 1))
            if current is None or not any(current["t_send"] - 0.001 <= r["t_start"]
                                          <= (nxt["t_end"] if nxt else session["ended"])
                                          for r in session_extract):
                missing_extract.append(f"session {s} after turn {boundary}")
    extract_overlap = _overlap_s(observed_main, extracts)
    expected_turns = ctx.repeats * SESS_TURNS
    if not sessions:
        turn_reason = "no sessions attempted"
    elif (failures or len(turns) != expected_turns or len(sessions) != ctx.repeats
          or any(x["turns"] != SESS_TURNS or x["rc"] != 0 for x in sessions)
          or any([t["turn"] for t in turns if t["session"] == s] != list(range(1, SESS_TURNS + 1))
                 for s in range(ctx.repeats))):
        turn_reason = "incomplete turns or failed session"
    elif acks != expected_turns:
        turn_reason = ACK_SLIP
    elif not all(t["main_call_ok"] for t in turns):
        turn_reason = "main call trace missing or incomplete"
    else:
        turn_reason = None
    # A declared smol model must serve every memory call: one served by the model under test (omp resolving its
    # memory role elsewhere) would credit the candidate memory model with the main model's extractions.
    misrouted = (sorted({str(r.get("model")) for r in mem_rows if r.get("model") != ctx.smol_model})
                 if ctx.smol_model else [])
    if llm_mode == "none":
        if mem_rows:
            memory_reason = f"{len(mem_rows)} memory LLM call(s) under llmMode none: the overlay did not take effect"
        elif not observed_main:
            memory_reason = "main call trace missing or incomplete"
        else:
            memory_reason = turn_reason
    elif bad_mem:
        memory_reason = "failed or aborted memory call (memory-extract unverified)"
    elif misrouted:
        memory_reason = f"memory call(s) served by {', '.join(misrouted)}, not the declared smol model {ctx.smol_model}"
    elif not mem_rows:
        memory_reason = "memory-extract trace missing"
    elif not extracts:
        memory_reason = "no successful memory-extract call"
    elif not observed_main:
        memory_reason = "main call trace missing or incomplete"
    elif missing_extract:
        memory_reason = f"memory-extract missing at scheduled boundary: {', '.join(missing_extract)}"
    elif sessions_without_overlap or extract_overlap <= 0:
        memory_reason = "no positive memory-extract/main overlap in each session"
    else:
        memory_reason = turn_reason
    calls_verdict = ("VOID" if memory_reason in ("memory-extract trace missing",
                                                  "main call trace missing or incomplete")
                     else "FAIL" if memory_reason else "PASS")
    memory_detail = {"reason": memory_reason, "llm_mode": llm_mode, "memory_calls": len(mem_rows)}
    memory_case = (
        Result("sess.memory_calls_none", "sess", "SHOULD", verdict=calls_verdict,
               detail={**memory_detail, "expected_memory_calls": 0, "extraction": NO_LLM_EXTRACTION})
        if llm_mode == "none" else
        Result("sess.memory_calls_ok", "sess", "SHOULD", verdict=calls_verdict,
               detail={**memory_detail, "successful_extracts": len(extracts),
                       "extract_overlap_with_main_s": extract_overlap, "not_ok": bad_mem}))
    return [
        Result("sess.turn", "sess", "perf", {
            **_turn_metrics(turns),
            "wall_s": metric([t["wall"] for t in regular], "lower"),
            "post_retain_wall_s": metric([t["wall"] for t in post], "lower"),
            "pre_main_s": metric([t["pre_main"] for t in regular], "lower"),
            "post_retain_pre_main_s": metric([t["pre_main"] for t in post], "lower"),
            "ack_rate": {"value": ack_rate, "better": "higher", "spread": [ack_rate, ack_rate], "n": len(turns)}},
            # Signed and near 0 when retention costs nothing, so not a golden metric (a relative band on ~0 is
            # meaningless): post-retain turn wall minus the mean of its two neighbours.
            detail={"neighbour_delta_s": {"median": round(statistics.median(deltas), 3), "min": round(min(deltas), 3),
                                          "max": round(max(deltas), 3), "n": len(deltas)} if deltas else None,
                    "config": str(ctx.mem_config), "llm_mode": llm_mode, "smol_model": ctx.smol_model,
                    "turns": [{k: (round(v, 3) if isinstance(v, float) and k not in ("t_send", "t_end") else v)
                               for k, v in t.items()} for t in turns]}),
        Result("sess.memory", "sess", "perf", {
            "extract_s": metric([r["total_s"] for r in extracts if r.get("total_s") is not None], "lower")},
            detail={"by_purpose": by_purpose, "overlap_with_main_s": _overlap_s(main, mem_rows),
                    "extract_overlap_with_main_s": extract_overlap,
                    "exit_s": [x["exit_s"] for x in sessions], "sessions": sessions}),
        Result("sess.turns_complete", "sess", "MUST",
               verdict="VOID" if turn_reason in ("no sessions attempted", "main call trace missing or incomplete")
               else "FAIL" if turn_reason else "PASS",
               detail={"reason": turn_reason, "failures": failures, "turns": len(turns), "acks": acks,
                       "expected": expected_turns}),
        memory_case,
    ]


# ---------------------------------------------------------------- memory proof

# Today's route for omp's memory feature: every profile's smol. It has not yet been measured under the g73 oracle; its
# legs, run under that oracle in the same invocation as the candidate's, are the baseline arm.
# Matched by digest (features.same_digest), not name: while parked, the incumbent runs under its park alias
# localbench-parked:<digest12> (AGENTS.md), which is the same build.
MEMORY_BASELINE = {"kind": "route", "id": "ollama/qwen3.8:27b-mlx", "digest": "5642e97495e1"}
# The declared non-local alternatives of the memory family (registries/presets.json; features.alternatives) a baseline
# arm may run instead of MEMORY_BASELINE, named as features.incumbent names the route each gives (`fixed` = the gating
# setting's value while the route is off), with the leg configuration (_leg_memory key, value) that identifies it.
MEMORY_ALTERNATIVES = (
    {"kind": "fixed", "id": "none", "preset": "memory:none", "leg": ("llm_mode", "none")},
    {"kind": "fixed", "id": "true", "preset": "embeddings:fts", "leg": ("embeddings", "off")},
)
GPU_BUSY = "system.during.gpu_device_pct"
# Higher is better; a loss beyond the legs' A/A noise makes the comparison WORSE, and one unmeasured keeps it from
# BETTER (fresh-process recall is the proof that memory works; a config that did not measure it proves nothing).
# Conformance cases count 1.0 (PASS) / 0.0 (FAIL) per leg, VOID unmeasured. sess.memory_calls_ok is the leg's
# memory-calls case, which under llmMode none is sess.memory_calls_none (zero memory-LLM calls, as declared).
MEMORY_QUALITY = ("mem.recall.hit_rate", "mem.no_leak", "mem.derail.ok_rate", "sess.memory_calls_ok")
# Lower is better; one better beyond noise, with no quality loss, is the measured win BETTER needs.
MEMORY_WINS = (("mem.recall.pre_main_s", "recall_latency"), ("sess.turn.post_retain_wall_s", "post_retain_latency"),
               ("sess.memory.extract_s", "extract_latency"), (GPU_BUSY, "gpu_busy"))
_MEMORY_CASES = {"mem.no_leak": ("mem.no_leak",),
                 "sess.memory_calls_ok": ("sess.memory_calls_ok", "sess.memory_calls_none")}


def _real_value(v) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def _memory_leg_value(leg: dict, key: str) -> float | None:
    """One leg's (an `execute` summary's) value of a MEMORY_QUALITY/MEMORY_WINS key; None when it did not measure it.
    A memory-calls case whose FAIL reason is only the main model's acknowledgement slip (sess falls through to the
    turn reason when every memory check held) counts as PASS on an ack-only-slip leg: the slip is
    memory.main_ack_slips, not a memory failure."""
    if key == GPU_BUSY:
        return _real_value((((leg.get("system") or {}).get("during") or {}).get("gpu_device_pct") or {}).get("mean"))
    if key in _MEMORY_CASES:
        conformance = leg.get("conformance") or {}
        case = next((c for c in _MEMORY_CASES[key] if c in conformance), None)
        entry = conformance.get(case) or {}
        if (entry.get("verdict") == "FAIL" and _leg_details(leg).get(case, {}).get("reason") == ACK_SLIP
                and _ack_only_slips(leg) is not None):
            return 1.0
        return {"PASS": 1.0, "FAIL": 0.0}.get(entry.get("verdict"))
    m = (leg.get("metrics") or {}).get(key) or {}
    return None if m.get("void") or m.get("n") == 0 else _real_value(m.get("value"))


# sess.turns_complete's reason when every turn completed but the main model did not reply exactly NOTED.
ACK_SLIP = "incorrect acknowledgement"
# The ack-slip exemption is pre-registered "for the next run, before any new data, never applied to past receipts":
# bead kit-memory-study-vce, DECISIONS comment (the owner, via ask), created_at 2026-10-01T14:30:14Z. Only legs whose
# provenance.created is this stamp or later get it; earlier or unrecorded legs keep the MUST failure.
ACK_SLIP_RULE_FROM = "20261001T143014Z"
_STAMP = re.compile(r"\d{8}T\d{6}Z")
# The side-model regime for memory-route proofs (bead kit-memory-study-vce, DECISION comment, the owner via ask,
# created_at 2026-10-02T01:31:49Z): no park, no fence, the real smol model stays resident and shared; co-resident load
# is recorded, never used to void; quality gates unchanged. Legs carry it as provenance/pins `regime: side` (ab
# --side-regime); it applies only to legs created at or after SIDE_REGIME_FROM ("never applies to earlier legs").
SIDE_REGIME = "side"
SIDE_REGIME_FROM = "20261002T013149Z"


def side_regime(leg: dict) -> bool:
    """Whether a leg (an execute summary or its banked view) ran under the side-model regime and the pre-registered
    rule covers it: pins.regime side and a created stamp at or after SIDE_REGIME_FROM. An unrecorded stamp: no."""
    prov = leg.get("provenance") or {}
    created = prov.get("created")
    return ((prov.get("pins") or {}).get("regime") == SIDE_REGIME and isinstance(created, str)
            and bool(_STAMP.fullmatch(created)) and created >= SIDE_REGIME_FROM)


def _co_resident(leg: dict) -> dict:
    """What shared the machine with a side-regime leg (its sampler's system.contention episodes): the foreign models
    seen (resident `server/model`, or the GPU process row's model or name), the mean GPU % of the foreign model
    processes at each episode's first sample (None without any), and the unreadable-residency sample count."""
    events = (leg.get("system") or {}).get("contention") or []
    models: set[str] = set()
    gpu: list[float] = []
    for ev in events if isinstance(events, list) else []:
        for server, names in (ev.get("foreign") or {}).items():
            models.update(f"{server}/{name}" for name in names or [])
        for row in ev.get("foreign_gpu") or []:
            models.add(str(row.get("model") or row.get("name")))
            if isinstance(row.get("pct"), int | float):
                gpu.append(float(row["pct"]))
    return {"models": sorted(models), "episodes": len(events) if isinstance(events, list) else 0,
            "foreign_gpu_mean_pct": round(statistics.fmean(gpu), 1) if gpu else None,
            "resident_unknown_samples": _resident_unknown(leg)}




def _ack_only_slips(leg: dict) -> int | None:
    """Turns a leg's main model slipped on, when its MUST sess.turns_complete FAILed ONLY on acknowledgement
    (pre-registered 2026-10-01, bead kit-memory-study-vce: reported as memory.main_ack_slips, not voided): reason
    ACK_SLIP, no session failure, every turn completed (turns == expected), zero sess turn timeouts, and the leg created
    at or after ACK_SLIP_RULE_FROM. None for any other leg: a PASS, a failure with any other cause, or a leg created
    before the rule (or with no created stamp), whose failure stays a MUST failure."""
    created = (leg.get("provenance") or {}).get("created")
    if not isinstance(created, str) or not _STAMP.fullmatch(created) or created < ACK_SLIP_RULE_FROM:
        return None
    entry = (leg.get("conformance") or {}).get("sess.turns_complete") or {}
    d = _leg_details(leg).get("sess.turns_complete") or {}
    expected, acks = d.get("expected"), d.get("acks")
    timeouts = ((leg.get("metrics") or {}).get("sess.turn.timeouts") or {}).get("value")
    if (entry.get("verdict") != "FAIL" or d.get("reason") != ACK_SLIP or d.get("failures") != []
            or not isinstance(expected, int) or expected <= 0 or d.get("turns") != expected
            or not isinstance(acks, int) or not 0 <= acks < expected or timeouts != 0):
        return None
    return expected - acks


def _memory_call_count(leg: dict) -> int | None:
    """Memory-LLM calls a leg's sess sessions made (sess.memory by_purpose calls); None when unrecorded."""
    by_purpose = _leg_details(leg).get("sess.memory", {}).get("by_purpose")
    if not isinstance(by_purpose, dict) or any(not isinstance(p, dict) or not isinstance(p.get("calls"), int)
                                               for p in by_purpose.values()):
        return None
    return sum(p["calls"] for p in by_purpose.values())


def _leg_details(leg: dict) -> dict:
    """case -> detail of one leg: an `execute` summary carries them in `results`, a banked receipt leg
    (__main__._receipt_view) only in `details`; both are read, banked details winning."""
    out = {r["case"]: r["detail"] for r in leg.get("results") or []
           if isinstance(r, dict) and r.get("case") and isinstance(r.get("detail"), dict)}
    out.update({k: v for k, v in (leg.get("details") or {}).items() if isinstance(v, dict)})
    return out


def _served_memory_models(leg: dict) -> set[str] | None:
    """The models that served a leg's memory-LLM calls (sess.memory by_purpose, from the proxy rows' `model`); None
    when the leg did not record them (no sess.memory detail, or one banked before by_purpose carried `models`)."""
    by_purpose = _leg_details(leg).get("sess.memory", {}).get("by_purpose")
    if not isinstance(by_purpose, dict):
        return None
    if any(not isinstance(p, dict) or not isinstance(p.get("models"), list) for p in by_purpose.values()):
        return None
    return {str(m) for p in by_purpose.values() for m in p["models"]}


def _resident_unknown(leg: dict) -> int | None:
    """Sampler samples with unreadable resident-model state (as __main__._resident_unknown); None when unrecorded."""
    n = (((leg.get("system") or {}).get("during") or {}).get("resident_unknown_samples"))
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None


def _leg_embeddings(leg: dict) -> str | None:
    """"on" or "off": whether the leg's omp children ran fastembed recall, read from the memory overlay its mem/sess
    details name (`config`; mnemopi.noEmbeddings, omp's default false) after checking that file still hashes to the
    leg's omp_mem_config pin. None when no single config is recorded, the file is gone or changed, or unreadable."""
    from .backends import sha16  # backends imports this module's callers
    paths = {d["config"] for d in _leg_details(leg).values() if isinstance(d.get("config"), str)}
    pin = ((leg.get("provenance") or {}).get("pins") or {}).get("omp_mem_config")
    if len(paths) != 1 or not pin:
        return None
    path = next(iter(paths))
    try:
        if sha16(path) != pin:
            return None
        section = mem_overlay(Path(path)).get("mnemopi", {})
    except (OSError, ValueError):
        return None
    if not isinstance(section, dict):
        return None
    return {"true": "off", "false": "on"}.get(section.get("noEmbeddings", "false"))


def _leg_memory(leg: dict) -> dict:
    """Which memory configuration a leg ran: the llmMode its mem/sess results record (None unless exactly one), the
    memory model (none under llmMode none, else the declared smol model, else the model under test), embeddings
    (_leg_embeddings) and the models that actually served its memory calls."""
    prov = leg.get("provenance") or {}
    pins = prov.get("pins") or {}
    modes = {d["llm_mode"] for d in _leg_details(leg).values() if d.get("llm_mode")}
    mode = next(iter(modes)) if len(modes) == 1 else None
    if mode == "none":
        model, digest = None, None
    elif pins.get("smol_model"):
        model, digest = pins["smol_model"], pins.get("smol_digest")
    else:
        model, digest = pins.get("model"), pins.get("model_digest")
    return {"label": prov.get("label"), "llm_mode": mode, "backend": pins.get("backend"), "model": model,
            "digest": digest, "embeddings": _leg_embeddings(leg), "embedder": pins.get("embedder"),
            "embedder_digest": pins.get("embedder_digest"), "served": _served_memory_models(leg)}


def _served_problem(side: str, m: dict, also: frozenset[str] = frozenset()) -> str | None:
    """A leg whose memory calls were not all served by its pinned memory model (or a name in `also`, another name of
    the same build), or by no model at all; under llmMode none, a leg with any memory call."""
    served = m["served"]
    if served is None:
        return (f"{side} leg {m['label']} records no models for its memory calls (no sess.memory by_purpose[*].models "
                "in its results or banked details): refused")
    accepted = set() if m["llm_mode"] == "none" else {str(m["model"]), *also}
    if served - accepted or (accepted and not served):
        return (f"{side} leg {m['label']} memory calls served by {sorted(served) or 'no model'}, "
                f"not its pinned memory model {sorted(accepted) or 'none (llmMode none)'}")
    return None


# The ledger's stage-2 loop gate (NEGATIVE_EVIDENCE 2026-09-28 DROP row): a candidate turn may make at most this many
# times the largest baseline leg's tool calls in the same tier, and no candidate turn may end at the turn timeout.
LOOP_FACTOR = 3


def _loop_gate(candidate_legs: list[dict], baseline_legs: list[dict], label) -> tuple[list[str], list[str]]:
    """(failures, unmeasured) of the loop gate per tier (mem, sess). A missing or VOID metric on any leg the gate
    reads is unmeasured: a problem, never a pass."""
    failures, unmeasured = [], []

    def value(leg: dict, key: str) -> float | None:
        m = (leg.get("metrics") or {}).get(key) or {}
        return None if m.get("void") else _real_value(m.get("value"))

    for tier in ("mem", "sess"):
        calls, timeouts = f"{tier}.turn.max_tool_calls", f"{tier}.turn.timeouts"
        base = [value(leg, calls) for leg in baseline_legs]
        if not base or any(v is None for v in base):
            unmeasured.append(f"{calls} (baseline)")
            bound = None
        else:
            bound = LOOP_FACTOR * max(base)
        for leg in candidate_legs:
            got, timed_out = value(leg, calls), value(leg, timeouts)
            if got is None:
                unmeasured.append(f"{calls} ({label(leg)})")
            elif bound is not None and got > bound:
                failures.append(f"{label(leg)} {calls} {got:g} > {LOOP_FACTOR}x largest baseline leg ({bound:g})")
            if timed_out is None:
                unmeasured.append(f"{timeouts} ({label(leg)})")
            elif timed_out > 0:
                failures.append(f"{label(leg)} {timeouts} {timed_out:g} > 0")
    return failures, unmeasured


def _memory_row(c_vals: list, b_vals: list, better: str, eps: float) -> dict:
    """decision.compare's A/A rule on leg values: each arm's median; the band is the wider of the two arms' spreads
    (max - min over legs); a gain or loss must clear it. Fewer than two legs per arm, or a leg without the value, is
    unmeasured: no noise estimate, no judgement."""
    row = {"candidate_legs": c_vals, "baseline_legs": b_vals, "better": better, "judgement": "unmeasured"}
    if len(c_vals) < 2 or len(b_vals) < 2 or any(v is None for v in c_vals + b_vals):
        return row
    cv, bv = statistics.median(c_vals), statistics.median(b_vals)
    band = max(max(c_vals) - min(c_vals), max(b_vals) - min(b_vals))
    gain = cv - bv if better == "higher" else bv - cv
    return {**row, "candidate": cv, "baseline": bv, "delta": cv - bv, "band": band,
            "judgement": "gain" if gain > band + eps else "loss" if gain < -(band + eps) else "within_noise"}


def memory_verdict(candidate_legs: list[dict], baseline_legs: list[dict], *, feature: str, omp_module_sha: str,
                   baseline: dict | None = None, rev: str | None = None) -> dict:
    """A memory proof receipt (features.grade's contract, run.label "memory") from interleaved mem+sess legs of a
    candidate memory configuration and of the baseline route, each arm >= 2 legs (its A/A null). Verdict, as
    decision.compare: WORSE on a MEMORY_QUALITY loss beyond noise; BETTER with no loss, every quality metric
    measured, and a MEMORY_WINS gain beyond noise; else NOT_BETTER. NONE when the comparison is void: a leg of either
    arm contended (memory runs are chat-model runs: anything resident beyond the declared main and smol models is a
    contender) or with unknown or unrecorded residency, or a baseline MUST not PASS. run.provenance.pins.model_digest
    is the candidate memory model's digest (main model kept as main_model/main_model_digest). Problems, any of which
    keeps the receipt evidence rather than proof: not BETTER, a void comparison, a quality metric unmeasured, a
    candidate MUST not PASS, legs whose memory configuration is unrecorded or differs within an arm (pins included),
    memory calls not served by exactly the pinned memory model, a baseline leg that did not run the baseline route, a
    pin that moved mid-leg, a remote candidate, and an llmMode none candidate whose legs made any memory-model call.
    llmMode none proves a route that targets no model: pins.route {kind fixed, id none} and pins.model_calls 0 (the
    count of memory-model calls), graded by features.grade against a profile whose route is off at `none`.
    A sess.turns_complete FAIL caused only by main-model acknowledgement slips (_ack_only_slips) is not a MUST
    failure: it is the SHOULD diagnostic memory.main_ack_slips {arm: {leg: slipped turns}}.
    For side-regime legs (side_regime: pins.regime side, created >= SIDE_REGIME_FROM) a CONTENDED leg or unknown
    residency does not void: it is the SHOULD diagnostic memory.co_resident {arm: {leg: _co_resident}}; every other
    leg keeps the one-model rule, and legs mixing the two regimes are a problem.
    The baseline arm (`baseline`, else inferred from the baseline legs) is the incumbent route MEMORY_BASELINE (legs on
    its digest with llmMode smol) or a declared alternative of MEMORY_ALTERNATIVES (every baseline leg runs its
    configuration, no candidate leg does, and the arms agree on the other memory settings: embeddings for memory:none,
    llmMode and memory-model digest for embeddings:fts), so the same legs
    score both directions: qwen3.8 vs llmMode none, and llmMode none vs qwen3.8. pins.embeddings is the candidate legs'
    "on"/"off" (_leg_embeddings); for the incumbent route the arms must agree on it.
    The receipt has decision.run_suite's run shape (`localbench validate` / `show`): provenance created (UTC, now),
    localbench_rev (`rev`, else __main__._rev()), fingerprint (the candidate legs' shared pins plus each leg's source),
    run_dir None, and conformance: this verdict's own checks as MUST cases (memory.baseline_route, .served_model,
    .comparison_valid, .loop_gate, .llm_mode), PASS or FAIL with their values, and memory.main_ack_slips (SHOULD)."""
    from . import golden, park
    from .decision import EPS
    from .features import same_digest

    problems: list[str] = []
    if len(candidate_legs) < 2 or len(baseline_legs) < 2:
        problems.append(f"A/A noise needs >= 2 legs per arm (candidate {len(candidate_legs)}, "
                        f"baseline {len(baseline_legs)})")
    deltas, losses, wins, unmeasured = {}, [], [], []
    for key in MEMORY_QUALITY:
        deltas[key] = {"class": "quality", **_memory_row([_memory_leg_value(leg, key) for leg in candidate_legs],
                                                         [_memory_leg_value(leg, key) for leg in baseline_legs],
                                                         "higher", EPS)}
        if deltas[key]["judgement"] == "loss":
            losses.append(key)
        elif deltas[key]["judgement"] == "unmeasured":
            unmeasured.append(key)
    for key, label in MEMORY_WINS:
        deltas[key] = {"class": "win", **_memory_row([_memory_leg_value(leg, key) for leg in candidate_legs],
                                                     [_memory_leg_value(leg, key) for leg in baseline_legs],
                                                     "lower", EPS)}
        if deltas[key]["judgement"] == "gain":
            wins.append(label)
    computed = "WORSE" if losses else "BETTER" if wins and not unmeasured else "NOT_BETTER"
    if losses:
        problems.append(f"quality loss beyond A/A noise: {', '.join(losses)}")
    if unmeasured:
        problems.append(f"quality unmeasured (a leg lacks it, or < 2 legs per arm): {', '.join(unmeasured)}")

    def label(leg: dict) -> str:
        return str((leg.get("provenance") or {}).get("label"))

    loop_failures, loop_unmeasured = _loop_gate(candidate_legs, baseline_legs, label)
    if loop_failures:
        computed = "WORSE"
        problems.append(f"tool-call loop gate failed: {'; '.join(loop_failures)}")
    if loop_unmeasured:
        problems.append(f"tool-call loop gate unmeasured: {', '.join(loop_unmeasured)}")

    def must_failures(legs: list[dict]) -> list[str]:
        # An ack-only sess.turns_complete FAIL is memory.main_ack_slips (below), not a MUST failure.
        return sorted(f"{label(leg)}:{case}" for leg in legs for case, entry in (leg.get("conformance") or {}).items()
                      if entry.get("level") == "MUST" and entry.get("verdict") != "PASS"
                      and not (case == "sess.turns_complete" and _ack_only_slips(leg) is not None))

    def slips(leg: dict) -> int | None:
        """Slipped turns per leg: 0 on a passing sess.turns_complete, None when it failed for another cause."""
        entry = (leg.get("conformance") or {}).get("sess.turns_complete") or {}
        return 0 if entry.get("verdict") == "PASS" else _ack_only_slips(leg)

    ack_slips = {side: {label(leg): slips(leg) for leg in legs}
                 for side, legs in (("candidate", candidate_legs), ("baseline", baseline_legs))}

    must_fail, baseline_must_fail = must_failures(candidate_legs), must_failures(baseline_legs)
    if must_fail:
        problems.append(f"MUST FAIL: {', '.join(must_fail)}")
    void: list[str] = []
    if baseline_must_fail:
        void.append(f"baseline MUST FAIL: {', '.join(baseline_must_fail)}")
    # Side-regime legs (side_regime) record co-resident models and unreadable residency as the diagnostic
    # memory.co_resident; every other leg keeps the one-model rule, where either voids the comparison.
    co_resident: dict[str, dict[str, dict]] = {}
    for side, legs in (("candidate", candidate_legs), ("baseline", baseline_legs)):
        strict = [leg for leg in legs if not side_regime(leg)]
        co_resident[side] = {label(leg): _co_resident(leg) for leg in legs if side_regime(leg)}
        contended = [label(leg) for leg in strict if (leg.get("verdicts") or {}).get("contended")]
        if contended:
            void.append(f"{side} leg(s) {', '.join(contended)} CONTENDED: a model beyond the declared main and smol "
                        "was resident or running")
        unknown_residency = [label(leg) for leg in strict if _resident_unknown(leg) != 0]
        if unknown_residency:
            void.append(f"{side} leg(s) {', '.join(unknown_residency)} have unknown or unrecorded resident-model "
                        "state")
    problems += [f"comparison void: {reason}" for reason in void]
    verdict = "NONE" if void else computed
    regimes = {label(leg): SIDE_REGIME if side_regime(leg) else "one-model" for leg in candidate_legs + baseline_legs}
    if len(set(regimes.values())) > 1:
        problems.append("legs mix measurement regimes (side-model vs one-model): "
                        + "; ".join(f"{k}={v}" for k, v in regimes.items()))
    cand, base = [_leg_memory(leg) for leg in candidate_legs], [_leg_memory(leg) for leg in baseline_legs]
    if baseline is None:
        # The baseline arm is the incumbent route, or the declared alternative whose configuration every baseline leg
        # recorded and no candidate leg did (memory:none: llm_mode none; embeddings:fts: embeddings off): with both
        # arms on llmMode none, an embeddings-off arm is embeddings:fts, not memory:none.
        def runs(legs: list[dict], a: dict) -> bool:
            return bool(legs) and all(m[a["leg"][0]] == a["leg"][1] for m in legs)
        baseline = next((a for a in MEMORY_ALTERNATIVES if runs(base, a) and not any(runs([m], a) for m in cand)),
                        next((a for a in MEMORY_ALTERNATIVES if runs(base, a)), MEMORY_BASELINE))
    base_backend, _, base_model = baseline["id"].partition("/")
    base_digest = baseline.get("digest")

    def base_names(m: dict) -> frozenset[str]:
        """Names of the baseline build a digest-matched baseline leg's memory calls may carry: the route's model
        and its park alias."""
        if not same_digest(m["digest"], base_digest):
            return frozenset()
        return frozenset({base_model, park.PREFIX + base_digest.lower().removeprefix("sha256:")[:12]})

    served_problems, mode_problems = [], []
    for side, legs in (("candidate", cand), ("baseline", base)):
        unknown = [m["label"] for m in legs if m["llm_mode"] is None]
        if unknown:
            mode_problems.append(f"{side} leg(s) {', '.join(map(str, unknown))} record no single mnemopi llmMode (no "
                                 "llm_mode in mem/sess results or banked details): refused")
        served_problems += [p for m in legs
                            if (p := _served_problem(side, m, base_names(m) if side == "baseline" else frozenset()))]
    if len({(m["llm_mode"], m["model"], m["digest"]) for m in cand}) > 1:
        problems.append("candidate legs ran different memory configurations: "
                        + "; ".join(f"{m['label']}={m['llm_mode']}/{m['model']}@{m['digest']}" for m in cand))
    first = cand[0] if cand else {"llm_mode": None, "model": None, "digest": None}
    if first["llm_mode"] == "remote":
        mode_problems.append("llmMode remote is not a local route: its memory calls leave this host's models")
    elif first["llm_mode"] == "none":
        # The no-model route's pin: zero memory-model calls in every candidate leg (sess.memory by_purpose).
        counts = [_memory_call_count(leg) for leg in candidate_legs]
        model_calls = None if any(c is None for c in counts) else sum(counts)
        if model_calls != 0:
            mode_problems.append(f"llmMode none candidate legs made {model_calls!r} memory-model calls (None: "
                                 "unrecorded); the no-model route's pin is 0")
    elif not first["digest"]:
        problems.append(f"candidate memory model {first['model']!r} has no digest in the run pins")
    problems += mode_problems + served_problems
    route_problems: list[str] = []
    if "leg" in baseline:
        key, value = baseline["leg"]
        off_route = [m["label"] for m in base if m[key] != value]
        if off_route:
            problems.append(f"baseline leg(s) {', '.join(map(str, off_route))} did not run the declared alternative "
                            f"{baseline['preset']} ({key} {value})")
        own = [m["label"] for m in cand if m[key] == value]
        if own:
            route_problems.append(f"candidate leg(s) {', '.join(map(str, own))} ran the baseline alternative's own "
                                  f"configuration ({key} {value})")
        # Only the alternative's own setting may differ between the arms. Under llmMode none in every leg there is no
        # memory model, so its digest is None on both sides and equal.
        no_model = all(m["llm_mode"] == "none" for m in cand + base)
        for other in (("embeddings",) if key == "llm_mode" else ("llm_mode", "digest")):
            seen = {m[other] for m in cand + base}
            if len(seen) > 1 or (None in seen and not (other == "digest" and no_model)):
                route_problems.append(f"the arms differ in {other}, or a leg does not record it "
                                      f"({' / '.join(sorted(map(str, seen)))}): the comparison is not {key} alone")
    else:
        off_route = [m["label"] for m in base if (m["backend"], m["llm_mode"]) != (base_backend, "smol")
                     or not same_digest(m["digest"], base_digest)]
        if off_route:
            problems.append(f"baseline leg(s) {', '.join(map(str, off_route))} did not run {baseline['id']} "
                            f"(digest {base_digest}, under any name) with llmMode smol")
        seen = {m["embeddings"] for m in cand + base}
        if None in seen or len(seen) > 1:
            route_problems.append(f"the arms differ in embeddings, or a leg does not record it "
                                  f"({' / '.join(sorted(map(str, seen)))}): the comparison is not the memory route "
                                  "alone")
    # An embeddings comparison (baseline embeddings:fts) proves recall-embeddings: its model is the candidate legs'
    # embedding model, pinned by name and on-disk digest (embedding_pins), which every candidate leg must share.
    embedder = None
    if baseline.get("preset") == "embeddings:fts":
        seen = {(m["embedder"], m["embedder_digest"]) for m in cand}
        embedder = next(iter(seen)) if len(seen) == 1 else None
        if embedder is None or None in embedder:
            route_problems.append(f"candidate legs pin no single embedding model with a digest "
                                  f"({' / '.join(sorted(f'{e}@{d}' for e, d in seen))})")
            embedder = None
    problems += route_problems
    all_pins = [[(leg.get("provenance") or {}).get("pins") or {} for leg in legs]
                for legs in (candidate_legs, baseline_legs)]
    for tier, reason in sorted(golden.arm_pin_drift(*all_pins, ["mem", "sess"]).items()):
        problems.append(f"{tier}: {reason} (arm A = candidate, B = baseline)")
    for side, legs, pins in (("candidate", candidate_legs, all_pins[0]), ("baseline", baseline_legs, all_pins[1])):
        for key in ("smol_model", "smol_digest", "embedder", "embedder_digest"):
            seen = list(dict.fromkeys(str(p.get(key)) for p in pins))
            if len(seen) > 1:
                problems.append(f"{side} legs ran different {key}: {' / '.join(seen)}")
        moved = {label(leg): (leg.get("verdicts") or {}).get("pins_changed") for leg in legs
                 if (leg.get("verdicts") or {}).get("pins_changed")}
        if moved:
            problems.append(f"PINS CHANGED mid-run in {side} leg(s): {moved}")
    against = {"kind": baseline["kind"], "id": baseline["id"]}
    if verdict != "BETTER":
        problems.append(f"not better than {against['kind']} {against['id']}: {verdict}")

    main_pins = all_pins[0][0] if candidate_legs else {}
    pins = {**main_pins, "model": first["model"], "model_digest": first["digest"],
            "main_model": main_pins.get("model"), "main_model_digest": main_pins.get("model_digest"),
            "embeddings": next(iter(e)) if len(e := {m["embeddings"] for m in cand}) == 1 else None,
            # llmMode none proves a route that targets no model: the registry row's route off at setting value
            # `none` (features.grade, kind fixed as in a fixed baseline), pinned by its zero model calls.
            **({"route": {"kind": "fixed", "id": "none"}, "model_calls": model_calls}
               if first["llm_mode"] == "none" and baseline.get("preset") != "embeddings:fts" else {}),
            # An embeddings receipt proves the embedding model; the memory model it ran beside is kept apart.
            **({"model": embedder[0], "model_digest": embedder[1], "memory_model": first["model"],
                "memory_model_digest": first["digest"]} if embedder else {})}
    comparison = {"verdict": verdict, "computed": computed, "void": void, "quality_losses": losses, "wins": wins,
                  "baseline": baseline,
                  "unmeasured_quality": unmeasured,
                  "noise": "aa_spread_over_legs", "deltas": deltas}

    def check(failed, level: str = "MUST", **values) -> dict:
        return {"level": level, "verdict": "FAIL" if failed else "PASS", **values}

    conformance = {
        "memory.baseline_route": check(off_route or route_problems, baseline=baseline["id"], digest=base_digest,
                                       off_route=off_route, problems=route_problems),
        "memory.served_model": check(served_problems, problems=served_problems),
        "memory.comparison_valid": check(void, void=void),
        "memory.loop_gate": check(loop_failures or loop_unmeasured, factor=LOOP_FACTOR, failures=loop_failures,
                                  unmeasured=loop_unmeasured),
        "memory.llm_mode": check(mode_problems, candidate=first["llm_mode"], problems=mode_problems),
        # Diagnostic (SHOULD): main-model acknowledgement slips are reported, not voided (pre-registered 2026-10-01).
        "memory.main_ack_slips": check(any(n for arm in ack_slips.values() for n in arm.values()), level="SHOULD",
                                       slips=ack_slips),
        # Diagnostic (SHOULD): co-resident load of side-regime legs, recorded, never a void (DECISION 2026-10-02).
        "memory.co_resident": check(any(c["models"] or c["resident_unknown_samples"]
                                        for arm in co_resident.values() for c in arm.values()),
                                    level="SHOULD", legs=co_resident, regimes=regimes),
    }
    if rev is None:
        from localbench.__main__ import _rev  # lazy: __main__ imports this module
        rev = _rev()

    def source(arm: str, leg: dict) -> dict:
        prov = leg.get("provenance") or {}
        return {"arm": arm, "label": prov.get("label"), "run_dir": leg.get("run_dir"), "created": prov.get("created"),
                "localbench_rev": prov.get("localbench_rev")}

    shared = {k: v for k, v in main_pins.items() if all(p.get(k) == v for p in all_pins[0])}
    run = {
        "label": "memory",
        "provenance": {"pins": pins, "label": "memory", "tiers": ["mem", "sess"], "llm_mode": first["llm_mode"],
                       "created": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "localbench_rev": rev,
                       "fingerprint": {**shared, "legs": [source("candidate", leg) for leg in candidate_legs]
                                       + [source("baseline", leg) for leg in baseline_legs]},
                       "legs": {"candidate": [label(leg) for leg in candidate_legs],
                                "baseline": [label(leg) for leg in baseline_legs]}},
        # Contended or residency-unknown legs void the comparison above (verdict NONE); listed here per leg.
        "verdicts": {"must_fail": must_fail, "baseline_must_fail": baseline_must_fail,
                     "contended": [label(leg) for leg in candidate_legs + baseline_legs
                                   if (leg.get("verdicts") or {}).get("contended")]},
        "metrics": {k: {"value": row.get("candidate"), "better": row["better"], "n": len(candidate_legs)}
                    for k, row in deltas.items()},
        "conformance": conformance,
        "run_dir": None,
        "memory": {"compare": comparison},
    }
    return {"kind": "run", "feature": feature, "omp_module_sha": omp_module_sha, "problems": problems,
            "verdict": {"compare": verdict, "baseline": against}, "run": run}


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
