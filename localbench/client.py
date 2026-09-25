"""OpenAI-compatible streaming client that measures what a user feels.

One request -> one Sample: time-to-first-token (any token: reasoning, content,
or tool call), prefill throughput implied by TTFT, and steady decode rate.
Works against any `/v1/chat/completions` server (Ollama, mlx-serve, llama.cpp).
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field


@dataclass
class Sample:
    backend: str
    model: str
    label: str
    prompt_tokens: int = 0
    cached_tokens: int | None = None
    completion_tokens: int = 0
    ttft_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    finish_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    text: str = ""
    reasoning: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def chat_stream(
    base_url: str,
    body: dict,
    *,
    backend: str,
    label: str,
    api_key: str = "local",
    timeout: float = 1800,
) -> Sample:
    """POST a chat completion with stream=true and time every chunk."""
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    s = Sample(backend=backend, model=body.get("model", ""), label=label)
    parts: list[str] = []
    thought: list[str] = []
    calls: dict[int, dict] = {}
    token_chunks = 0
    t0 = time.perf_counter()
    t_first = t_last = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                usage = ev.get("usage")
                if usage:
                    s.prompt_tokens = usage.get("prompt_tokens") or s.prompt_tokens
                    s.completion_tokens = usage.get("completion_tokens") or s.completion_tokens
                    details = usage.get("prompt_tokens_details") or {}
                    if "cached_tokens" in details:
                        s.cached_tokens = details["cached_tokens"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    got = False
                    for key in ("content", "reasoning_content", "reasoning"):
                        if delta.get(key):
                            got = True
                            if key == "content":
                                parts.append(delta[key])
                            else:
                                thought.append(delta[key])
                    for tc in delta.get("tool_calls") or []:
                        got = True
                        slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if got:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        token_chunks += 1
                    if ch.get("finish_reason"):
                        s.finish_reason = ch["finish_reason"]
    except Exception as exc:  # noqa: BLE001 - recorded, never raised: a failed sample is data
        s.error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    s.total_s = t_end - t0
    s.text = "".join(parts)
    s.reasoning = "".join(thought)
    s.tool_calls = [calls[i] for i in sorted(calls)]
    if not s.completion_tokens:
        s.completion_tokens = token_chunks
    if t_first is not None:
        s.ttft_s = t_first - t0
        s.decode_s = (t_last or t_first) - t_first
        if s.prompt_tokens and s.ttft_s > 0:
            fresh = s.prompt_tokens - (s.cached_tokens or 0)
            s.prefill_tps = fresh / s.ttft_s
        if s.completion_tokens > 1 and s.decode_s > 0:
            s.decode_tps = (s.completion_tokens - 1) / s.decode_s
    return s
