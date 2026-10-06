"""OpenAI-compatible streaming client that measures what a user feels.

One request -> one Sample: time-to-first-token (any token: reasoning, content,
or tool call), prefill throughput implied by TTFT, and steady decode rate.
Works against any `/v1/chat/completions` server (Ollama, mlx-serve, llama.cpp).
"""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field


class RunAborted(RuntimeError):
    """A run watchdog stopped an active local request."""


class RequestCancellation:
    def __init__(self):
        self._lock = threading.Lock()
        self._response = None
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def register(self, response) -> None:
        with self._lock:
            self._response = response
            cancelled = self._reason is not None
        if cancelled:
            self._interrupt(response)

    def unregister(self, response) -> None:
        with self._lock:
            if self._response is response:
                self._response = None

    def cancel(self, reason: str) -> None:
        with self._lock:
            if self._reason is not None:
                return
            self._reason = reason
            response = self._response
        if response is not None:
            self._interrupt(response)

    @staticmethod
    def _interrupt(response) -> None:
        raw = getattr(getattr(response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        response.close()

    def raise_if_cancelled(self) -> None:
        reason = self.reason
        if reason is not None:
            raise RunAborted(reason)


def run_cancellable(args: list[str], *, cancellation: RequestCancellation | None = None, **kwargs
                    ) -> subprocess.CompletedProcess:
    """Run a child process while allowing a sampler thread to request prompt termination."""
    timeout = kwargs.pop("timeout", None)
    deadline = time.monotonic() + timeout if timeout is not None else None
    process = subprocess.Popen(args, **kwargs)
    try:
        while True:
            if cancellation:
                cancellation.raise_if_cancelled()
            wait = min(1.0, max(0.0, deadline - time.monotonic())) if deadline is not None else (
                1.0 if cancellation else None)
            try:
                stdout, stderr = process.communicate(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    raise
        if cancellation:
            cancellation.raise_if_cancelled()
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.communicate()
        raise



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
    cancellation: RequestCancellation | None = None,
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
    if cancellation:
        cancellation.raise_if_cancelled()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if cancellation:
                cancellation.register(resp)
            try:
                for raw in resp:
                    if cancellation:
                        cancellation.raise_if_cancelled()
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
            finally:
                if cancellation:
                    cancellation.unregister(resp)
    except RunAborted:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded, never raised: a failed sample is data
        if cancellation and cancellation.reason is not None:
            raise RunAborted(cancellation.reason) from exc
        s.error = f"{type(exc).__name__}: {exc}"
    if cancellation:
        cancellation.raise_if_cancelled()
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
