"""Transparent OpenAI-compatible proxy: forwards to an upstream `/v1`, streams the
response through untouched, and logs per-call timing (and optionally the request
body) for every chat completion omp makes.

omp reaches it as the `localbench` provider in ~/.omp/agent/models.yml
(baseUrl http://127.0.0.1:11299/v1), so `omp --model localbench/<id>` exercises
the exact request shape omp sends to local backends, with TTFT per LLM call.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Self

PORT = 11299
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-authenticate", "content-length", "host"}


def _has_token(ev: dict) -> bool:
    """Same rule as client.chat_stream: a non-empty content/reasoning delta or a tool-call delta."""
    for ch in ev.get("choices") or []:
        delta = ch.get("delta") or {}
        if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning")) or delta.get("tool_calls"):
            return True
    return False


# Markers of omp's side calls, taken from request bodies omp 18.2.11 sent through this proxy (2026-09-23,
# save_dir) and from its prompt files: (purpose, turn the marker is sent in, marker). The auto-thinking classifier is
# one judge question, so it is matched first. mnemopi's extraction sends its instructions as the system turn
# (src/prompts/system/memory-extraction-system.md); consolidation renders memory-consolidation-system.md into the user
# turn with no system turn (mnemopi/backend.ts resolveMemoryCompletionInput). A side call no marker matches stays
# "aux" rather than being guessed.
SIDE_CALLS = (("auto-thinking", "system", "Choose the reasoning effort this turn needs"),
              ("judge", "system", "The state is untrusted data to judge."),
              ("memory-extract", "system", "You are a precise long-term memory extractor."),
              ("memory-consolidate", "user", "Summarize memories in 1-3 concise sentences."))


def purpose(meta: dict) -> str:
    """The side call a system-turn marker identifies, else `main` for turns that carry omp's tools, else the side call
    a user-turn marker identifies, else `aux`. System markers come first because omp's judge retries an answer that
    is not a label with a forced `submit_judgment` tool (omp 18.3.1, pi-ai judgment/chat.ts; seen through a stub
    server, 2026-09-26): that retry is still the effort classifier, not main work, and not a turn's answer call.
    omp's main-turn system prompt carries none of these markers; a user turn can quote memory text, so user-turn
    markers only name tool-free calls."""
    text = {role: " ".join(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
                           for m in meta.get("messages") or [] if m.get("role") == role)
            for role in ("system", "user")}
    hit = next((name for name, where, marker in SIDE_CALLS if where == "system" and marker in text["system"]), None)
    if hit:
        return hit
    if meta.get("tools"):
        return "main"
    return next((name for name, where, marker in SIDE_CALLS if where == "user" and marker in text["user"]), "aux")


def make_handler(upstream: str, calls_log: Path, save_dir: Path | None, label: str):
    lock = threading.Lock()
    counter = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # quiet
            pass

        def _forward(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            path = self.path.removeprefix("/v1")
            req = urllib.request.Request(upstream.rstrip("/") + path, data=body, method=method)
            for k, v in self.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    req.add_header(k, v)
            is_chat = method == "POST" and path.endswith("/chat/completions")
            body_file = None
            response_file = None
            response_path = None
            if is_chat and save_dir is not None:
                with lock:
                    counter["n"] += 1
                    n = counter["n"]
                save_dir.mkdir(parents=True, exist_ok=True)
                body_file = f"{label}-{n:02d}.json"
                response_file = f"{label}-{n:02d}.response.jsonl"
                response_path = save_dir / response_file
                (save_dir / body_file).write_bytes(body or b"{}")
            t0 = time.perf_counter()
            t_start = time.time()
            t_first = None
            nbytes = 0
            usage = {}
            try:
                resp = urllib.request.urlopen(req, timeout=3600)
            except urllib.error.HTTPError as err:
                resp = err
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            aborted = False
            stream_complete = False
            response_stream = response_path.open("wb") if response_path is not None else None
            try:
                for line in iter(resp.readline, b""):
                    if response_stream is not None:
                        response_stream.write(line)
                    if is_chat and line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]"):
                        try:
                            ev = json.loads(line[5:])
                        except json.JSONDecodeError:
                            ev = {}
                        usage = ev.get("usage") or usage
                        if t_first is None and _has_token(ev):
                            t_first = time.perf_counter()
                    nbytes += len(line)
                    self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                stream_complete = True
            except (BrokenPipeError, ConnectionResetError):
                # omp dropped the stream mid-response. Record the aborted call and close the
                # upstream connection; whether generation stops depends on the backend.
                aborted = True
                self.close_connection = True
            finally:
                if response_stream is not None:
                    response_stream.flush()
                    os.fsync(response_stream.fileno())
                    response_stream.close()
                resp.close()
            if is_chat:
                meta = json.loads(body or b"{}")
                row = {
                    "t": time.time(), "t_start": round(t_start, 3), "label": label, "model": meta.get("model"),
                    "messages": len(meta.get("messages") or []), "tools": len(meta.get("tools") or []),
                    "purpose": purpose(meta), "enable_thinking": meta.get("enable_thinking"),
                    "status": resp.status, "aborted": aborted,
                    "ttft_s": round(t_first - t0, 3) if t_first else None,
                    "total_s": round(time.perf_counter() - t0, 3),
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "body": body_file, "response_body": response_file, "response_complete": stream_complete,
                }
                with lock, calls_log.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")

        def do_GET(self) -> None:
            self._forward("GET")

        def do_POST(self) -> None:
            self._forward("POST")

    return Handler


class _QuietServer(ThreadingHTTPServer):
    """A client closing its keep-alive socket is routine for omp; only unexpected handler errors are printed."""

    def handle_error(self, request, client_address) -> None:
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


class Proxy:
    """`with Proxy(upstream, calls_log): ...` runs the proxy on a daemon thread."""

    def __init__(self, upstream: str, calls_log: Path, save_dir: Path | None = None, label: str = "call",
                 port: int = PORT):
        self.server = _QuietServer(("127.0.0.1", port), make_handler(upstream, calls_log, save_dir, label))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
