"""Loopback-only, canned OMP API server for offline request-shape captures."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_REPLY = "OK"
MAX_BODY_BYTES = 128 * 1024 * 1024
_SERVERS: dict[str, MockOmpServer] = {}
_SERVERS_LOCK = threading.Lock()


@dataclass(frozen=True)
class RequestRecord:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def json(self) -> dict:
        value = json.loads(self.body)
        if not isinstance(value, dict):
            raise ValueError("captured request body is not a JSON object")
        return value


def _systemone_response(request: dict, preferred_choice: str, noul_score: float) -> dict:
    answers = {}
    for name, question in request.get("questions", {}).items():
        kind = question.get("type")
        if kind == "noul":
            answers[name] = {"type": "noul", "noul": noul_score}
            continue
        if kind == "choice":
            options = list(question.get("criteria", {}))
        elif kind == "score":
            options = [str(index) for index, _ in enumerate(question.get("criteria", []))]
        else:
            continue
        if not options:
            continue
        selected = preferred_choice if preferred_choice in options else options[0]
        rest = (1.0 - 0.9) / (len(options) - 1) if len(options) > 1 else 0.0
        probabilities = {option: 0.9 if option == selected else rest for option in options}
        if kind == "choice":
            answers[name] = {"type": kind, "choice": selected, "probabilities": probabilities, "confidence": 0.9}
        else:
            answers[name] = {"type": kind, "score": float(options.index(selected)),
                             "legend": dict(zip(options, question["criteria"], strict=True)),
                             "probabilities": probabilities, "confidence": 0.9}
    return {"model": request.get("model", "localbench/mock"), "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def _chat_message(request: dict, server) -> tuple[dict, str]:
    tools = request.get("tools", [])
    has_find = any((tool.get("function") or {}).get("name") == "find" for tool in tools)
    prior_find_call = any(message.get("role") == "assistant" and message.get("tool_calls")
                          for message in request.get("messages", []))
    if has_find and not prior_find_call:
        arguments = json.dumps({"query": "localbench behavior", "grep_keywords": ["feature", "route"]},
                               separators=(",", ":"))
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_mock_find", "type": "function",
             "function": {"name": "find", "arguments": arguments}}]}, "tool_calls"
    text = server.canned_responses.get(request.get("model"), server.reply)
    return {"role": "assistant", "content": text}, "stop"


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, reply: str, systemone_choice: str, systemone_noul_score: float,
                 canned_responses: dict[str, str]):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.reply = reply
        self.systemone_choice = systemone_choice
        self.systemone_noul_score = systemone_noul_score
        self.canned_responses = canned_responses
        self.records: list[RequestRecord] = []
        self.records_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "localbench/mock", "model": "localbench/mock"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            self._json(400, {"error": "invalid content length"})
            return
        if not 0 <= length <= MAX_BODY_BYTES:
            self._json(413, {"error": "request body size is invalid or too large"})
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._json(400, {"error": "truncated request body"})
            return
        server = cast(_Server, self.server)
        record = RequestRecord("POST", self.path, dict(self.headers.items()), body)
        with server.records_lock:
            server.records.append(record)

        if self.path == "/api/tags/show":
            self._json(200, {"modelfile": "FROM localbench/mock", "parameters": "", "details": {"family": "mock"}})
        elif self.path == "/v1/systemone":
            try:
                self._json(200, _systemone_response(json.loads(body), server.systemone_choice,
                                                    server.systemone_noul_score))
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                self._json(400, {"error": "invalid System One request"})
        elif self.path in ("/v1/chat/completions", "/chat/completions"):
            try:
                request_data = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(400, {"error": "request body must be JSON"})
                return
            message, finish_reason = _chat_message(request_data, server)
            model = request_data.get("model", "localbench/mock")
            if request_data.get("stream"):
                self._stream(model, message, finish_reason)
            else:
                self._json(200, {
                    "id": "chatcmpl-localbench-mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })
            return
        else:
            self._json(404, {"error": "not found"})

    def _stream(self, model: str, message: dict, finish_reason: str) -> None:
        delta = {"role": "assistant"}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        else:
            delta["content"] = message.get("content", "")
        chunks = [
            {"id": "chatcmpl-localbench-mock", "object": "chat.completion.chunk", "created": 0,
             "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"id": "chatcmpl-localbench-mock", "object": "chat.completion.chunk", "created": 0,
             "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
        ]
        payload = b"".join(b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n"
                          for chunk in chunks) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        del format, args


class MockOmpServer:
    """Threaded HTTP server bound to IPv4 loopback, retaining every POST body byte-for-byte."""

    def __init__(self, reply: str = DEFAULT_REPLY, systemone_choice: str = "high",
                 systemone_noul_score: float = 1.0, canned_responses: dict[str, str] | None = None):
        # Keys are wire model ids: omp sends its models.yml entry id ("tiny"), not
        # the provider-qualified name. The <title> markers let omp's title parser
        # accept the first attempt, keeping title captures to one request.
        canned = {"localbench/smol": '{"facts":["fixture marker is cobalt"],"instructions":[],"preferences":[],'
                 '"timelines":[],"kg":[]}',
                 "localbench/tiny": '{"title":"Mock OMP session"}',
                 "tiny": "<title>Mock OMP session</title>"}
        if canned_responses:
            canned.update(canned_responses)
        self._server = _Server(reply, systemone_choice, systemone_noul_score, canned)
        self._thread: threading.Thread | None = None

    @property
    def systemone_choice(self) -> str:
        return self._server.systemone_choice

    @property
    def systemone_noul_score(self) -> float:
        return self._server.systemone_noul_score

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list[RequestRecord]:
        with self._server.records_lock:
            return list(self._server.records)

    def start(self) -> MockOmpServer:
        if self._thread is None:
            self._thread = threading.Thread(target=self._server.serve_forever, name="mockomp", daemon=True)
            self._thread.start()
            with _SERVERS_LOCK:
                _SERVERS[self.url] = self
        return self

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()
        with _SERVERS_LOCK:
            _SERVERS.pop(self.url, None)

    def __enter__(self) -> MockOmpServer:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


def server_for_url(url: str) -> MockOmpServer | None:
    """Return the live in-process server owning an exact loopback URL."""
    with _SERVERS_LOCK:
        return _SERVERS.get(url.rstrip("/"))


def post_json(url: str, body: bytes, timeout: float = 5) -> dict:
    """Issue one local JSON POST and return the decoded JSON response."""
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"mockomp returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
