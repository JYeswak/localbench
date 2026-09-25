"""A closed pipe must be a BrokenPipeError where it happens, never a silent kill of the harness. 2026-09-23: main() set
SIGPIPE to SIG_DFL for quiet `| head`; the sess tier then died (exit 141, no log) when omp abandoned a classifier stream
queued behind a memory extraction and the proxy wrote to the closed socket."""

import contextlib
import io
import json
import signal
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from localbench.__main__ import main
from localbench.proxy import Proxy


class SlowSSE(BaseHTTPRequestHandler):
    """Upstream that streams one chunk every 50 ms for ~2 s: long enough for the client to walk away mid-stream."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            for i in range(40):
                line = f'data: {{"choices":[{{"delta":{{"content":"t{i}"}}}}]}}\n\n'.encode()
                self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"0\r\n\r\n")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ProxyAbort(unittest.TestCase):
    def test_client_abort_is_logged_as_aborted_and_the_proxy_survives(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), SlowSSE)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            log, port = Path(tmp) / "calls.jsonl", free_port()
            with Proxy(f"http://127.0.0.1:{upstream.server_port}/v1", log, port=port):
                body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
                sock = socket.create_connection(("127.0.0.1", port))
                sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                             + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
                sock.recv(512)                                   # first bytes arrive: the stream is live
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                sock.close()                                     # RST: the next proxy write hits a dead socket
                deadline = time.time() + 10
                while not (log.exists() and log.read_text().strip()) and time.time() < deadline:
                    time.sleep(0.05)
            rows = [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []
        upstream.shutdown()
        upstream.server_close()
        self.assertEqual(len(rows), 1, "the abandoned call must still be logged")
        self.assertTrue(rows[0]["aborted"])
        self.assertEqual(rows[0]["status"], 200)


class SignalDisposition(unittest.TestCase):
    def test_main_leaves_sigpipe_ignored(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main(["memory", "--json"])
        self.assertIs(signal.getsignal(signal.SIGPIPE), signal.SIG_IGN)


if __name__ == "__main__":
    unittest.main()
