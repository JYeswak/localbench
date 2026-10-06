import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

from localbench.client import RequestCancellation, RunAborted, chat_stream, run_cancellable


class QuietHTTPServer(ThreadingHTTPServer):
    headers_sent: threading.Event
    release: threading.Event


class QuietStreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        server = cast(QuietHTTPServer, self.server)
        server.headers_sent.set()
        server.release.wait(5)

    def log_message(self, format: str, *args: object) -> None:
        pass


class RequestCancellationTests(unittest.TestCase):
    def test_watchdog_cancellation_interrupts_a_blocked_stream(self):
        server = QuietHTTPServer(("127.0.0.1", 0), QuietStreamHandler)
        server.headers_sent = threading.Event()
        server.release = threading.Event()
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        cancellation = RequestCancellation()
        outcome = []

        def request():
            try:
                chat_stream(f"http://127.0.0.1:{server.server_port}/v1", {"model": "fixture"},
                            backend="fixture", label="blocked", cancellation=cancellation)
            except Exception as exc:
                outcome.append(exc)

        request_thread = threading.Thread(target=request)
        request_thread.start()
        try:
            self.assertTrue(server.headers_sent.wait(2), "mock server did not send response headers")
            cancellation.cancel("foreign GPU client")
            request_thread.join(2)
            self.assertFalse(request_thread.is_alive(), "cancel did not interrupt the blocked response read")
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], RunAborted)
            self.assertEqual(str(outcome[0]), "foreign GPU client")
        finally:
            server.release.set()
            server.shutdown()
            server.server_close()
            server_thread.join(2)
            request_thread.join(2)

    def test_watchdog_cancellation_stops_a_running_child_process(self):
        cancellation = RequestCancellation()
        outcome = []

        def child():
            try:
                run_cancellable([sys.executable, "-c", "import time; time.sleep(30)"],
                                cancellation=cancellation, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
            except Exception as exc:
                outcome.append(exc)

        child_thread = threading.Thread(target=child)
        child_thread.start()
        time.sleep(0.1)
        cancellation.cancel("residency became unknown")
        child_thread.join(2)

        self.assertFalse(child_thread.is_alive(), "cancellation did not stop the active child")
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], RunAborted)
        self.assertEqual(str(outcome[0]), "residency became unknown")


if __name__ == "__main__":
    unittest.main()
