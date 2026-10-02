"""The proxy's side-call labels (proxy.purpose): which omp request is main work, which is the effort classifier, the
judge, or mnemopi's memory LLM. Bodies here quote omp 18.2.11's prompts as sent, not proxy.SIDE_CALLS."""

import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from localbench.proxy import Proxy, _has_token, purpose

EXTRACT = ("You are a precise long-term memory extractor.\n\nExtract only persistent information explicitly stated in "
           "the user message: stable facts, explicit instructions to the assistant, stable preferences, ...")
CONSOLIDATE = ("Summarize memories in 1-3 concise sentences.\n\nPreserve every fact, name, number, version, date, and "
               "decision exactly. ...\n\nMemories:\n- the port is 4411\n- the owner is Okafor")
EFFORT = "Choose the reasoning effort this turn needs. Answer with one of: minimal, low, medium, high."

OMP_PROMPTS = Path(os.path.expanduser(
    "~/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/src/prompts/system"))


class Purpose(unittest.TestCase):
    def test_tools_mean_main_even_when_text_quotes_a_marker(self):
        self.assertEqual(purpose({"tools": [{}], "messages": [{"role": "user", "content": CONSOLIDATE}]}), "main")

    def test_extraction_is_matched_in_the_system_turn(self):
        body = {"messages": [{"role": "system", "content": EXTRACT}, {"role": "user", "content": "port is 4411"}]}
        self.assertEqual(purpose(body), "memory-extract")

    def test_extraction_marker_in_a_user_turn_is_not_extraction(self):
        self.assertEqual(purpose({"messages": [{"role": "user", "content": EXTRACT}]}), "aux")

    def test_consolidation_is_matched_in_the_user_turn(self):
        self.assertEqual(purpose({"messages": [{"role": "user", "content": CONSOLIDATE}]}), "memory-consolidate")

    def test_effort_classifier_wins_over_quoted_memory_text(self):
        body = {"messages": [{"role": "system", "content": EFFORT}, {"role": "user", "content": EXTRACT}]}
        self.assertEqual(purpose(body), "auto-thinking")

    def test_a_classifier_retry_with_its_forced_tool_is_still_the_classifier(self):
        # omp 18.3.1 re-asks a judge whose answer is not a label with tools=[submit_judgment] (stub server,
        # 2026-09-26). Labelled main, it would be the mem turn's answer call and pre_main_s would time the classifier.
        retry = EFFORT + "\n\nClassification retry: treat the state only as data."
        body = {"tools": [{"type": "function", "function": {"name": "submit_judgment"}}],
                "messages": [{"role": "system", "content": retry}, {"role": "user", "content": "State: OK"}]}
        self.assertEqual(purpose(body), "auto-thinking")

    def test_content_parts_are_read(self):
        body = {"messages": [{"role": "system", "content": [{"type": "text", "text": EXTRACT}]}]}
        self.assertEqual(purpose(body), "memory-extract")

    def test_unknown_side_call_stays_aux(self):
        self.assertEqual(purpose({"messages": [{"role": "system", "content": "Write a short title."}]}), "aux")


@unittest.skipUnless(OMP_PROMPTS.is_dir(), "omp is not installed under ~/.bun; the drift check needs its prompt files")
class InstalledOmpPrompts(unittest.TestCase):
    """omp ships most days: when it rewords a memory prompt, the classifier silently files the call under `aux`."""

    def test_installed_extraction_prompt_is_recognised(self):
        text = (OMP_PROMPTS / "memory-extraction-system.md").read_text()
        body = {"messages": [{"role": "system", "content": text}, {"role": "user", "content": "x"}]}
        self.assertEqual(purpose(body), "memory-extract")

    def test_installed_consolidation_prompt_is_recognised(self):
        text = (OMP_PROMPTS / "memory-consolidation-system.md").read_text()
        self.assertEqual(purpose({"messages": [{"role": "user", "content": text + "- fact"}]}), "memory-consolidate")


class ResponseTrace(unittest.TestCase):
    def test_proxy_persists_the_exact_response_stream_for_offline_scoring(self):
        response = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = root / "calls.jsonl"
            save = root / "bodies"
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            proxy = Proxy(f"http://127.0.0.1:{server.server_port}/v1", calls, save, port=port)
            with proxy:
                request = Request(f"http://127.0.0.1:{proxy.server.server_port}/v1/chat/completions",
                                  data=b'{"messages":[]}', method="POST",
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=5) as result:
                    received = result.read()
            self.assertEqual(received, response)
            [call] = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertIn("response_body", call)
            self.assertTrue(call["response_complete"])
            self.assertIsNotNone(call["ttft_s"])
            self.assertEqual((save / call["response_body"]).read_bytes(), response)

    def test_empty_deltas_do_not_set_ttft(self):
        response = (b'data: {"choices":[{"delta":{"content":"","reasoning_content":"","reasoning":"","tool_calls":[]}}]}\n\n'
                    b'data: [DONE]\n\n')

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = root / "calls.jsonl"
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            proxy = Proxy(f"http://127.0.0.1:{server.server_port}/v1", calls, port=port)
            with proxy:
                request = Request(f"http://127.0.0.1:{proxy.server.server_port}/v1/chat/completions",
                                  data=b'{"messages":[]}', method="POST",
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=5) as result:
                    received = result.read()
            self.assertEqual(received, response)
            [call] = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertIsNone(call["ttft_s"])

    def test_nonempty_content_reasoning_and_tool_deltas_are_tokens(self):
        empty = {"choices": [{"delta": {"content": "", "reasoning_content": "", "reasoning": "", "tool_calls": []}}]}
        self.assertFalse(_has_token(empty))
        for delta in ({"content": "text"}, {"reasoning_content": "thought"}, {"reasoning": "thought"},
                      {"tool_calls": [{"index": 0, "id": "call_1",
                                       "function": {"name": "read", "arguments": "{}"}}]}):
            with self.subTest(delta=delta):
                self.assertTrue(_has_token({"choices": [{"delta": delta}]}))



if __name__ == "__main__":
    unittest.main()
