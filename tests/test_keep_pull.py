"""`localbench keep`, `pull` (ollama and hf:) and `create` against a fake ollama: what they ask ollama for, what they
report, and that none of them touches ollama while a localbench run is alive."""

import contextlib
import io
import json
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest import mock

from localbench import __main__ as cli
from localbench import sysstats

FOREVER = "2318-01-02T03:04:05.123456789-07:00"
REAL_RESIDENTS = sysstats.ollama_residents


class FakeOllama(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []
    loaded: ClassVar[dict] = {}
    tags: ClassVar[dict] = {}
    pull_events: ClassVar[list] = []
    sticky: ClassVar[set] = set()      # models an unload request does not unload (another client is mid-request)
    ps_stall_s: ClassVar[float] = 0.0  # measured: 0.34.4 held /api/ps 8.9 s while its scheduler loaded a model

    def log_message(self, format, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/ps":
            time.sleep(self.ps_stall_s)
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):   # the client gave up on a stall
                self._json({"models": [{"name": n, "expires_at": e} for n, e in self.loaded.items()]})
        elif self.path == "/api/tags":
            self._json({"models": [{"name": n, "digest": d} for n, d in self.tags.items()]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((self.path, body))
        if self.path == "/api/generate":
            if body["keep_alive"] == 0:
                if body["model"] not in self.sticky:
                    self.loaded.pop(body["model"], None)
            else:
                self.loaded[body["model"]] = FOREVER if body["keep_alive"] == -1 else "2026-09-24T10:30:00-06:00"
            self._json({"done": True})
        elif self.path == "/api/pull":
            self.send_response(200)
            self.end_headers()
            for ev in self.pull_events:
                self.wfile.write(json.dumps(ev).encode() + b"\n")
                if ev.get("status") == "success":
                    # Measured: ollama lists a bare library pull as <name>:latest (`localbench models`:
                    # nomic-embed-text:latest). Untagged hf.co pulls are not measured here, so the fake only
                    # models names with no colon at all instead of copying cmd_pull's own tag test.
                    name = body["model"]
                    self.tags[name if ":" in name else name + ":latest"] = "sha256:8b1474be6e54aabb"
        elif self.path == "/api/show":
            self._json({"renderer": "qwen3.8", "parser": "qwen3.5", "capabilities": ["completion", "tools", "thinking"]})


class CliAgainstFake(unittest.TestCase):
    def setUp(self):
        FakeOllama.requests, FakeOllama.loaded, FakeOllama.tags, FakeOllama.pull_events = [], {}, {}, []
        FakeOllama.sticky, FakeOllama.ps_stall_s = set(), 0.0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        root = f"http://127.0.0.1:{self.server.server_port}"
        fake = type("O", (), {"root": root})
        self.patches = [mock.patch.object(cli, "Ollama", fake), mock.patch.object(cli, "_run_alive", return_value=False)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.server.shutdown()
        self.server.server_close()

    def run_cli(self, *argv):
        """(exit code, stdout); stderr lands in self.err. A failure's message belongs on stderr, so a caller piping
        stdout into another tool never reads it as a result."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(list(argv))
        self.err = err.getvalue()
        return rc, out.getvalue()


class Keep(CliAgainstFake):
    def test_forever_asks_for_keep_alive_minus_one_and_reports_it(self):
        rc, out = self.run_cli("keep", "ollama:qwen3.6:35b-mlx")
        self.assertEqual(FakeOllama.requests, [("/api/generate", {"model": "qwen3.6:35b-mlx", "keep_alive": -1})])
        self.assertEqual((rc, out.strip()), (0, "qwen3.6:35b-mlx: loaded until forever"))

    def test_a_duration_passes_through_and_zero_unloads(self):
        self.run_cli("keep", "ollama:m:1", "30m")
        rc, out = self.run_cli("keep", "ollama:m:1", "0")
        self.assertEqual([b["keep_alive"] for _, b in FakeOllama.requests], ["30m", 0])
        self.assertEqual((rc, out.strip()), (0, "m:1: unloaded"))

    def test_the_unload_alias_unloads_like_zero(self):
        FakeOllama.loaded = {"m:1": FOREVER}
        rc, out = self.run_cli("keep", "ollama:m:1", "unload")
        self.assertEqual(FakeOllama.requests, [("/api/generate", {"model": "m:1", "keep_alive": 0})])
        self.assertEqual((rc, out.strip()), (0, "m:1: unloaded"))

    def test_an_unload_that_leaves_the_model_resident_fails(self):
        FakeOllama.loaded, FakeOllama.sticky = {"m:1": FOREVER}, {"m:1"}
        rc, out = self.run_cli("keep", "ollama:m:1", "0")
        self.assertEqual((rc, out, self.err.strip()), (1, "", "m:1: still loaded"))

    def test_nothing_is_loaded_while_a_run_is_alive(self):
        with mock.patch.object(cli, "_run_alive", return_value=True):
            rc, _ = self.run_cli("keep", "ollama:qwen3.6:35b-mlx")
        self.assertEqual((rc, FakeOllama.requests), (1, []))

    def test_a_non_ollama_spec_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as stop:
            self.run_cli("keep", "mlx-serve:/x/model")
        self.assertEqual(stop.exception.code, 2)
        self.assertEqual(FakeOllama.requests, [])

    def test_a_stalled_ps_after_the_load_is_unknown_not_a_verdict(self):
        FakeOllama.ps_stall_s = 0.5      # keep waits 120 s; the stand-in shortens only the wait, not the logic
        with mock.patch.object(sysstats, "ollama_residents", lambda root, timeout: REAL_RESIDENTS(root, timeout=0.1)):
            rc, out = self.run_cli("keep", "ollama:m:1")
        self.assertEqual((rc, out, self.err.strip()),
                         (1, "", "m:1: requested, but ollama did not answer /api/ps within 120 s; loaded state unknown"))


class Residents(CliAgainstFake):
    def test_stalled_empty_and_down_are_three_different_answers(self):
        root = f"http://127.0.0.1:{self.server.server_port}"
        FakeOllama.loaded = {"m:1": FOREVER}
        self.assertEqual(sysstats.ollama_residents(root, timeout=2.0), [("m:1", "forever")])
        FakeOllama.ps_stall_s = 0.5
        self.assertIsNone(sysstats.ollama_residents(root, timeout=0.1))     # stalled: unknown, not "none"
        FakeOllama.ps_stall_s, FakeOllama.loaded = 0.0, {}
        self.assertEqual(sysstats.ollama_residents(root, timeout=2.0), [])  # answered: nothing loaded
        self.server.shutdown()
        self.server.server_close()
        self.assertEqual(sysstats.ollama_residents(root, timeout=2.0), [])  # refused: down, nothing loaded


class KeepUntil(unittest.TestCase):
    def test_forever_timer_and_unknown(self):
        self.assertEqual(sysstats.keep_until(FOREVER), "forever")
        self.assertRegex(sysstats.keep_until("2026-09-24T10:30:00-06:00"), r"^09-2[45] \d\d:\d\d$")
        self.assertEqual(sysstats.keep_until(None), "unknown")
        self.assertEqual(sysstats.keep_until("not a date"), "unknown")


class Pull(CliAgainstFake):
    NAME = "hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q4_K_M"

    def test_a_pull_shows_progress_and_the_installed_digest(self):
        FakeOllama.pull_events = [{"status": "pulling manifest"},
                                  *({"status": "pulling abc", "total": 1000, "completed": c} for c in range(0, 1001, 50)),
                                  {"status": "verifying sha256 digest"}, {"status": "success"}]
        rc, out = self.run_cli("pull", "ollama:" + self.NAME)
        self.assertEqual(FakeOllama.requests, [("/api/pull", {"model": self.NAME, "stream": True})])
        lines = out.strip().splitlines()
        self.assertEqual(rc, 0)
        self.assertEqual(lines[-1], f"installed {self.NAME} (digest 8b1474be6e54)")
        self.assertEqual(len([ln for ln in lines if ln.startswith("pulling abc")]), 11)   # one line per 10%

    def test_an_untagged_name_is_confirmed_as_latest(self):
        FakeOllama.pull_events = [{"status": "pulling manifest"}, {"status": "success"}]
        rc, out = self.run_cli("pull", "ollama:nomic-embed-text")
        self.assertEqual(FakeOllama.requests, [("/api/pull", {"model": "nomic-embed-text", "stream": True})])
        self.assertEqual((rc, out.strip().splitlines()[-1]), (0, "installed nomic-embed-text:latest (digest 8b1474be6e54)"))

    def test_an_error_event_fails_the_pull(self):
        FakeOllama.pull_events = [{"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"}]
        rc, out = self.run_cli("pull", "ollama:nope:1")
        self.assertEqual(rc, 1)
        self.assertIn("pull failed: pull model manifest", self.err)
        self.assertNotIn("pull failed", out)

    def test_a_pull_that_leaves_no_tag_fails(self):
        FakeOllama.pull_events = [{"status": "pulling manifest"}]
        rc, _ = self.run_cli("pull", "ollama:" + self.NAME)
        self.assertEqual(rc, 1)
        self.assertIn("not in ollama's model list", self.err)

    def test_nothing_is_pulled_while_a_run_is_alive(self):
        with mock.patch.object(cli, "_run_alive", return_value=True):
            rc, _ = self.run_cli("pull", "ollama:" + self.NAME)
        self.assertEqual((rc, FakeOllama.requests), (1, []))


class Create(CliAgainstFake):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "ThinkingCap"
        self.src.mkdir()
        (self.src / "config.json").write_text("{}")
        self.calls = []

        def fake_run(argv, env=None, **_):
            self.calls.append((argv, env.get("OLLAMA_MODELS"), Path(argv[4]).read_text()))
            FakeOllama.tags[argv[2]] = "sha256:0ca43d94af99aaaa"
            return subprocess.CompletedProcess(argv, 0)

        self.patches += [mock.patch.object(cli.subprocess, "run", fake_run),
                         mock.patch.object(cli.sysstats, "ollama_models_dir", return_value=Path("/Volumes/Z/m"))]
        for p in self.patches[-2:]:
            p.start()

    def tearDown(self):
        super().tearDown()
        self.tmp.cleanup()

    def test_import_goes_to_the_servers_store_with_the_replaced_models_renderer_and_parser(self):
        rc, out = self.run_cli("create", "ollama:thinkingcap:27b-nvfp4", "--from", str(self.src), "--quantize", "nvfp4",
                               "--like", "ollama:qwen3.8:27b-mlx")
        self.assertEqual(rc, 0, out)
        argv, store, modelfile = self.calls[0]
        self.assertEqual((argv[:4], argv[5:], store), (["ollama", "create", "thinkingcap:27b-nvfp4", "-f"],
                                                       ["--quantize", "nvfp4"], "/Volumes/Z/m"))
        self.assertEqual(modelfile, f"FROM {self.src}\nRENDERER qwen3.8\nPARSER qwen3.5\n")
        self.assertIn("created thinkingcap:27b-nvfp4 (digest 0ca43d94af99", out)

    def test_an_existing_name_is_refused_before_ollama_runs(self):
        FakeOllama.tags = {"thinkingcap:27b-nvfp4": "sha256:1"}
        rc, out = self.run_cli("create", "ollama:thinkingcap:27b-nvfp4", "--from", str(self.src))
        self.assertEqual((rc, self.calls, out), (1, [], ""))
        self.assertIn("already exists", self.err)


class PullHF(CliAgainstFake):
    INFO = {"sha": "93944fefc63a6c6eb3f649bb003286691e78a40c", "usedStorage": 22_808_633_766, "gated": False}

    def run_pull(self, free_bytes):
        calls = []
        real_get = cli.backends._get
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cli.backends, "_get", lambda url, timeout=10: self.INFO if "huggingface.co" in url
                                  else real_get(url, timeout)), \
                mock.patch.object(cli.shutil, "disk_usage", return_value=mock.Mock(free=free_bytes)), \
                mock.patch.object(cli.subprocess, "run",
                                  lambda argv, **_: calls.append(argv) or subprocess.CompletedProcess(argv, 0)):
            rc, out = self.run_cli("pull", "hf:bottlecapai/TC-MLX", "--to", str(Path(tmp) / "TC-MLX"))
        return rc, out, calls

    def test_too_little_space_refuses_before_downloading(self):
        rc, out, calls = self.run_pull(free_bytes=30e9)       # 22.8 GB repo + 20 GB headroom does not fit
        self.assertEqual((rc, calls, out), (1, [], ""))
        self.assertIn("needs 22.8 GB plus 20 GB headroom", self.err)

    def test_the_download_is_pinned_to_the_revision_seen_at_start(self):
        rc, out, calls = self.run_pull(free_bytes=100e9)
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls[0][:5], ["hf", "download", "bottlecapai/TC-MLX", "--revision", self.INFO["sha"]])


if __name__ == "__main__":
    unittest.main()
