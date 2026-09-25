"""The oMLX backend: it never measures a server it did not start, every start is cold, it counts as another model
for every other run, and its pins come from what is installed."""

import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import backends, sysstats
from localbench.backends import OMLX, MlxServe

TARGET = ("ollama", "qwen3.6:35b-mlx")
OMLX_ROW = {"pid": 9, "name": "python3.13", "pct": 60.0,
            "cmd": "~/.local/share/uv/tools/omlx/bin/python ~/.local/bin/omlx serve --model-dir /tmp/x"}


class _Proc:
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


class Model(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "Qwen3.6-35B-A3B-MLX-Serve-4bit"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps(
            {"model_type": "qwen3_5_moe_vl", "text_config": {"model_type": "qwen3_5_moe"}, "quantization": {"bits": 4}}))

    def tearDown(self):
        self.tmp.cleanup()


class NeverAnotherProgram(Model):
    def test_a_port_held_by_another_program_refuses_the_start(self):
        # 2026-09-24: :8765 answered 401 from the Agent Mail MCP server.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            threading.Thread(target=lambda: listener.accept()[0].close(), daemon=True).start()
            srv = OMLX(self.model, port=listener.getsockname()[1])
            with mock.patch.object(backends.subprocess, "Popen") as popen, \
                    self.assertRaisesRegex(RuntimeError, "in use by another program"):
                srv.start(ready_timeout=5)
            popen.assert_not_called()


class EveryStartIsCold(Model):
    def start(self, srv):
        with mock.patch.object(OMLX, "up", side_effect=[False, True]), \
                mock.patch.object(OMLX, "port_taken", return_value=False), \
                mock.patch.object(backends.subprocess, "Popen", return_value=_Proc()) as popen:
            srv.start(ready_timeout=5)
        return popen.call_args.args[0]

    def test_each_start_serves_only_this_model_with_its_own_cache_and_stop_removes_it(self):
        srv = OMLX(self.model)
        first = self.start(srv)
        self.assertEqual(first[:2], ["omlx", "serve"])
        models = Path(first[first.index("--model-dir") + 1])
        cache = Path(first[first.index("--paged-ssd-cache-dir") + 1])
        self.assertEqual([p.name for p in models.iterdir()], [self.model.name])
        self.assertEqual((models / self.model.name).resolve(), self.model.resolve())
        self.assertIn("--no-hf-cache", first)
        self.assertEqual(first[first.index("--port") + 1], "11236")
        self.assertTrue(cache.is_dir())
        srv.stop()
        self.assertFalse(cache.exists())
        second = self.start(srv)
        self.assertNotEqual(Path(second[second.index("--paged-ssd-cache-dir") + 1]), cache)
        srv.stop()


class OneModelAtATime(unittest.TestCase):
    def test_a_serving_omlx_blocks_an_ollama_run_and_a_serving_mlx_serve_blocks_an_omlx_run(self):
        with mock.patch.object(OMLX, "up", return_value=True), mock.patch.object(MlxServe, "up", return_value=False), \
                self.assertRaisesRegex(SystemExit, "omlx on :11236"):
            with cli.open_backend("ollama:qwen3.6:35b-mlx"):
                pass
        with mock.patch.object(MlxServe, "up", lambda self: type(self) is MlxServe), \
                self.assertRaisesRegex(SystemExit, "mlx-serve on :11234"):
            with cli.open_backend("omlx:/tmp/some-model"):
                pass

    def test_an_omlx_server_busy_during_another_run_is_contention_not_app_load(self):
        _, models, apps = sysstats.classify([OMLX_ROW], {}, TARGET, 25.0)
        self.assertEqual((models, apps), ([OMLX_ROW], []))

    def test_a_model_loaded_in_omlx_is_resident(self):
        def fake(url, timeout=2.0):
            if url.endswith("/api/status"):
                return {"loaded_models": ["Qwen3.6-35B-A3B-MLX-Serve-4bit", {"id": "Other"}]}
            return {}
        with mock.patch.object(sysstats, "_json", side_effect=fake):
            resident = sysstats.resident_models()
        self.assertEqual(resident["omlx"], ["Other", "Qwen3.6-35B-A3B-MLX-Serve-4bit"])
        self.assertEqual(sysstats.foreign_models(resident, *TARGET), {"omlx": resident["omlx"]})


class Pins(Model):
    def test_fingerprint_reads_the_model_config_and_the_servers_context(self):
        srv = OMLX(self.model)
        with mock.patch.object(OMLX, "models", return_value=[{"id": self.model.name, "max_model_len": 262144}]), \
                mock.patch.object(OMLX, "status", return_value={"version": "0.7.0rc1"}), \
                mock.patch.object(backends, "omlx_sha", return_value="abcd"):
            fp = srv.fingerprint(self.model.name)
        self.assertEqual((fp["backend"], fp["backend_version"], fp["backend_sha"], fp["architecture"],
                          fp["quantization"], fp["loaded_context"]),
                         ("omlx", "0.7.0rc1", "abcd", "qwen3_5_moe", "4-bit", 262144))

    def test_the_sha_follows_the_installed_package_not_the_launcher(self):
        env = Path(self.tmp.name) / "env"
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "omlx").write_text("#!/bin/sh\n")
        record = env / "lib" / "python3.13" / "site-packages" / "omlx-0.7.0rc1.dist-info" / "RECORD"
        record.parent.mkdir(parents=True)
        record.write_text("omlx/server.py,sha256=aaa,10\n")
        with mock.patch.object(backends.shutil, "which", return_value=str(env / "bin" / "omlx")):
            before = backends.omlx_sha()
            record.write_text("omlx/server.py,sha256=bbb,10\n")
            record.touch()
            after = backends.omlx_sha()
        self.assertIsNotNone(before)
        self.assertNotEqual(before, after)



class MlxServeBinary(unittest.TestCase):
    def test_the_override_is_what_launches_and_what_the_pins_name(self):
        # A side-by-side release (LOCALBENCH_MLX_SERVE) must be the process that serves AND the sha/version on the
        # receipt; launching one binary while pinning PATH's would bank the wrong generation.
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "mlx-serve"
            exe.write_text("#!/bin/sh\necho 'mlx-serve 26.9.5'\n")
            exe.chmod(0o755)
            argv = []
            with mock.patch.dict("os.environ", {"LOCALBENCH_MLX_SERVE": str(exe)}):
                server = MlxServe(tmp)
                with mock.patch.object(backends.subprocess, "Popen", lambda a, **_: argv.append(a)):
                    server._spawn(None)
                pins = server.pins("m")                             # runs the fake binary for --version
            want_sha = backends.sha16(str(exe))
        self.assertEqual(argv[0][0], str(exe))
        self.assertEqual((pins["backend_version"], pins["backend_sha"]), ("26.9.5", want_sha))


if __name__ == "__main__":
    unittest.main()
