"""The mlxfast backend (Layr-Labs mlx-server): it launches the binary the environment names with the parsers that make
tool calls structured, refuses a binary that would die silently, pins the commit that binary was built from, and its
process counts as the run's own work only on an mlxfast run."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import backends, golden, sysstats
from localbench.backends import OMLX, MlxFast, MlxServe

ROOT = Path(__file__).resolve().parents[1]
SERVER = "/Volumes/Models/src/mlxfast-baseline/Vendor/mlx-swift-lm/.build/release/mlx-server"
SERVER_ROW = {"pid": 11, "name": "mlx-server", "pct": 80.0,
              "cmd": f"{SERVER} --model /m/Ternary-Bonsai-2-27B-mlx-2bit --host 127.0.0.1 --port 11237"}
MLX_SERVE_ROW = {"pid": 12, "name": "mlx-serve", "pct": 80.0,
                 "cmd": "/opt/homebrew/bin/mlx-serve --model /m/Qwen3.6 --serve --host 127.0.0.1 --port 11234"}


class _Proc:
    pid = 4242
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


class Build(unittest.TestCase):
    """A fake mlx-server in its build dir, optionally inside a git tree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.release = self.root / "engine" / "Vendor" / ".build" / "release"
        self.release.mkdir(parents=True)
        self.binary = self.release / "mlx-server"
        self.binary.write_text("#!/bin/sh\n")
        self.binary.chmod(0o755)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text", "quantization": {"bits": 2}}))
        env = mock.patch.dict(os.environ, {"LOCALBENCH_MLXFAST": str(self.binary)})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, srv):
        with mock.patch.object(MlxFast, "up", side_effect=[False, True]), \
                mock.patch.object(MlxFast, "port_taken", return_value=False), \
                mock.patch.object(backends.subprocess, "Popen", return_value=_Proc()) as popen:
            srv.start(ready_timeout=5)
        return popen


class Launch(Build):
    def test_the_named_binary_serves_the_model_on_its_port_with_structured_tool_calls_and_reasoning(self):
        (self.release / "mlx.metallib").write_bytes(b"")
        argv = self.start(MlxFast(self.model, ("--max-kv", "8192"))).call_args.args[0]
        self.assertEqual(argv[0], str(self.binary))
        self.assertEqual(argv[argv.index("--model") + 1], str(self.model))
        self.assertEqual(argv[argv.index("--port") + 1], "11237")
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "xml_function")
        self.assertEqual(argv[argv.index("--reasoning-parser") + 1], "qwen3")
        self.assertEqual(argv[-2:], ["--max-kv", "8192"])

    def test_a_binary_without_its_metallib_is_refused_before_it_runs(self):
        # 2026-09-26: without mlx.metallib beside it mlx-server exits rc=133 and writes nothing to its log.
        with mock.patch.object(MlxFast, "up", return_value=False), \
                mock.patch.object(MlxFast, "port_taken", return_value=False), \
                mock.patch.object(backends.subprocess, "Popen", return_value=_Proc()) as popen, \
                self.assertRaisesRegex(RuntimeError, "mlx.metallib"):
            MlxFast(self.model).start(ready_timeout=5)
        popen.assert_not_called()


class Pins(Build):
    def git(self, *args) -> str:
        # The user's global hooks (a commit-subject gate) and signing, and a calling hook's GIT_DIR/GIT_INDEX_FILE (the
        # pre-commit hook runs this suite), are not part of the tree being pinned.
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                               "-C", str(self.root / "engine"), *args], check=True, capture_output=True, text=True,
                              env={**env, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                                   "GIT_COMMITTER_EMAIL": "t@t"}).stdout.strip()

    def test_the_generation_is_the_binary_hash_and_the_commit_of_the_tree_it_was_built_in(self):
        self.git("init", "-q")
        (self.root / "engine" / "README").write_text("x\n")
        self.git("add", "README")
        self.git("commit", "-q", "-m", "x")
        head = self.git("rev-parse", "HEAD")
        # As under a git hook (the pre-commit hook runs this suite): the caller's GIT_DIR names another repository.
        with mock.patch.dict(os.environ, {"GIT_DIR": str(ROOT / ".git")}):
            pins = MlxFast(self.model, ("--max-kv", "8192")).pins("M")
        self.assertEqual(pins["backend"], "mlxfast")
        self.assertEqual(pins["backend_version"], head)
        self.assertEqual(pins["backend_sha"], backends.sha16(str(self.binary)))
        self.assertEqual(pins["backend_args"], "--max-kv 8192")

    def test_outside_a_git_tree_or_without_a_binary_the_commit_is_unknown_not_localbenchs(self):
        self.assertIsNone(MlxFast(self.model).pins("M")["backend_version"])
        # A bare name that PATH did not resolve must not be read as the caller's cwd (this checkout, a git tree).
        cwd = os.getcwd()
        with mock.patch.object(backends, "mlxfast_bin", return_value="mlx-server"):
            os.chdir(ROOT)
            try:
                pins = MlxFast(self.model).pins("M")
            finally:
                os.chdir(cwd)
        self.assertEqual((pins["backend_version"], pins["backend_sha"]), (None, None))


class LoadedContext(Build):
    """omp's tiers refuse to run without a loaded context (2026-09-26 screen: mem and sess MUST-FAILed on mlxfast).
    mlx-server applies no cap, so the context is the checkpoint's max_position_embeddings."""

    def fingerprint(self, cfg: dict) -> dict:
        (self.model / "config.json").write_text(json.dumps(cfg))
        with mock.patch.object(MlxFast, "models", return_value=[{"id": "M"}]):
            return MlxFast(self.model).fingerprint("M")

    def test_the_text_configs_position_limit_wins_over_the_top_level_one(self):
        fp = self.fingerprint({"max_position_embeddings": 4096, "text_config": {"max_position_embeddings": 262144}})
        self.assertEqual(fp["loaded_context"], 262144)
        self.assertEqual(fp["loaded_context_source"], "config.max_position_embeddings (server applies no cap)")

    def test_a_top_level_position_limit_is_used_when_the_text_config_has_none(self):
        self.assertEqual(self.fingerprint({"text_config": {}, "max_position_embeddings": 32768})["loaded_context"],
                         32768)

    def test_no_position_limit_means_no_context_not_a_guess(self):
        fp = self.fingerprint({"model_type": "x"})
        self.assertEqual((fp["loaded_context"], fp["loaded_context_source"]), (None, None))



class OwnWorkOnlyOnItsOwnRun(unittest.TestCase):
    def test_mlx_server_is_the_mlxfast_runs_own_gpu_work(self):
        self.assertEqual(sysstats.classify([SERVER_ROW], {}, ("mlxfast", "/m/Ternary"), 25.0), ({}, [], []))

    def test_mlx_server_is_another_model_during_an_mlx_serve_or_ollama_run(self):
        # "mlx-serve" is a prefix of "mlx-server": a substring match would call it the mlx-serve run's own work.
        for target in (("mlx-serve", "Qwen3.6"), ("ollama", "qwen3.8:27b-mlx")):
            _, models, apps = sysstats.classify([SERVER_ROW], {}, target, 25.0)
            self.assertEqual((models, apps), ([SERVER_ROW], []), target)

    def test_mlx_serve_is_another_model_during_an_mlxfast_run(self):
        _, models, _ = sysstats.classify([MLX_SERVE_ROW], {}, ("mlxfast", "/m/Ternary"), 25.0)
        self.assertEqual(models, [MLX_SERVE_ROW])

    def test_its_model_is_resident_under_mlxfast_and_foreign_to_any_other_run(self):
        def fake(url, timeout=2.0, *, probe=None):
            return ({"data": [{"id": "/m/Ternary", "object": "model", "created": 0, "owned_by": "mlx-swift-lm"}]}
                    if ":11237/" in url else {})
        with mock.patch.object(sysstats, "_json", fake):
            resident = sysstats.resident_models()
        self.assertEqual(resident["mlxfast"], ["/m/Ternary"])
        self.assertEqual({k: v for k, v in sysstats.foreign_models(resident, "mlxfast", "/m/Ternary").items() if v}, {})
        self.assertEqual(sysstats.foreign_models(resident, "ollama", "m")["mlxfast"], ["/m/Ternary"])

    def test_a_serving_mlxfast_blocks_every_other_run(self):
        with mock.patch.object(MlxServe, "up", lambda self: type(self) is MlxFast), \
                mock.patch.object(OMLX, "up", return_value=False), \
                self.assertRaisesRegex(SystemExit, "mlxfast on :11237"):
            with cli.open_backend("ollama:qwen3.8:27b-mlx"):
                pass


class StatusReadsTheGoldensOwnBackend(unittest.TestCase):
    def test_an_mlxfast_golden_is_checked_against_mlxfast_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "aa.json"
            receipt.write_text(json.dumps({"runs": [{"provenance": {"fingerprint": {"model_dir": "/m/Ternary"}}}]}))
            (Path(tmp) / "h").mkdir()
            golden.write(Path(tmp) / "h" / "mlxfast__m.json",
                         {"pins": {"backend": "mlxfast", "model": "/m/Ternary"}, "aa_receipt": str(receipt)})
            seen = []

            def fake_pins(backend, model, host):
                seen.append((type(backend), backend.model_dir))
                return {"model_digest": None}

            with mock.patch.object(golden, "GOLDENS", Path(tmp)), mock.patch.object(cli, "run_pins", fake_pins):
                cli.golden_states({"host_id": "h"})
        self.assertEqual(seen, [(MlxFast, Path("/m/Ternary"))])


if __name__ == "__main__":
    unittest.main()
