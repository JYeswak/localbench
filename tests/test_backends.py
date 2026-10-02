import subprocess
import threading
import unittest
from unittest import mock

from localbench import backends


class OllamaTokenizerIdentity(unittest.TestCase):
    def identity(self, model_info: dict | None) -> str | None:
        ollama = backends.Ollama()
        with mock.patch.object(backends, "_post", return_value={"model_info": model_info}) as post:
            identity = ollama.tokenizer_identity("m")
        post.assert_called_once_with("http://127.0.0.1:11434/api/show", {"model": "m", "verbose": True})
        return identity

    def test_identity_comes_from_reported_tokenizer_tables_not_model_generation(self):
        report = {
            "general.architecture": "qwen3",
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["<unk>", "hello", "world"],
            "tokenizer.ggml.merges": ["h e", "he llo"],
        }
        first = self.identity(report)
        second = self.identity({**report, "general.architecture": "nemotron"})
        self.assertTrue(first.startswith("ollama-ggml-sha256:"))
        self.assertEqual(first, second)

    def test_a_changed_token_table_has_a_different_identity(self):
        base = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": ["a", "b"]}
        changed = {**base, "tokenizer.ggml.tokens": ["a", "c"]}
        self.assertNotEqual(self.identity(base), self.identity(changed))

    def test_missing_or_incomplete_tokenizer_report_does_not_guess_an_identity(self):
        for report in (None, {"general.architecture": "qwen3"},
                       {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": []}):
            with self.subTest(report=report):
                self.assertIsNone(self.identity(report))


class _FakeStopProcess:
    def __init__(self, pid: int, timeout_first_wait: bool = False):
        self.pid = pid
        self.returncode = None
        self.timeout_first_wait = timeout_first_wait
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.timeout_first_wait and timeout is not None:
            self.timeout_first_wait = False
            raise subprocess.TimeoutExpired("server", timeout)
        self.returncode = -9 if self.killed else 0
        return self.returncode


class OwnedStopProvenance(unittest.TestCase):
    def stop_records(self, servers):
        records = []
        with mock.patch.object(backends, "_OWNED_STOP_LOG", records), \
                mock.patch.object(backends, "_OWNED_STOP_LOCK", threading.Lock()), \
                mock.patch.object(backends, "_get", return_value={
                    "data": [{"id": "target", "loaded": True}, {"id": "other", "loaded": False}]
                }):
            for pid, server in enumerate(servers, 40):
                server._proc = _FakeStopProcess(pid)
                server.stop()
            return backends.owned_stop_events()

    def test_stop_records_loaded_prestate_and_exit_for_shared_backend_lifecycle(self):
        servers = [backends.MlxServe("/models/target"), backends.OMLX("/models/target"),
                   backends.MlxFast("/models/target")]

        records = self.stop_records(servers)

        self.assertEqual([row["server"] for row in records], ["mlx-serve", "omlx", "mlxfast"])
        for row, pid in zip(records, (40, 41, 42)):
            with self.subTest(server=row["server"]):
                self.assertEqual(row["port"], servers[pid - 40].port)
                self.assertEqual(row["pid"], pid)
                self.assertEqual(row["pre"], ["target"])
                self.assertLessEqual(row["t_term"], row["t_exit"])
                self.assertEqual(row["rc"], 0)
                self.assertFalse(row["killed"])

    def test_unreadable_prestate_is_none_and_kill_path_is_not_a_clean_exit(self):
        records = []
        server = backends.MlxServe("/models/target")
        server._proc = _FakeStopProcess(50, timeout_first_wait=True)
        with mock.patch.object(backends, "_OWNED_STOP_LOG", records), \
                mock.patch.object(backends, "_OWNED_STOP_LOCK", threading.Lock()), \
                mock.patch.object(backends, "_get", side_effect=OSError("unreadable")):
            server.stop()
            rows = backends.owned_stop_events()

        (row,) = rows
        self.assertIsNone(row["pre"])
        self.assertEqual(row["rc"], -9)
        self.assertTrue(row["killed"])




if __name__ == "__main__":
    unittest.main()
