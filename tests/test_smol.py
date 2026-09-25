"""`localbench smol`: moving omp's smol role to a dedicated server edits only what it must in every profile, and revert
gives back exactly what was there; park stops that server and unpark restarts it."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import park, smol

SELECTOR = "mlx-smol/Qwen3.8-27B-MLX-Serve-4bit"
PROFILES = {
    # default: an existing provider file with another local provider in it
    ".omp/agent": ("modelRoles:\n  smol: ollama/qwen3.8:27b-mlx\n  default: openai-codex/gpt-6-luna:xhigh\n",
                   "# Local inference providers.\nproviders:\n  mlx-serve:\n    baseUrl: http://127.0.0.1:11234/v1\n"),
    # omp-test: its DEFAULT role also names the ollama smol model and must not move; no models.yml at all
    ".omp/profiles/omp-test/agent": ("modelRoles:\n  default: ollama/qwen3.8:27b-mlx:max\n  smol: ollama/qwen3.8:27b-mlx\n",
                                     None),
    # claude: a credential-only providers file
    ".omp/profiles/claude/agent": ("theme: dark\nmodelRoles:\n  smol: ollama/qwen3.8:27b-mlx\n  judge: typesafe/proj-b-latest\n",
                                   "# TypeSafe credential.\nproviders:\n  typesafe:\n    apiKey: \"!cmd\"\n"),
    # a profile with no smol line is left alone
    ".omp/profiles/nosmol/agent": ("modelRoles:\n  default: xai-oauth/grok-4.7\n", None),
}


class ProfileEdits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.before = {}
        for rel, (cfg, mdl) in PROFILES.items():
            d = self.home / rel
            d.mkdir(parents=True)
            (d / "config.yml").write_text(cfg)
            if mdl is not None:
                (d / "models.yml").write_text(mdl)
            self.before[rel] = (cfg, mdl)
        self.dirs = smol.profile_dirs(self.home)
        self.block = smol.provider_block("Qwen3.8-27B-MLX-Serve-4bit", 262144)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, rel):
        d = self.home / rel
        return (d / "config.yml").read_text(), (d / "models.yml").read_text() if (d / "models.yml").exists() else None

    def test_set_changes_only_smol_lines_and_adds_the_provider_under_providers(self):
        record = smol.set_profiles(SELECTOR, self.block, self.dirs, self.home / "backup")
        self.assertEqual(set(self.dirs), {"default", "omp-test", "claude", "nosmol"})
        cfg, mdl = self.read(".omp/profiles/omp-test/agent")
        self.assertEqual(cfg, f"modelRoles:\n  default: ollama/qwen3.8:27b-mlx:max\n  smol: {SELECTOR}\n")
        self.assertTrue(mdl.startswith("# Local providers written by `localbench smol set`") and self.block in mdl)
        cfg, mdl = self.read(".omp/profiles/claude/agent")
        self.assertEqual(cfg, f"theme: dark\nmodelRoles:\n  smol: {SELECTOR}\n  judge: typesafe/proj-b-latest\n")
        self.assertEqual(mdl, "# TypeSafe credential.\nproviders:\n" + self.block + "  typesafe:\n    apiKey: \"!cmd\"\n")
        self.assertEqual(self.read(".omp/profiles/nosmol/agent"), self.before[".omp/profiles/nosmol/agent"])
        self.assertEqual(record["omp-test"], {"previous": "ollama/qwen3.8:27b-mlx", "created_models_yml": True})
        self.assertEqual(record["nosmol"]["previous"], None)

    def test_revert_gives_back_every_file_byte_for_byte(self):
        record = smol.set_profiles(SELECTOR, self.block, self.dirs, self.home / "backup")
        self.assertEqual(smol.revert_profiles(record, SELECTOR, self.dirs), [])
        for rel, want in self.before.items():
            self.assertEqual(self.read(rel), want, rel)

    def test_a_second_set_keeps_the_original_smol_for_revert(self):
        first = smol.set_profiles(SELECTOR, self.block, self.dirs, self.home / "b1")
        second = smol.set_profiles(SELECTOR, self.block, self.dirs, self.home / "b2", prior=first)
        self.assertEqual(second, first)
        self.assertEqual(self.read(".omp/agent")[1].count(smol.BEGIN), 1)          # one block, not two
        smol.revert_profiles(second, SELECTOR, self.dirs)
        for rel, want in self.before.items():
            self.assertEqual(self.read(rel), want, rel)

    def test_a_smol_line_changed_by_hand_since_set_is_left_alone(self):
        record = smol.set_profiles(SELECTOR, self.block, self.dirs, self.home / "backup")
        cfg = self.home / ".omp/profiles/claude/agent/config.yml"
        cfg.write_text(cfg.read_text().replace(SELECTOR, "anthropic/claude-haiku-5"))
        self.assertEqual(smol.revert_profiles(record, SELECTOR, self.dirs), ["claude"])
        self.assertIn("smol: anthropic/claude-haiku-5", cfg.read_text())
        self.assertNotIn(smol.BEGIN, self.read(".omp/profiles/claude/agent")[1])   # our provider block still goes


class ParkStopsTheServer(unittest.TestCase):
    def test_park_stops_it_and_unpark_restarts_it(self):
        live = {"port": smol.PORT, "model_id": "Qwen3.8-27B-MLX-Serve-4bit", "pid": 42}
        calls = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(park, "STATE", Path(tmp) / "PARKED.json"), \
                mock.patch.object(park, "HISTORY", Path(tmp) / "hist.jsonl"), \
                mock.patch.object(park, "_tags", return_value={}), \
                mock.patch.object(park, "fallbacks", return_value=[]), \
                mock.patch.object(park.smol, "load_state", return_value=live), \
                mock.patch.object(park.smol, "server_up", return_value=True), \
                mock.patch.object(park.smol, "stop_server", lambda st: calls.append(("stop", st["pid"]))), \
                mock.patch.object(park.smol, "start_server", lambda st: calls.append(("start", st["pid"]))):
            parked = park.park()
            recorded = json.loads((Path(tmp) / "PARKED.json").read_text())
            park.park()                                                   # a second park does not stop it twice
            park.unpark()
        self.assertEqual([p["kind"] for p in parked], [park.SMOL_SERVER])
        self.assertEqual(recorded, parked)
        self.assertEqual(calls, [("stop", 42), ("start", 42)])


class FakeServer:
    """A server whose port closes at SIGTERM but whose process takes `linger` more polls to exit, as mlx-serve does."""

    def __init__(self, pid, linger):
        self.pid, self.linger, self.port_open, self.signals = pid, linger, True, []

    def kill(self, pid, sig):
        if pid != self.pid or (self.linger < 0 and not self.port_open):
            raise ProcessLookupError(pid)
        if sig == 0:
            self.linger -= 0 if self.port_open else 1
            if self.linger < 0:
                raise ProcessLookupError(pid)
            return
        self.signals.append((pid, sig))
        self.port_open = False

    def patches(self):
        return (mock.patch.object(smol, "listener_pid", lambda port: self.pid if self.port_open else None),
                mock.patch.object(smol, "server_up", lambda port: self.port_open),
                mock.patch.object(smol.os, "kill", self.kill),
                mock.patch.object(smol.time, "sleep", lambda s: None))


class StopSignalsTheListener(unittest.TestCase):
    def run_stop(self, server, state):
        with server.patches()[0], server.patches()[1], server.patches()[2], server.patches()[3]:
            smol.stop_server(state)

    def test_a_stale_state_pid_is_never_signalled(self):
        # After a reboot or a launchd restart the saved pid names some other process; stop must hit the port's owner.
        server = FakeServer(222, linger=0)
        self.run_stop(server, {"port": smol.PORT, "pid": 111})
        self.assertEqual([p for p, _ in server.signals], [222])

    def test_nothing_listening_signals_nobody(self):
        server = FakeServer(222, linger=0)
        server.port_open = False
        self.run_stop(server, {"port": smol.PORT, "pid": 111})
        self.assertEqual(server.signals, [])

    def test_stop_returns_only_after_the_process_has_exited(self):
        # mlx-serve closes its port, then shuts down for seconds; a kickstart in that gap was a no-op (2026-09-25).
        server = FakeServer(222, linger=3)
        self.run_stop(server, {"port": smol.PORT})
        self.assertLess(server.linger, 0, "stop returned while the server process was still running")


class SuiteIsolation(unittest.TestCase):
    def test_the_suite_never_reads_the_live_smol_state(self):
        # tests/__init__.py points STATE_DIR at a temp dir; were it the live one, every commit's suite run would park
        # (stop) the real smol server.
        self.assertNotEqual(smol.STATE_DIR, Path.home() / ".localbench" / "smol")
        self.assertEqual(smol.state_path().parent, smol.STATE_DIR)
        self.assertNotEqual(smol.LAUNCH_AGENTS, Path.home() / "Library" / "LaunchAgents")
        self.assertEqual(smol.plist_path().parent, smol.LAUNCH_AGENTS)


if __name__ == "__main__":
    unittest.main()
