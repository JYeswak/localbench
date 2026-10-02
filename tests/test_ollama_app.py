"""`localbench ollama-app`: Ollama.app is aligned only when it runs from /Applications and its server is up; a
restart gets there even though the app refuses to quit (2026-09-27: AppleScript -128, SIGTERM ignored)."""

import contextlib
import io
import signal
import subprocess
import unittest
from unittest import mock

from localbench import __main__ as cli
from localbench import ollama_app as oa

STALE = "/private/tmp/Ollama-0.34.2-replaced.app/Contents/MacOS/Ollama"


class FakeMac:
    """A process table with Ollama.app and its server. The app ignores quit and SIGTERM, dies on SIGKILL; `open`
    starts a fresh app from APP, which spawns a server `spawn_after` polls later (never, when None)."""

    def __init__(self, image=STALE, spawn_after: int | None = 1):
        self.app, self.image, self.serve = 100, image, 200
        self.spawn_after, self.polls, self.next_pid = spawn_after, 0, 300
        self.log: list = []

    def run(self, argv, **kw):
        out = ""
        if argv[0] == "pgrep":
            pattern = argv[-1]
            if pattern == oa.APP_MATCH and self.app:
                out = f"{self.app}\n"
            elif pattern == oa.SERVE_MATCH:
                if self.app and not self.serve and self.spawn_after is not None:
                    self.polls += 1
                    if self.polls > self.spawn_after:
                        self.serve, self.next_pid = self.next_pid, self.next_pid + 1
                out = f"{self.serve}\n" if self.serve else ""
        elif argv[0] == "lsof" and self.app:
            out = f"p{self.app}\nfcwd\ntDIR\nn/\nftxt\ntREG\nn{self.image}\n"
        elif argv[0] in ("osascript", "open"):
            self.log.append(argv[0])
            if argv[0] == "open" and not self.app:
                self.app, self.image = self.next_pid, oa.APP_EXE
                self.next_pid += 1
        return subprocess.CompletedProcess(argv, 0, out, "")

    def kill(self, pid, sig):
        self.log.append((pid, sig))
        if pid == self.app and sig == signal.SIGKILL:
            self.app = None
        elif pid == self.serve and sig in (signal.SIGTERM, signal.SIGKILL):
            self.serve = None


class Alignment(unittest.TestCase):
    def test_an_app_running_a_replaced_image_is_not_aligned(self):
        for image, aligned in ((STALE, False), (oa.APP_EXE, True)):
            with mock.patch.object(oa.subprocess, "run", FakeMac(image=image).run):
                self.assertEqual(oa.state().aligned, aligned)


def oa_restart(mac, up, **kw):
    with mock.patch.object(oa.subprocess, "run", mac.run):
        return oa.restart(up, **kw)


class Restart(unittest.TestCase):
    def test_a_refusing_app_is_killed_its_server_stopped_and_the_bundle_reopened(self):
        mac = FakeMac()
        st = oa_restart(mac, lambda: True, kill=mac.kill, sleep=lambda s: None)
        self.assertTrue(st.aligned)
        self.assertEqual(st.app_image, oa.APP_EXE)
        self.assertIn((100, signal.SIGKILL), mac.log, "the app ignores quit; it must be killed")
        self.assertIn((200, signal.SIGTERM), mac.log)
        self.assertLess(mac.log.index((200, signal.SIGTERM)), mac.log.index("open"), "reopen after the server stops")

    def test_an_app_that_never_brings_its_server_back_is_a_named_failure(self):
        mac = FakeMac(spawn_after=None)
        with self.assertRaises(RuntimeError) as err:
            oa_restart(mac, lambda: True, kill=mac.kill, sleep=lambda s: None, ready_timeout=6)
        self.assertIn("did not come back", str(err.exception))

    def test_a_server_that_does_not_answer_is_not_back(self):
        mac = FakeMac()
        with self.assertRaises(RuntimeError):
            oa_restart(mac, lambda: False, kill=mac.kill, sleep=lambda s: None, ready_timeout=6)


class Cli(unittest.TestCase):
    """`localbench ollama-app restart` restarts the server every omp session and a run depend on: refused while a run
    is alive or models are parked, and a dry run touches nothing."""

    def setUp(self):
        self.restart = mock.patch.object(oa, "restart").start()
        mock.patch.object(oa, "state", return_value=oa.State(100, STALE, 200)).start()
        self.alive = mock.patch.object(cli, "_run_alive", return_value=False).start()
        self.parked = mock.patch.object(cli.park, "parked_now", return_value=[]).start()
        self.addCleanup(mock.patch.stopall)

    def main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(["ollama-app", *argv])

    def test_refused_while_models_are_parked(self):
        self.parked.return_value = [{"name": "qwen3.8:27b-mlx"}]
        self.assertEqual(self.main("restart"), 1)
        self.restart.assert_not_called()

    def test_refused_while_a_run_is_alive(self):
        self.alive.return_value = True
        self.assertEqual(self.main("restart"), 1)
        self.restart.assert_not_called()

    def test_a_dry_run_restarts_nothing(self):
        self.assertEqual(self.main("restart", "--dry-run"), 0)
        self.restart.assert_not_called()

    def test_status_of_a_stale_app_exits_1(self):
        self.assertEqual(self.main("status"), 1)


if __name__ == "__main__":
    unittest.main()
