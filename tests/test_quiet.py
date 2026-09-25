"""`localbench quiet`: which processes it may pause (omp's managed browser only), that a pause and a resume really
stop and continue a process, that a reused pid is never signalled, and that a resume cannot land mid-run."""

import contextlib
import io
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import quiet

OMP_GPU = ("~/.omp/puppeteer/chrome/mac_arm-150.0.7871.24/chrome-mac-arm64/Google Chrome for Testing.app/"
           "Contents/Frameworks/Google Chrome for Testing Framework.framework/Helpers/Google Chrome for Testing Helper "
           "(GPU).app/Contents/MacOS/Google Chrome for Testing Helper (GPU) --type=gpu-process")
USER_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TERMINAL = "/System/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal"
WINDOWSERVER = "/System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer -daemon"


def reaches(pid: int, flag: int, check, within_s: float = 3.0) -> bool:
    """Poll a child's wait status without blocking: a pause or resume that never lands fails the test instead of
    hanging it (a blocking WCONTINUED wait hung a mutation run for 5 minutes, 2026-09-24)."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        got, status = os.waitpid(pid, flag | os.WNOHANG)
        if got and check(status):
            return True
        time.sleep(0.02)
    return False


class State(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(quiet, "STATE", Path(self.tmp.name) / "quiet.json")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()


class Candidates(unittest.TestCase):
    def test_only_omps_managed_browser_is_pausable(self):
        procs = {1: OMP_GPU, 2: USER_CHROME, 3: TERMINAL, 4: WINDOWSERVER, 5: "ollama runner --model qwen3.6:35b-mlx"}
        self.assertEqual(quiet.candidates(procs), {1: OMP_GPU})


class PauseResume(State):
    def test_a_paused_process_is_stopped_and_resume_continues_it(self):
        child = subprocess.Popen(["sleep", "30"])
        try:
            procs = {child.pid: OMP_GPU}
            self.assertEqual([r["pid"] for r in quiet.pause(procs)], [child.pid])
            self.assertTrue(reaches(child.pid, os.WUNTRACED, os.WIFSTOPPED))
            self.assertEqual([r["pid"] for r in quiet.resume(procs)], [child.pid])
            self.assertTrue(reaches(child.pid, os.WCONTINUED, os.WIFCONTINUED))
            self.assertEqual(quiet.paused(), [])
        finally:
            child.kill()
            child.wait()

    def test_a_second_pause_signals_nothing_new(self):
        sent = []
        quiet.pause({7: OMP_GPU}, kill=lambda pid, sig: sent.append((pid, sig)))
        quiet.pause({7: OMP_GPU}, kill=lambda pid, sig: sent.append((pid, sig)))
        self.assertEqual(sent, [(7, signal.SIGSTOP)])
        self.assertEqual([r["pid"] for r in quiet.paused()], [7])

    def test_a_reused_pid_is_never_signalled(self):
        quiet.pause({7: OMP_GPU}, kill=lambda pid, sig: None)
        sent = []
        resumed = quiet.resume({7: "/usr/bin/some-other-program"}, kill=lambda pid, sig: sent.append((pid, sig)))
        self.assertEqual((resumed, sent, quiet.paused()), ([], [], []))


class ResumeDuringARun(State):
    def test_resume_is_refused_while_a_run_is_alive(self):
        quiet.pause({7: OMP_GPU}, kill=lambda pid, sig: None)
        with mock.patch.object(cli, "_run_alive", return_value=True), \
                mock.patch.object(quiet, "resume") as resume, contextlib.redirect_stderr(io.StringIO()) as err:
            rc = cli.main(["quiet", "--resume"])
        self.assertEqual(rc, 1)
        resume.assert_not_called()
        self.assertIn("run is alive", err.getvalue())
        self.assertEqual([r["pid"] for r in quiet.paused()], [7])


class DisplaySleep(State):
    def run_display(self, rc):
        done = subprocess.CompletedProcess([], rc)
        with mock.patch.object(cli.subprocess, "run", return_value=done) as run, \
                mock.patch.object(quiet, "processes", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = cli.main(["quiet", "--display"])
        return code, out.getvalue() + err.getvalue(), run

    def test_a_failed_display_sleep_is_reported_not_claimed(self):
        code, out, _ = self.run_display(1)
        self.assertEqual(code, 1)
        self.assertIn("the display is still on", out)
        self.assertNotIn("display asleep", out)

    def test_the_display_sleeps_only_when_asked(self):
        code, out, run = self.run_display(0)
        self.assertEqual((code, run.call_args.args[0]), (0, ["pmset", "displaysleepnow"]))
        with mock.patch.object(cli.subprocess, "run") as run, mock.patch.object(quiet, "processes", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main(["quiet"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
