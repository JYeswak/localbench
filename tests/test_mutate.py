"""The mutation runner is the instrument every cross-grade rests on: a plant must really run, a restore must really
restore, and two graders must not plant in one tree at once."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location("mutate", Path(__file__).resolve().parent.parent / "scripts" / "mutate.py")
assert SPEC and SPEC.loader, "scripts/mutate.py not found"
mut = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mut
SPEC.loader.exec_module(mut)

CASE = {"label": "x", "file": "m.py", "old": "X = 1", "new": "X = 2", "tests": ["test_m"]}


class Tree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "m.py").write_text("X = 1\n")
        (self.root / "test_m.py").write_text(
            "import unittest\nimport m\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
            "        self.assertEqual(m.X, 1)\n")

    def tearDown(self):
        self.tmp.cleanup()

    def sha(self) -> str:
        return hashlib.sha256((self.root / "m.py").read_bytes()).hexdigest()

    def detects(self, tests):
        """A runner whose tests catch the plant: they pass only while m.py says X = 1."""
        return (0, []) if "X = 1" in (self.root / "m.py").read_text() else (1, ["FAIL: test_x"])


class FreshBytecode(Tree):
    def test_a_same_size_plant_with_the_cached_mtime_is_executed(self):
        plain = [sys.executable, "-m", "unittest", "test_m"]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE")}
        subprocess.run(plain, cwd=self.root, capture_output=True, check=True, env=env)   # caches X = 1
        m = self.root / "m.py"
        st = m.stat()
        m.write_text("X = 2\n")                                                          # same size
        os.utime(m, ns=(st.st_atime_ns, st.st_mtime_ns))                                 # same mtime
        self.assertEqual(subprocess.run(plain, cwd=self.root, capture_output=True, env=env, check=False).returncode, 0,
                         "control: a plain run should load the stale cache and miss the plant")
        # Under the mutation runner this process inherits a fresh cache prefix; the runner must bring its own.
        with mock.patch.dict(os.environ, env, clear=True):
            rc, fails = mut.unittest_runner(self.root)(["test_m"])
        self.assertEqual(rc, 1)
        self.assertEqual(fails, ["FAIL: test_x"])


class RunCase(Tree):
    def test_a_plant_the_tests_catch_is_caught_and_restored(self):
        before = self.sha()
        r = mut.run_case(CASE, self.root, self.detects)
        self.assertEqual((r["caught"], r["restored"], r["fails"]), (True, True, ["FAIL: test_x"]))
        self.assertEqual(self.sha(), before)

    def test_a_plant_the_tests_miss_is_not_caught(self):
        self.assertFalse(mut.run_case(CASE, self.root, lambda tests: (0, []))["caught"])

    def test_a_failing_known_good_is_not_a_catch(self):
        self.assertFalse(mut.run_case(CASE, self.root, lambda tests: (1, ["FAIL: test_x"]))["caught"])

    def test_an_anchor_that_is_not_unique_is_skipped_untouched(self):
        (self.root / "m.py").write_text("X = 1\nX = 1\n")
        before = self.sha()
        r = mut.run_case(CASE, self.root, self.detects)
        self.assertIn("skipped", r)
        self.assertEqual(self.sha(), before)

    def test_the_file_is_restored_when_the_planted_run_blows_up(self):
        before = self.sha()
        calls = []

        def boom(tests):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("runner died mid-plant")
            return (0, [])
        with self.assertRaises(RuntimeError):
            mut.run_case(CASE, self.root, boom)
        self.assertEqual(self.sha(), before)


class Main(Tree):
    """The exit code is what a grader reads: 0 must mean every plant was caught and restored."""

    def main(self, runner, cases=None) -> tuple[int, list[dict]]:
        path = self.root / "cases.json"
        path.write_text(json.dumps(cases or [CASE]))
        buf = io.StringIO()
        with mock.patch.object(mut, "ROOT", self.root), \
                mock.patch.object(mut, "LOCK", self.root / ".mutation.lock"), \
                mock.patch.object(mut, "unittest_runner", return_value=runner), \
                contextlib.redirect_stdout(buf):
            rc = mut.main([str(path)])
        return rc, [json.loads(ln) for ln in buf.getvalue().splitlines()]

    def test_exit_0_only_when_every_plant_is_caught(self):
        rc, rows = self.main(self.detects)
        self.assertEqual((rc, rows[0]["caught"]), (0, True))

    def test_an_escaped_plant_exits_1(self):
        self.assertEqual(self.main(lambda tests: (0, []))[0], 1)

    def test_a_later_catch_does_not_hide_an_earlier_escape(self):
        # n.py's plant escapes (the tests only read m.py); m.py's is caught. Last-case-wins would exit 0.
        (self.root / "n.py").write_text("Y = 1\n")
        escape = {"label": "escapes", "file": "n.py", "old": "Y = 1", "new": "Y = 2", "tests": ["test_m"]}
        rc, rows = self.main(self.detects, [escape, CASE])
        self.assertEqual([r["caught"] for r in rows], [False, True])
        self.assertEqual(rc, 1)

    def test_a_caught_plant_left_in_the_tree_exits_1(self):
        # A restore that did not match the original sha means the plant may still be in the code.
        left = {"label": "x", "caught": True, "restored": False, "good_rc": 0, "bad_rc": 1, "fails": []}
        with mock.patch.object(mut, "run_case", return_value=left):
            rc, rows = self.main(self.detects)
        self.assertEqual((rc, rows[0]["restored"]), (1, False))

    def test_an_empty_case_list_is_a_usage_error_not_success(self):
        path = self.root / "cases.json"
        path.write_text("[]")
        with mock.patch.object(mut, "LOCK", self.root / ".mutation.lock"), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(mut.main([str(path)]), 2)
        self.assertIn("nothing was planted", err.getvalue())

    def test_an_empty_list_is_refused_at_once_while_another_grader_holds_the_lock(self):
        # The refusal must not wait on the lock: inside held() an empty file would sit out another grader's run.
        lock = self.root / ".mutation.lock"
        lock.mkdir()                                               # another grader holds it
        (lock / "owner").write_text("pid 4242 grok (%21) since Wed Sep 23 20:40:00 2026\n")
        path = self.root / "cases.json"
        path.write_text("[]")
        snapshot = lambda: {p.name: p.read_bytes() for p in sorted(lock.iterdir())}
        before = snapshot()
        real = mut.held
        with mock.patch.object(mut, "LOCK", lock), \
                mock.patch.object(mut, "held", lambda lk: real(lk, wait_s=0.2, poll_s=0.05)), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(mut.main([str(path)]), 2)
        # Every entry and its bytes: a stray file would make the holder's release (unlink owner, rmdir) fail and
        # leave the lock stuck for everyone after it.
        self.assertEqual(snapshot(), before, "the other grader's lock entries and their bytes must be left alone")

    def test_a_skipped_case_exits_1(self):
        rc, rows = self.main(self.detects, [{**CASE, "old": "not in the file"}])
        self.assertEqual(rc, 1)
        self.assertIn("skipped", rows[0])


class Lock(unittest.TestCase):
    def test_a_second_grader_waits_and_then_gives_up_while_the_first_holds_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".mutation.lock"
            with mut.held(lock), self.assertRaises(TimeoutError), mut.held(lock, wait_s=0.2, poll_s=0.05):
                pass
            with mut.held(lock, wait_s=0.2, poll_s=0.05):
                self.assertTrue(lock.is_dir())
            self.assertFalse(lock.exists())

    def test_the_second_grader_proceeds_once_the_first_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".mutation.lock"
            order = []

            def first():
                with mut.held(lock):
                    order.append("first in")
                    time.sleep(0.3)
                    order.append("first out")
            t = threading.Thread(target=first)
            t.start()
            time.sleep(0.1)
            with mut.held(lock, wait_s=5, poll_s=0.05):
                order.append("second in")
            t.join()
            self.assertEqual(order, ["first in", "first out", "second in"])

    def test_the_lock_is_released_when_the_holder_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".mutation.lock"
            with self.assertRaises(ValueError), mut.held(lock):
                raise ValueError
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
