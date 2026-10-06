"""scripts/land.py: lock refusal, hunk selection, mutation-gate selection."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import land


def git(*args, cwd, env=None):
    base = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    if env:
        base.update(env)
    # A fixture repo, not a commit of this project: the global commit-msg hook does not apply.
    if args and args[0] in ("commit", "add"):
        args = ("-c", "core.hooksPath=/dev/null", *args)
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=base,
                          check=True)


class LockState(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="land-lock-"))
        (self.tmp / ".git").mkdir()

    def lock(self, age_s):
        path = self.tmp / ".git" / "index.lock"
        path.write_text("x")
        old = __import__("time").time() - age_s
        os.utime(path, (old, old))
        return str(path)

    def test_no_lock_proceeds(self):
        self.assertIsNone(land.lock_state(str(self.tmp)))

    def test_stale_lock_refuses_with_rm_command(self):
        lock = self.lock(300)
        real, land._lock_holders = land._lock_holders, lambda p: []
        try:
            state = land.lock_state(str(self.tmp))
        finally:
            land._lock_holders = real
        self.assertIn("stale", state)
        self.assertIn(f"rm {lock}", state)
        self.assertNotIn("lsof is unavailable", state)

    def test_stale_lock_refusal_never_removes_lock(self):
        lock = self.lock(300)
        real, land._lock_holders = land._lock_holders, lambda p: []
        try:
            state = land.lock_state(str(self.tmp))
        finally:
            land._lock_holders = real
        self.assertIn(f"rm {lock}", state)
        self.assertTrue(Path(lock).is_file(), "land.py must never self-remove a stale lock")

    def test_held_lock_names_holder_and_never_suggests_rm(self):
        self.lock(300)
        real, land._lock_holders = land._lock_holders, lambda p: ["1234"]
        try:
            state = land.lock_state(str(self.tmp))
        finally:
            land._lock_holders = real
        self.assertIn("1234", state)
        self.assertNotIn("rm ", state)

    def test_fresh_lock_asks_for_retry(self):
        self.lock(5)
        real, land._lock_holders = land._lock_holders, lambda p: []
        try:
            state = land.lock_state(str(self.tmp))
        finally:
            land._lock_holders = real
        self.assertIn("retry", state)
        self.assertNotIn("rm ", state)

    def test_missing_lsof_fails_closed(self):
        self.lock(300)
        real, land._lock_holders = land._lock_holders, lambda p: None
        try:
            state = land.lock_state(str(self.tmp))
        finally:
            land._lock_holders = real
        self.assertIn("lsof is unavailable", state)
        self.assertNotIn("rm ", state)


class EnvAndSpecs(unittest.TestCase):
    def test_clean_env_drops_git_vars_keeps_rest(self):
        env = {"PATH": "/x", "GIT_DIR": "/y", "GIT_INDEX_FILE": "/z", "HOME": "/h"}
        real = dict(os.environ)
        os.environ.clear()
        os.environ.update(env)
        try:
            self.assertEqual(land.clean_env(), {"PATH": "/x", "HOME": "/h"})
        finally:
            os.environ.clear()
            os.environ.update(real)

    def test_split_spec_whole_and_hunk(self):
        self.assertEqual(land.split_spec("a/b.py"), ("a/b.py", None))
        path, pattern = land.split_spec(r"a/b.py::def foo")
        self.assertEqual(path, "a/b.py")
        self.assertIn("def foo", pattern)

    def test_changed_hunks_parses_new_side_ranges(self):
        diff = ("+++ b/x.py\n@@ -3,0 +4,2 @@\n+a\n+b\n"
                "+++ b/y.py\n@@ -10,3 +11,1 @@\n+c\n")
        self.assertEqual(land.changed_hunks(diff), {"x.py": [(4, 5)], "y.py": [(11, 11)]})

    def test_select_hunks_keeps_only_matches(self):
        old = "one\ntwo\nthree\nfour\n"
        diff = ("--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n-one\n+ONE\n two\n"
                "@@ -3,1 +3,1 @@\n-three\n+THREE\n")
        self.assertEqual(land.select_hunks(diff, old, "THREE"),
                         "one\ntwo\nTHREE\nfour\n")

    def test_selector_treats_regex_metacharacters_as_literal_text(self):
        old = "one\nONE\nfour\n"
        diff = "--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n-one\n+ONE\n"
        self.assertEqual(land.select_hunks(diff, old, "O.E"), old)

    def test_select_hunks_refuses_a_moved_tree(self):
        old = "one\nTWO\nthree\nfour\n"
        diff = "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n-one\n+ONE\n two\n"
        with self.assertRaises(SystemExit):
            land.select_hunks(diff, old, "ONE")


class AdvanceMain(unittest.TestCase):
    class Result:
        def __init__(self, rc=0, stdout=""):
            self.returncode, self.stdout, self.stderr = rc, stdout, ""

    def test_compare_and_swap_prevents_rewinding_a_racing_main(self):
        calls = []
        def runner(argv, cwd, **kwargs):
            calls.append(argv)
            if argv[1] == "rev-parse":
                return self.Result(stdout="oldsha\n")
            if argv[1] == "merge-base":
                return self.Result()
            if argv[1] == "update-ref":
                return self.Result(rc=1)  # main advanced after the ancestry check
            return self.Result()
        ok, reason = land.advance_main("/repo", "newsha", ["a.py"], runner)
        self.assertFalse(ok)
        self.assertIn("compare-and-swap", reason)
        self.assertEqual([c[1] for c in calls], ["rev-parse", "merge-base", "update-ref"])
        self.assertEqual(calls[-1][-1], "oldsha")

    def test_main_advanced_after_worktree_cut_is_not_rewound(self):
        with tempfile.TemporaryDirectory(prefix="land-cas-") as tmp:
            root = Path(tmp)
            git("init", "-q", cwd=root)
            git("config", "user.name", "t", cwd=root)
            git("config", "user.email", "t@t", cwd=root)
            git("config", "commit.gpgsign", "false", cwd=root)
            (root / "value.txt").write_text("base\n")
            git("add", "value.txt", cwd=root)
            git("commit", "-q", "-m", "base", cwd=root)
            git("branch", "-M", "main", cwd=root)
            base = git("rev-parse", "HEAD", cwd=root).stdout.strip()
            # Mimic land.py's detached worktree commit from the observed main.
            git("checkout", "-q", "-b", "worker", cwd=root)
            (root / "worker.txt").write_text("landed candidate\n")
            git("add", "worker.txt", cwd=root)
            git("commit", "-q", "-m", "candidate", cwd=root)
            candidate = git("rev-parse", "HEAD", cwd=root).stdout.strip()
            git("checkout", "-q", "main", cwd=root)
            (root / "other.txt").write_text("racing main commit\n")
            git("add", "other.txt", cwd=root)
            git("commit", "-q", "-m", "racing main", cwd=root)
            before = git("rev-parse", "refs/heads/main", cwd=root).stdout.strip()
            ok, reason = land.advance_main(str(root), candidate, ["worker.txt"])
            after = git("rev-parse", "refs/heads/main", cwd=root).stdout.strip()
            self.assertFalse(ok)
            self.assertIn("not an ancestor", reason)
            self.assertEqual(after, before)
            self.assertNotEqual(after, base)

    def test_real_race_after_ancestry_check_is_rejected_by_cas(self):
        with tempfile.TemporaryDirectory(prefix="land-cas-race-") as tmp:
            root = Path(tmp)
            git("init", "-q", cwd=root)
            git("config", "user.name", "t", cwd=root)
            git("config", "user.email", "t@t", cwd=root)
            (root / "base.txt").write_text("base\n")
            git("add", "base.txt", cwd=root)
            git("commit", "-q", "-m", "base", cwd=root)
            git("branch", "-M", "main", cwd=root)
            git("checkout", "-q", "-b", "worker", cwd=root)
            (root / "candidate.txt").write_text("candidate\n")
            git("add", "candidate.txt", cwd=root)
            git("commit", "-q", "-m", "candidate", cwd=root)
            candidate = git("rev-parse", "HEAD", cwd=root).stdout.strip()
            git("checkout", "-q", "main", cwd=root)
            raced = []

            def runner(argv, cwd, **kwargs):
                result = land.land_cmd(argv, cwd, **kwargs)
                if argv[1] == "merge-base":
                    (root / "racer.txt").write_text("new main commit\n")
                    git("add", "racer.txt", cwd=root)
                    git("commit", "-q", "-m", "concurrent main", cwd=root)
                    raced.append(git("rev-parse", "HEAD", cwd=root).stdout.strip())
                return result

            ok, reason = land.advance_main(str(root), candidate, ["candidate.txt"], runner)
            after = git("rev-parse", "refs/heads/main", cwd=root).stdout.strip()
            self.assertFalse(ok)
            self.assertIn("compare-and-swap", reason)
            self.assertEqual(after, raced[0])
            self.assertNotEqual(after, candidate)
            self.assertEqual((root / "racer.txt").read_text(), "new main commit\n")

    def test_valid_fast_forward_advances_main(self):
        with tempfile.TemporaryDirectory(prefix="land-cas-fast-forward-") as tmp:
            root = Path(tmp)
            git("init", "-q", cwd=root)
            git("config", "user.name", "t", cwd=root)
            git("config", "user.email", "t@t", cwd=root)
            (root / "base.txt").write_text("base\n")
            git("add", "base.txt", cwd=root)
            git("commit", "-q", "-m", "base", cwd=root)
            git("branch", "-M", "main", cwd=root)
            git("checkout", "-q", "-b", "worker", cwd=root)
            (root / "candidate.txt").write_text("candidate\n")
            git("add", "candidate.txt", cwd=root)
            git("commit", "-q", "-m", "candidate", cwd=root)
            candidate = git("rev-parse", "HEAD", cwd=root).stdout.strip()
            git("checkout", "-q", "main", cwd=root)

            ok, reason = land.advance_main(str(root), candidate, ["candidate.txt"])

            self.assertTrue(ok, reason)
            self.assertEqual(git("rev-parse", "refs/heads/main", cwd=root).stdout.strip(), candidate)
    def test_success_updates_ref_before_reset(self):
        calls = []
        def runner(argv, cwd, **kwargs):
            calls.append(argv)
            return self.Result(stdout="oldsha\n" if argv[1] == "rev-parse" else "")
        ok, reason = land.advance_main("/repo", "newsha", ["a.py"], runner)
        self.assertTrue(ok, reason)
        self.assertEqual([c[1] for c in calls], ["rev-parse", "merge-base", "update-ref", "reset"])
        self.assertEqual(calls[2][-2:], ["newsha", "oldsha"])


class GateHelpers(unittest.TestCase):
    def test_last_line_empty_is_not_an_index_error(self):
        self.assertEqual(land.last_line(""), "(empty)")
        self.assertEqual(land.last_line("a\nb\n"), "b")

    def test_mutation_problems_flags_bad_lines(self):
        out = '{"label": "x", "caught": true}\n{"label": "y", "caught": false}\n'
        self.assertEqual(len(land.mutation_problems(out)), 1)
        self.assertEqual(land.mutation_problems('{"caught": true}\n'), [])
        skipped = 'result line\n"skipped": true\n'
        self.assertEqual(len(land.mutation_problems(skipped)), 1)

    def test_parse_conflicts_ignores_own_holds(self):
        text = ("1 conflict(s) found:\n"
                "PATH          HOLDER       PATTERN       EXPIRES\n"
                "scripts/a.py  SnowyBeacon  scripts/a.py  2026-10-02T22:20:21.\n"
                "reservation_read_attestation: state=current\n")
        self.assertEqual(land.parse_conflicts(text, "snowybeacon"), [])
        blockers = land.parse_conflicts(text, "OtherPane")
        self.assertEqual(blockers, ["scripts/a.py held by SnowyBeacon"])

    def test_parse_conflicts_empty_is_clear(self):
        self.assertEqual(land.parse_conflicts("no conflicts found\n", "Anyone"), [])

    def test_unreachable_am_refuses_without_owner_confirm(self):
        real_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ""
        try:
            self.assertEqual(land.check_reservations("/repo", "me", ["f.py"], False), 1)
        finally:
            os.environ["PATH"] = real_path

    def test_unreachable_am_records_owner_confirmed_bypass(self):
        real_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ""
        try:
            self.assertEqual(land.check_reservations("/repo", "me", ["f.py"], True), 0)
        finally:
            os.environ["PATH"] = real_path

    def test_gate_fails_on_content_not_just_rc(self):
        self.assertTrue(land.gate_failed(0, ['{"caught": false}']))
        self.assertTrue(land.gate_failed(1, []))
        self.assertFalse(land.gate_failed(0, []))


class MutationSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="land-mut-"))
        git("init", "-q", cwd=self.tmp)
        git("config", "user.name", "t", cwd=self.tmp)
        git("config", "user.email", "t@t", cwd=self.tmp)
        git("config", "commit.gpgsign", "false", cwd=self.tmp)
        (self.tmp / "tests").mkdir()
        (self.tmp / "mod.py").write_text("alpha = 1\nbeta = 2\n")
        inside = [{"label": "flip gamma", "file": "mod.py", "old": "beta = 2\ngamma = 3",
                   "new": "beta = 2\ngamma = 4", "tests": ["tests.test_land"]}]
        outside = [{"label": "flip alpha", "file": "mod.py", "old": "alpha = 1",
                    "new": "alpha = 2", "tests": ["tests.test_land"]}]
        (self.tmp / "tests" / "a-mutations.json").write_text(json.dumps(inside))
        (self.tmp / "tests" / "b-mutations.json").write_text(json.dumps(outside))
        git("add", ".", cwd=self.tmp)
        git("commit", "-q", "-m", "base", cwd=self.tmp)
        (self.tmp / "mod.py").write_text("alpha = 1\nbeta = 2\ngamma = 3\n")

    def test_case_in_changed_hunk_is_selected(self):
        self.assertEqual(land.select_mutations(str(self.tmp), []), ["a"])

    def test_case_outside_changed_hunk_is_not_selected(self):
        (self.tmp / "tests" / "a-mutations.json").unlink()
        self.assertEqual(land.select_mutations(str(self.tmp), []), [])

    def test_duplicate_anchor_aborts(self):
        (self.tmp / "mod.py").write_text("alpha = 1\nbeta = 2\ngamma = 3\nbeta = 2\ngamma = 3\n")
        git("add", ".", cwd=self.tmp)
        with self.assertRaises(SystemExit):
            land.select_mutations(str(self.tmp), [])


if __name__ == "__main__":
    unittest.main()
