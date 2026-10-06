import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "suite_verdict.sh"


class SuiteVerdict(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        (self.root / "tests").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "tests" / "__init__.py").write_text("")
        (self.root / "marker.txt").write_text("initial\n")
        (self.root / "tests" / "test_probe.py").write_text(
            "import os\nimport unittest\nfrom pathlib import Path\n"
            "class Probe(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        run_marker = os.environ.get(\"SUITE_VERDICT_TEST_RUN_MARKER\")\n"
            "        if run_marker:\n"
            "            Path(run_marker).write_text(\"ran\\n\")\n"
            "        self.assertTrue(Path(\"marker.txt\").read_text().strip() == \"tree-green\")\n"
        )
        (self.root / "scripts" / "suite_verdict.sh").write_text(SCRIPT.read_text())
        (self.root / "scripts" / "suite_verdict.sh").chmod(0o755)
        # The slot itself is tested in test_heavyslot; here heavy_run only has to run the suite in the cwd.
        (self.root / "scripts" / "heavy_run.py").write_text(
            "import subprocess, sys\n"
            "raise SystemExit(subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'])"
            ".returncode)\n")
        (self.root / "pyproject.toml").write_text("[project]\nname='probe'\nversion='0'\nrequires-python='>=3.12'\n")
        self.git(["init", "-q"])
        self.git(["config", "user.email", "test@example.invalid"])
        self.git(["config", "user.name", "test"])
        self.git(["config", "core.hooksPath", os.devnull])
        self.git(["add", "."])
        self.git(["commit", "-qm", "initial"])
        (self.root / "marker.txt").write_text("tree-green\n")
        self.git(["add", "marker.txt"])
        self.fake_uv = self.root / "fake-bin" / "uv"
        self.fake_uv.parent.mkdir()
        self.fake_uv.write_text(
            "#!/bin/sh\n"
            "if [ \"${FAKE_UV_PYVER:-}\" != \"\" ]; then\n"
            "  for arg in \"$@\"; do if [ \"$arg\" = \"-c\" ]; then shift; shift; "
            "printf '%s\\n' \"$FAKE_UV_PYVER\"; exit 0; fi; done\n"
            "fi\n"
            "while [ $# -gt 0 ] && [ \"$1\" != python ] && [ \"$1\" != python3 ]; do shift; done\n"
            "[ $# -gt 0 ] || exit 2\n"
            "shift\n"
            "exec python3 \"$@\"\n"
        )
        self.fake_uv.chmod(0o755)
        self.env = {**os.environ, "HOME": str(self.root / "home"),
                    "PATH": f"{self.fake_uv.parent}:{os.environ['PATH']}"}
        Path(self.env["HOME"]).mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, args):
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=True,
                               env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")})

    def tree(self):
        return self.git(["write-tree"]).stdout.strip()

    def run_verdict(self, tree, env=None):
        return subprocess.run(["sh", str(self.root / "scripts" / "suite_verdict.sh"), tree], cwd=self.root,
                              env={**self.env, **(env or {})}, capture_output=True, text=True, check=False)

    def test_unstaged_red_worktree_does_not_change_index_tree_verdict(self):
        tree = self.tree()
        (self.root / "tests" / "__init__.py").write_text("")
        (self.root / "tests" / "test_probe.py").write_text("import unittest\nclass Probe(unittest.TestCase):\n def test_bad(self): self.fail()\n")
        result = self.run_verdict(tree)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_same_tree_second_run_is_cached(self):
        tree = self.tree()
        first = self.run_verdict(tree)
        second = self.run_verdict(tree)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 0)
        self.assertIn("cached PASS", second.stdout)

    def test_fresh_flag_reexecutes_cached_tree_and_rewrites_cache(self):
        tree = self.tree()
        first = self.run_verdict(tree)
        self.assertEqual(first.returncode, 0, first.stderr)
        cache_dir = Path(self.env["HOME"]) / ".localbench" / "suite-verdicts"
        cache_file = next(cache_dir.glob(f"{tree}-*"))
        cached = self.run_verdict(tree)
        self.assertEqual(cached.returncode, 0, cached.stderr)
        self.assertIn("cached PASS", cached.stdout)
        run_marker = self.root / "fresh-run-marker"
        fresh = self.run_verdict(
            tree,
            {
                "SUITE_VERDICT_FRESH": "1",
                "SUITE_VERDICT_TEST_RUN_MARKER": str(run_marker),
            },
        )
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertNotIn("cached PASS", fresh.stdout)
        self.assertIn("suite_verdict: PASS for tree", fresh.stdout)
        self.assertTrue(run_marker.is_file())
        self.assertTrue(cache_file.is_file())
        cache_file.unlink()
        fresh_without_cache = self.run_verdict(tree, {"SUITE_VERDICT_FRESH": "1"})
        self.assertEqual(fresh_without_cache.returncode, 0, fresh_without_cache.stderr)
        self.assertTrue(cache_file.is_file())
        self.assertIn("cached PASS", self.run_verdict(tree).stdout)

    def test_red_tree_fails_and_is_not_cached(self):
        (self.root / "tests" / "__init__.py").write_text("")
        (self.root / "tests" / "test_probe.py").write_text("import unittest\nclass Probe(unittest.TestCase):\n def test_bad(self): self.fail()\n")
        self.git(["add", "."])
        tree = self.tree()
        first, second = self.run_verdict(tree), self.run_verdict(tree)
        for result in (first, second):
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("cached PASS", result.stdout)
            self.assertIn("suite_verdict: FAIL", result.stderr)

    def test_python_version_is_part_of_cache_key(self):
        tree = self.tree()
        first = self.run_verdict(tree, {"FAKE_UV_PYVER": "3.14.0"})
        second = self.run_verdict(tree, {"FAKE_UV_PYVER": "3.14.1"})
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 0)
        self.assertNotIn("cached PASS", second.stdout)

    def test_busy_slot_is_not_a_verdict_and_is_not_cached(self):
        """The commit hook must not queue for the slot while holding index.lock (2026-10-03: two stale locks)."""
        (self.root / "scripts" / "heavy_run.py").write_text("raise SystemExit(75)\n")
        tree = self.tree()
        busy = self.run_verdict(tree, {"SUITE_VERDICT_WAIT": "0"})
        self.assertEqual(busy.returncode, 75)
        self.assertIn("heavy slot busy", busy.stderr)
        self.assertNotIn("FAIL", busy.stderr)
        (self.root / "scripts" / "heavy_run.py").write_text(
            "import subprocess, sys\n"
            "raise SystemExit(subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'])"
            ".returncode)\n")
        self.assertNotIn("cached PASS", self.run_verdict(tree).stdout)



    def test_fresh_zero_keeps_cached_pass(self):
        tree = self.tree()
        first = self.run_verdict(tree)
        self.assertEqual(first.returncode, 0, first.stderr)
        zero = self.run_verdict(tree, {"SUITE_VERDICT_FRESH": "0"})
        self.assertEqual(zero.returncode, 0, zero.stderr)
        self.assertIn("cached PASS", zero.stdout)

if __name__ == "__main__":
    unittest.main()
