"""CLI contracts that must hold before any backend is touched: overlay parsing, the golden-write refusals (the golden
for a spec is its default launch; variants are measured as ab's B leg), and what a banked receipt keeps."""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as main_mod
from localbench import smol
from localbench.__main__ import DETAILS_NOT_BANKED, _overlay, _receipt_view
from localbench.workloads import FIXTURES, MEM_CONFIG, TIERS

ROOT = Path(__file__).resolve().parent.parent
FTS = FIXTURES / "omp" / "child-config-mem-fts.yml"


# A CLI subprocess does not see tests/__init__.py's redirects; under this HOME its audit rows and smol state are
# scratch, never the user's ~/.localbench (2026-09-25: the aa refusal tests wrote rows into the live ledger).
SCRATCH_HOME = tempfile.mkdtemp(prefix="localbench-test-home-")


def cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "localbench", *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=60, env={**os.environ, "HOME": SCRATCH_HOME, **(env or {})})


class Overlay(unittest.TestCase):
    def test_explicit_default_path_is_the_default(self):
        self.assertEqual(_overlay("fixtures/omp/child-config-mem.yml"), MEM_CONFIG)

    def test_missing_file_is_a_usage_error(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _overlay("/nonexistent/overlay.yml")


class GoldenWriteRefusals(unittest.TestCase):
    """Refusals exit before open_backend: a nonexistent model name proves nothing was contacted."""

    def test_variant_overlay_cannot_write_a_golden(self):
        p = cli("aa", "ollama:no-such-model", "--tiers", "mem", "--mem-config", str(FTS), "--write-golden")
        self.assertEqual(p.returncode, 1)
        self.assertIn("refusing --write-golden", p.stderr)

    def test_server_flag_cannot_write_a_golden(self):
        p = cli("aa", "mlx-serve:/nonexistent", "--server-arg=--mtp", "--write-golden")
        self.assertEqual(p.returncode, 1)
        self.assertIn("refusing --write-golden", p.stderr)

    def test_a_non_default_round_count_cannot_write_a_golden(self):
        p = cli("aa", "ollama:no-such-model", "--tiers", "mem", "--mem-rounds", "9", "--write-golden")
        self.assertEqual(p.returncode, 1)
        self.assertIn("refusing --write-golden", p.stderr)
        self.assertIn("--mem-rounds", p.stderr)

    def test_zero_rounds_is_a_usage_error_before_any_backend(self):
        p = cli("aa", "ollama:no-such-model", "--mem-rounds", "0", "--write-golden")
        self.assertEqual(p.returncode, 2)
        self.assertIn("at least 1", p.stderr)
        self.assertNotIn("refusing --write-golden", p.stderr)


class ReceiptView(unittest.TestCase):
    def test_receipts_keep_the_details_of_every_tier_a_reader_audits(self):
        # Dropped until caught: mem and sess (2026-09-23), think (2026-09-24: per-question answers and cut-offs, the
        # evidence behind think.accuracy). The kept set is every tier outside DETAILS_NOT_BANKED, so a new tier in
        # TIERS is banked without anyone remembering to list it.
        results = [{"case": f"{t}.x", "tier": t, "detail": {"why": t}} for t in TIERS]
        summary = {"provenance": {}, "verdicts": {}, "metrics": {}, "conformance": {}, "run_dir": "r",
                   "results": results, "system": {"before": {"live": {}, "host": {}}}}
        kept = set(_receipt_view(summary)["details"])
        for t in ("e2e", "rel", "relcold", "relfresh", "mem", "sess", "think"):
            self.assertIn(f"{t}.x", kept)
        self.assertEqual(kept, {f"{t}.x" for t in TIERS} - {f"{t}.x" for t in DETAILS_NOT_BANKED})
        self.assertLessEqual(DETAILS_NOT_BANKED, set(TIERS))


class PureJsonStdout(unittest.TestCase):
    """`--json` read commands print one JSON document on stdout and nothing else (agents pipe them into jq)."""

    def test_memory_json(self):
        json.loads(cli("memory", "--json").stdout)

    @unittest.skipUnless((ROOT / "runs" / "observe.db").exists(), "no runs/observe.db (localbench watch never ran)")
    def test_report_json(self):
        json.loads(cli("report", "--since", "1h", "--json").stdout)


class Version(unittest.TestCase):
    def test_version_needs_no_subcommand_and_names_the_installed_release(self):
        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        p = cli("--version")
        self.assertEqual((p.returncode, p.stdout.strip()), (0, f"localbench {version}"))


class DataRoot(unittest.TestCase):
    """A non-editable install resolves the data root to site-packages; every answer from there is silently empty."""

    def test_a_root_without_fixtures_is_refused_with_a_way_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = cli("status", env={"LOCALBENCH_HOME": tmp})
        self.assertEqual((p.returncode, p.stdout), (2, ""))
        self.assertIn("LOCALBENCH_HOME=<clone>", p.stderr)
        self.assertIn("uv tool install -e .", p.stderr)

    def test_localbench_home_is_where_state_is_read_and_an_empty_window_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "fixtures" / "omp").mkdir(parents=True)
            as_json = cli("report", "--since", "1h", "--json", env={"LOCALBENCH_HOME": tmp})
            text = cli("report", "--since", "1h", env={"LOCALBENCH_HOME": tmp})
            db = (Path(tmp) / "runs" / "observe.db").exists()
        self.assertTrue(db, "report read another root's runs/observe.db")
        r = json.loads(as_json.stdout)
        self.assertEqual((as_json.returncode, r["samples"], r["gpu"], r["traffic"]), (0, 0, [], []))
        self.assertEqual((text.returncode, text.stdout), (0, ""))
        self.assertIn("no samples", text.stderr)


class TeachingErrors(unittest.TestCase):
    """Bad input is a usage error (exit 2) that says what to give instead, never a traceback, and never on stdout."""

    def assert_usage(self, p: subprocess.CompletedProcess, hint: str):
        self.assertEqual((p.returncode, p.stdout), (2, ""), p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn(hint, p.stderr)

    def test_a_window_that_is_not_a_duration(self):
        self.assert_usage(cli("report", "--since", "banana"), "90m, 24h, 7d")

    def test_a_run_dir_without_a_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_usage(cli("compare", tmp), "runs/")
            self.assert_usage(cli("bank", tmp, "x"), "runs/")

    def test_a_file_that_is_not_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md") as f:
            f.write("# a markdown receipt\n")
            f.flush()
            self.assert_usage(cli("show", f.name), "receipt .json")


class SmolActionsParse(unittest.TestCase):
    def test_every_action_reaches_its_handler(self):
        # A flag read by cmd_smol but defined on no parser crashed `smol revert` with AttributeError (2026-09-25).
        # With a run alive every changing action refuses before touching a server or a profile.
        for action in ("set", "start", "stop", "autostart", "revert", "status"):
            err = io.StringIO()
            with self.subTest(action), mock.patch.object(smol, "load_state", return_value=None), \
                    mock.patch.object(main_mod, "_run_alive", return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(err):
                rc = main_mod.main(["smol", action])
                if action == "status":
                    self.assertEqual((rc, out.getvalue().startswith("smol: not managed")), (0, True))
                else:
                    self.assertEqual((rc, out.getvalue()), (1, ""))
                    self.assertIn("run is alive", err.getvalue())


if __name__ == "__main__":
    unittest.main()
