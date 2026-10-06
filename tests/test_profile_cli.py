from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import profile_cli


class CliProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec_path = self.root / "profile.json"
        self.output = self.root / "profile-run"
        self.spec = {
            "schema_version": "localbench.cli-profile.v1",
            "name": "synthetic",
            "samples": 2,
            "warmups": 0,
            "commands": [
                {"id": "ok", "argv": ["ok"]},
                {"id": "mixed", "argv": ["mixed"]},
            ],
        }
        self.spec_path.write_text(json.dumps(self.spec))

    def test_resume_skips_complete_command_and_ranks_exit_cohorts_separately(self):
        calls = {"ok": 0, "mixed": 0}

        def runner(argv):
            command = argv[0]
            calls[command] += 1
            if command == "mixed" and calls[command] == 1:
                raise KeyboardInterrupt
            return {
                "elapsed_s": 0.25 if command == "ok" else 0.5,
                "stdout": f"{command} stdout {calls[command]}",
                "stderr": f"{command} stderr {calls[command]}",
                "returncode": 0 if command == "ok" or calls[command] == 2 else 7,
            }

        with self.assertRaises(KeyboardInterrupt):
            profile_cli.run_profile(self.spec_path, self.output, runner=runner)
        self.assertEqual(calls, {"ok": 2, "mixed": 1})
        self.assertTrue((self.output / "commands" / "ok.json").is_file())
        self.assertFalse((self.output / "commands" / "mixed.json").exists())
        interrupted = profile_cli.build_report(self.output)
        self.assertEqual((interrupted["status"], interrupted["missing_commands"]), ("INCOMPLETE", ["mixed"]))

        profile_cli.run_profile(self.spec_path, self.output, resume=True, runner=runner)
        self.assertEqual(calls, {"ok": 2, "mixed": 3})
        report = profile_cli.build_report(self.output)
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual([row["command"] for row in report["cohorts"]["success"]], ["ok", "mixed"])
        self.assertEqual([row["command"] for row in report["cohorts"]["failure"]], ["mixed"])
        saved = json.loads((self.output / "commands" / "mixed.json").read_text())
        self.assertEqual([(sample["returncode"], sample["exit_class"]) for sample in saved["samples"]],
                         [(0, "success"), (7, "failure")])
        self.assertEqual([sample["stdout"] for sample in saved["samples"]],
                         ["mixed stdout 2", "mixed stdout 3"])
        self.assertEqual([sample["stderr"] for sample in saved["samples"]],
                         ["mixed stderr 2", "mixed stderr 3"])

    def test_cli_runs_subprocesses_and_reports_raw_outcomes(self):
        self.spec["samples"] = 1
        self.spec["commands"] = [
            {"id": "success", "argv": [sys.executable, "-c", "print('ok-output')"]},
            {"id": "failure", "argv": [sys.executable, "-c", "import sys; print('bad-output'); print('bad-error', file=sys.stderr); sys.exit(7)"]},
        ]
        self.spec_path.write_text(json.dumps(self.spec))
        output = io.StringIO()
        with mock.patch.object(profile_cli, "acquire", return_value=contextlib.nullcontext()), \
                contextlib.redirect_stdout(output):
            self.assertEqual(profile_cli.main(["run", str(self.spec_path), "--output", str(self.output)]), 0)
        self.assertIn('"status": "COMPLETE"', output.getvalue())
        report_output = io.StringIO()
        with contextlib.redirect_stdout(report_output):
            self.assertEqual(profile_cli.main(["report", str(self.output)]), 0)
        report = json.loads(report_output.getvalue())
        self.assertEqual([row["command"] for row in report["cohorts"]["success"]], ["success"])
        self.assertEqual([row["command"] for row in report["cohorts"]["failure"]], ["failure"])
        failure = json.loads((self.output / "commands" / "failure.json").read_text())["samples"][0]
        self.assertEqual((failure["stdout"], failure["stderr"], failure["returncode"], failure["exit_class"]),
                         ("bad-output\n", "bad-error\n", 7, "failure"))
