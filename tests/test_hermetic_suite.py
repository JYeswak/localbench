import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from localbench.sysstats import INFERENCE_PORTS
from scripts import hermetic_suite


class HermeticSuitePolicy(unittest.TestCase):
    def test_external_egress_is_blocked_and_loopback_fixtures_work(self):
        if os.environ.get(hermetic_suite.CHILD_FLAG) == "1":
            summary = hermetic_suite.probe_current_sandbox()
        else:
            result = hermetic_suite.run_isolated_probe()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = result.stdout
        self.assertIn("omp=absent-from-PATH", summary)
        self.assertIn("ollama=absent-from-PATH", summary)
        self.assertIn("spawn=refused(FileNotFoundError)", summary)
        self.assertIn("external-egress=blocked", summary)
        self.assertIn("egress-control=blocked", summary)
        self.assertIn("loopback-fixture=passed", summary)
        self.assertIn("model-service-ports=denied", summary)

    def test_model_service_ports_cover_inference_servers(self):
        self.assertEqual(
            set(hermetic_suite.MODEL_SERVICE_PORTS),
            set(INFERENCE_PORTS),
        )
        self.assertIn(11235, hermetic_suite.MODEL_SERVICE_PORTS)

    def test_ci_suite_temporary_files_use_runner_temp_outside_checkout(self):
        runner_temp = Path("/ci-runner-temp")
        self.assertEqual(
            hermetic_suite.suite_temp_root({"GITHUB_ACTIONS": "true", "RUNNER_TEMP": str(runner_temp)}),
            runner_temp,
        )
        self.assertEqual(
            hermetic_suite.suite_temp_root({"GITHUB_ACTIONS": "false", "RUNNER_TEMP": str(hermetic_suite.ROOT)}),
            hermetic_suite.SCRATCH_ROOT,
        )
        with self.assertRaises(hermetic_suite.HermeticError):
            hermetic_suite.suite_temp_root({"GITHUB_ACTIONS": "true"})
        with self.assertRaises(hermetic_suite.HermeticError):
            hermetic_suite.suite_temp_root({
                "GITHUB_ACTIONS": "true",
                "RUNNER_TEMP": str(hermetic_suite.ROOT / "runner-temp"),
            })

    def test_sandbox_scratch_survives_with_owner_records(self):
        with hermetic_suite.sandbox_context(parent_env={}) as sandbox:
            profile_dir = sandbox.profile.parent
            suite_tmp = Path(sandbox.env["TMPDIR"])
        self.assertTrue(sandbox.profile.is_file())
        for path, label in (
            (profile_dir, "hermetic-suite"),
            (suite_tmp, "localbench-suite-tmp"),
        ):
            with self.subTest(label=label):
                self.assertTrue(path.is_dir())
                owner = (path / ".owner").read_text()
                self.assertIn(f"label={label}", owner)
                self.assertIn("created=", owner)

    def test_child_path_omits_model_tools_even_if_parent_path_has_them(self):
        tool_dir = hermetic_suite._owned_scratch(
            hermetic_suite.suite_temp_root(os.environ), "hermetic-tools")
        for name in hermetic_suite.TOOLS:
            tool = tool_dir / name
            tool.write_text("#!/bin/sh\nexit 0\n")
            tool.chmod(0o755)
        with mock.patch.dict(os.environ, {
            "PATH": str(tool_dir),
            "LOCALBENCH_OMP": str(tool_dir / "omp"),
            "OLLAMA_HOST": "http://127.0.0.1:11434",
        }):
            env = hermetic_suite.child_environment(sys.executable)
        for name in hermetic_suite.TOOLS:
            self.assertIsNone(shutil.which(name, path=env["PATH"]), name)
        self.assertNotIn("LOCALBENCH_OMP", env)
        self.assertNotIn("OLLAMA_HOST", env)

    def test_service_skips_and_expected_failures_are_not_allowed(self):
        self.assertFalse(hermetic_suite.skip_is_allowed(
            "tests.test_isolation.OmpHonoursTheHarnessDir", "omp not installed"))
        self.assertFalse(hermetic_suite.skip_is_allowed(
            "tests.test_proxy.InstalledOmpPrompts", "network unavailable"))
        self.assertTrue(hermetic_suite.skip_is_allowed(
            hermetic_suite.REPORT_JSON_TEST, hermetic_suite.REPORT_JSON_SKIP))

        result = unittest.TestResult()
        result.addSkip(self, "omp not installed")
        result.expectedFailures.append((self, "synthetic expected failure"))
        self.assertEqual(len(hermetic_suite.suppression_violations(result)), 2)


if __name__ == "__main__":
    unittest.main()
