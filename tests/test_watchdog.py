import unittest

from localbench.watchdog import RunWatchdog


class RunWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.sha = "pinned-sha"
        self.watchdog = RunWatchdog(("ollama", "target"), self.sha, lambda: self.sha)

    @staticmethod
    def sample():
        return {"resident": {"ollama": ["target"], "mlx-serve": [], "splash": []}, "gpu_procs": []}

    def test_valid_sample_has_no_violations(self):
        self.assertEqual(self.watchdog.violations(self.sample()), [])

    def test_foreign_residency_and_gpu_client_abort(self):
        sample = self.sample()
        sample["resident"]["ollama"].append("other")
        sample["gpu_procs"] = [{"name": "ollama", "cmd": "ollama runner --model other", "model": "other",
                                "pct": 30.0}]

        reasons = self.watchdog.violations(sample)

        self.assertTrue(any(reason.startswith("foreign_resident:") for reason in reasons))
        self.assertTrue(any(reason.startswith("foreign_gpu_client:") for reason in reasons))

    def test_missing_expected_residency_unknown_state_and_omp_drift_fail_closed(self):
        self.watchdog.current_omp_sha = lambda: "changed-sha"
        sample = {"resident": {"ollama": [], "mlx-serve": None, "splash": []}, "gpu_procs": []}

        reasons = self.watchdog.violations(sample)

        self.assertIn("residency_unknown", reasons)
        self.assertIn("expected_model_not_resident:target", reasons)
        self.assertIn("omp_sha_changed:pinned-sha->changed-sha", reasons)

    def test_unreadable_omp_identity_fails_closed(self):
        def unavailable():
            raise OSError("omp binary disappeared")

        self.watchdog.current_omp_sha = unavailable

        self.assertEqual(self.watchdog.violations(self.sample()), ["omp_sha_unavailable:OSError"])


if __name__ == "__main__":
    unittest.main()
