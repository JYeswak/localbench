import json
import tempfile
import unittest
from pathlib import Path

from scripts.replay_void_runs import replay_run


class VoidRunReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def candidate(self, *, run_dir="runs/fixture", contended=True, pins_changed=None):
        return {
            "run": {
                "run_dir": run_dir,
                "provenance": {"pins": {"backend": "ollama", "model": "target"}},
                "verdicts": {"contended": contended, "pins_changed": pins_changed or {},
                             "preflight_problems": []},
                "system": {"during": {"resident_unknown_samples": 0}},
            },
            "receipts": ["fixture.json"],
            "reasons": ["contended"] if contended else ["pins_changed"],
        }

    def write_telemetry(self, candidate, samples, *, done=200):
        run_dir = self.root / candidate["run"]["run_dir"]
        run_dir.mkdir(parents=True)
        progress = [{"t": 100, "event": "start"}, {"t": done, "event": "done"}]
        (run_dir / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in progress))
        (run_dir / "sampler.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
        return replay_run(self.root, candidate)

    @staticmethod
    def sample(t, *, resident=None, gpu_procs=None):
        return {
            "t": t,
            "resident": resident or {"ollama": ["target"], "mlx-serve": [], "splash": [],
                                     "omlx": [], "mlx-smol": [], "mlxfast": []},
            "gpu_procs": gpu_procs or [],
        }

    def test_first_foreign_gpu_sample_reports_remaining_fraction(self):
        candidate = self.candidate()
        foreign = {"name": "ollama", "cmd": "ollama runner --model other", "model": "other", "pct": 30.0}
        report = self.write_telemetry(candidate, [self.sample(110), self.sample(150, gpu_procs=[foreign])])

        self.assertEqual(report["status"], "timed")
        self.assertEqual(report["first_violation"]["reasons"], ["foreign_gpu_client"])
        self.assertEqual(report["elapsed_s"], 50)
        self.assertEqual(report["duration_s"], 100)
        self.assertEqual(report["saved_fraction"], 0.5)

    def test_missing_gpu_samples_do_not_fabricate_an_early_violation(self):
        candidate = self.candidate()
        incomplete = self.sample(110)
        del incomplete["gpu_procs"]
        foreign = {"name": "ollama", "cmd": "ollama runner --model other", "model": "other", "pct": 30.0}

        report = self.write_telemetry(candidate, [incomplete, self.sample(150, gpu_procs=[foreign])])

        self.assertEqual(report["status"], "incomplete_evidence")
        self.assertEqual(report["data_gaps"],
                         ["first=110.000; samples=1; reasons=gpu_process_samples_missing"])
        self.assertIsNone(report["saved_fraction"])

    def test_expected_model_missing_is_a_sampler_violation(self):
        candidate = self.candidate()
        candidate["run"]["verdicts"]["contended"] = False
        report = self.write_telemetry(candidate, [self.sample(150, resident={"ollama": [], "mlx-serve": [],
                                                                             "splash": []})])

        self.assertEqual(report["first_violation"]["reasons"], ["expected_model_not_resident:target"])
        self.assertEqual(report["saved_fraction"], 0.5)

    def test_pin_drift_without_per_second_pin_samples_has_no_invented_time(self):
        candidate = self.candidate(contended=False, pins_changed={"omp_sha": ["old", "new"]})
        report = self.write_telemetry(candidate, [self.sample(150)])

        self.assertEqual(report["status"], "first_time_uncertain_pin_drift")
        self.assertIsNone(report["first_violation"])
        self.assertIsNone(report["saved_fraction"])

    def test_missing_sampler_artifact_is_not_counted_as_zero_savings(self):
        candidate = self.candidate()
        run_dir = self.root / candidate["run"]["run_dir"]
        run_dir.mkdir(parents=True)
        (run_dir / "progress.jsonl").write_text('{"t":100,"event":"start"}\n')

        report = replay_run(self.root, candidate)

        self.assertEqual(report["status"], "missing_artifacts")
        self.assertEqual(report["missing"], ["sampler.jsonl"])
        self.assertIsNone(report["saved_fraction"])


if __name__ == "__main__":
    unittest.main()
