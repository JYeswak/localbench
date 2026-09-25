"""Testing while the machine works (the owner, 2026-09-24): apps and the person at the keyboard are recorded as load, not
a veto; only another model voids a run; interleaved A/B legs keep a busy machine fair; one pair reproduces the original
A,B,A rule exactly."""

import json
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import golden, park, sysstats
from localbench.__main__ import _busy_check, ab_order, unsound

ROOT = Path(__file__).resolve().parent.parent
TARGET = ("ollama", "qwen3.6:35b-mlx")
OURS = {"pid": 1, "name": "ollama", "pct": 90.0, "model": "qwen3.6:35b-mlx",
        "cmd": "/Applications/Ollama.app/Contents/Resources/ollama runner --mlx-engine --model qwen3.6:35b-mlx --port 51"}
OTHER_MODEL = {"pid": 2, "name": "ollama", "pct": 40.0, "model": "qwen3.8:27b-mlx",
               "cmd": "/Applications/Ollama.app/Contents/Resources/ollama runner --mlx-engine --model qwen3.8:27b-mlx --port 52"}
CHROME = {"pid": 3, "name": "Google Chrome He", "pct": 31.0,
          "cmd": "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Helpers"}
WINDOWSERVER = {"pid": 4, "name": "WindowServer", "pct": 12.0,
                "cmd": "/System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer -daemon"}


def one(v, better="lower"):
    return {"x": {"value": v, "better": better}}


class Classify(unittest.TestCase):
    def test_apps_are_load_and_another_model_is_contention(self):
        foreign, models, apps = sysstats.classify([OURS, OTHER_MODEL, CHROME, WINDOWSERVER],
                                                  {"ollama": ["qwen3.6:35b-mlx"]}, TARGET, 25.0)
        self.assertEqual((foreign, models, apps), ({}, [OTHER_MODEL], [CHROME]))

    def test_a_resident_foreign_model_is_contention(self):
        foreign, _, _ = sysstats.classify([], {"ollama": ["qwen3.6:35b-mlx", "qwen3.8:27b-mlx"]}, TARGET, 25.0)
        self.assertEqual(foreign, {"ollama": ["qwen3.8:27b-mlx"]})


class LoadSummary(unittest.TestCase):
    def test_app_gpu_spikes_and_user_activity(self):
        series = [{"gpu_procs": [OURS, CHROME, WINDOWSERVER], "user_idle_s": 0.5},
                  {"gpu_procs": [OURS, WINDOWSERVER], "user_idle_s": 3.0},
                  {"gpu_procs": [OURS, OTHER_MODEL], "user_idle_s": 45.0},
                  {"gpu_procs": [OURS], "user_idle_s": None}]
        load = sysstats.load_summary(series, TARGET, 25.0)
        self.assertEqual(load, {"samples": 4, "app_gpu_mean_pct": round((43 + 12 + 0 + 0) / 4, 1),
                                "app_gpu_p95_pct": 43.0, "app_spike_seconds": 1, "user_active_pct": 66.7})
        # p95 is nearest-rank: with 4 samples it is the largest, never below the mean (the first smoke read
        # p95 10.6 under mean 11.8 with an interpolating index).


class SamplerLoop(unittest.TestCase):
    def test_a_live_sampler_records_app_spikes_as_load_and_a_model_runner_as_contention(self):
        script = [[OURS, CHROME], [OURS], [OURS, OTHER_MODEL], [OURS]]
        calls = iter(range(1_000_000))

        def share(*_a, **_k):
            i = next(calls)
            return script[i] if i < len(script) else [OURS]

        def live():
            return {"t": time.time(), "resident": {"ollama": ["qwen3.6:35b-mlx"]}, "user_idle_s": 1.0}
        events = []
        with mock.patch.object(sysstats, "live", side_effect=live), \
                mock.patch.object(sysstats, "gpu_time_by_pid", return_value={}), \
                mock.patch.object(sysstats, "gpu_share", side_effect=share):
            with sysstats.Sampler(0.01, target=TARGET, on_contention=events.append, gpu_foreign_max_pct=25.0) as s:
                deadline = time.monotonic() + 3
                while len(s.series) < len(script) + 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
        self.assertEqual([e["foreign_gpu"] for e in events], [[OTHER_MODEL]])
        self.assertEqual([spike["apps"] for spike in s.load_spikes], [[CHROME]])


class _BusySampler:
    """Sampler stand-in: the device is busy, split between the rows given."""
    rows: list = []

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def summary(self):
        return {"gpu_device_pct": {"mean": 97.0}, "gpu_by_process": self.rows, "swap_used_mb": {"max": 0}}


class Preflight(unittest.TestCase):
    def problems(self, rows: list[dict], cpu: float = 88.0) -> list[str]:
        _BusySampler.rows = rows
        with mock.patch.object(sysstats, "Sampler", _BusySampler), \
                mock.patch.object(sysstats, "cpu_busy_pct", return_value=cpu), \
                mock.patch.object(park, "reachable_smol", return_value=[]), \
                mock.patch.object(sysstats, "omp_processes", return_value=[]), \
                mock.patch.object(park, "stuck_sessions", return_value=[]):
            return _busy_check()[2]

    def test_a_busy_machine_of_apps_starts(self):
        self.assertEqual(self.problems([CHROME, {**WINDOWSERVER, "pct": 60.0}]), [])

    def test_another_models_runner_refuses(self):
        (problem,) = self.problems([CHROME, OTHER_MODEL])
        self.assertIn("another model is running", problem)
        self.assertIn("qwen3.8:27b-mlx", problem)


class Verdict(unittest.TestCase):
    def test_only_contention_by_a_model_is_unsound(self):
        summary = {"verdicts": {"contended": True, "must_fail": [], "preflight_problems": [], "pins_changed": {}}}
        (reason,) = unsound(summary)
        self.assertIn("another model was resident or running", reason)


class ABOrder(unittest.TestCase):
    def test_one_pair_keeps_the_original_legs_and_more_pairs_interleave(self):
        self.assertEqual(ab_order(1), [("A", "ab_a1"), ("B", "ab_b"), ("A", "ab_a2")])
        self.assertEqual([arm for arm, _ in ab_order(2)], ["A", "B", "A", "B", "A"])


class ABTable(unittest.TestCase):
    def test_one_pair_reproduces_a_banked_receipt(self):
        # Oracle: the ThinkingCap receipt was judged by the original A,B,A code (c1a78a4).
        r = json.loads((ROOT / "docs/evidence/receipts/ab-thinkingcap-q4km-20260924.json").read_text())
        legs = [leg["metrics"] for leg in r["legs"]]
        table = golden.ab_table([legs[0], legs[2]], [legs[1]])
        for key, want in r["table"].items():
            got = table[key]
            self.assertEqual(got["verdict"], want["verdict"], key)
            for field in ("band", "b_over_a", "aa_rel_spread", "a1", "a2", "b"):
                if field in want:
                    self.assertEqual(got[field], want[field], (key, field))

    def test_a_loaded_leg_moves_the_median_less_than_the_mean(self):
        # A legs 10, 10.4, 30 (one leg caught a burst); B legs 9.6, 9.8.
        table = golden.ab_table([one(10.0), one(10.4), one(30.0)], [one(9.6), one(9.8)])["x"]
        self.assertEqual((table["a_median"], table["b_median"]), (10.4, 9.7))
        self.assertAlmostEqual(table["aa_rel_spread"], (30.0 - 10.0) / 10.4, places=4)

    def test_b_legs_that_disagree_widen_the_band(self):
        calm = golden.ab_table([one(10.0), one(10.1), one(10.0)], [one(8.0), one(8.1)])["x"]
        noisy = golden.ab_table([one(10.0), one(10.1), one(10.0)], [one(6.0), one(10.0)])["x"]
        self.assertEqual(calm["verdict"], "B-BETTER")
        self.assertGreater(noisy["band"], calm["band"])
        self.assertEqual(noisy["verdict"], "WITHIN-NOISE")

    def test_a_void_leg_voids_the_row(self):
        void = {"x": {"value": None, "void": "no reasoning streamed", "better": "lower"}}
        row = golden.ab_table([one(10.0), one(10.0), one(10.0)], [void, one(9.0)])["x"]
        self.assertEqual((row["verdict"], row["void"]), ("VOID", "no reasoning streamed"))


class ArmPinDrift(unittest.TestCase):
    def test_omp_rebuilt_inside_arm_a_voids_only_omp_bound_rows(self):
        # Oracle: the oMLX receipt. omp replaced its 18.3.1 build during leg A1 (omp_sha 46cb390b -> cacaf572),
        # while B's backend differs from A's by design (mlx-serve vs oMLX).
        r = json.loads((ROOT / "docs/evidence/receipts/ab-mlxserve-vs-omlx-qwen36-20260925.json").read_text())
        a, b = [r["legs"][0], r["legs"][2]], [r["legs"][1]]
        drift = golden.arm_pin_drift([x["provenance"]["pins"] for x in a], [x["provenance"]["pins"] for x in b],
                                     ["conf", "micro", "replay", "e2e"])
        self.assertEqual(drift, {"e2e": "arm A legs ran different omp_sha: 46cb390bb7e6def5 / cacaf5726609a21b"})
        table = golden.ab_table([x["metrics"] for x in a], [x["metrics"] for x in b], void_tiers=drift)
        self.assertEqual({k: v["verdict"] for k, v in table.items() if k.startswith("e2e.")},
                         {k: "VOID" for k in table if k.startswith("e2e.")})
        self.assertEqual(table["replay.full.cold_ttft_s"]["verdict"], "B-BETTER")   # other tiers judged as banked

    def test_a_backend_updated_between_a_legs_voids_every_tier(self):
        # ollama auto-updates itself; an update between A1 and A2 leaves no A/A null for any row.
        pins = {"backend": "ollama", "backend_version": "0.34.4", "omp_sha": "x"}
        drift = golden.arm_pin_drift([pins, {**pins, "backend_version": "0.34.5"}], [pins], ["micro", "e2e"])
        self.assertEqual(set(drift), {"micro", "e2e"})
        self.assertIn("backend_version: 0.34.4 / 0.34.5", drift["micro"])


class LoadBalance(unittest.TestCase):
    def rows(self, favours):
        def legs(wall, fast, acc):
            return {"x.wall_s": {"value": wall, "better": "lower"}, "y.fast_s": {"value": fast, "better": "lower"},
                    "z.accuracy": {"value": acc, "better": "higher"}}
        return golden.ab_table([legs(10.0, 10.0, 1.0), legs(10.0, 10.0, 1.0)], [legs(20.0, 5.0, 0.5)],
                               load_favours=favours)

    def test_heavier_b_withholds_only_time_rows_that_lean_with_the_load(self):
        favours = golden.load_balance([10.0, 12.0], [30.0])["favours"]
        t = self.rows(favours)
        self.assertEqual((favours, t["x.wall_s"]["verdict"], t["x.wall_s"]["withheld"]), ("A", "LOAD-FAVOURED", "B-WORSE"))
        self.assertEqual(t["y.fast_s"]["verdict"], "B-BETTER")      # faster while heavier: load cannot explain it
        self.assertEqual(t["z.accuracy"]["verdict"], "B-WORSE")     # not a time row

    def test_lighter_b_withholds_b_better_time_rows(self):
        favours = golden.load_balance([30.0, 32.0], [10.0])["favours"]
        t = self.rows(favours)
        self.assertEqual((favours, t["y.fast_s"]["verdict"], t["x.wall_s"]["verdict"]), ("B", "LOAD-FAVOURED", "B-WORSE"))

    def test_bracketed_boundary_or_missing_load_changes_nothing(self):
        self.assertIsNone(golden.load_balance([10.0, 12.0], [17.0])["favours"])   # exactly +5: within tolerance
        self.assertIsNone(golden.load_balance([10.0, 12.0], [5.0])["favours"])    # exactly -5 below the lower A leg
        missing = golden.load_balance([10.0, None], [40.0])
        self.assertEqual((missing["favours"], missing["note"]), (None, "CPU busy missing on a leg: balance not checked"))
        self.assertEqual(self.rows(None)["x.wall_s"]["verdict"], "B-WORSE")


class PerArmBinaries(unittest.TestCase):
    def test_b_binaries_run_only_the_b_legs_and_a_legs_stay_on_the_default(self):
        import contextlib
        import io
        import os
        import tempfile

        from localbench import __main__ as cli

        seen = []

        def fake_execute(backend, model, *, label, **_):
            omp = os.environ.get("LOCALBENCH_OMP")
            seen.append((label, omp, os.environ.get("LOCALBENCH_MLX_SERVE")))
            return {"provenance": {"label": label, "created": "t", "pins": {"model": "m", "omp_sha": omp or "default"}},
                    "verdicts": {"contended": False, "must_fail": [], "preflight_problems": []},
                    "metrics": {"e2e.ok.first_startup_s": {"value": 1.0 if omp else 0.6, "better": "lower"}},
                    "conformance": {}, "run_dir": "r", "results": [],
                    "system": {"before": {"live": {}, "host": {}}, "cpu": {"busy_pct": {"mean": 5.0}}}}

        @contextlib.contextmanager
        def fake_backend(spec, args=()):
            yield None, "m"

        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:     # cmd_ab prints the receipt path relative to ROOT
            old, new_mlx = Path(tmp) / "omp-old", Path(tmp) / "mlx-serve-new"
            for exe in (old, new_mlx):
                exe.write_text("#!/bin/sh\n")
                exe.chmod(0o755)
            with mock.patch.object(cli, "execute", fake_execute), mock.patch.object(cli, "open_backend", fake_backend), \
                    mock.patch.object(cli, "RECEIPTS", Path(tmp)), contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.dict(os.environ, {"LOCALBENCH_MLX_SERVE": "/opt/default/mlx-serve"}):
                os.environ.pop("LOCALBENCH_OMP", None)
                rc = cli.main(["ab", "ollama:m", "ollama:m", "--tiers", "e2e", "--pairs", "2", "--b-omp", str(old),
                               "--b-mlx-serve", str(new_mlx), "--bank", "t"])
                after = (os.environ.get("LOCALBENCH_OMP"), os.environ.get("LOCALBENCH_MLX_SERVE"))
            receipt = json.loads((Path(tmp) / "t.json").read_text())
        a, b = (None, "/opt/default/mlx-serve"), (str(old), str(new_mlx))
        self.assertEqual(rc, 0)
        self.assertEqual(seen, [("ab_a1", *a), ("ab_b1", *b), ("ab_a2", *a), ("ab_b2", *b), ("ab_a3", *a)])
        self.assertEqual(after, a)                                   # unset stays unset; a default comes back
        self.assertEqual(receipt["pin_drift"], {})                  # each arm ran one omp throughout
        self.assertEqual(receipt["table"]["e2e.ok.first_startup_s"]["verdict"], "B-WORSE")


if __name__ == "__main__":
    unittest.main()
