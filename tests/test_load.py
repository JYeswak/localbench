"""Testing while the machine works: apps and the person at the keyboard are recorded as load, not a veto; another
model contends, unknown residency withholds proof; interleaved A/B legs keep a busy machine fair."""

import json
import time
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest import mock

from localbench import golden, park, smol, sysstats
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

    def test_declared_smol_residency_is_expected_but_other_model_still_contends(self):
        smol = "qwen3.8:27b-mlx"
        target = (*TARGET, smol)

        foreign, _, _ = sysstats.classify(
            [], {"ollama": [TARGET[1]], "mlx-smol": [smol]}, target, 25.0)
        self.assertEqual(foreign, {})

        foreign, _, _ = sysstats.classify(
            [], {"ollama": [TARGET[1]], "mlx-smol": [smol, "other-model"]}, target, 25.0)
        self.assertEqual(foreign, {"mlx-smol": ["other-model"]})


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

    def test_declared_smol_is_not_contention_but_an_undeclared_model_is(self):
        smol = "qwen3.8:27b-mlx"
        target = (*TARGET, smol)
        residents = [
            {"ollama": [TARGET[1]], "mlx-smol": [smol]},
            {"ollama": [TARGET[1]], "mlx-smol": [smol, "other-model"]},
        ]
        calls = 0

        def live():
            nonlocal calls
            resident = residents[min(calls, 1)]
            calls += 1
            return {"t": time.time(), "resident": resident}

        events = []
        with mock.patch.object(sysstats, "live", side_effect=live), \
                mock.patch.object(sysstats, "gpu_time_by_pid", return_value={}), \
                mock.patch.object(sysstats, "gpu_share", return_value=[]):
            with sysstats.Sampler(0.005, target=target, on_contention=events.append) as sampler:
                deadline = time.monotonic() + 2
                while len(sampler.series) < 2 and time.monotonic() < deadline:
                    time.sleep(0.005)

        self.assertGreaterEqual(len(sampler.series), 2)
        self.assertEqual([event["foreign"] for event in events], [{"mlx-smol": ["other-model"]}])


class _BusySampler:
    """Sampler stand-in: the device is busy, split between the rows given."""
    rows: list = []
    resident_unknown_samples = 0

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def summary(self):
        return {"gpu_device_pct": {"mean": 97.0}, "gpu_by_process": self.rows, "swap_used_mb": {"max": 0},
                "resident_unknown_samples": self.resident_unknown_samples}


class Preflight(unittest.TestCase):
    def problems(self, rows: list[dict], cpu: float = 88.0, unknown: int = 0) -> list[str]:
        _BusySampler.rows = rows
        _BusySampler.resident_unknown_samples = unknown
        with mock.patch.object(sysstats, "Sampler", _BusySampler), \
                mock.patch.object(sysstats, "cpu_busy_pct", return_value=cpu), \
                mock.patch.object(park, "reachable_smol", return_value=[]), \
                mock.patch.object(sysstats, "omp_processes", return_value=[]), \
                mock.patch.object(park, "stuck_sessions", return_value=[]), \
                mock.patch.object(smol, "load_state", return_value=None):
            return _busy_check()[2]

    def test_a_busy_machine_of_apps_starts(self):
        self.assertEqual(self.problems([CHROME, {**WINDOWSERVER, "pct": 60.0}]), [])

    def test_another_models_runner_refuses(self):
        (problem,) = self.problems([CHROME, OTHER_MODEL])
        self.assertIn("another model is running", problem)
        self.assertIn("qwen3.8:27b-mlx", problem)

    def test_unreadable_residency_in_preflight_is_not_proof_of_another_model(self):
        (problem,) = self.problems([], unknown=1)
        self.assertIn("resident", problem.lower())
        self.assertNotIn("CONTENDED", problem)


class Verdict(unittest.TestCase):
    def test_only_contention_by_a_model_is_unsound(self):
        summary = {"conformance": {}, "system": {"during": {"resident_unknown_samples": 0}},
                   "verdicts": {"contended": True, "must_fail": [], "preflight_problems": [], "pins_changed": {}}}
        (reason,) = unsound(summary)
        self.assertIn("another model was resident or running", reason)

    def test_unknown_residency_withholds_proof_without_inventing_contention(self):
        cases = (
            ("timeout", None, [OURS], 1, False, True),
            ("mlx timeout", ["qwen3.6:35b-mlx"], [OURS], 1, False, True),
            ("own model", ["qwen3.6:35b-mlx"], [OURS], 0, False, False),
            ("app load", ["qwen3.6:35b-mlx"], [OURS, CHROME], 0, False, False),
            ("other model", ["qwen3.6:35b-mlx", "qwen3.8:27b-mlx"], [OURS], 0, True, True),
            ("server down", "down", [], 0, False, False),
        )
        for label, models, rows, unknown, contended, rejected in cases:
            with self.subTest(label=label):
                def endpoint(url, timeout):
                    if url.endswith("/api/ps"):
                        if models is None:
                            raise urllib.error.URLError(TimeoutError("fake Ollama timeout"))
                        if models == "down":
                            raise urllib.error.URLError(ConnectionRefusedError("fake Ollama down"))
                        return BytesIO(json.dumps({"models": [{"name": name} for name in models]}).encode())
                    if label == "mlx timeout" and url.startswith("http://127.0.0.1:11234/"):
                        raise urllib.error.URLError(TimeoutError("fake mlx-serve timeout"))
                    raise urllib.error.URLError(ConnectionRefusedError("fake inactive server"))

                with mock.patch.object(sysstats.urllib.request, "urlopen", side_effect=endpoint):
                    resident = sysstats.resident_models()
                foreign, runners, apps = sysstats.classify(rows, resident, TARGET, 25.0)
                sampler = sysstats.Sampler(target=TARGET, gpu_foreign_max_pct=25.0)
                sampler.series = [{"resident": resident, "gpu_procs": rows}]
                summary = {"system": {"during": sampler.summary()}, "conformance": {},
                           "verdicts": {"contended": bool(foreign or runners), "must_fail": [],
                                        "preflight_problems": [], "pins_changed": {}}}
                reasons = unsound(summary)
                self.assertEqual(summary["system"]["during"]["resident_unknown_samples"], unknown)
                self.assertEqual(bool(foreign or runners), contended)
                self.assertEqual(bool(reasons), rejected)
                self.assertEqual(any("CONTENDED" in reason for reason in reasons), contended)
                if unknown:
                    self.assertTrue(any("residen" in reason.lower() for reason in reasons))
                if label == "app load":
                    self.assertEqual(apps, [CHROME])


    def test_hidden_second_server_timeout_below_gpu_threshold_is_nonproof(self):
        probes = {}
        # The fake second server has an idle qwen3.8 resident; its status endpoint times out.
        # Its process is absent from the low-GPU process sample, so only residency sampling detects uncertainty.

        def endpoint(url, timeout):
            if url.endswith("/api/ps"):
                return BytesIO(json.dumps({"models": [{"name": TARGET[1]}]}).encode())
            if url.startswith("http://127.0.0.1:11234/"):
                raise urllib.error.URLError(TimeoutError("fake MLX residency timeout"))
            raise urllib.error.URLError(ConnectionRefusedError("fake inactive server"))

        with mock.patch.object(sysstats.urllib.request, "urlopen", side_effect=endpoint):
            resident = sysstats.resident_models(probes)
        self.assertEqual(resident["ollama"], [TARGET[1]])
        self.assertIsNone(resident["mlx-serve"])
        probe = probes["mlx-serve"]
        self.assertEqual(probe["error_class"], "timeout")
        self.assertLessEqual(probe["probe_start"], probe["probe_end"])

        rows = []
        foreign, runners, _apps = sysstats.classify(rows, resident, TARGET, 25.0)
        sampler = sysstats.Sampler(target=TARGET, gpu_foreign_max_pct=25.0)
        sampler.series = [{"resident": resident, "resident_probes": probes, "gpu_procs": rows,
                           "gpu_device_pct": 5.0}]
        during = sampler.summary()
        summary = {"system": {"during": during}, "conformance": {},
                   "verdicts": {"contended": bool(foreign or runners), "must_fail": [],
                                "preflight_problems": [], "pins_changed": {}}}
        self.assertEqual(during["resident_unknown_samples"], 1)
        self.assertEqual(during["gpu_device_pct"]["mean"], 5.0)
        self.assertFalse(summary["verdicts"]["contended"])
        reasons = unsound(summary)
        self.assertTrue(any("resident model state unknown" in reason for reason in reasons), reasons)
        self.assertFalse(any(reason.startswith("CONTENDED:") for reason in reasons), reasons)

    def test_campaign_runs_cannot_be_used_as_performance_or_golden_evidence(self):
        summary = {"evaluation_campaign": {"profile": "omp-e2e.v1"}, "conformance": {},
                   "verdicts": {"contended": False, "must_fail": [], "preflight_problems": [], "pins_changed": {}}}
        reasons = unsound(summary)
        self.assertTrue(any("behavioral evaluation campaign" in reason for reason in reasons))


class OwnedStopClassification(unittest.TestCase):
    OWNED_TARGET = ("mlx-serve", "target")

    @staticmethod
    def probe(server="mlx-serve", port=11234, start=11.0, end=12.0):
        return {"server": server, "port": port, "probe_start": start, "probe_end": end,
                "error_class": "ConnectionResetError", "http_status": None}

    @staticmethod
    def stop_event(*, server="mlx-serve", port=11234, pre=("target",), rc=0, killed=False,
                   t_term=10.0, t_exit=13.0):
        return {"server": server, "port": port, "pid": 17, "t_term": t_term, "t_exit": t_exit,
                "pre": None if pre is None else list(pre), "rc": rc, "killed": killed}

    @staticmethod
    def sample(resident, probe=None):
        probes = {probe["server"]: probe} if probe else {}
        return {"resident": resident, "resident_probes": probes, "gpu_procs": [], "user_idle_s": None}

    def summarize(self, rows, stops):
        sampler = sysstats.Sampler(target=self.OWNED_TARGET)
        sampler.series = rows
        with mock.patch.object(sysstats, "owned_stop_events", return_value=stops):
            during = sampler.summary()
        verdict = {"system": {"during": during}, "conformance": {},
                   "verdicts": {"contended": False, "must_fail": [], "preflight_problems": [],
                                "pins_changed": {}}}
        return during, unsound(verdict)

    def test_target_only_restart_window_keeps_list_none_empty_samples_known(self):
        stop = self.stop_event()
        rows = [self.sample({"mlx-serve": ["target"]}),
                self.sample({"mlx-serve": None}, self.probe()),
                self.sample({"mlx-serve": []})]

        during, reasons = self.summarize(rows, [stop])

        self.assertEqual(during["resident_unknown_samples"], 0)
        self.assertEqual(during["resident_owned_stop_samples"], 1)
        self.assertEqual(rows[1]["resident_owned_stop"], stop)
        self.assertEqual(reasons, [])


    def test_owned_stop_with_smol_prestate_requires_declared_smol(self):
        smol = "smol-model"
        stop = self.stop_event(pre=sorted(("target", smol)))
        for target, expected in (
            (("mlx-serve", "target", smol), (0, 1)),
            (("mlx-serve", "target"), (1, 0)),
        ):
            with self.subTest(target=target):
                row = self.sample({"mlx-serve": None}, self.probe())
                sampler = sysstats.Sampler(target=target)
                sampler.series = [row]
                with mock.patch.object(sysstats, "owned_stop_events", return_value=[stop]):
                    during = sampler.summary()
                actual = (during["resident_unknown_samples"], during["resident_owned_stop_samples"])
                self.assertEqual(actual, expected)

    def test_rejects_untrusted_none_samples(self):
        probe = self.probe()
        cases = (
            ("outside stop interval", self.probe(start=8.0, end=9.0), self.stop_event(),
             {"mlx-serve": None}),
            ("other model in prestate", probe, self.stop_event(pre=["target", "other"]),
             {"mlx-serve": None}),
            ("unreadable prestate", probe, self.stop_event(pre=None),
             {"mlx-serve": None}),
            ("non-target server", self.probe(server="omlx", port=11236), self.stop_event(),
             {"mlx-serve": ["target"], "omlx": None}),
            ("missing exit status", probe, self.stop_event(rc=None),
             {"mlx-serve": None}),
            ("killed server", probe, self.stop_event(killed=True),
             {"mlx-serve": None}),
        )
        for label, sample_probe, stop, resident in cases:
            with self.subTest(label=label):
                during, reasons = self.summarize([self.sample(resident, sample_probe)], [stop])
                self.assertEqual(during["resident_unknown_samples"], 1)
                self.assertEqual(during["resident_owned_stop_samples"], 0)
                self.assertTrue(any("resident model state unknown" in reason for reason in reasons))
                self.assertFalse(any("CONTENDED" in reason for reason in reasons))


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
            seen.append((label, omp, os.environ.get("LOCALBENCH_MLX_SERVE"), os.environ.get("LOCALBENCH_MLXFAST")))
            return {"provenance": {"label": label, "created": "t", "pins": {"model": "m", "omp_sha": omp or "default"}},
                    "verdicts": {"contended": False, "must_fail": [], "preflight_problems": []},
                    "metrics": {"e2e.ok.first_startup_s": {"value": 1.0 if omp else 0.6, "better": "lower"}},
                    "conformance": {}, "run_dir": "r", "results": [],
                    "system": {"before": {"live": {}, "host": {}}, "during": {"resident_unknown_samples": 0},
                               "cpu": {"busy_pct": {"mean": 5.0}}}}

        @contextlib.contextmanager
        def fake_backend(spec, args=()):
            yield None, "m"

        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:     # cmd_ab prints the receipt path relative to ROOT
            old, new_mlx, new_fast = Path(tmp) / "omp-old", Path(tmp) / "mlx-serve-new", Path(tmp) / "mlx-server"
            for exe in (old, new_mlx, new_fast):
                exe.write_text("#!/bin/sh\n")
                exe.chmod(0o755)
            with mock.patch.object(cli, "execute", fake_execute), mock.patch.object(cli, "open_backend", fake_backend), \
                    mock.patch.object(cli, "RECEIPTS", Path(tmp)), contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.dict(os.environ, {"LOCALBENCH_MLX_SERVE": "/opt/default/mlx-serve",
                                                 "LOCALBENCH_MLXFAST": "/opt/default/mlx-server"}):
                os.environ.pop("LOCALBENCH_OMP", None)
                rc = cli.main(["ab", "ollama:m", "ollama:m", "--tiers", "e2e", "--pairs", "2", "--b-omp", str(old),
                               "--b-mlx-serve", str(new_mlx), "--b-mlxfast", str(new_fast), "--bank", "t"])
                after = (os.environ.get("LOCALBENCH_OMP"), os.environ.get("LOCALBENCH_MLX_SERVE"),
                         os.environ.get("LOCALBENCH_MLXFAST"))
            receipt = json.loads((Path(tmp) / "t.json").read_text())
        a, b = (None, "/opt/default/mlx-serve", "/opt/default/mlx-server"), (str(old), str(new_mlx), str(new_fast))
        self.assertEqual(rc, 0)
        self.assertEqual(seen, [("ab_a1", *a), ("ab_b1", *b), ("ab_a2", *a), ("ab_b2", *b), ("ab_a3", *a)])
        self.assertEqual(after, a)                                   # unset stays unset; a default comes back
        self.assertEqual(receipt["pin_drift"], {})                  # each arm ran one omp throughout
        self.assertEqual(receipt["table"]["e2e.ok.first_startup_s"]["verdict"], "B-WORSE")


if __name__ == "__main__":
    unittest.main()
