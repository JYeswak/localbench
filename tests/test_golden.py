"""Generation binding and merge: which pins invalidate which golden rows (golden.py). Promoted from the /tmp checks
that caught the None == None false PASS and the merge() tier drop of 2026-09-23 (0b6832e)."""

import unittest
from pathlib import Path
from typing import ClassVar

from localbench import golden

BASE = {"host_id": "h", "backend": "ollama", "backend_version": "0.32.15", "backend_sha": "b1", "model": "m",
        "model_digest": "d1", "macos_build": "25F", "backend_args": ""}
OLD_OMP = {"omp_version": "18.2.10", "omp_sha": "o1", "omp_child_config": "c1", "omp_agent_config": "a1"}
NEW_OMP = {"omp_version": "18.2.11", "omp_sha": "o2", "omp_child_config": "c2", "omp_agent_config": "a1"}
CONF = {"conf.tools": {"level": "MUST", "verdict": "PASS"}, "e2e.ok.correct": {"level": "MUST", "verdict": "PASS"}}


def row(v: float) -> dict:
    return {"value": v, "better": "lower", "tol": 0.1, "tol_source": "r", "spread": None}


def golden_with(pins: dict, keys=("micro.decode.tps", "replay.lean.cold_ttft_s", "e2e.ok.first_wall_s")) -> dict:
    return {"pins": pins, "aa_receipt": "r", "metrics": {k: row(1.0) for k in keys}, "conformance": dict(CONF)}


def states(g: dict, pins: dict) -> dict:
    metrics = {k: {"value": v["value"], "better": "lower"} for k, v in g["metrics"].items()}
    conf = {k: v for k, v in CONF.items() if k in g["conformance"]}
    return {r["key"]: r["status"] for r in golden.compare(metrics, conf, g, pins)}


class TierBinding(unittest.TestCase):
    def setUp(self):
        self.g = golden_with({**BASE, **OLD_OMP, "fixtures_sha": "f1"})

    def test_nothing_moved_nothing_stale(self):
        self.assertNotIn("GENERATION-MISMATCH", states(self.g, {**BASE, **OLD_OMP, "fixtures_sha": "f1"}).values())

    def test_an_omp_version_bump_does_not_stale_e2e(self):
        bumped = {**OLD_OMP, "omp_version": "18.3.0", "omp_sha": "o9"}
        s = states(self.g, {**BASE, **bumped, "fixtures_sha": "f1"})
        self.assertEqual(s["e2e.ok.first_wall_s"], "PASS")
        self.assertEqual(s["e2e.ok.correct"], "PASS")

    def test_a_child_config_change_stales_e2e_only(self):
        s = states(self.g, {**BASE, **OLD_OMP, "omp_child_config": "c9", "fixtures_sha": "f1"})
        self.assertEqual(s["e2e.ok.first_wall_s"], "GENERATION-MISMATCH")
        self.assertEqual(s["e2e.ok.correct"], "GENERATION-MISMATCH")
        for key in ("micro.decode.tps", "conf.tools", "replay.lean.cold_ttft_s"):
            self.assertEqual(s[key], "PASS", key)

    def test_fixture_rerecord_stales_only_replay(self):
        s = states(self.g, {**BASE, **OLD_OMP, "fixtures_sha": "f2"})
        self.assertEqual(s["replay.lean.cold_ttft_s"], "GENERATION-MISMATCH")
        self.assertEqual(s["micro.decode.tps"], "PASS")
        self.assertEqual(s["e2e.ok.first_wall_s"], "PASS")

    def test_an_agent_config_change_stales_e2e_only(self):
        # Isolation (2bdcdfc) changed the prompt omp sends by ~17% and moved only this pin.
        s = states(self.g, {**BASE, **OLD_OMP, "omp_agent_config": "a9", "fixtures_sha": "f1"})
        self.assertEqual(s["e2e.ok.first_wall_s"], "GENERATION-MISMATCH")
        self.assertEqual(s["micro.decode.tps"], "PASS")

    def test_mem_rows_follow_the_omp_configs_and_their_overlay_not_the_version(self):
        pins = {**BASE, **OLD_OMP, "fixtures_sha": "f1", "omp_mem_config": "m1"}
        g = golden_with(pins, keys=("mem.recall.wall_s", "micro.decode.tps"))
        for moved in ({"omp_child_config": "c9"}, {"omp_agent_config": "a9"}, {"omp_mem_config": "m9"}):
            s = states(g, {**pins, **moved})
            self.assertEqual(s["mem.recall.wall_s"], "GENERATION-MISMATCH", moved)
            self.assertEqual(s["micro.decode.tps"], "PASS", moved)
        self.assertEqual(states(g, {**pins, "omp_version": "18.3.0", "omp_sha": "o9"})["mem.recall.wall_s"], "PASS")

    def test_backend_change_stales_every_tier(self):
        s = states(self.g, {**BASE, "backend_sha": "b2", **OLD_OMP, "fixtures_sha": "f1"})
        self.assertEqual(set(s.values()), {"GENERATION-MISMATCH"})

    def test_unrecorded_pin_is_an_unknown_generation(self):
        # A golden banked before fixtures_sha existed cannot vouch for replay rows, even when the run lacks it too.
        legacy = golden_with({**BASE, **OLD_OMP, "fixtures_sha": None})
        s = states(legacy, {**BASE, **OLD_OMP})
        self.assertEqual(s["replay.lean.cold_ttft_s"], "GENERATION-MISMATCH")
        self.assertEqual(s["micro.decode.tps"], "PASS")


class BackendArgs(unittest.TestCase):
    PINS = {k: v for k, v in BASE.items() if k != "backend_args"}

    def test_legacy_golden_means_default_launch(self):
        legacy = golden_with(dict(self.PINS), keys=("micro.decode.tps",))
        self.assertEqual(states(legacy, {**self.PINS, "backend_args": ""})["micro.decode.tps"], "PASS")
        self.assertEqual(states(legacy, {**self.PINS, "backend_args": "--mtp"})["micro.decode.tps"],
                         "GENERATION-MISMATCH")

    def test_flagged_golden_needs_the_same_flags(self):
        flagged = golden_with({**self.PINS, "backend_args": "--mtp"}, keys=("micro.decode.tps",))
        self.assertEqual(states(flagged, {**self.PINS, "backend_args": "--mtp"})["micro.decode.tps"], "PASS")
        self.assertEqual(states(flagged, {**self.PINS, "backend_args": ""})["micro.decode.tps"], "GENERATION-MISMATCH")


class SplashPin(unittest.TestCase):
    SPLASH: ClassVar[dict] = {"splash_version": "1.0", "splash_sha": "6158ce6f1d4b2eb2"}

    def test_a_changed_splash_sha_stales_tiers_that_measured_splash(self):
        pins = {**BASE, "backend": "splash", **self.SPLASH}
        g = golden_with(pins, keys=("micro.decode.tps", "e2e.ok.first_wall_s"))
        moved = {**pins, "splash_sha": "0000000000000000"}
        s = states(g, moved)
        self.assertEqual(s["micro.decode.tps"], "GENERATION-MISMATCH")
        self.assertEqual(s["e2e.ok.first_wall_s"], "GENERATION-MISMATCH")

    def test_the_same_splash_sha_is_not_a_mismatch(self):
        pins = {**BASE, "backend": "splash", **self.SPLASH}
        g = golden_with(pins, keys=("micro.decode.tps",))
        self.assertEqual(states(g, pins)["micro.decode.tps"], "PASS")

    def test_a_resident_splash_does_not_stale_an_ollama_golden(self):
        g = golden_with({**BASE, **OLD_OMP, "fixtures_sha": "f1"}, keys=("micro.decode.tps",))
        run = {**BASE, **OLD_OMP, "fixtures_sha": "f1", **self.SPLASH}
        self.assertEqual(states(g, run)["micro.decode.tps"], "PASS")
        self.assertNotIn("splash_sha", golden.tier_diff(g, "micro", run))

    def test_attach_records_splash_only_when_it_is_the_backend_or_resident(self):
        base = {"backend": "ollama", "model": "m"}
        self.assertEqual(golden.attach_splash(base, backend="ollama", resident=False, splash=self.SPLASH), base)
        resident = golden.attach_splash(base, backend="ollama", resident=True, splash=self.SPLASH)
        self.assertEqual(resident["splash_sha"], self.SPLASH["splash_sha"])
        served = golden.attach_splash(base, backend="splash", resident=False, splash=self.SPLASH)
        self.assertEqual(served["splash_version"], "1.0")


class Merge(unittest.TestCase):
    def new_golden(self, pins: dict, tier: str, key: str) -> dict:
        return {"pins": pins, "aa_receipt": "r2", "tier_pins": {tier: {k: pins.get(k) for k in golden.tier_keys(tier)}},
                "metrics": {key: row(2.0)}, "conformance": {}}

    def test_rebank_one_tier_keeps_others_with_their_banked_pins(self):
        old = golden_with({**BASE, **OLD_OMP, "fixtures_sha": "f1"})
        new_pins = {**BASE, **NEW_OMP, "fixtures_sha": "f1"}
        merged, dropped = golden.merge(old, self.new_golden(new_pins, "e2e", "e2e.ok.first_wall_s"), ["e2e"])
        self.assertEqual(dropped, [])
        self.assertEqual(merged["metrics"]["micro.decode.tps"], row(1.0))
        self.assertEqual(merged["metrics"]["e2e.ok.first_wall_s"], row(2.0))
        self.assertNotIn("GENERATION-MISMATCH", states(merged, new_pins).values())
        # the kept replay rows still carry their own pins: a later fixture re-record invalidates them
        s = states(merged, {**new_pins, "fixtures_sha": "f9"})
        self.assertEqual(s["replay.lean.cold_ttft_s"], "GENERATION-MISMATCH")
        self.assertEqual(s["micro.decode.tps"], "PASS")

    def test_model_change_drops_untouched_tiers(self):
        old = golden_with({**BASE, **OLD_OMP, "fixtures_sha": "f1"})
        new_pins = {**BASE, **NEW_OMP, "fixtures_sha": "f1", "model_digest": "d2"}
        merged, dropped = golden.merge(old, self.new_golden(new_pins, "e2e", "e2e.ok.first_wall_s"), ["e2e"])
        self.assertEqual(dropped, ["conf", "micro", "replay"])
        self.assertEqual(set(merged["metrics"]), {"e2e.ok.first_wall_s"})

    def test_legacy_golden_without_backend_args_keeps_tiers_on_default_rebank(self):
        # 2026-09-23 regression: merge read the legacy golden's missing backend_args as a backend change.
        pins = {k: v for k, v in BASE.items() if k != "backend_args"}
        old = golden_with(dict(pins), keys=("micro.decode.tps", "e2e.ok.first_wall_s"))
        new = self.new_golden({**pins, "backend_args": ""}, "replay", "replay.lean.cold_ttft_s")
        merged, dropped = golden.merge(old, new, ["replay"])
        self.assertEqual(dropped, [])
        self.assertEqual(set(merged["metrics"]), {"micro.decode.tps", "e2e.ok.first_wall_s", "replay.lean.cold_ttft_s"})

    def test_legacy_golden_drops_tiers_when_flags_change(self):
        pins = {k: v for k, v in BASE.items() if k != "backend_args"}
        old = golden_with(dict(pins), keys=("micro.decode.tps", "e2e.ok.first_wall_s"))
        new = self.new_golden({**pins, "backend_args": "--mtp"}, "replay", "replay.lean.cold_ttft_s")
        _, dropped = golden.merge(old, new, ["replay"])
        self.assertEqual(dropped, ["conf", "e2e", "micro"])



class TokenizerVoid(unittest.TestCase):
    """A VOID conformance case is not a regression. kit-l7l's tokenizer mismatch must not red a golden."""

    VOID: ClassVar[dict] = {"level": "SHOULD", "verdict": "VOID"}

    def test_a_new_void_case_is_not_a_fail(self):
        g = golden_with({**BASE, "fixtures_sha": "f1"})
        rows = golden.compare({}, {"replay.lean.prompt_tokens": self.VOID}, g, {**BASE, "fixtures_sha": "f1"})
        self.assertEqual(rows[-1]["status"], "VOID")
        self.assertNotIn("FAIL", {r["status"] for r in rows if r["key"] == "replay.lean.prompt_tokens"})

    def test_a_matching_void_stays_void(self):
        g = golden_with({**BASE, "fixtures_sha": "f1"})
        g["conformance"]["replay.lean.prompt_tokens"] = self.VOID
        rows = golden.compare({}, {"replay.lean.prompt_tokens": dict(self.VOID)}, g,
                              {**BASE, "fixtures_sha": "f1"})
        hit = next(r for r in rows if r["key"] == "replay.lean.prompt_tokens")
        self.assertEqual(hit["status"], "VOID")

    def test_a_pass_golden_that_voids_still_fails(self):
        g = golden_with({**BASE, "fixtures_sha": "f1"})
        g["conformance"]["replay.lean.prompt_tokens"] = {"level": "SHOULD", "verdict": "PASS"}
        rows = golden.compare({}, {"replay.lean.prompt_tokens": self.VOID}, g, {**BASE, "fixtures_sha": "f1"})
        hit = next(r for r in rows if r["key"] == "replay.lean.prompt_tokens")
        self.assertEqual(hit["status"], "FAIL")



class BankedTokenizerFail(unittest.TestCase):
    """The mlx-serve golden banked this case as FAIL. A VOID run must not exit 1 on it."""

    PATH = Path(__file__).resolve().parents[1] / "goldens" / "mac-studio-apple-m3-ultra-512gb" / (
        "mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit.json")

    def rows(self, verdict: str) -> list[dict]:
        g = golden.load(self.PATH)
        conf = {k: dict(v) for k, v in g["conformance"].items()}
        conf["replay.lean.prompt_tokens"] = {"level": "SHOULD", "verdict": verdict}
        metrics = {k: {"value": v["value"], "better": v["better"]} for k, v in g["metrics"].items()}
        return golden.compare(metrics, conf, g, g["pins"])

    def test_a_void_run_against_the_banked_fail_is_not_failing(self):
        hit = next(r for r in self.rows("VOID") if r["key"] == "replay.lean.prompt_tokens")
        self.assertEqual(hit["status"], "VOID")
        self.assertNotIn(hit["status"], golden.FAILING)

    def test_the_resolved_listing_no_longer_excuses_a_fail(self):
        hit = next(r for r in self.rows("FAIL") if r["key"] == "replay.lean.prompt_tokens")
        self.assertEqual(hit["status"], "FAIL")
        self.assertIn(hit["status"], golden.FAILING)

if __name__ == "__main__":
    unittest.main()
