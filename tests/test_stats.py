"""Pre-registered sequential statistics tests (beads kit-r12, kit-v8t). Pure functions;
known values are textbook/closed-form, never fitted to the implementation."""

from __future__ import annotations

import math
import random
import unittest

from localbench import stats


class TCdf(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(stats.t_cdf(0.0, 8), 0.5)
        self.assertAlmostEqual(stats.t_cdf(2.306, 8), 0.975, places=3)
        self.assertAlmostEqual(stats.t_cdf(-2.306, 8), 0.025, places=3)
        self.assertAlmostEqual(stats.t_cdf(1.0, 1), 0.75, places=6)
        with self.assertRaises(ValueError):
            stats.t_cdf(0.0, 0)


class PairedT(unittest.TestCase):
    def test_clear_win_and_clear_loss(self):
        self.assertLess(stats.paired_t([-1.0, -2.0, -3.0])["p"], 0.05)
        self.assertGreater(stats.paired_t([1.0, 2.0, 3.0])["p"], 0.95)
        self.assertEqual(stats.paired_t([0.0, 0.0])["p"], 1.0)

    def test_needs_two_differences(self):
        with self.assertRaises(ValueError):
            stats.paired_t([0.5])


class WelchT(unittest.TestCase):
    def test_separated_arms_win(self):
        out = stats.welch_t_lower([1.0, 1.1, 0.9], [2.0, 2.1, 1.9])
        self.assertLess(out["p"], 0.01)
        self.assertGreater(out["df"], 0)

    def test_identical_arms_tie(self):
        self.assertEqual(stats.welch_t_lower([1.0, 1.0], [1.0, 1.0])["p"], 1.0)


class DriftPaired(unittest.TestCase):
    def test_fifteen_percent_win_under_drift(self):
        rng = random.Random(7)
        a = [10.0 * (1 + 0.02 * i) + rng.uniform(-0.2, 0.2) for i in range(7)]
        b = [0.85 * (x0 + x1) / 2 + rng.uniform(-0.2, 0.2)
             for x0, x1 in zip(a, a[1:])]
        out = stats.drift_paired(b, a)
        self.assertEqual(len(out["diffs"]), 6)
        self.assertLess(out["p"], 0.05)

    def test_symmetric_differences_do_not_win(self):
        ups = [math.e ** 0.5, math.e ** -0.5] * 3
        out = stats.drift_paired(ups, [1.0] * 7)
        self.assertEqual(out["p"], 0.5)

    def test_shape_and_positivity_are_refused(self):
        with self.assertRaises(ValueError):
            stats.drift_paired([1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            stats.drift_paired([0.0, 0.0], [1.0, 1.0, 1.0])

    def test_null_simulation_holds_type_one(self):
        # F2: no arm effect under smooth drift; the one-sided 5% test must not
        # reject far above its size. Seeded: deterministic outcome, generous bound.
        rng = random.Random(20261002)
        rejects = 0
        sims = 200
        for _ in range(sims):
            base = rng.uniform(5.0, 15.0)
            slope = rng.uniform(-0.05, 0.05)
            a = [base * (1 + slope * i) * rng.lognormvariate(0.0, 0.1) for i in range(7)]
            b = [(x0 + x1) / 2 * rng.lognormvariate(0.0, 0.1) for x0, x1 in zip(a, a[1:])]
            if stats.drift_paired(b, a)["p"] < 0.05:
                rejects += 1
        self.assertLessEqual(rejects / sims, 0.15)

    def test_fifteen_percent_win_has_power(self):
        # PowerPlan sizing spot-check: P=6 drift pairs detect a 15% win most runs.
        rng = random.Random(99)
        rejects = 0
        sims = 60
        for _ in range(sims):
            base = rng.uniform(5.0, 15.0)
            a = [base * rng.lognormvariate(0.0, 0.05) for _ in range(7)]
            b = [0.85 * (x0 + x1) / 2 * rng.lognormvariate(0.0, 0.05)
                 for x0, x1 in zip(a, a[1:])]
            if stats.drift_paired(b, a)["p"] < 0.05:
                rejects += 1
    def test_factor_is_exact_at_equal_variances(self):
        rng = random.Random(7)
        a = [10.0 * (1 + 0.02 * i) + rng.uniform(-0.2, 0.2) for i in range(7)]
        b = [0.85 * (x0 + x1) / 2 + rng.uniform(-0.2, 0.2)
             for x0, x1 in zip(a, a[1:])]
        out = stats.drift_paired(b, a)
        self.assertAlmostEqual(out["factor"], 1.2, delta=0.2)
        self.assertLess(out["p"], 0.05)

    def test_factor_grows_with_a_noisier_baseline(self):
        rng = random.Random(11)
        a = [10.0 + rng.uniform(-2.0, 2.0) for _ in range(7)]
        b = [8.5 + rng.uniform(-0.05, 0.05) for _ in range(6)]
        out = stats.drift_paired(b, a)
        self.assertGreater(out["factor"], 1.2)
        self.assertLessEqual(out["factor"], 2.0)

    def test_factor_shrinks_with_a_noisier_candidate(self):
        rng = random.Random(13)
        a = [10.0 + rng.uniform(-0.05, 0.05) for _ in range(7)]
        b = [8.5 + rng.uniform(-2.0, 2.0) for _ in range(6)]
        out = stats.drift_paired(b, a)
        self.assertLess(out["factor"], 1.2)
        self.assertGreaterEqual(out["factor"], 1.0)

    def test_unequal_variance_null_holds_type_one(self):
        # %pane re-review: with a noisier A arm the fixed 1.2 understated the
        # variance and inflated t toward false wins; the computed factor must
        # hold size. Seeded, generous bound, no arm effect.
        rng = random.Random(20261003)
        rejects = 0
        sims = 200
        for _ in range(sims):
            base = rng.uniform(5.0, 15.0)
            a = [base * rng.lognormvariate(0.0, 0.2) for _ in range(7)]
            b = [(x0 + x1) / 2 * rng.lognormvariate(0.0, 0.05)
                 for x0, x1 in zip(a, a[1:])]
            if stats.drift_paired(b, a)["p"] < 0.05:
                rejects += 1
        self.assertLessEqual(rejects / sims, 0.15)


class Fisher(unittest.TestCase):
    def test_balanced_tables_do_not_reject(self):
        self.assertAlmostEqual(stats.fisher_b_worse(9, 10, 9, 10), 0.7632, places=3)
        self.assertEqual(stats.fisher_b_worse(9, 10, 10, 10), 1.0)

    def test_shutout_rejects(self):
        self.assertLess(stats.fisher_b_worse(10, 10, 0, 10), 0.001)

    def test_counts_are_validated(self):
        with self.assertRaises(ValueError):
            stats.fisher_b_worse(11, 10, 0, 10)
        with self.assertRaises(ValueError):
            stats.fisher_b_worse(5, 0, 5, 10)


class IccCmh(unittest.TestCase):
    def test_independent_legs_have_no_clustering(self):
        groups = [(8, 9), (9, 9), (7, 9), (9, 9)]
        self.assertLess(stats.icc_binary(groups), 0.1)

    def test_clustered_legs_fall_back_to_cmh(self):
        groups = [(9, 9)] * 3 + [(0, 9)] * 3
        self.assertGreater(stats.icc_binary(groups), 0.1)

    def test_recall_test_selects_fisher_when_flat(self):
        a = [(8, 9), (9, 9), (9, 9)]
        b = [(8, 9), (9, 9), (9, 9)]
        out = stats.recall_test(a, b)
        self.assertEqual(out["method"], "pooled-fisher")
        self.assertAlmostEqual(out["p"], 0.7547, places=3)

    def test_recall_test_selects_cmh_when_clustered(self):
        a = [(9, 9)] * 4
        b = [(9, 9), (9, 9), (0, 9), (0, 9)]
        out = stats.recall_test(a, b)
        self.assertEqual(out["method"], "cmh")
        self.assertLess(out["p"], 0.05)
        self.assertEqual(out["dropped_legs"], (0, 0))

    def test_recall_test_names_dropped_legs(self):
        out = stats.recall_test([(9, 9)] * 5, [(0, 9)] * 4)
        self.assertEqual(out["method"], "cmh")
        self.assertEqual(out["dropped_legs"], (1, 0))
        self.assertEqual(out["clusters"], 4)
        self.assertFalse(out["few_clusters"])

    def test_cmh_needs_an_informative_pair(self):
        with self.assertRaises(ValueError):
            stats.cmh_b_worse([((9, 9), (9, 9))])

    def test_design_effect_is_one_without_clustering(self):
        self.assertEqual(stats.design_effect(0.0, [9, 9, 9]), 1.0)
        self.assertEqual(stats.design_effect(0.5, [1, 1, 1]), 1.0)
        self.assertAlmostEqual(stats.design_effect(1.0, [9, 9]), 9.0)
        with self.assertRaises(ValueError):
            stats.design_effect(0.5, [])
        with self.assertRaises(ValueError):
            stats.design_effect(0.5, [9, 0])

    def test_few_clustered_legs_shrink_to_deff_fisher(self):
        # Pooled Fisher screams on 18/18 vs 0/18; with 2 legs it is 2 observations.
        pooled = stats.fisher_b_worse(18, 18, 0, 18)
        self.assertLess(pooled, 0.05)
        out = stats.recall_test([(9, 9)] * 2, [(0, 9)] * 2)
        self.assertEqual(out["method"], "deff-fisher")
        self.assertTrue(out["few_clusters"])
        self.assertEqual(out["clusters"], 2)
        self.assertGreaterEqual(out["p"], 0.05)
        self.assertEqual(out["a_eff"], [2, 2])
        self.assertEqual(out["b_eff"], [0, 2])

    def test_three_clustered_pairs_still_shrink(self):
        out = stats.recall_test([(9, 9)] * 3, [(0, 9)] * 3)
        self.assertEqual(out["method"], "deff-fisher")
        self.assertTrue(out["few_clusters"])

    def test_min_clusters_is_validated(self):
        with self.assertRaises(ValueError):
            stats.recall_test([(9, 9)], [(9, 9)], min_clusters=0)
        with self.assertRaises(ValueError):
            stats.recall_test([(9, 9)], [(9, 9)], min_clusters=True)


class Obf(unittest.TestCase):
    def test_nominal_alphas_are_obrien_fleming(self):
        self.assertAlmostEqual(stats.OBF_ALPHA[0.25], 4.4e-5, delta=1e-5)
        self.assertAlmostEqual(stats.OBF_ALPHA[0.5], 0.0028, delta=0.0002)
        self.assertAlmostEqual(stats.OBF_ALPHA[0.75], 0.0119, delta=0.001)
        self.assertEqual(tuple(sorted(stats.OBF_INFO_FRACTIONS)), (0.25, 0.5, 0.75, 1.0))

    def test_hopeless_interims_reject_clean_interims_do_not(self):
        self.assertTrue(stats.obf_reject(0, 25, 0.9, 0.25))
        self.assertFalse(stats.obf_reject(20, 25, 0.9, 0.25))
        self.assertFalse(stats.obf_reject(22, 25, 0.9, 0.5))
        self.assertAlmostEqual(stats.obf_p(25, 25, 0.9), 1.0)

    def test_obf_p_is_a_probability(self):
        p = stats.obf_p(22, 25, 0.9)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_bad_looks_and_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            stats.obf_reject(0, 25, 0.9, 1.0)
        with self.assertRaises(ValueError):
            stats.obf_p(26, 25, 0.9)


class BettingCs(unittest.TestCase):
    def test_strong_pass_rate_excludes_a_low_bar(self):
        self.assertLess(stats.eb_p([1] * 50, 0.8), 0.05)
        self.assertGreater(stats.eb_lcb([1] * 50), 0.85)

    def test_all_fail_bounds_at_zero(self):
        self.assertEqual(stats.eb_lcb([0] * 50), 0.0)
        self.assertEqual(stats.eb_p([0] * 50, 0.0), 1.0)

    def test_null_holds_size_under_continuous_monitoring(self):
        # Optional stopping at every item: reject when the running-max capital
        # ever crosses 1/alpha. Seeded, generous bound.
        rng = random.Random(20261004)
        rejects = 0
        sims = 40
        for _ in range(sims):
            xs = [1 if rng.random() < 0.5 else 0 for _ in range(40)]
            if max(stats.eb_capital_path(xs, 0.5)) >= math.log(20.0):
                rejects += 1
        self.assertLessEqual(rejects / sims, 0.2)

    def test_find_scale_all_pass_stays_finite(self):
        # %pane WIP review: raw-exp mixture overflowed past n ~= 2000; the find
        # suite scale (n = 6626) must run log-safe.
        xs = [1] * 6626
        self.assertEqual(stats.eb_p(xs, 0.8), 0.0)
        path = stats.eb_capital_path(xs, 0.8)
        self.assertTrue(all(math.isfinite(v) for v in path))

    def test_lcb_covers_in_a_null_sim(self):
        rng = random.Random(20261005)
        misses = 0
        sims = 40
        for _ in range(sims):
            xs = [1 if rng.random() < 0.6 else 0 for _ in range(40)]
            if stats.eb_lcb(xs) > 0.6:
                misses += 1
        self.assertLessEqual(misses / sims, 0.2)

    def test_bad_scores_and_bars_are_refused(self):
        for bad in ([], [1.5], [float("nan")], [True]):
            with self.assertRaises(ValueError):
                stats.eb_p(bad, 0.5)
        for m0 in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                stats.eb_p([1, 0], m0)
        with self.assertRaises(ValueError):
            stats.eb_lcb([1, 0], alpha=1.0)
        with self.assertRaises(ValueError):
            stats.eb_capital_path([1, 0], 0.5, lambdas=[])

    def test_betting_cs_rejects_invalid_lambdas(self):
        for lam in (-1.01, 1.01, float("inf"), float("nan"), True):
            with self.subTest(lam=lam):
                with self.assertRaisesRegex(ValueError, "lambdas"):
                    stats.eb_capital_path([0, 1], 0.5, lambdas=(lam,))



class MixtureSprt(unittest.TestCase):
    def test_uniform_mixture_is_exact_on_five_of_five(self):
        out = stats.mixture_sprt_paired(5, 5)
        self.assertAlmostEqual(out["p"], 6 / 32, places=6)
        self.assertAlmostEqual(out["log_p"], math.log(6 / 32), places=6)

    def test_null_holds_size(self):
        rng = random.Random(20261006)
        rejects = 0
        sims = 60
        for _ in range(sims):
            b = sum(rng.random() < 0.5 for _ in range(30))
            if stats.mixture_sprt_paired(b, 30)["p"] < 0.05:
                rejects += 1
        self.assertLessEqual(rejects / sims, 0.2)

    def test_large_n_stays_finite_in_log_space(self):
        out = stats.mixture_sprt_paired(1880, 1880)
        self.assertTrue(math.isfinite(out["log_p"]))
        self.assertLess(out["log_p"], -700.0)
        self.assertEqual(out["p"], 0.0)

    def test_counts_and_prior_are_validated(self):
        for b, n in ((-1, 5), (6, 5), (0, 0)):
            with self.assertRaises(ValueError):
                stats.mixture_sprt_paired(b, n)
        with self.assertRaises(ValueError):
            stats.mixture_sprt_paired(3, 5, a=0.0)


class NormalQuantile(unittest.TestCase):
    def test_spot_values(self):
        self.assertAlmostEqual(stats.normal_quantile(0.975), 1.95996, places=4)
        self.assertAlmostEqual(stats.normal_quantile(0.025), -1.95996, places=4)
        self.assertEqual(stats.normal_quantile(0.5), 0.0)

    def test_unit_interval_is_enforced(self):
        for p in (0.0, 1.0, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                stats.normal_quantile(p)


class Ppi(unittest.TestCase):
    def test_perfect_judge_recovers_the_unlabeled_mean(self):
        yl = [1] * 70 + [0] * 30
        out = stats.ppi_mean([0.7] * 500, yl, list(yl))
        self.assertAlmostEqual(out["estimate"], 0.7, places=9)
        self.assertAlmostEqual(out["rectifier"], 0.0, places=9)
        self.assertLess(out["halfwidth"], 0.1)

    def test_rectifier_corrects_a_biased_judge(self):
        rng = random.Random(20261007)
        yl = [1 if rng.random() < 0.7 else 0 for _ in range(100)]
        fl = [y if rng.random() < 0.8 else 1 - y for y in yl]
        yu = [1 if rng.random() < 0.7 else 0 for _ in range(500)]
        fu = [y if rng.random() < 0.8 else 1 - y for y in yu]
        out = stats.ppi_mean(fu, yl, fl)
        expect = sum(fu) / len(fu) + (sum(yl) / len(yl) - sum(fl) / len(fl))
        self.assertAlmostEqual(out["estimate"], expect, places=9)
        self.assertAlmostEqual(out["estimate"], 0.7, delta=0.05)

    def test_interval_covers_in_simulation(self):
        rng = random.Random(20261008)
        misses = 0
        sims = 60
        for _ in range(sims):
            yl = [1 if rng.random() < 0.7 else 0 for _ in range(100)]
            fl = [y if rng.random() < 0.8 else 1 - y for y in yl]
            yu = [1 if rng.random() < 0.7 else 0 for _ in range(500)]
            fu = [y if rng.random() < 0.8 else 1 - y for y in yu]
            out = stats.ppi_mean(fu, yl, fl)
            if not out["lo"] <= 0.7 <= out["hi"]:
                misses += 1
        self.assertLessEqual(misses / sims, 0.2)

    def test_unpaired_and_degenerate_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            stats.ppi_mean([0.5] * 10, [1] * 5, [1] * 4)
        with self.assertRaises(ValueError):
            stats.ppi_mean([0.5], [1, 0], [1, 0])
        with self.assertRaises(ValueError):
            stats.ppi_mean([0.5] * 10 + [float("inf")], [1] * 30, [1] * 30)

    def test_ppi_rejects_out_of_range_rates_before_overflow(self):
        cases = (
            ([1e308, 1e308], [0, 1], [0, 1]),
            ([0, 1], [1e308, 1e308], [0, 1]),
            ([0, 1], [0, 1], [1e308, 1e308]),
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "PPI needs .* in \\[0, 1\\]"):
                    stats.ppi_mean(*values)

if __name__ == "__main__":
    unittest.main()


class HolmMultiplicity(unittest.TestCase):
    def test_holm_controls_two_benefit_endpoints(self):
        self.assertEqual(stats.holm_reject({"error": 0.04, "latency": 0.001}),
                         {"latency": True, "error": True})
        self.assertEqual(stats.holm_reject({"latency": 0.001, "error": 0.04}),
                         {"latency": True, "error": True})
