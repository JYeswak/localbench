"""Pre-registered sequential statistics, stdlib math only: exact small-sample tests for
the memory leg verdict and decision-screen early stops (beads kit-r12, kit-v8t;
PowerPlan bead kit-memory-study-vce comment 204).

All tests are one-sided at ALPHA = 0.05 unless stated. Float never holds a 2**n
denominator (kit-r12): tail sums run in log space or as exact integer ratios."""

from __future__ import annotations

import math

ALPHA = 0.05
# O'Brien-Fleming interim looks (information fractions of planned facts) and nominal
# one-sided alphas, Lan-DeMets z_k = 1.96 * sqrt(4/k): P(Z > z) = 4.4e-5, 0.0028,
# 0.0119 at 25/50/75%; overall one-sided ~= 0.04 <= 0.05. Reject-only: a look stops
# a screen only on a failure signal, never to accept. The final 1.0 look is the
# suite's own gate, not a p-value.
OBF_INFO_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
OBF_Z = {0.25: 3.92, 0.5: 2.77, 0.75: 2.26}
OBF_ALPHA = {f: 0.5 * math.erfc(z / math.sqrt(2.0)) for f, z in OBF_Z.items()}
# Facts are clustered within legs/sessions, so a pooled rate overstates precision past
# this intraclass correlation: above it the verdict falls back to Cochran-Mantel-Haenszel
# across matched leg pairs. 0.1 is the pre-registered line, not a fitted one.
ICC_FALLBACK = 0.1
# CMH's normal approximation needs enough matched strata: with fewer pairs the
# verdict shrinks pooled counts by the Kish design effect instead of pretending
# to cluster-adjust. 4 is the pre-registered line, not a fitted one.
MIN_CLUSTERS = 4


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes betacf)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) for 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(a * math.log(x) + b * math.log1p(-x) + log_beta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(b * math.log1p(-x) + a * math.log(x) + log_beta) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """P(T <= t) for Student t with df degrees of freedom."""
    if df < 1:
        raise ValueError(f"t needs df >= 1, got {df}")
    x = df / (df + t * t)
    lower = 0.5 * _betai(0.5 * df, 0.5, x)
    return lower if t <= 0 else 1.0 - lower


def _mean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs)


def paired_t(ds: list[float]) -> dict:
    """One-sided paired t for improvement (mean < 0): t, df, p. Needs >= 2 differences;
    zero variance is decisive (all differences share the sign) or a tie (p = 1)."""
    if len(ds) < 2:
        raise ValueError(f"paired t needs >= 2 differences, got {len(ds)}")
    mean = _mean(ds)
    var = math.fsum((d - mean) ** 2 for d in ds) / (len(ds) - 1)
    if var <= 0.0:
        return {"t": float("-inf") if mean < 0 else float("inf") if mean > 0 else 0.0,
                "df": len(ds) - 1, "p": 0.0 if mean < 0 else 1.0}
    t = mean / math.sqrt(var / len(ds))
    return {"t": t, "df": len(ds) - 1, "p": t_cdf(t, len(ds) - 1)}


def welch_t_lower(xs: list[float], ys: list[float]) -> dict:
    """One-sided Welch t for mean(xs) < mean(ys): t, df, p. Each arm needs >= 2 values."""
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(f"welch t needs >= 2 values per arm, got {len(xs)} and {len(ys)}")
    mx, my = _mean(xs), _mean(ys)
    vx = math.fsum((v - mx) ** 2 for v in xs) / (len(xs) - 1)
    vy = math.fsum((v - my) ** 2 for v in ys) / (len(ys) - 1)
    if vx <= 0.0 and vy <= 0.0:
        return {"t": 0.0 if mx == my else (float("-inf") if mx < my else float("inf")),
                "df": len(xs) + len(ys) - 2, "p": 1.0 if mx >= my else 0.0}
    se2 = vx / len(xs) + vy / len(ys)
    t = (mx - my) / math.sqrt(se2)
    df = se2 * se2 / ((vx / len(xs)) ** 2 / (len(xs) - 1) + (vy / len(ys)) ** 2 / (len(ys) - 1)) \
        if se2 > 0 else float(len(xs) + len(ys) - 2)
    return {"t": t, "df": df, "p": t_cdf(t, int(round(df))) if df >= 1 else 1.0}


def drift_paired(b_meds: list[float], a_meds: list[float]) -> dict:
    """Neighbour-paired log-scale leg test (PowerPlan B): d_i = lnB_i - (lnA_i + lnA_{i+1})/2
    over baseline-first A,B,A legs. Var(d_i) = sy2 + sx2/2 while successive differences
    estimate sy2 + sx2/4 (adjacent d share one A leg), so Var(d) = factor * mean of
    (d_{i+1} - d_i)^2/2 with factor = (sy2 + sx2/2)/(sy2 + sx2/4), in [1, 2] by
    construction. The fixed 1.2 was this factor at equal variances; with a noisier A
    arm the true factor is larger, so 1.2 understated the variance and inflated t
    toward false wins. Per-arm log variances come from successive leg differences
    (drift contaminates both arms about equally, keeping the ratio honest); 1.2 when
    both estimate zero. Var(mean) = Var(d)/P * (1 + (P - 1)/(3P)); t on P - 1 df,
    one-sided for a win (mean < 0 on the log scale). Needs len(a) == len(b) + 1 >= 3
    and positive values."""
    if len(a_meds) != len(b_meds) + 1 or len(b_meds) < 2:
        raise ValueError(f"drift pairing needs len(a) == len(b) + 1 >= 3, got {len(a_meds)} and {len(b_meds)}")
    logs = [math.log(v) for v in a_meds + b_meds]
    if any(not math.isfinite(v) for v in logs):
        raise ValueError("drift pairing needs positive leg values for the log scale")
    la, lb = logs[:len(a_meds)], logs[len(a_meds):]
    ds = [y - (x0 + x1) / 2 for y, x0, x1 in zip(lb, la, la[1:])]
    mean = _mean(ds)
    var_x = _mean([(p - q) ** 2 / 2 for p, q in zip(la[1:], la)])
    var_y = _mean([(p - q) ** 2 / 2 for p, q in zip(lb[1:], lb)])
    denom = var_y + var_x / 4
    factor = (var_y + var_x / 2) / denom if denom > 0 else 1.2
    var_d = factor * _mean([(p - q) ** 2 / 2 for p, q in zip(ds[1:], ds)])
    var_mean = var_d / len(ds) * (1 + (len(ds) - 1) / (3 * len(ds)))
    base = {"diffs": ds, "factor": factor, "var_x": var_x, "var_y": var_y, "var_d": var_d}
    if var_mean <= 0.0:
        return {**base, "t": 0.0 if mean == 0 else (float("-inf") if mean < 0 else float("inf")),
                "df": len(ds) - 1, "p": 1.0 if mean >= 0 else 0.0}
    t = mean / math.sqrt(var_mean)
    return {**base, "t": t, "df": len(ds) - 1, "p": t_cdf(t, len(ds) - 1)}


def _hypergeometric_logpmf(k: int, K: int, n: int, N: int) -> float:
    """log P(drawing k successes in n draws from K successes in N)."""
    if not (max(0, n + K - N) <= k <= min(n, K)):
        return float("-inf")
    return (math.lgamma(K + 1) - math.lgamma(k + 1) - math.lgamma(K - k + 1)
            + math.lgamma(N - K + 1) - math.lgamma(n - k + 1) - math.lgamma(N - K - n + k + 1)
            - (math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1)))


def fisher_b_worse(a_hits: int, a_n: int, b_hits: int, b_n: int) -> float:
    """Exact one-sided Fisher p that the candidate (B) rate is worse: P(B hits <= observed
    | margins) under equal rates, hypergeometric tail by recurrence from the mode side."""
    for name, v in (("a_hits", a_hits), ("a_n", a_n), ("b_hits", b_hits), ("b_n", b_n)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"fisher needs non-negative int counts, {name}={v!r}")
    if a_hits > a_n or b_hits > b_n or a_n == 0 or b_n == 0:
        raise ValueError(f"fisher needs 0 <= hits <= n per arm with n >= 1, got ({a_hits}/{a_n}, {b_hits}/{b_n})")
    N, K, n = a_n + b_n, a_hits + b_hits, b_n
    lo = max(0, n + K - N)
    logp = _hypergeometric_logpmf(b_hits, K, n, N)
    total = 0.0
    prob = math.exp(logp)
    for k in range(b_hits, lo - 1, -1):
        if k < b_hits:
            prob *= (k + 1) * (N - K - n + k + 1) / ((K - k) * (n - k)) if (K - k) * (n - k) else 0.0
        total += prob
    return min(1.0, total)


def icc_binary(groups: list[tuple[int, int]]) -> float:
    """One-way ANOVA ICC(1) of binary outcomes given as per-group (hits, n): 0 when
    unmeasurable (< 2 groups or no within-group room), else clamped to [0, 1]."""
    groups = [(h, n) for h, n in groups if n > 0]
    if len(groups) < 2:
        return 0.0
    total_hits = sum(h for h, _ in groups)
    total_n = sum(n for _, n in groups)
    if total_n - len(groups) <= 0:
        return 0.0
    mean = total_hits / total_n
    ss_between = math.fsum(n * (h / n - mean) ** 2 for h, n in groups)
    ss_within = math.fsum(n * (h / n) * (1 - h / n) for h, n in groups)
    df_between, df_within = len(groups) - 1, total_n - len(groups)
    if df_between <= 0 or df_within <= 0:
        return 0.0
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    n0 = (total_n - math.fsum(n * n for _, n in groups) / total_n) / df_between
    if ms_within <= 0:
        return 1.0 if ms_between > 0 else 0.0
    return max(0.0, min(1.0, (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)))


def cmh_b_worse(pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> dict:
    """Cochran-Mantel-Haenszel one-sided p that B is worse, across matched (A, B) leg
    pairs given as ((a_hits, a_n), (b_hits, b_n)). Cluster-robust fallback when facts
    cluster within legs (ICC above threshold). Needs >= 1 pair with variance."""
    num, den = 0.0, 0.0
    used = 0
    for (a_hits, a_n), (b_hits, b_n) in pairs:
        n = a_n + b_n
        if a_n <= 0 or b_n <= 0:
            continue
        hits = a_hits + b_hits
        expected = b_n * hits / n
        var = (a_n * b_n * hits * (n - hits) / (n * n * (n - 1))) if n > 1 else 0.0
        if var <= 0:
            continue
        num += b_hits - expected
        den += var
        used += 1
    if den <= 0 or used == 0:
        raise ValueError("cmh needs >= 1 informative matched pair")
    z = num / math.sqrt(den)
    return {"z": z, "pairs": used, "p": 0.5 * math.erfc(-z / math.sqrt(2.0))}


def design_effect(icc: float, sizes: list[int]) -> float:
    """Kish design effect 1 + (mean cluster size - 1) * ICC: the factor by which
    clustering inflates variance. 1.0 for singleton clusters or zero ICC."""
    if not sizes:
        raise ValueError("design_effect needs >= 1 cluster size")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in sizes):
        raise ValueError(f"design_effect needs positive cluster sizes, got {sizes!r}")
    icc = max(0.0, min(1.0, float(icc)))
    return 1.0 + (math.fsum(sizes) / len(sizes) - 1.0) * icc


def recall_test(a_counts: list[tuple[int, int]], b_counts: list[tuple[int, int]],
                icc_threshold: float = ICC_FALLBACK, min_clusters: int = MIN_CLUSTERS) -> dict:
    """Pooled-facts quality test with clustering fallbacks: ICC over per-leg (hits, n)
    at or below threshold -> exact one-sided Fisher on pooled counts; above with
    enough matched leg pairs -> CMH across them (extra legs are named, not silently
    kept); above with too few pairs -> Fisher on design-effect-shrunk counts, flagged
    few_clusters (a pilot cannot buy precision by pooling: the effective sample is
    near the leg count, so a pooled-significant split correctly stops rejecting).
    Returns {method, icc, p, clusters, few_clusters, ...}."""
    for counts in (a_counts, b_counts):
        for hits, n in counts:
            if not isinstance(hits, int) or isinstance(hits, bool) or not isinstance(n, int) \
                    or isinstance(n, bool) or hits < 0 or n < 1 or hits > n:
                raise ValueError(f"recall_test needs per-leg (hits, n) counts, got {(hits, n)!r}")
    if not a_counts or not b_counts:
        raise ValueError("recall_test needs >= 1 leg per arm")
    if not isinstance(min_clusters, int) or isinstance(min_clusters, bool) or min_clusters < 1:
        raise ValueError(f"recall_test needs min_clusters >= 1, got {min_clusters!r}")
    icc = icc_binary([(h, n) for h, n in a_counts + b_counts])
    a_hits, a_n = sum(h for h, _ in a_counts), sum(n for _, n in a_counts)
    b_hits, b_n = sum(h for h, _ in b_counts), sum(n for _, n in b_counts)
    pairs = list(zip(a_counts, b_counts))
    if icc <= icc_threshold:
        return {"method": "pooled-fisher", "icc": icc,
                "p": fisher_b_worse(a_hits, a_n, b_hits, b_n),
                "a": [a_hits, a_n], "b": [b_hits, b_n],
                "clusters": len(pairs), "few_clusters": False}
    if len(pairs) >= min_clusters:
        dropped = (len(a_counts) - len(pairs), len(b_counts) - len(pairs))
        out = cmh_b_worse([(a, b) for a, b in pairs])
        return {"method": "cmh", "icc": icc, **out,
                "paired": len(pairs), "dropped_legs": dropped,
                "clusters": len(pairs), "few_clusters": False}
    deff = design_effect(icc, [n for _, n in a_counts + b_counts])
    a_eff = [int(a_hits / deff), int(a_n / deff)]
    b_eff = [int(b_hits / deff), int(b_n / deff)]
    if a_eff[1] < 1 or b_eff[1] < 1:
        p = 1.0
    else:
        p = fisher_b_worse(min(a_eff[0], a_eff[1]), a_eff[1], min(b_eff[0], b_eff[1]), b_eff[1])
    return {"method": "deff-fisher", "icc": icc, "deff": deff, "p": p,
            "a": [a_hits, a_n], "b": [b_hits, b_n],
            "a_eff": a_eff, "b_eff": b_eff,
            "clusters": len(pairs), "few_clusters": True}


def obf_p(successes: int, n: int, p0: float) -> float:
    """Exact one-sided binomial lower tail P(X <= successes | p0): the interim evidence
    against meeting a gate, in float (terms stay <= 1, no overflow at any n)."""
    if n <= 0 or not 0.0 < p0 < 1.0:
        raise ValueError(f"obf needs n >= 1 and 0 < p0 < 1, got n={n} p0={p0!r}")
    if not isinstance(successes, int) or isinstance(successes, bool) \
            or not 0 <= successes <= n:
        raise ValueError(f"obf needs 0 <= successes <= n, got {successes!r} of {n}")
    log_p0, log_q0 = math.log(p0), math.log1p(-p0)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * log_p0 + (n - i) * log_q0)
                         for i in range(successes + 1)))


def obf_reject(successes: int, n: int, p0: float, look: float) -> bool:
    """Reject-only O'Brien-Fleming interim: True when the exact one-sided binomial lower
    tail is below the nominal alpha for `look` in (0.25, 0.5, 0.75). Final looks use
    the suite gate itself, never this function."""
    if look not in OBF_ALPHA:
        raise ValueError(f"obf interim looks are {sorted(OBF_ALPHA)}, got {look}")
    return obf_p(successes, n, p0) < OBF_ALPHA[look]


# --- anytime-valid sequential tests (Waudby-Smith & Ramdas betting CS; Robbins mixture
# SPRT). Screens may stop at any item with no alpha inflation: every p below holds under
# optional stopping (Ville), replacing the 4-look OBF above, which stays as the
# pre-registered fallback until the betting CS passes review. All log-space; n=1880 safe.

# Betting grid for [0, 1] scores: 1 + lam * (x - m) stays >= 0.1 on this range
# (x - m in [-1, 1]), so log1p never touches its -1 pole; Ville holds for any grid.
EB_LAMBDAS = tuple(0.9 * i / 20 for i in range(-20, 21))
# Inversion grid for the lower bound (pointwise running-max capitals, no monotonicity ask).
EB_M_GRID = tuple(i / 100 for i in range(101))


def _bounded(xs, name: str) -> list[float]:
    """[0, 1] floats, bools refused; every screen score this module gates is a rate."""
    if not xs:
        raise ValueError(f"betting CS needs >= 1 observation, got {name!r}")
    out = []
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) \
                or not 0.0 <= x <= 1.0:
            raise ValueError(f"betting CS needs [0, 1] scores, got {x!r} in {name}")
        out.append(float(x))
    return out


def eb_capital_path(xs: list[float], m: float, lambdas: tuple = EB_LAMBDAS) -> list[float]:
    """LOG running-max mixture capital testing mean(xs) == m: per-prefix log of the
    lambda-grid mean of prod(1 + lam * (x - m)), maximized over prefixes in log space
    (Ville: P(ever >= 1/alpha) <= alpha under the null). Log-sum-exp throughout, so
    no exp() ever sees a raw log-capital: n=6626 all-pass safe. Custom lambdas must
    be finite and lie strictly inside (-1, 1). m in [0, 1]."""
    xs = _bounded(xs, "xs")
    if not isinstance(m, (int, float)) or isinstance(m, bool) or not math.isfinite(m) \
            or not 0.0 <= m <= 1.0:
        raise ValueError(f"betting CS needs m in [0, 1], got {m!r}")
    if not lambdas:
        raise ValueError("betting CS needs >= 1 lambda")
    if any(isinstance(lam, bool) or not isinstance(lam, (int, float))
           or not -1.0 < lam < 1.0 or not math.isfinite(lam) for lam in lambdas):
        raise ValueError("betting CS needs finite lambdas in (-1, 1)")
    log_cap = [0.0] * len(lambdas)
    path, running = [], float("-inf")
    for x in xs:
        for i, lam in enumerate(lambdas):
            log_cap[i] += math.log1p(lam * (x - m))
        peak = max(log_cap)
        log_mix = peak + math.log(math.fsum(math.exp(c - peak) for c in log_cap)
                                  / len(lambdas))
        running = max(running, log_mix)
        path.append(running)
    return path


def eb_p(xs: list[float], m0: float, lambdas: tuple = EB_LAMBDAS) -> float:
    """Anytime p against mean <= m0: exp(-final log capital) (Ville: rejecting when
    p < alpha holds its size however the stopping item was chosen). Underflows to 0.0
    on overwhelming evidence, like the mixture SPRT. Small p excludes means at or
    below m0, i.e. evidence the true mean exceeds m0."""
    final = eb_capital_path(xs, m0, lambdas)[-1]
    return min(1.0, math.exp(-final)) if final > 0 else 1.0


def eb_lcb(xs: list[float], alpha: float = ALPHA, m_grid: tuple = EB_M_GRID,
           lambdas: tuple = EB_LAMBDAS) -> float:
    """Lower (1 - alpha) confidence bound by grid inversion: the smallest grid m whose
    running-max LOG capital stays below log(1/alpha). 0.0 when even m = 0 is plausible
    (weak evidence); near 1.0 on all-pass runs. Valid however the stopping item was
    chosen. Cost is O(grid * n * lambdas) pure Python (~27M log1p at n = 6626:
    fine rarely, not per-item)."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
            or not 0.0 < alpha < 1.0:
        raise ValueError(f"betting CS needs 0 < alpha < 1, got {alpha!r}")
    xs = _bounded(xs, "xs")
    log_level = -math.log(alpha)
    for m in sorted(m_grid):
        if eb_capital_path(xs, m, lambdas)[-1] < log_level:
            return float(m)
    return 1.0


def mixture_sprt_paired(b: int, n: int, a: float = 1.0, b_: float = 1.0) -> dict:
    """Robbins mixture-SPRT p for paired binary outcomes: b local-only wins of n
    discordant pairs, H0 split 1/2, uniform-mixture capital M = 2^n B(b+a, n-b+b_)/B(a,b_).
    Two-sided by construction (either direction grows capital); the sign of b - n/2 gives
    the direction. Log-space throughout: n=1880 safe. p = min(1, 1/M); log_p stays
    finite where p underflows to 0.0."""
    if not isinstance(b, int) or isinstance(b, bool) or not isinstance(n, int) \
            or isinstance(n, bool) or not 0 <= b <= n or n < 1:
        raise ValueError(f"mixture SPRT needs 0 <= b <= n with n >= 1, got b={b!r} n={n!r}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
               for v in (a, b_)):
        raise ValueError(f"mixture SPRT needs positive Beta prior, got {(a, b_)!r}")
    log_m = (n * math.log(2.0) + math.lgamma(b + a) + math.lgamma(n - b + b_)
             - math.lgamma(n + a + b_) + math.lgamma(a + b_) - math.lgamma(a) - math.lgamma(b_))
    return {"p": min(1.0, math.exp(-log_m)) if log_m > 0 else 1.0, "log_p": -log_m,
            "log_capital": log_m, "b": b, "n": n}


def normal_quantile(p: float) -> float:
    """Standard-normal quantile (Acklam's approximation, ~1e-9): the CI need PPI has
    that stdlib omits. Valid for the planned n ~= 100 gold set; proof claims must not
    use the normal CI under n = 30 (no exact fallback here)."""
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 < p < 1.0:
        raise ValueError(f"normal quantile needs 0 < p < 1, got {p!r}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log1p(-p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def ppi_mean(f_unlabeled: list[float], y_labeled: list[float], f_labeled: list[float],
             alpha: float = ALPHA) -> dict:
    """Prediction-powered win-rate estimate (Angelopoulos et al. 2023): the judge's mean
    over N unlabeled pairs rectified by its mean error on n trusted gold labels.
    estimate = mean(f_u) + mean(y_l - f_l); variance = var(f_u)/N + var(resid)/n
    (sample variances); normal CI at alpha. PPI++ weighting rejected until the gold set
    exceeds ~200 (tuning weights on n = 100 overfits). Gold labels rectify the judge,
    never train it. Inputs are finite rates in [0, 1]; lists paired where paired; n, N >= 2."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
            or not 0.0 < alpha < 1.0:
        raise ValueError(f"PPI needs 0 < alpha < 1, got {alpha!r}")
    for name, vals in (("f_unlabeled", f_unlabeled), ("y_labeled", y_labeled),
                        ("f_labeled", f_labeled)):
        if len(vals) < 2:
            raise ValueError(f"PPI needs >= 2 {name} values, got {len(vals)}")
        for v in vals:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
                    or not 0.0 <= v <= 1.0:
                raise ValueError(f"PPI needs {name} values in [0, 1], got {v!r}")
    if len(y_labeled) != len(f_labeled):
        raise ValueError(f"PPI labels and judge values pair up: {len(y_labeled)} != {len(f_labeled)}")
    n, n_un = len(y_labeled), len(f_unlabeled)
    mean_u = math.fsum(f_unlabeled) / n_un
    resid = [y - f for y, f in zip(y_labeled, f_labeled)]
    rectifier = math.fsum(resid) / n
    var_u = math.fsum((v - mean_u) ** 2 for v in f_unlabeled) / (n_un - 1)
    mean_r = rectifier
    var_r = math.fsum((v - mean_r) ** 2 for v in resid) / (n - 1)
    half = normal_quantile(1 - alpha / 2) * math.sqrt(var_u / n_un + var_r / n)
    estimate = mean_u + rectifier
    return {"estimate": estimate, "halfwidth": half, "lo": estimate - half,
            "hi": estimate + half, "rectifier": rectifier,
            "n_labeled": n, "n_unlabeled": n_un}


def holm_reject(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down family-wise rejection for a named benefit family."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("Holm alpha must be between 0 and 1")
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    out = {name: False for name in pvalues}
    for rank, (name, pvalue) in enumerate(ordered):
        if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise ValueError(f"Holm p-value must be in [0, 1]: {name}={pvalue!r}")
        if pvalue > alpha / (len(ordered) - rank):
            break
        out[name] = True
    return out
