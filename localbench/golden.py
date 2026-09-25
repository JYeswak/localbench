"""Goldens: one file per (host, backend, model), banked ONLY from an A/A pair (`localbench aa`).

- Perf metrics are FUZZY goldens. The band is derived, never guessed: tol = max(3 x the relative A/A
  spread, a noise floor), and every tol names the receipt that produced it (`tol_source`). A metric
  without one is TOL-UNPROVEN, not PASS.
- Conformance cases are STRUCTURAL goldens (verdict + shape), exact match. A FAIL that stays a FAIL is
  XFAIL only when docs/evidence/DISCREPANCIES.md lists the case; otherwise it is FAIL.
- Pins (backend version + binary, model digest, omp version + binary, macOS build, host) are the
  generation. Any pin difference turns every row into GENERATION-MISMATCH: a golden from another
  generation is not evidence for this one.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .workloads import ROOT

GOLDENS = ROOT / "goldens"
DISCREPANCIES = ROOT / "docs" / "evidence" / "DISCREPANCIES.md"

# Below these relative bands the measurement cannot resolve a change on this desktop; see packet §7.
TOL_FLOOR = {"higher": 0.05, "lower": 0.10}
AA_MULTIPLIER = 3.0
# Latencies under a few hundred ms are dominated by jitter; latency bands get an absolute slack.
ABS_SLACK_S = 0.15

# What a row exercised decides which pins can invalidate it. conf and micro rows talk to the backend directly;
# replay rows send recorded omp request bodies, so their identity is those bodies (fixtures_sha), not the omp
# installed today. omp ships most days: its version and sha are recorded on the receipt and are not generation
# keys. The child overlay and the agent config are ours; those still bind the tiers that run live omp.
BACKEND_KEYS = ("host_id", "backend", "backend_version", "backend_sha", "backend_args", "model", "model_digest",
                "macos_build")
OMP_KEYS = ("omp_child_config", "omp_agent_config")
OMP_TIERS = frozenset({"e2e", "rel", "relcold", "relfresh"})
PIN_KEYS = BACKEND_KEYS + OMP_KEYS + ("fixtures_sha", "omp_mem_config")
# Splash is not a measured backend yet. These pins are recorded when Splash is the backend or a
# resident, and they invalidate only tiers whose golden says the backend was Splash. Putting them in
# PIN_KEYS would stale every ollama/mlx golden the moment the Splash binary moved, including goldens
# banked before the key existed (an unrecorded pin counts as moved).
SPLASH_KEYS = ("splash_version", "splash_sha")


def tier_keys(tier: str) -> tuple[str, ...]:
    """The pins a row of `tier` depends on. mem and sess run omp children with the mem overlay."""
    if tier in ("mem", "sess"):
        return BACKEND_KEYS + OMP_KEYS + ("omp_mem_config",)
    if tier in OMP_TIERS:
        return BACKEND_KEYS + OMP_KEYS
    if tier == "replay":
        return BACKEND_KEYS + ("fixtures_sha",)
    return BACKEND_KEYS


def tiers_of(g: dict) -> list[str]:
    return sorted({k.split(".", 1)[0] for part in ("metrics", "conformance") for k in g[part]})


def with_pin_defaults(pins: dict) -> dict:
    """backend_args entered the pins 2026-09-23. Goldens banked before then were launched without server flags (aa
    never received --server-arg; their mlx-serve logs show only default flags `[args] ... ctx-size=0, pld=on`), so for
    this one key an unrecorded value means "", not unknown. Every pin comparison against a golden goes through here:
    the first version applied it in tier_diff only, and merge() then read None != "" as a backend change and dropped a
    golden's conf/e2e/micro tiers (2026-09-23, caught before commit)."""
    return pins if pins.get("backend_args") is not None else {**pins, "backend_args": ""}


def golden_tier_pins(g: dict, tier: str) -> dict:
    """The pins `tier`'s rows were banked under: `tier_pins` when present, else the golden's single `pins` block
    (goldens banked before per-tier binding). A Splash golden's binary pin is part of that block even when the
    golden predates per-tier binding."""
    recorded = (g.get("tier_pins") or {}).get(tier)
    if recorded is not None:
        out = with_pin_defaults(recorded)
    else:
        out = with_pin_defaults({k: g["pins"].get(k) for k in tier_keys(tier)})
    if (g.get("pins") or {}).get("backend") == "splash":
        for k in SPLASH_KEYS:
            out.setdefault(k, (recorded or g["pins"]).get(k))
    return out


def tier_diff(g: dict, tier: str, pins: dict) -> dict:
    """The pins `tier` depends on that moved since the golden banked its rows. A pin the golden never recorded
    (None: banked before that pin existed) counts as moved — an unknown generation cannot vouch for a row. Without
    this, a pre-fixtures_sha golden and a pre-fixtures_sha run compared None == None and judged replay rows measured
    on different request bodies as PASS (caught on a real run dir, 2026-09-23).

    Splash keys join that comparison only when the golden or the run says the backend was Splash. A resident
    Splash during an ollama run is provenance, not a generation those rows exercised."""
    gp = golden_tier_pins(g, tier)
    keys = tier_keys(tier)
    if (g.get("pins") or {}).get("backend") == "splash" or pins.get("backend") == "splash":
        keys = keys + SPLASH_KEYS
    return {k: [gp.get(k), pins.get(k)] for k in keys if gp.get(k) is None or gp.get(k) != pins.get(k)}


def attach_splash(pins: dict, *, backend: str, resident: bool, splash: dict) -> dict:
    """Copy Splash's binary pin onto a run when Splash is the backend or a resident. Otherwise leave the
    pins unchanged, so an idle Splash install cannot become part of an ollama generation."""
    if backend == "splash" or resident:
        return {**pins, **{k: splash.get(k) for k in SPLASH_KEYS}}
    return pins


def stored_pin_block(pins: dict, tier: str | None = None) -> dict:
    """Pins written into a golden. Splash keys are kept only for a Splash backend, on the top-level block
    and on each tier block."""
    keys = tier_keys(tier) if tier is not None else PIN_KEYS
    block = {k: pins.get(k) for k in keys}
    if pins.get("backend") == "splash":
        block.update({k: pins.get(k) for k in SPLASH_KEYS})
    return block


FAILING = {"REGRESSED", "FAIL", "MISSING", "GENERATION-MISMATCH", "TOL-UNPROVEN"}


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


DRIFT_KEYS = ("omp_version", "omp_sha")


def arm_pin_drift(a_pins: list[dict], b_pins: list[dict], tiers: list[str]) -> dict[str, str]:
    """{tier: reason} for tiers whose A/A null is not a null: the legs of one arm ran different values of a pin that
    tier depends on. Across arms a pin may differ on purpose (it is the variable under test); within an arm it may
    not. omp's version and binary join the check for omp-bound tiers although goldens ignore them (4b9532d): a
    golden spans days of omp updates, an A/B spans minutes, and on 2026-09-25 omp replaced its 18.3.1 build inside
    leg A1 (startup 5.4 -> 1.5 s), which read as A/A noise and widened every e2e band."""
    drift: dict[str, str] = {}
    for tier in tiers:
        keys = tier_keys(tier)
        keys += DRIFT_KEYS if set(OMP_KEYS) <= set(keys) else ()
        for arm, pins in (("A", a_pins), ("B", b_pins)):
            for k in keys:
                seen = list(dict.fromkeys(str(p.get(k)) for p in pins))
                if len(seen) > 1:
                    drift.setdefault(tier, f"arm {arm} legs ran different {k}: {' / '.join(seen)}")
    return drift


LOAD_TOL_PCT = 5.0            # above c's own jitter: 0.0-4.3 points over 16 A/A pairs (ledger, load-balance row)
TIME_SUFFIXES = ("_s", "_tps")


def load_balance(a_busy: list[float | None], b_busy: list[float | None]) -> dict:
    """Which arm the whole-machine CPU load favoured, from each leg's mean CPU busy %. B's median must sit within
    LOAD_TOL_PCT of the A legs' range; below it B ran lighter ("B"), above it heavier ("A"). None when bracketed or
    when a leg did not record CPU busy (then nothing is withheld, and the receipt says so)."""
    out = {"a": a_busy, "b": b_busy, "tol_pct": LOAD_TOL_PCT, "favours": None}
    if not a_busy or not b_busy or any(c is None for c in a_busy + b_busy):
        return out | {"note": "CPU busy missing on a leg: balance not checked"}
    b = _median(b_busy)
    out["favours"] = "B" if b < min(a_busy) - LOAD_TOL_PCT else "A" if b > max(a_busy) + LOAD_TOL_PCT else None
    return out


def ab_table(a_legs: list[dict], b_legs: list[dict], void_tiers: dict[str, str] | None = None,
             load_favours: str | None = None) -> dict:
    """Per metric of interleaved A/B legs (metrics dicts, A,B,...,A order): each arm is the median of its legs; the
    noise is the wider relative range of the two arms (an arm with one leg has none); band = max(AA_MULTIPLIER x
    noise, TOL_FLOOR). With legs A1, B, A2 this is the original A,B,A rule: the median of two is their mean.
    Interleaving is how a busy machine stays fair: a burst of user activity lands in one leg, not in one arm.
    Rows of a tier in `void_tiers` (arm_pin_drift) are VOID with that reason. With `load_favours` (load_balance), a
    time row whose verdict points the way the load leaned (B-BETTER when B ran lighter, B-WORSE when heavier) is
    LOAD-FAVOURED: load alone could have produced it. A verdict against the load stands."""
    withheld = {"B": "B-BETTER", "A": "B-WORSE"}.get(load_favours or "")
    table = {}
    for key, m1 in a_legs[0].items():
        ms_a = [legs.get(key, {}) for legs in a_legs]
        ms_b = [legs.get(key, {}) for legs in b_legs]
        cells = {"a": [m.get("value") for m in ms_a], "b": [m.get("value") for m in ms_b]}
        if len(a_legs) == 2 and len(b_legs) == 1:
            cells |= {"a1": cells["a"][0], "a2": cells["a"][1], "b": cells["b"][0], "b_legs": cells["b"]}
            cells["a_legs"] = cells.pop("a")
        else:
            cells = {"a_legs": cells["a"], "b_legs": cells["b"]}
        drifted = (void_tiers or {}).get(key.split(".", 1)[0])
        if drifted or any(m.get("void") or not m.get("value") for m in ms_a + ms_b):
            table[key] = {**cells, "verdict": "VOID",
                          "void": drifted or next((m["void"] for m in ms_a + ms_b if m.get("void")), None)}
            continue
        a_vals, b_vals = [m["value"] for m in ms_a], [m["value"] for m in ms_b]
        a_med, b_med = _median(a_vals), _median(b_vals)
        aa_rel = (max(a_vals) - min(a_vals)) / a_med
        bb_rel = (max(b_vals) - min(b_vals)) / b_med if len(b_vals) > 1 else 0.0
        ratio = b_med / a_med
        band = max(AA_MULTIPLIER * max(aa_rel, bb_rel), TOL_FLOOR[m1["better"]])
        better = (ratio > 1) == (m1["better"] == "higher")
        verdict = "WITHIN-NOISE" if abs(ratio - 1) <= band else ("B-BETTER" if better else "B-WORSE")
        if verdict == withheld and key.endswith(TIME_SUFFIXES):
            verdict = "LOAD-FAVOURED"
        table[key] = {**cells, "a_median": a_med, "b_median": b_med, "better": m1["better"],
                      "aa_rel_spread": round(aa_rel, 4), "bb_rel_spread": round(bb_rel, 4), "band": round(band, 4),
                      "b_over_a": round(ratio, 3), "verdict": verdict,
                      **({"withheld": withheld} if verdict == "LOAD-FAVOURED" else {})}
    return table


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def golden_path(host_id: str, backend: str, model: str) -> Path:
    return GOLDENS / host_id / f"{slug(backend)}__{slug(model)}.json"


def flatten(results: list) -> tuple[dict, dict]:
    """Result objects (or their dicts) -> ({"case.metric": metric}, {case: {level, verdict, shape?}})."""
    metrics, conformance = {}, {}
    for r in results:
        r = r if isinstance(r, dict) else r.__dict__
        for name, m in r["metrics"].items():
            metrics[f"{r['case']}.{name}"] = m
        if r["verdict"] is not None:
            entry = {"level": r["level"], "verdict": r["verdict"]}
            if "shape" in r["detail"]:
                entry["shape"] = r["detail"]["shape"]
            conformance[r["case"]] = entry
    return metrics, conformance


def listed_discrepancies(backend: str, model: str) -> set[str]:
    """Case ids listed XFAIL in DISCREPANCIES.md for exactly this backend/model. An entry is a
    `## DISC-NNN: <case id> on <backend>/<model>` header whose block has a `Status: XFAIL <case id>` line; an
    XFAIL for one backend or model never excuses the same case on another."""
    if not DISCREPANCIES.exists():
        return set()
    cases, scope = set(), None
    for line in DISCREPANCIES.read_text().splitlines():
        head = re.match(r"##\s+DISC-\d+:\s+(\S+)\s+on\s+([^/\s]+)/(\S+)\s*$", line)
        if head:
            scope = head.groups()
            continue
        status = re.match(r"-\s+Status:\s+XFAIL\s+((?:conf|e2e|replay|micro)\.[A-Za-z0-9_.]+)", line)
        if status and scope and scope[0] == status.group(1) and scope[1:] == (backend, model):
            cases.add(status.group(1))
    return cases


def from_aa(run1: list, run2: list, pins: dict, receipt: str, tiers: list[str]) -> tuple[dict | None, list[str]]:
    """Build a golden from two same-config runs. Returns (golden, refusals); golden is None if refused."""
    m1, c1 = flatten(run1)
    m2, c2 = flatten(run2)
    refusals = []
    listed = listed_discrepancies(pins.get("backend"), pins.get("model"))
    for case, entry in c1.items():
        other = c2.get(case)
        if (other is None or other["verdict"] != entry["verdict"]) and entry["level"] == "MUST":
            refusals.append(f"{case}: MUST verdict differs between A/A runs ({entry['verdict']} vs "
                            f"{other and other['verdict']})")
        if entry["level"] == "MUST" and entry["verdict"] == "FAIL" and case not in listed:
            refusals.append(f"{case}: MUST FAIL not listed as XFAIL in docs/evidence/DISCREPANCIES.md")
    metrics = {}
    for key, a in m1.items():
        b = m2.get(key)
        if not b or a.get("void") or b.get("void") or not a.get("value") or not b.get("value"):
            continue
        mean = statistics.fmean([a["value"], b["value"]])
        rel = abs(a["value"] - b["value"]) / mean
        spreads = [x for m in (a, b) for x in m.get("spread", [])]
        metrics[key] = {
            "value": round(mean, 4), "better": a["better"],
            "tol": round(max(AA_MULTIPLIER * rel, TOL_FLOOR[a["better"]]), 4),
            "aa_rel_spread": round(rel, 4), "tol_source": receipt,
            "spread": [min(spreads), max(spreads)] if spreads else None,
        }
    conformance = {case: e if c2.get(case, {}).get("verdict") == e["verdict"] else {**e, "verdict": "FLAKY"}
                   for case, e in c1.items()}
    if refusals:
        return None, refusals
    return {"pins": stored_pin_block(pins), "aa_receipt": receipt,
            "tier_pins": {t: stored_pin_block(pins, t) for t in tiers},
            "metrics": metrics, "conformance": conformance}, []


def merge(old: dict | None, new: dict, tiers: list[str]) -> tuple[dict, list[str]]:
    """Fold a golden banked from an A/A of `tiers` into the existing one: rows of those tiers are replaced, rows of
    other tiers are kept with the pins they were banked under — but only while the backend pins still match (another
    backend build or model digest makes every old row another generation). Returns (golden, dropped tiers)."""
    if old is None:
        return new, []
    if pin_diff(with_pin_defaults(old["pins"]), new["pins"], BACKEND_KEYS):
        return new, [t for t in tiers_of(old) if t not in tiers]
    keep = [t for t in tiers_of(old) if t not in tiers]
    return {
        "pins": new["pins"], "aa_receipt": new["aa_receipt"],
        "tier_pins": {**{t: golden_tier_pins(old, t) for t in keep}, **new["tier_pins"]},
        **{part: {**{k: v for k, v in old[part].items() if k.split(".", 1)[0] in keep}, **new[part]}
           for part in ("metrics", "conformance")},
    }, []


def pin_diff(golden_pins: dict, pins: dict, keys: tuple[str, ...] = PIN_KEYS) -> dict:
    return {k: [golden_pins.get(k), pins.get(k)] for k in keys if golden_pins.get(k) != pins.get(k)}


def compare(metrics: dict, conformance: dict, golden: dict, pins: dict, tiers: list[str] | None = None) -> list[dict]:
    """Row per metric/case: PASS | IMPROVED | REGRESSED | VOID | NEW | MISSING | FAIL | XFAIL | FLAKY |
    TOL-UNPROVEN | GENERATION-MISMATCH. A row is GENERATION-MISMATCH only when a pin ITS tier depends on moved
    (tier_keys). With `tiers`, only golden rows of those tiers are judged (a `--tiers rel` run is not MISSING every
    micro row); a row of a tier that ran and produced nothing is still MISSING."""
    if tiers is not None:
        golden = {**golden, **{part: {k: v for k, v in golden[part].items() if k.split(".", 1)[0] in tiers}
                               for part in ("metrics", "conformance")}}
    diffs: dict[str, dict] = {}

    def stale(key: str) -> dict:
        tier = key.split(".", 1)[0]
        if tier not in diffs:
            diffs[tier] = tier_diff(golden, tier, pins)
        return diffs[tier]

    rows = []
    listed = listed_discrepancies(pins.get("backend"), pins.get("model"))
    for key, g in golden["metrics"].items():
        if stale(key):
            rows.append({"key": key, "status": "GENERATION-MISMATCH", "pins": stale(key)})
            continue
        a = metrics.get(key)
        if a is None:
            rows.append({"key": key, "status": "MISSING", "golden": g["value"]})
            continue
        if not g.get("tol_source"):
            rows.append({"key": key, "status": "TOL-UNPROVEN", "golden": g["value"], "actual": a.get("value")})
            continue
        if a.get("void"):
            rows.append({"key": key, "status": "VOID", "golden": g["value"], "reason": a["void"]})
            continue
        gv, av, tol = g["value"], a["value"], g["tol"]
        slack = ABS_SLACK_S if key.endswith("_s") else 0.0
        if g["better"] == "higher":
            status = "REGRESSED" if av < gv * (1 - tol) else "IMPROVED" if av > gv * (1 + tol) else "PASS"
        else:
            status = ("REGRESSED" if av > gv * (1 + tol) + slack else
                      "IMPROVED" if av < gv * (1 - tol) - slack else "PASS")
        rows.append({"key": key, "status": status, "golden": gv, "actual": av, "tol": tol,
                     "delta_pct": round((av - gv) / gv * 100, 1) if gv else None})
    for key, a in metrics.items():
        if key not in golden["metrics"]:
            rows.append({"key": key, "status": "VOID" if a.get("void") else "NEW", "actual": a.get("value")})
    for key, g in golden["conformance"].items():
        if stale(key):
            rows.append({"key": key, "status": "GENERATION-MISMATCH", "pins": stale(key)})
            continue
        a = conformance.get(key)
        if a is None:
            rows.append({"key": key, "status": "MISSING", "golden": g["verdict"]})
        elif g["verdict"] == "FLAKY":
            rows.append({"key": key, "status": "FLAKY", "actual": a})
        elif a == g and a["verdict"] == "VOID":
            rows.append({"key": key, "status": "VOID", "actual": a})
        elif a == g:
            status = "PASS" if a["verdict"] == "PASS" else ("XFAIL" if key in listed else "FAIL")
            rows.append({"key": key, "status": status, "actual": a})
        elif a["verdict"] == "PASS" and g["verdict"] == "FAIL":
            rows.append({"key": key, "status": "IMPROVED", "golden": g, "actual": a})
        elif a["verdict"] == "VOID" and g["verdict"] == "FAIL":
            rows.append({"key": key, "status": "VOID", "golden": g, "actual": a})
        else:
            rows.append({"key": key, "status": "FAIL", "golden": g, "actual": a})
    for key, a in conformance.items():
        if key not in golden["conformance"]:
            status = "NEW" if a["verdict"] == "PASS" else ("VOID" if a["verdict"] == "VOID" else "FAIL")
            rows.append({"key": key, "status": status, "actual": a})
    return rows


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def write(path: Path, golden: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
