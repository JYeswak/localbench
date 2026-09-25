"""Text projections of localbench's JSON records for a reader, a person or a model, at a fraction of the tokens.

The JSON on disk stays the record: progress.jsonl, summary.json, receipts and goldens are what code reads and what
claims cite. This module only projects it: numbers at 4 significant digits, pins stated once, the verdict first,
unchanged rows counted instead of listed, long free text clipped with an explicit `…[+N chars]` marker, long tables
paged with the command that continues them, and every row citable as `<file>#<RFC 6901 pointer>` into the JSON it
came from; `localbench show <file> --path <pointer>` prints that subtree unrounded. Nothing here is read back by code.
Audit 2026-09-23 (o200k tokens): a 3-leg A/B receipt read raw is 10.5k, its run stream 8.8k, a golden 3.8k."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from itertools import pairwise

CLIP = 160
PAGE = 50


# ---------------------------------------------------------------- values

def num(v) -> str:
    """4 significant digits; >= 1000 as a whole number (prefill tok/s, token counts); None as —."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        if v == 0:
            return "0"
        return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def clip(s: str, limit: int = CLIP) -> str:
    return s if len(s) <= limit else f"{s[:limit]}…[+{len(s) - limit} chars]"


def rounded(v):
    """`v` for display: floats at num()'s precision, strings clipped, structure and keys unchanged."""
    if isinstance(v, bool) or v is None or isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return v
        return round(v) if abs(v) >= 1000 else float(f"{v:.4g}")
    if isinstance(v, str):
        return clip(v)
    if isinstance(v, dict):
        return {k: rounded(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [rounded(x) for x in v]
    return str(v)


def val(v) -> str:
    """One value in a `key=value` list: bare when unambiguous, JSON-quoted when it holds spaces or separators."""
    if isinstance(v, str):
        s = clip(v)
        return s if s and not any(c in s for c in ' ="\n\t') else json.dumps(s, ensure_ascii=False)
    if isinstance(v, dict | list | tuple):
        return json.dumps(rounded(v), separators=(",", ":"), ensure_ascii=False)
    return num(v)


def kv(d: dict, skip: tuple = ()) -> str:
    return " ".join(f"{k}={val(v)}" for k, v in d.items() if k not in skip)


def members(who: list[str], universe: list[str], noun: str) -> str:
    """A subset of `universe` (listed once elsewhere) in the fewest characters that still name it exactly:
    `all N <noun>`, `all but X, Y`, or the members themselves."""
    missing = [n for n in universe if n not in who]
    if not missing:
        return f"all {len(universe)} {noun}"
    listed, but = ", ".join(who), f"all but {', '.join(missing)}"
    return but if len(but) < len(listed) else listed


def pointer(*parts) -> str:
    """RFC 6901 JSON pointer."""
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def resolve(doc, ptr: str):
    """The value at RFC 6901 pointer `ptr`. A miss names the segment and what exists at that level."""
    if ptr in ("", "/"):
        return doc
    cur, walked = doc, ""
    for raw in ptr.lstrip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        try:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, ValueError, TypeError):
            have = (", ".join(list(cur)[:25]) if isinstance(cur, dict) else
                    f"indexes 0..{len(cur) - 1}" if isinstance(cur, list) else f"a {type(cur).__name__}, not a container")
            raise KeyError(f"no {part!r} under {walked or '/'}; there: {have}") from None
        walked += "/" + raw
    return cur


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(r) + " |" for r in rows)]


def page(rows: list, cursor: int, limit: int, more: str) -> tuple[list, list[str]]:
    """Rows [cursor, cursor+limit) and, when rows remain, the line naming the command that continues."""
    part = rows[cursor:cursor + limit]
    end = cursor + len(part)
    tail = [f"rows {cursor + 1}-{end} of {len(rows)}; next: {more} --cursor {end}"] if end < len(rows) else []
    return part, tail


# ---------------------------------------------------------------- run stream

def event_line(ev: dict) -> str:
    """One progress event as `event k=v …`: every non-null field but the timestamp, rounded and clipped by val().
    The unrounded event is the same line of the run's progress.jsonl."""
    return " ".join([str(ev.get("event", "event")),
                     *(f"{k}={val(v)}" for k, v in ev.items() if k not in ("t", "event") and v is not None)])


# ---------------------------------------------------------------- records

def kind_of(doc: dict) -> str:
    if doc.get("kind") in ("aa", "ab", "run"):
        return doc["kind"]
    if {"pins", "metrics", "aa_receipt"} <= doc.keys():
        return "golden"
    if {"provenance", "verdicts", "metrics"} <= doc.keys():
        return "summary"
    return "json"


def _legs(doc: dict, kind: str) -> list[dict]:
    return {"ab": doc.get("legs"), "aa": doc.get("runs"), "run": [doc.get("run")], "summary": [doc]}.get(kind) or []


def _leg_ptr(kind: str, i: int) -> str:
    return {"ab": pointer("legs", i), "aa": pointer("runs", i), "run": pointer("run"), "summary": ""}[kind]


def _pins(sets: list[tuple[str, dict]]) -> list[str]:
    keys = list(dict.fromkeys(k for _, p in sets for k in p))
    first = sets[0][1]
    shared = {k: first.get(k) for k in keys if all(p.get(k) == first.get(k) for _, p in sets)}
    out = [f"pins{' (all legs)' if len(sets) > 1 else ''}: {kv(shared)}"]
    differ = [k for k in keys if k not in shared]
    if differ:
        out.append("pins that differ: " + "; ".join(
            f"{k}: " + " ".join(f"{label}={val(p.get(k))}" for label, p in sets) for k in differ))
    return out


def _system(leg: dict) -> dict:
    sysd, v = leg.get("system") or {}, leg.get("verdicts") or {}
    during = sysd.get("during") or {}
    gpu = during.get("gpu_device_pct") or {}
    others = [p for p in during.get("gpu_by_process") or [] if not p.get("model")]
    top = max(others, key=lambda p: p.get("pct") or 0, default=None)
    pre = (sysd.get("preflight") or {}).get("problems") or v.get("preflight_problems") or []
    # Whole-machine CPU busy (top). Across 35 legs 2026-09-23..25 it tracked mlx-serve decode (160 tok/s at <=12%
    # busy, 29-79 at 37-40%); a leg table without it hides the largest confounder on this host.
    busy = (sysd.get("cpu") or {}).get("busy_pct") or {}
    return {"contended": "yes" if v.get("contended") else "no", "must_fail": ", ".join(v.get("must_fail") or []) or "0",
            "preflight": "ok" if not pre else clip("; ".join(pre), 80),
            "gpu": f"{num(gpu.get('mean'))}/{num(gpu.get('max'))}",
            "other_gpu": f"{top['name']} {num(top.get('pct'))}" if top else "—",
            "pins_changed": val(v.get("pins_changed")) if v.get("pins_changed") else "no",
            "user_active": num((during.get("load") or {}).get("user_active_pct")),
            "app_gpu": (f"{num((during.get('load') or {}).get('app_gpu_mean_pct'))}/"
                        f"{num((during.get('load') or {}).get('app_gpu_p95_pct'))}") if during.get("load") else "—",
            "cpu": num(busy.get("mean")) if busy else "—", "cpu_max": num(busy.get("max")) if busy else "—"}


def _conformance(legs: list[tuple[str, dict]]) -> list[str]:
    out = []
    for label, leg in legs:
        conf = leg.get("conformance") or {}
        passed = sum(1 for e in conf.values() if e.get("verdict") == "PASS")
        rest = [f"{e.get('verdict')} {c} ({e.get('level')})" for c, e in conf.items() if e.get("verdict") != "PASS"]
        out.append(f"{label}: {passed}/{len(conf)} PASS" + (f"; {'; '.join(rest)}" if rest else ""))
    return out


def show(doc: dict, label: str, *, cursor: int = 0, limit: int = PAGE, cmd: str | None = None) -> str:
    """The reading view of a receipt (aa, ab, run), a golden, or a run's summary.json. `label` names the file in
    citations; `cmd` is the command that re-renders it (for the continuation line)."""
    kind = kind_of(doc)
    cmd = cmd or f"localbench show {label}"
    if kind == "golden":
        return _show_golden(doc, label, cursor, limit, cmd)
    if kind == "json":
        return _show_json(doc, label)
    legs = _legs(doc, kind)
    labels = [(leg.get("provenance") or {}).get("label") or f"leg{i}" for i, leg in enumerate(legs)]
    problems = doc.get("problems") if kind != "summary" else None
    sound = "SOUND" if not problems else "UNSOUND"
    if kind == "summary":
        sysrow = _system(legs[0])
        sound = "UNSOUND" if any(sysrow[k] not in ("no", "0", "ok") for k in ("contended", "must_fail", "preflight",
                                                                                "pins_changed")) else "SOUND"
    lines = [f"# {kind} · {sound} · {label}"]
    lines += [f"UNSOUND: {clip(p, 300)}" for p in problems or []]
    if kind == "ab":
        lines.append(f"A: {doc.get('a')}   B: {doc.get('b')}{'  (same spec)' if doc.get('a') == doc.get('b') else ''}"
                     f"   order {','.join(doc.get('order') or [])}")
    prov0 = legs[0].get("provenance") or {}
    lines.append(f"tiers {','.join(prov0.get('tiers') or [])} · repeats {prov0.get('repeats')}")
    lines += _pins([(lab, (leg.get("provenance") or {}).get("pins") or {}) for lab, leg in zip(labels, legs)])
    fps = [(leg.get("provenance") or {}).get("fingerprint") or {} for leg in legs]
    pinned = (prov0.get("pins") or {}).keys()
    extra = {k: v for k, v in fps[0].items() if k not in pinned and all(fp.get(k) == v for fp in fps)}
    if extra:
        lines.append(f"backend: {kv(extra)}")
    sysrows = [_system(leg) for leg in legs]
    lines += ["", *table(["leg", "created", "rev", "contended", "MUST fail", "preflight", "GPU mean/max %",
                          "top other GPU %", "app GPU mean/p95 %", "CPU busy mean/max %", "user active %",
                          "pins changed", "run_dir"],
                         [[lab, str((leg.get("provenance") or {}).get("created")),
                           str((leg.get("provenance") or {}).get("localbench_rev")), s["contended"], s["must_fail"],
                           s["preflight"], s["gpu"], s["other_gpu"], s["app_gpu"], f"{s['cpu']}/{s['cpu_max']}",
                           s["user_active"], s["pins_changed"], str(leg.get("run_dir"))]
                          for lab, leg, s in zip(labels, legs, sysrows)])]
    if kind == "ab":
        tbl = doc.get("table") or {}
        counts: dict[str, int] = {}
        for r in tbl.values():
            counts[r.get("verdict")] = counts.get(r.get("verdict"), 0) + 1
        lines += ["", "verdicts: " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
        arms = {arm: [s for s, a in zip(sysrows, doc.get("order") or []) if a == arm] for arm in ("A", "B")}
        if any(s["user_active"] != "—" or s["cpu"] != "—" for s in sysrows):
            lines.append("load by leg (reported, not gated): " + " · ".join(
                f"{arm} CPU-busy {'/'.join(s['cpu'] for s in rows)}% user-active "
                f"{'/'.join(s['user_active'] for s in rows)}% app-GPU {'/'.join(s['app_gpu'] for s in rows)}"
                for arm, rows in arms.items() if rows))
        if doc.get("pin_drift"):
            lines.append("pin drift within an arm (those tiers VOID): " + "; ".join(
                f"{t}: {r}" for t, r in doc["pin_drift"].items()))
        bal = doc.get("load_balance")
        if bal:
            lines.append("load balance: " + (bal.get("note") or (
                f"B ran {'lighter' if bal['favours'] == 'B' else 'heavier'} than every A leg by more than "
                f"{num(bal['tol_pct'])} CPU-busy points: time rows judged "
                f"{'B-BETTER' if bal['favours'] == 'B' else 'B-WORSE'} are LOAD-FAVOURED (withheld)"
                if bal.get("favours") else f"B within {num(bal['tol_pct'])} CPU-busy points of the A legs: no verdict withheld")))
        note = lambda r: r.get("verdict") + (f" ({clip(str(r['void']), 60)})" if r.get("void") else "") + (
            f" (withheld {r['withheld']})" if r.get("withheld") else "")
        if all("a1" in r for r in tbl.values()):
            rows = [[k, num(r.get("a1")), num(r.get("a2")), num(r.get("b")), num(r.get("b_over_a")), num(r.get("band")),
                     note(r)] for k, r in tbl.items()]
            head = ["metric", "A1", "A2", "B", "B/A", "band", "verdict"]
        else:
            legs_of = lambda xs: "/".join(num(x) for x in xs or [])
            rows = [[k, num(r.get("a_median")), legs_of(r.get("a_legs")), num(r.get("b_median")), legs_of(r.get("b_legs")),
                     num(r.get("b_over_a")), num(r.get("band")), note(r)] for k, r in tbl.items()]
            head = ["metric", "A median", "A legs", "B median", "B legs", "B/A", "band", "verdict"]
        part, tail = page(rows, cursor, limit, cmd)
        lines += table(head, part) + tail
        cite = f"cite: {label}#{pointer('table', '<metric>')} · leg values {label}#{pointer('legs', '<i>', 'metrics', '<metric>')}"
    else:
        keys = list(dict.fromkeys(k for leg in legs for k in (leg.get("metrics") or {})))
        rows = []
        for k in keys:
            ms = [(leg.get("metrics") or {}).get(k) or {} for leg in legs]
            vals = [m.get("value") for m in ms]
            cells = [f"VOID ({clip(str(m['void']), 50)})" if m.get("void") else num(m.get("value")) for m in ms]
            if len(legs) == 2:
                a, b = vals
                sp = (f"{abs(a - b) / ((a + b) / 2) * 100:.1f}" if isinstance(a, int | float) and isinstance(b, int | float)
                      and (a + b) else "—")
                rows.append([k, *cells, sp, num(ms[0].get("n"))])
            else:
                spread = ms[0].get("spread")
                rows.append([k, *cells, val(spread) if spread else "—", num(ms[0].get("n"))])
        part, tail = page(rows, cursor, limit, cmd)
        head = ["metric", *labels, "A/A spread %", "n"] if len(legs) == 2 else ["metric", "value", "spread", "n"]
        lines += [""] + table(head, part) + tail
        leg_ptr = _leg_ptr(kind, 0).replace("/0", "/<i>") if kind == "aa" else _leg_ptr(kind, 0)
        cite = f"cite: {label}#{leg_ptr}{pointer('metrics', '<metric>')}"
    lines += ["", "conformance: " + " | ".join(_conformance(list(zip(labels, legs))))]
    lines += _details([(lab, leg.get("details") or {}, _leg_ptr(kind, i)) for i, (lab, leg) in enumerate(zip(labels, legs))])
    lines += [cite + f" · raw subtree: localbench show {label} --path <pointer>"]
    return "\n".join(lines) + "\n"


INLINE = 120


def _details(legs: list[tuple[str, dict, str]]) -> list[str]:
    """Per-case detail fields of omp-driven tiers: a field whose rendering fits INLINE chars is shown; a larger one is
    named with its size and the pointer that prints it whole."""
    out = []
    for lab, details, leg_ptr in legs:
        for case, d in details.items():
            if not isinstance(d, dict) or not d:
                continue
            small, big = [], []
            for k, v in d.items():
                s = val(v)
                if len(s) <= INLINE:
                    small.append(f"{k}={s}")
                else:
                    size = f"{len(v)} items" if isinstance(v, dict | list) else f"{len(s)} chars"
                    big.append(f"{k} ({size}) {leg_ptr}{pointer('details', case, k)}")
            out.append(f"  {lab} {case}: " + " ".join(small) + (" · via --path: " + ", ".join(big) if big else ""))
    return ["details:", *out] if out else []


def _show_golden(g: dict, label: str, cursor: int, limit: int, cmd: str) -> str:
    lines = [f"# golden · {label}", f"banked from: {g.get('aa_receipt')}", f"pins: {kv(g.get('pins') or {})}"]
    for tier, tp in (g.get("tier_pins") or {}).items():
        moved = {k: v for k, v in tp.items() if (g.get("pins") or {}).get(k) != v}
        if moved:
            lines.append(f"tier {tier} banked under: {kv(moved)}")
    sources = {m.get("tol_source") for m in g["metrics"].values()}
    one_source = len(sources) == 1
    if one_source:
        lines.append(f"tol source (every row): {next(iter(sources))}")
    rows = [[k, num(m.get("value")), num(m.get("tol")), val(m.get("spread")) if m.get("spread") else "—",
             m.get("better") or "—", *([] if one_source else [str(m.get("tol_source"))])]
            for k, m in g["metrics"].items()]
    part, tail = page(rows, cursor, limit, cmd)
    lines += [""] + table(["metric", "value", "tol", "A/A spread", "better", *([] if one_source else ["tol source"])],
                          part) + tail
    conf = g.get("conformance") or {}
    rest = [f"{e.get('verdict')} {c} ({e.get('level')})" for c, e in conf.items() if e.get("verdict") != "PASS"]
    lines += ["", f"conformance: {sum(1 for e in conf.values() if e.get('verdict') == 'PASS')}/{len(conf)} PASS"
              + (f"; {'; '.join(rest)}" if rest else ""),
              f"cite: {label}#{pointer('metrics', '<metric>')} · raw subtree: localbench show {label} --path <pointer>"]
    return "\n".join(lines) + "\n"


def golden_diff(old: dict, new: dict, label: str, rev: str) -> str:
    """What a re-bank changed: pins that moved, tiers whose rows changed / did not / disappeared / appeared, every
    metric row whose value or tol changed (added and removed rows included), and conformance verdict changes.
    Unchanged rows are counted. A tier re-run with identical rows reads as unchanged: the golden cannot tell."""
    lines = [f"# golden diff · {label} · {rev} → working tree"]
    op, np_ = old.get("pins") or {}, new.get("pins") or {}
    moved = [f"{k}: {val(op.get(k))} → {val(np_.get(k))}" for k in dict.fromkeys([*op, *np_]) if op.get(k) != np_.get(k)]
    lines.append("pins moved: " + ("; ".join(moved) if moved else "none"))
    ot = {k.split(".", 1)[0] for k in [*old["metrics"], *(old.get("conformance") or {})]}
    nt = {k.split(".", 1)[0] for k in [*new["metrics"], *(new.get("conformance") or {})]}
    changed = sorted(t for t in ot & nt if any(
        old["metrics"].get(k) != new["metrics"].get(k) or (old.get("conformance") or {}).get(k)
        != (new.get("conformance") or {}).get(k)
        for k in [*old["metrics"], *new["metrics"], *(old.get("conformance") or {}), *(new.get("conformance") or {})]
        if k.startswith(t + ".")))
    lines.append(f"tiers: changed {', '.join(changed) or '—'} · unchanged {', '.join(sorted(ot & nt - set(changed))) or '—'}"
                 f" · dropped {', '.join(sorted(ot - nt)) or '—'} · new {', '.join(sorted(nt - ot)) or '—'}")
    rows, same = [], 0
    for k in dict.fromkeys([*old["metrics"], *new["metrics"]]):
        o, n = old["metrics"].get(k), new["metrics"].get(k)
        if o and n and o.get("value") == n.get("value") and o.get("tol") == n.get("tol"):
            same += 1
            continue
        ov, nv = (o or {}).get("value"), (n or {}).get("value")
        delta = f"{(nv - ov) / ov * 100:+.1f}" if isinstance(ov, int | float) and isinstance(nv, int | float) and ov else "—"
        rows.append([k, num(ov) if o else "(new)", num(nv) if n else "(removed)", delta,
                     num((o or {}).get("tol")), num((n or {}).get("tol"))])
    lines += [""] + table(["metric", "old", "new", "Δ%", "old tol", "new tol"], rows) if rows else ["", "no metric changed"]
    lines.append(f"unchanged rows: {same}")
    oc, nc = old.get("conformance") or {}, new.get("conformance") or {}
    changes = [f"{c}: {(oc.get(c) or {}).get('verdict', '(new)')} → {(nc.get(c) or {}).get('verdict', '(removed)')}"
               for c in dict.fromkeys([*oc, *nc]) if (oc.get(c) or {}).get("verdict") != (nc.get(c) or {}).get("verdict")]
    lines.append("conformance changes: " + ("; ".join(changes) if changes else "none"))
    return "\n".join(lines) + "\n"


def _show_json(doc, label: str) -> str:
    """Unknown JSON: its top-level shape, so the reader can pick a --path instead of reading it whole."""
    items = doc.items() if isinstance(doc, dict) else enumerate(doc)
    rows = [[str(k), type(v).__name__, str(len(v)) if isinstance(v, dict | list | str) else num(v)] for k, v in items]
    return "\n".join([f"# json · {label}", *table(["key", "type", "len/value"], rows),
                      f"raw subtree: localbench show {label} --path /<key>"]) + "\n"


def subtree(doc, ptr: str, label: str, *, cursor: int = 0, limit: int = PAGE) -> str:
    """The value at `ptr`, unrounded and unclipped (compact JSON). A dict or list longer than `limit` is paged by
    entry, with the command that continues."""
    v = resolve(doc, ptr)
    if isinstance(v, dict | list) and len(v) > limit:
        items = list(v.items()) if isinstance(v, dict) else list(enumerate(v))
        part = items[cursor:cursor + limit]
        body = dict(part) if isinstance(v, dict) else [x for _, x in part]
        end = cursor + len(part)
        more = (f"\nentries {cursor + 1}-{end} of {len(items)}; next: localbench show {label} --path {ptr} --cursor {end}"
                if end < len(items) else "")
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False) + more + "\n"
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False) + "\n"



# ---------------------------------------------------------------- run report

def run_report(summary: dict, rows: list[dict], gstate: str, bad: list[str]) -> str:
    """report.md for one run: verdict first, then provenance and machine state as key=value lines, then every metric
    against its golden and every conformance case."""
    pins = summary["provenance"]["pins"]
    sysd = summary["system"]
    host = sysd["before"]["host"]
    during = sysd["during"]

    def span(d: dict | None, unit: str = "") -> str:
        return f"mean {num(d.get('mean'))}{unit} max {num(d.get('max'))}{unit}" if d else "—"

    lines = [(f"# localbench · {pins['backend']} / {pins['model']} ({summary['provenance']['label']}) · "
              f"{'UNSOUND' if bad else 'SOUND'}"), "",
             f"- golden: {gstate}", *(f"- UNSOUND: {clip(b, 300)}" for b in bad),
             f"- verdicts: {kv(summary['verdicts'])}",
             f"- pins: {kv(pins)}",
             f"- host: {host['chip']} · {host['gpu_cores']} GPU cores · {host['mem_gb']} GB · macOS {host['macos']}",
             (f"- during run: GPU {span(during.get('gpu_device_pct'), '%')} · CPU busy "
              f"{span((sysd.get('cpu') or {}).get('busy_pct'), '%')} · load1 {span(during.get('load1'))} · swap "
              f"{span(during.get('swap_used_mb'), ' MB')} · pressure {', '.join(during.get('pressure_levels') or []) or '—'}"),
             "- GPU by process (whole run): " + (", ".join(
                 f"{r['name']} pid {r['pid']}{' (' + r['model'] + ')' if r.get('model') else ''} {num(r['pct'])}%"
                 for r in during.get("gpu_by_process", [])) or "—"),
             f"- power: {kv(sysd.get('power') or {})}", ""]
    by_key = {r["key"]: r for r in rows}
    trs = []
    for key, m in summary["metrics"].items():
        row = by_key.get(key, {})
        value = f"VOID ({clip(str(m['void']), 80)})" if m.get("void") else num(m.get("value"))
        trs.append([key, value, val(m["spread"]) if m.get("spread") else "", num(m.get("n")) if m.get("n") else "",
                    num(row.get("golden")) if "golden" in row else "", num(row.get("tol")) if "tol" in row else "",
                    num(row.get("delta_pct")) if "delta_pct" in row else "", row.get("status", "—")])
    lines += table(["metric", "value", "spread", "n", "golden", "tol", "Δ%", "status"], trs)
    lines += [""] + table(["conformance", "level", "verdict", "status"],
                          [[c, e["level"], e["verdict"], by_key.get(c, {}).get("status", "—")]
                           for c, e in summary["conformance"].items()])
    return "\n".join(lines) + "\n"


def _utc(t: float) -> str:
    """UTC stamp of an event's unix `t`, truncated to the second, matching `date -u -r`."""
    return datetime.fromtimestamp(int(t), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def monitor_findings(events: list[dict], run_name: str) -> list[str]:
    """MONITOR.md lines for one progress.jsonl. Observes; does not grade.

    A contention, a sample error, an e2e failure, a gap over 10 minutes, a preflight_wait series, or a
    stream that never reaches `done` is a finding. A clean `done` with none of those is one line. PASS
    rows are not events here and are not findings.
    """
    lines = []
    for ev in events:
        kind, t = ev.get("event"), ev.get("t")
        if t is None:
            continue
        if kind == "contention":
            lines.append(f"{_utc(t)} {run_name} contention {json.dumps(ev, separators=(',', ':'))}")
        elif kind == "sample" and ev.get("error"):
            lines.append(f"{_utc(t)} {run_name} sample {json.dumps(ev, separators=(',', ':'))}")
        elif kind == "e2e" and ev.get("ok") is False:
            lines.append(f"{_utc(t)} {run_name} e2e {json.dumps(ev, separators=(',', ':'))}")
    stamped = [(ev["t"], ev.get("event")) for ev in events if ev.get("t") is not None]
    for (t0, e0), (t1, _e1) in pairwise(stamped):
        gap = t1 - t0
        if gap > 600:
            lines.append(f"{_utc(t1)} {run_name} stall last event t={t0} ({e0}), no event for {int(gap // 60)}m")
    waits = [ev for ev in events if ev.get("event") == "preflight_wait" and ev.get("t") is not None]
    if waits:
        first, last = waits[0], waits[-1]
        lines.append(
            f"{_utc(last['t'])} {run_name} preflight_wait count={len(waits)} "
            f"first={_utc(first['t'])} last={_utc(last['t'])} "
            f"problems={json.dumps(last.get('problems'), separators=(',', ':'))}")
    if events and events[-1].get("event") != "done" and events[-1].get("t") is not None:
        last = events[-1]
        lines.append(f"{_utc(last['t'])} {run_name} ended without done; last event {last.get('event')} at {_utc(last['t'])}")
    if not lines and events and events[-1].get("event") == "done" and events[-1].get("t") is not None:
        lines.append(f"{_utc(events[-1]['t'])} {run_name} done no findings")
    return lines
