"""The reading views (render.py) must not lose what a reader needs: every metric row, verdict, leg, differing pin,
problem and non-PASS case of every committed receipt and golden; every field of a progress event; the exact JSON of
any subtree. Rounding and clipping are the only allowed losses, and both are marked or reversible (--path)."""

import json
import unittest
from pathlib import Path

from localbench import render

ROOT = Path(__file__).resolve().parent.parent
RECEIPTS = sorted(p for p in (ROOT / "docs/evidence/receipts").glob("*.json")
                  if json.loads(p.read_text()).get("kind") in ("aa", "ab", "run"))
GOLDENS = sorted((ROOT / "goldens").glob("*/*.json"))
ALL = 10**6   # a page large enough to render every row


def legs_of(doc: dict) -> list[dict]:
    return doc.get("legs") or doc.get("runs") or [doc["run"]]


class Numbers(unittest.TestCase):
    def test_four_significant_digits_without_scientific_notation_above_1000(self):
        self.assertEqual(render.num(162.783), "162.8")
        self.assertEqual(render.num(0.07692), "0.07692")
        self.assertEqual(render.num(2648.8801), "2649")
        self.assertEqual(render.num(262144.0), "262144")
        self.assertEqual((render.num(None), render.num(True), render.num(0.0), render.num(7)), ("—", "yes", "0", "7"))


class Members(unittest.TestCase):
    UNIVERSE = ["default", "claude", "codex", "glm", "grok", "lab", "proj-b-scratch-p4", "muse", "omp-test"]

    def decode(self, label: str) -> set[str]:
        if label.startswith("all but "):
            return set(self.UNIVERSE) - set(label.removeprefix("all but ").split(", "))
        if label.startswith("all "):
            return set(self.UNIVERSE)
        return set(label.split(", "))

    def test_every_label_names_exactly_its_subset(self):
        import itertools
        for r in range(1, len(self.UNIVERSE) + 1):
            for who in itertools.combinations(self.UNIVERSE, r):
                label = render.members(list(who), self.UNIVERSE, "profiles")
                self.assertEqual(self.decode(label), set(who), label)
                self.assertLessEqual(len(label), len(", ".join(who)))


class EventLine(unittest.TestCase):
    ISOLATED = {"t": 1790180911.576, "event": "isolated", "evicted": [], "freed": [], "purged": None,
                "fingerprint": {"backend": "mlx-serve", "backend_version": "26.9.2", "backend_sha": "4d09a3beb8d4c9be",
                                "model": "Qwen3.6-35B-A3B-MLX-Serve-4bit", "model_digest": "files:68dedceb5da0",
                                "backend_args": "--mtp", "architecture": "qwen3_5_moe", "quantization": "4-bit",
                                "mtp_loaded": True, "loaded_context": 262144,
                                "model_dir": "~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit"}}
    SAMPLE = {"t": 1.0, "event": "sample", "label": "conf.tool_call", "prompt_tokens": 293, "cached_tokens": 0,
              "completion_tokens": 26, "ttft_s": 0.9502728750230744, "prefill_tps": 308.3324881738684,
              "decode_tps": 0.0, "total_s": 0.9503270410059486, "error": "HTTP 500: upstream reset the stream"}

    def test_every_field_but_the_timestamp_and_nulls_is_present(self):
        for ev in (self.ISOLATED, self.SAMPLE):
            line = render.event_line(ev)
            self.assertTrue(line.startswith(ev["event"] + " "))
            for k, v in ev.items():
                if k in ("t", "event") or v is None:
                    self.assertNotIn(f" {k}=", line)
                else:
                    self.assertIn(f" {k}=", line)

    def test_a_long_nested_value_stays_valid_complete_json(self):
        # The old echo cut this event at 300 characters: invalid JSON, model_dir and loaded_context lost.
        line = render.event_line(self.ISOLATED)
        nested = json.loads(line.split(" fingerprint=", 1)[1])
        self.assertEqual(nested, self.ISOLATED["fingerprint"])

    def test_floats_are_rounded_and_long_text_clipped_with_a_marker(self):
        line = render.event_line({**self.SAMPLE, "stderr": "x" * 500})
        self.assertIn("ttft_s=0.9503 ", line)
        self.assertIn(f"stderr=\"{'x' * render.CLIP}…[+{500 - render.CLIP} chars]\"", line)


class ReceiptViews(unittest.TestCase):
    def test_there_are_receipts_of_every_kind_to_check(self):
        kinds = {json.loads(p.read_text())["kind"] for p in RECEIPTS}
        self.assertEqual(kinds, {"aa", "ab", "run"})

    def test_every_row_leg_problem_and_non_pass_case_survives(self):
        for path in RECEIPTS:
            doc = json.loads(path.read_text())
            out = render.show(doc, path.name, limit=ALL)
            with self.subTest(receipt=path.name):
                keys = doc["table"] if doc["kind"] == "ab" else {k for leg in legs_of(doc) for k in leg["metrics"]}
                for k in keys:
                    self.assertIn(f"| {k} |", out)
                for leg in legs_of(doc):
                    self.assertIn(str(leg["run_dir"]), out)
                    for case, e in leg["conformance"].items():
                        if e["verdict"] != "PASS":
                            self.assertIn(f"{e['verdict']} {case}", out)
                for p in doc.get("problems") or []:
                    self.assertIn(p[:100], out)
                self.assertIn("UNSOUND" if doc.get("problems") else "SOUND", out.splitlines()[0])

    def test_ab_rows_carry_every_leg_value_and_the_verdict(self):
        for path in (p for p in RECEIPTS if json.loads(p.read_text())["kind"] == "ab"):
            doc = json.loads(path.read_text())
            out = render.show(doc, path.name, limit=ALL)
            for k, r in doc["table"].items():
                row = next(ln for ln in out.splitlines() if ln.startswith(f"| {k} |"))
                if "a1" in r:        # one pair: A1, A2, B columns
                    for field in ("a1", "a2", "b"):
                        self.assertIn(f" {render.num(r[field])} ", row, (path.name, k, field))
                else:                # --pairs N: arm medians plus every leg value
                    for field in ("a_median", "b_median"):
                        if field in r:
                            self.assertIn(f" {render.num(r[field])} ", row, (path.name, k, field))
                    for v in r["a_legs"] + r["b_legs"]:
                        self.assertIn(render.num(v), row, (path.name, k, v))
                self.assertIn(r["verdict"], row)

    def test_pins_that_differ_between_legs_are_named_per_leg(self):
        leg = {"provenance": {"label": "x", "pins": {"m": "q", "backend_args": ""}}, "metrics": {}, "conformance": {},
               "run_dir": "r", "verdicts": {}}
        doc = {"kind": "ab", "a": "s", "b": "s", "order": ["A", "B", "A"], "problems": [], "table": {},
               "legs": [leg, {**leg, "provenance": {"label": "y", "pins": {"m": "q", "backend_args": "--mtp"}}}]}
        out = render.show(doc, "r.json")
        self.assertIn('pins that differ: backend_args: x="" y=--mtp', out)
        self.assertIn("m=q", out)

    def test_a_void_leg_value_shows_its_reason_in_its_column(self):
        def leg(label, metric):
            return {"provenance": {"label": label, "pins": {}}, "metrics": {"e2e.ok.first_wall_s": metric},
                    "conformance": {}, "run_dir": label, "verdicts": {}}
        doc = {"kind": "aa", "problems": [], "runs": [
            leg("aa1", {"value": None, "void": "first call reported 11400 cached tokens; not cold"}),
            leg("aa2", {"value": 6.46, "n": 1})]}
        row = next(ln for ln in render.show(doc, "r.json").splitlines() if ln.startswith("| e2e.ok.first_wall_s |"))
        self.assertEqual(row.split(" | ")[1:3], ["VOID (first call reported 11400 cached tokens; not cold)", "6.46"])

    def test_long_tables_page_and_every_row_is_reachable_by_cursor(self):
        doc = json.loads(next(p for p in RECEIPTS if json.loads(p.read_text())["kind"] == "ab").read_text())
        seen, cursor = [], 0
        while True:
            out = render.show(doc, "r.json", cursor=cursor, limit=4)
            seen += [ln.split(" | ")[0][2:] for ln in out.splitlines() if ln.startswith("| ") and "." in ln.split(" | ")[0]]
            nxt = [ln for ln in out.splitlines() if "--cursor" in ln and ln.startswith("rows ")]
            if not nxt:
                break
            cursor = int(nxt[0].rsplit("--cursor ", 1)[1])
        self.assertEqual(seen, list(doc["table"]))


    def test_turn_gate_metrics_and_details_are_visible(self):
        leg = {
            "provenance": {"label": "run", "pins": {}, "tiers": ["mem", "sess"], "repeats": 1},
            "metrics": {
                "mem.turn.max_tool_calls": {"value": 356, "better": "lower", "spread": [0, 356], "n": 48},
                "mem.turn.timeouts": {"value": 1, "better": "lower", "spread": [1, 1], "n": 48},
                "sess.turn.max_tool_calls": {"value": 8, "better": "lower", "spread": [0, 8], "n": 12},
                "sess.turn.timeouts": {"value": 0, "better": "lower", "spread": [0, 0], "n": 12},
            },
            "details": {"mem.turn": {"turns": [{"tool_calls": 356, "timeout": True}]},
                        "sess.turn": {"turns": [{"tool_calls": 8, "timeout": False}]}},
            "conformance": {}, "run_dir": "runs/x", "verdicts": {},
        }
        out = render.show({"kind": "run", "run": leg}, "summary.json", limit=ALL)
        for metric in ("mem.turn.max_tool_calls", "mem.turn.timeouts",
                       "sess.turn.max_tool_calls", "sess.turn.timeouts"):
            self.assertIn(f"| {metric} |", out)
        self.assertIn("mem.turn", out)
        self.assertIn("sess.turn", out)

    def test_nonpassing_replay_detail_is_visible_and_reachable_by_path(self):
        from localbench.__main__ import _receipt_view

        detail = {"recorded": 11433, "replayed": 10853}
        summary = {"provenance": {"label": "run", "pins": {}, "tiers": ["replay"], "repeats": 1},
                   "verdicts": {}, "metrics": {}, "conformance": {
                       "replay.lean.prompt_tokens": {"level": "SHOULD", "verdict": "FAIL"},
                       "replay.full.prompt_tokens": {"level": "SHOULD", "verdict": "PASS"}},
                   "run_dir": "runs/x", "system": {"before": {"live": {}, "host": {}}},
                   "results": [
                       {"case": "replay.lean.prompt_tokens", "tier": "replay", "verdict": "FAIL", "detail": detail},
                       {"case": "replay.full.prompt_tokens", "tier": "replay", "verdict": "PASS",
                        "detail": {"recorded": 73779, "replayed": 73779}},
                       {"case": "replay.full", "tier": "replay", "verdict": None,
                        "detail": {"prompt_tokens": 73779, "fixture_prompt_tokens": 73779}},
                   ]}
        doc = {"kind": "run", "run": _receipt_view(summary)}
        shown = render.show(doc, "receipt.json")
        self.assertIn("replay.lean.prompt_tokens: recorded=11433 replayed=10853", shown)
        self.assertNotIn("replay.full.prompt_tokens:", shown)
        self.assertNotIn("replay.full:", shown)
        ptr = "/run/details/replay.lean.prompt_tokens"
        self.assertEqual(json.loads(render.subtree(doc, ptr, "receipt.json")), detail)
        perf_ptr = "/run/details/replay.full"
        self.assertEqual(json.loads(render.subtree(doc, perf_ptr, "receipt.json")),
                         {"prompt_tokens": 73779, "fixture_prompt_tokens": 73779})
        all_pass = {**summary,
                    "conformance": {case: {**entry, "verdict": "PASS"}
                                    for case, entry in summary["conformance"].items()},
                    "results": [{**r, "verdict": "PASS"} if r["verdict"] is not None else r
                                for r in summary["results"]]}
        pass_doc = {"kind": "run", "run": _receipt_view(all_pass)}
        empty_doc = {"kind": "run", "run": _receipt_view({**all_pass, "results": []})}
        self.assertEqual(len(render.show(pass_doc, "receipt.json")),
                         len(render.show(empty_doc, "receipt.json")))
        legacy_pass = {**pass_doc, "run": {**pass_doc["run"], "details": {
            **pass_doc["run"]["details"], "replay.full.prompt_tokens": {"recorded": 73779, "replayed": 73779}}}}
        self.assertNotIn("replay.full.prompt_tokens:", render.show(legacy_pass, "receipt.json"))
        ab_doc = {"kind": "ab", "a": "s", "b": "s", "order": ["A", "B", "A"],
                  "problems": [], "table": {}, "legs": [_receipt_view(all_pass), _receipt_view(summary),
                                                         _receipt_view(all_pass)]}
        self.assertIn("replay.lean.prompt_tokens: recorded=11433 replayed=10853",
                      render.show(ab_doc, "receipt.json"))
        self.assertEqual(json.loads(render.subtree(
            ab_doc, "/legs/1/details/replay.lean.prompt_tokens", "receipt.json")), detail)

class SummaryViews(unittest.TestCase):
    @staticmethod
    def summary(*, resident_unknown_samples: int, app_gpu_pct: float, contended: bool = False,
                resident_owned_stop_samples: int = 0) -> dict:
        return {
            "provenance": {"label": "run", "created": "2026-09-30T15:58:47Z", "localbench_rev": "rev",
                           "tiers": ["conf"], "repeats": 1, "pins": {}},
            "verdicts": {"contended": contended, "must_fail": [], "preflight_problems": [], "pins_changed": {}},
            "metrics": {},
            "conformance": {},
            "system": {"during": {"resident_unknown_samples": resident_unknown_samples,
                                  "resident_owned_stop_samples": resident_owned_stop_samples,
                                  "load": {"app_gpu_mean_pct": app_gpu_pct, "app_gpu_p95_pct": app_gpu_pct}}},
            "run_dir": "runs/test",
        }

    @staticmethod
    def contended_cell(out: str) -> str:
        columns = next(line for line in out.splitlines() if line.startswith("| leg |"))
        values = next(line for line in out.splitlines() if line.startswith("| run |"))
        column_names = [value.strip() for value in columns.strip("|").split("|")]
        cells = [value.strip() for value in values.strip("|").split("|")]
        return dict(zip(column_names, cells))["contended"]

    def test_unknown_residency_summary_is_unsound_without_claiming_contention(self):
        out = render.show(self.summary(resident_unknown_samples=1, app_gpu_pct=0), "summary.json")

        self.assertTrue(out.startswith("# summary · UNSOUND · summary.json"))
        self.assertIn("UNSOUND: resident model state unknown in 1 sampler sample(s); run is non-proof", out)
        self.assertEqual(self.contended_cell(out), "no")

    def test_missing_residency_sample_count_is_unsound_and_incomplete(self):
        doc = self.summary(resident_unknown_samples=0, app_gpu_pct=0)
        del doc["system"]["during"]["resident_unknown_samples"]
        out = render.show(doc, "summary.json")

        self.assertTrue(out.startswith("# summary · UNSOUND · summary.json"))
        self.assertIn("residency sampling incomplete: resident_unknown_samples is missing; run is non-proof", out)

    def test_known_isolation_with_app_gpu_load_remains_sound(self):
        out = render.show(self.summary(resident_unknown_samples=0, app_gpu_pct=80), "summary.json")

        self.assertTrue(out.startswith("# summary · SOUND · summary.json"))
        self.assertEqual(self.contended_cell(out), "no")
        row = next(line for line in out.splitlines() if line.startswith("| run |"))
        self.assertIn("| 80/80 |", row)

    def test_known_second_model_is_contended_and_unsound(self):
        out = render.show(self.summary(resident_unknown_samples=0, app_gpu_pct=0, contended=True), "summary.json")

        self.assertTrue(out.startswith("# summary · UNSOUND · summary.json"))
        self.assertIn("UNSOUND: CONTENDED: another model was resident or running during the run (system.contention)",
                      out)
        self.assertEqual(self.contended_cell(out), "yes")



    def test_owned_stop_samples_are_visible_without_marking_unknown(self):
        doc = self.summary(resident_unknown_samples=0, resident_owned_stop_samples=1, app_gpu_pct=0)
        out = render.show(doc, "summary.json")

        self.assertTrue(out.startswith("# summary · SOUND · summary.json"))
        header = next(line for line in out.splitlines() if line.startswith("| leg |"))
        row = next(line for line in out.splitlines() if line.startswith("| run |"))
        columns = [value.strip() for value in header.strip("|").split("|")]
        values = [value.strip() for value in row.strip("|").split("|")]
        self.assertEqual(dict(zip(columns, values))["owned-stop samples"], "1")

        doc["provenance"]["pins"] = {"backend": "mlx-serve", "model": "target"}
        doc["system"]["before"] = {"host": {"chip": "test", "gpu_cores": 1, "mem_gb": 1, "macos": "test"}}
        report = render.run_report(doc, [], "CURRENT", [])
        self.assertIn("- residency samples: 0 unknown · 1 owned-stop", report)


class GoldenViews(unittest.TestCase):
    def test_every_golden_row_keeps_value_and_tol(self):
        for path in GOLDENS:
            g = json.loads(path.read_text())
            out = render.show(g, path.name, limit=ALL)
            with self.subTest(golden=path.name):
                for k, m in g["metrics"].items():
                    row = next(ln for ln in out.splitlines() if ln.startswith(f"| {k} |"))
                    self.assertIn(f" {render.num(m['value'])} | {render.num(m['tol'])} ", row)
                for case, e in g["conformance"].items():
                    if e["verdict"] != "PASS":
                        self.assertIn(f"{e['verdict']} {case}", out)
                for k in g["pins"]:
                    self.assertIn(f"{k}=", out)

    def test_diff_lists_changed_added_removed_rows_moved_pins_and_verdicts(self):
        row = {"value": 1.0, "tol": 0.1, "better": "lower"}
        old = {"pins": {"omp_version": "18.2.10", "model": "m"}, "metrics": {
            "micro.a": row, "micro.b": row, "e2e.gone": row}, "conformance": {"conf.x": {"level": "MUST", "verdict": "PASS"}}}
        new = {"pins": {"omp_version": "18.2.11", "model": "m"}, "metrics": {
            "micro.a": row, "micro.b": {**row, "value": 1.25}, "replay.new": row},
            "conformance": {"conf.x": {"level": "MUST", "verdict": "FAIL"}}}
        out = render.golden_diff(old, new, "g.json", "HEAD")
        self.assertIn("omp_version: 18.2.10 → 18.2.11", out)
        self.assertNotIn("model:", out.split("\n")[1])
        self.assertIn("| micro.b | 1 | 1.25 | +25.0 |", out)
        self.assertIn("| e2e.gone | 1 | (removed) |", out)
        self.assertIn("| replay.new | (new) | 1 |", out)
        self.assertNotIn("| micro.a |", out)
        self.assertIn("unchanged rows: 1", out)
        self.assertIn("conf.x: PASS → FAIL", out)
        self.assertIn("dropped e2e", out)
        self.assertIn("new replay", out)


class Subtree(unittest.TestCase):
    def test_path_returns_the_exact_json(self):
        doc = json.loads(RECEIPTS[0].read_text())
        for ptr in ("/problems", render.pointer(*(["legs", 0] if "legs" in doc else ["runs", 0] if "runs" in doc
                                                 else ["run"]), "metrics")):
            self.assertEqual(json.loads(render.subtree(doc, ptr, "r.json", limit=ALL)), render.resolve(doc, ptr))

    def test_paged_subtree_reassembles_to_the_whole(self):
        doc = {"xs": [{"i": i, "v": i / 3} for i in range(11)]}
        parts, cursor = [], 0
        while True:
            out = render.subtree(doc, "/xs", "d.json", cursor=cursor, limit=4).splitlines()
            parts += json.loads(out[0])
            if len(out) == 1:
                break
            cursor = int(out[1].rsplit("--cursor ", 1)[1])
        self.assertEqual(parts, doc["xs"])

    def test_a_missing_segment_names_what_is_there(self):
        with self.assertRaises(KeyError) as ctx:
            render.resolve({"a": {"b": 1, "c": 2}}, "/a/z")
        self.assertIn("no 'z' under /a; there: b, c", ctx.exception.args[0])


class RunReport(unittest.TestCase):
    SUMMARY = {
        "provenance": {"pins": {"backend": "ollama", "model": "m", "omp_version": "18.2.11"}, "label": "run"},
        "verdicts": {"contended": False, "must_fail": [], "preflight_problems": [], "pins_changed": {}},
        "metrics": {"micro.decode.decode_tps": {"value": 146.08, "spread": [139.4, 153.0], "n": 3},
                    "e2e.ok.first_wall_s": {"value": None, "void": "first call reported 11400 cached tokens"}},
        "conformance": {"conf.tool_call": {"level": "MUST", "verdict": "PASS"},
                        "conf.greedy_deterministic": {"level": "SHOULD", "verdict": "FAIL"}},
        "system": {"before": {"host": {"chip": "Apple M3 Ultra", "gpu_cores": 80, "mem_gb": 512, "macos": "26.5.2"}},
                   "during": {"gpu_device_pct": {"mean": 70.4, "max": 99}, "gpu_by_process": []},
                   "power": {"available": False}}}

    def test_verdict_first_and_every_row_present(self):
        rows = [{"key": "micro.decode.decode_tps", "golden": 141.6, "tol": 0.18, "delta_pct": 3.2, "status": "PASS"}]
        out = render.run_report(self.SUMMARY, rows, "compared to g.json", ["micro.x: REGRESSED"])
        self.assertTrue(out.startswith("# localbench · ollama / m (run) · UNSOUND"))
        self.assertIn("- UNSOUND: micro.x: REGRESSED", out)
        self.assertIn("| micro.decode.decode_tps | 146.1 | [139.4,153.0] | 3 | 141.6 | 0.18 | 3.2 | PASS |", out)
        self.assertIn("VOID (first call reported 11400 cached tokens)", out)
        self.assertIn("| conf.greedy_deterministic | SHOULD | FAIL |", out)
        for k in (*self.SUMMARY["verdicts"], *self.SUMMARY["provenance"]["pins"]):
            self.assertIn(f"{k}=", out)
        self.assertIn("· SOUND", render.run_report(self.SUMMARY, rows, "g", []).splitlines()[0])



class MonitorFindings(unittest.TestCase):
    RUN = "20260922T211002Z__ollama__qwen3.6-35b"
    PLANTED = {"event": "contention", "t": 1790111643.2,
               "resident": {"ollama": ["qwen3.8:27b-mlx"], "mlx-serve": []}, "gpu_device_pct": 71}

    def test_a_planted_contention_is_named_with_its_timestamp(self):
        lines = render.monitor_findings([self.PLANTED], self.RUN)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(f"2026-09-22T21:14:03Z {self.RUN} contention "))
        self.assertIn("qwen3.8:27b-mlx", lines[0])
        self.assertIn('"gpu_device_pct":71', lines[0])
        self.assertIn("ended without done; last event contention at 2026-09-22T21:14:03Z", lines[1])

    def test_a_clean_done_is_not_a_contention(self):
        lines = render.monitor_findings([{"event": "done", "t": 1790111643.2}], self.RUN)
        self.assertEqual(lines, [f"2026-09-22T21:14:03Z {self.RUN} done no findings"])

    def test_a_refused_run_is_not_silence(self):
        waits = [{"event": "preflight_wait", "t": 1790111643.2 + i, "problems": ["GPU already 98.0% busy"]}
                 for i in range(3)]
        events = [{"event": "start", "t": 1790111643.2}, *waits]
        lines = render.monitor_findings(events, self.RUN)
        text = "\n".join(lines)
        self.assertIn("preflight_wait count=3", text)
        self.assertIn("first=2026-09-22T21:14:03Z", text)
        self.assertIn("last=2026-09-22T21:14:05Z", text)
        self.assertIn("GPU already 98.0% busy", text)
        self.assertIn("ended without done; last event preflight_wait at 2026-09-22T21:14:05Z", text)

if __name__ == "__main__":
    unittest.main()
