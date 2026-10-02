"""Unit A proof trigger: units, plan, argv-lists-only apply, dry-run planning. br is faked."""

import json
import unittest

from localbench import proofqueue


def _row(feature, **kw):
    routes = kw.get("routes", {})
    local = [p for p, r in routes.items() if r.get("local")]
    return {"feature": feature, "omp_package": "pi-coding-agent", "omp_module": "src/x.ts",
            "omp_module_sha": "a" * 64, "proof_suite": kw.get("proof_suite", "decision"),
            "route_kind": kw.get("route_kind", "setting"), "preset": kw.get("preset", "-"),
            "routes": routes, "local_profiles": local, "proofs": kw.get("proofs", {}),
            "proof": kw.get("proof", "UNPROVEN"), "receipt": kw.get("receipt"),
            "reason": kw.get("reason", "why")}


def _proof(model="m", proof="UNPROVEN", receipt=None, reason="why"):
    return {"model": model, "digest": "d" * 16, "bypass": False, "proof": proof, "receipt": receipt,
            "receipt_sha": None, "reason": reason, "incumbent": {"selector": model, "source": "test"}}


class FakeBr:
    """Recording br stand-in: argv lists in, (rc, stdout, stderr) out."""

    def __init__(self, issues=(), failures=()):
        self.calls = []
        self.issues = list(issues)
        self.failures = set(failures)

    def __call__(self, argv):
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv), argv
        self.calls.append(argv)
        if argv[0] in self.failures:
            return 1, "", "boom"
        if argv[0] == "list":
            return 0, json.dumps({"issues": self.issues}), ""
        if argv[0] == "create":
            return 0, "br-prove-test-123\n", ""
        return 0, "", ""


def _report_stub(rows):
    def fake_report(profiles=None, **kwargs):
        return rows
    return fake_report


def _bad_json_br(argv):
    return 0, "not json", ""

class ProofQueue(unittest.TestCase):
    def test_proof_kind_mapping(self):
        self.assertEqual(proofqueue.proof_kind({"proof_suite": "decision"}), "decision tier")
        self.assertEqual(proofqueue.proof_kind({"proof_suite": "mem"}), "memory study")
        self.assertEqual(proofqueue.proof_kind({"proof_suite": "-", "route_kind": "agent"}), "agent eval")
        self.assertEqual(proofqueue.proof_kind({"proof_suite": "-", "route_kind": "model_role"}), "generation tier")

    def test_units_group_live_profiles_by_model(self):
        row = _row("titles", proof_suite="-", route_kind="model_role",
                   routes={"a": {"local": True, "target": "tiny"}, "b": {"local": True, "target": "tiny"},
                           "c": {"local": True, "target": "smol"}, "d": {"local": False, "target": "x"}},
                   proofs={"a": _proof("tiny", "CARRIED"), "b": _proof("tiny", "BAD", "r.json", "bad"),
                           "c": _proof("smol", "UNPROVEN")})
        got = proofqueue.units([row], {})
        self.assertEqual([(u["route"], sorted(u["profiles"]), u["status"]) for u in got],
                         [("smol", ["c"], "UNPROVEN"), ("tiny", ["a", "b"], "BAD")])

    def test_units_file_candidates_for_routeless_family(self):
        row = _row("auto-thinking", preset="judge", proof="BAD", receipt="r.json", reason="bad compare")
        got = proofqueue.units([row], {"judge": ["judge:nimble", "judge:hosted"]})
        self.assertEqual([(u["route"], u["kind"], u["status"]) for u in got],
                         [("judge:hosted", "candidate", "BAD"), ("judge:nimble", "candidate", "BAD")])

    def test_plan_files_missing_bead(self):
        unit = {"feature": "f", "route": "m", "kind": "live", "profiles": ["p"], "target": "m",
                "model": "m", "digest": None, "status": "STALE", "reason": "old",
                "receipt": None, "row": {"proof_suite": "decision", "route_kind": "setting",
                                         "omp_package": "p", "omp_module": "m", "proofs": {}}}
        actions = proofqueue.plan([unit], [])
        self.assertEqual([(a["op"], a["title"]) for a in actions], [("create", "prove f on m")])
        self.assertIn("--silent", proofqueue.create_argv("t", "b"))

    def test_plan_noop_on_matching_marker(self):
        title, body = proofqueue.render("f", "m", "decision tier", "STALE", "old", {"profiles": ["p"]})
        unit = {"feature": "f", "route": "m", "kind": "live", "profiles": ["p"], "target": "m",
                "model": "m", "digest": None, "status": "STALE", "reason": "old",
                "receipt": None, "row": {"proof_suite": "decision", "route_kind": "setting",
                                         "omp_package": "p", "omp_module": "m", "proofs": {}}}
        actions = proofqueue.plan([unit], [{"id": "x", "title": title, "description": body}])
        self.assertEqual([a["op"] for a in actions], ["noop"])

    def test_plan_comments_and_rewrites_on_status_change(self):
        title, body = proofqueue.render("f", "m", "decision tier", "UNPROVEN", "none yet", {"profiles": ["p"]})
        unit = {"feature": "f", "route": "m", "kind": "live", "profiles": ["p"], "target": "m",
                "model": "m", "digest": None, "status": "STALE", "reason": "sha moved",
                "receipt": None, "row": {"proof_suite": "decision", "route_kind": "setting",
                                         "omp_package": "p", "omp_module": "m", "proofs": {}}}
        actions = proofqueue.plan([unit], [{"id": "x", "title": title, "description": body}])
        self.assertEqual([a["op"] for a in actions], ["comment"])
        fake = FakeBr()
        summary = proofqueue.apply(actions, fake)
        self.assertEqual(summary["updated"], [title])
        update, comments = fake.calls[0], fake.calls[1]
        self.assertEqual(update[:3], ["update", "x", "--description"])
        self.assertIn("proof-trigger status=STALE", update[3])
        self.assertEqual(comments[:3], ["comments", "add", "x"])

    def test_close_cites_receipt_and_carried_leaves_open(self):
        def mkunit(status):
            return {"feature": "f", "route": "m", "kind": "live", "profiles": ["p"],
                    "target": "m", "model": "m", "digest": None, "status": status,
                    "reason": "r", "receipt": "banked.json",
                    "row": {"proof_suite": "decision", "route_kind": "setting",
                            "omp_package": "p", "omp_module": "m", "proofs": {}}}
        title, _ = proofqueue.render("f", "m", "decision tier", "STALE", "old", {"profiles": ["p"]})
        open_bead = {"id": "x", "title": title, "description": "proof-trigger status=STALE | old"}
        proven = proofqueue.plan([mkunit("PROVEN")], [open_bead])
        self.assertEqual([a["op"] for a in proven], ["close"])
        fake = FakeBr()
        proofqueue.apply(proven, fake)
        close = fake.calls[0]
        self.assertEqual(close[:2], ["close", "x"])
        self.assertIn("banked.json", close[close.index("--reason") + 1])
        carried = proofqueue.plan([mkunit("CARRIED")], [open_bead])
        self.assertEqual(carried, [])

    def test_open_bead_without_a_unit_is_ignored(self):
        actions = proofqueue.plan([], [{"id": "x", "title": "prove f on m", "description": ""}])
        self.assertEqual(actions, [])

    def test_apply_records_argv_lists_and_fails_closed(self):
        unit = {"feature": "f", "route": "m", "kind": "live", "profiles": ["p"], "target": "m",
                "model": "m", "digest": None, "status": "UNPROVEN", "reason": "none",
                "receipt": None, "row": {"proof_suite": "decision", "route_kind": "setting",
                                         "omp_package": "p", "omp_module": "m", "proofs": {}}}
        fake = FakeBr()
        summary = proofqueue.apply(proofqueue.plan([unit], []), fake)
        self.assertEqual(summary["filed"], ["prove f on m"])
        create = fake.calls[0]
        self.assertEqual(create[0], "create")
        self.assertIn("--silent", create)
        self.assertEqual(create[create.index("--labels") + 1], "side-model")
        with self.assertRaises(proofqueue.BeadError):
            proofqueue.apply(proofqueue.plan([unit], []), FakeBr(failures={"create"}))

    def test_open_issues_rejects_br_failures(self):
        with self.assertRaises(proofqueue.BeadError):
            proofqueue.open_issues(FakeBr(failures={"list"}))
        with self.assertRaises(proofqueue.BeadError):
            proofqueue.open_issues(_bad_json_br)

    def test_queue_dry_run_plans_without_mutating(self):
        import localbench.features as features_module
        rows = [_row("titles", proof_suite="-", route_kind="model_role",
                     routes={"p": {"local": True, "target": "tiny"}},
                     proofs={"p": _proof("tiny", "UNPROVEN")})]
        real, fake = features_module.report, FakeBr()
        features_module.report = _report_stub(rows)
        try:
            summary = proofqueue.queue(br=fake, dry_run=True)
        finally:
            features_module.report = real
        self.assertEqual(summary["actions"], [("create", "prove titles on tiny")])
        self.assertEqual([c[0] for c in fake.calls], ["list"])

    def test_queue_applies_and_summarizes(self):
        import localbench.features as features_module
        rows = [_row("titles", proof_suite="-", route_kind="model_role",
                     routes={"p": {"local": True, "target": "tiny"}},
                     proofs={"p": _proof("tiny", "UNPROVEN")})]
        real, fake = features_module.report, FakeBr()
        features_module.report = _report_stub(rows)
        try:
            summary = proofqueue.queue(br=fake, dry_run=False)
        finally:
            features_module.report = real
        self.assertEqual(summary["filed"], ["prove titles on tiny"])
        self.assertEqual([c[0] for c in fake.calls], ["list", "create"])

    def test_opted_out_preset_is_skipped_with_its_reason(self):
        import localbench.features as features_module
        rows = [_row("scout-agent", preset="scout", route_kind="agent", proof="UNPROVEN",
                     reason="no receipt", routes={}, proofs={})]
        real, fake = features_module.report, FakeBr()
        features_module.report = _report_stub(rows)
        try:
            summary = proofqueue.queue(br=fake, dry_run=True)
        finally:
            features_module.report = real
        self.assertEqual(summary["actions"], [])
        self.assertEqual(summary["skipped"],
                         [{"preset": "scout:on", "reason": "the owner 2026-10-01: scouts off"}])
        self.assertNotIn("scout:on", [t for _, t in summary["actions"]])

    def _adopt_unit(self, status="STALE", reason="sha moved"):
        return {"feature": "f2", "route": "judge:nimble", "kind": "candidate", "profiles": [],
                "target": "preset judge:nimble", "model": None, "digest": None, "status": status,
                "reason": reason, "receipt": None, "preset": "judge:nimble",
                "row": {"proof_suite": "decision", "route_kind": "setting",
                        "omp_package": "p", "omp_module": "m", "proofs": {}}}

    def test_adopt_attaches_marker_by_comment_without_rewriting(self):
        open_bead = {"id": "k1", "title": "prove f2 on judge:nimble / judge:tev1",
                     "description": "hand-filed with a pre-registration"}
        actions = proofqueue.plan([self._adopt_unit()], [open_bead], FakeBr())
        self.assertEqual([(a["op"], a["id"]) for a in actions], [("adopt", "k1")])
        fake = FakeBr()
        summary = proofqueue.apply(actions, fake)
        self.assertEqual(summary["adopted"], ["prove f2 on judge:nimble / judge:tev1"])
        self.assertEqual([c[0] for c in fake.calls], ["comments"])
        self.assertIn("proof-trigger status=STALE", fake.calls[0][4])
        self.assertIn("left untouched", fake.calls[0][4])

    def test_adopted_marker_comment_is_then_a_noop(self):
        note = ("proof-trigger status=STALE | sha moved\nadopted: this bead already covers "
                "prove f2 on judge:nimble; body left untouched (pre-registration).")

        def comments_br(argv):
            if argv[0] == "comments":
                return 0, json.dumps([{"id": 1, "text": note}]), ""
            return 0, json.dumps({"issues": []}), ""

        open_bead = {"id": "k1", "title": "prove f2 on judge:nimble / judge:tev1",
                     "description": "hand-filed with a pre-registration"}
        actions = proofqueue.plan([self._adopt_unit()], [open_bead], comments_br)
        self.assertEqual([(a["op"], a["title"]) for a in actions],
                         [("noop", "prove f2 on judge:nimble / judge:tev1")])

    def test_proven_route_closes_an_adopted_bead(self):
        open_bead = {"id": "k1", "title": "prove f2 on judge:nimble / judge:tev1",
                     "description": "hand-filed with a pre-registration"}
        unit = self._adopt_unit(status="PROVEN", reason="receipt grades clean")
        unit["receipt"] = "banked.json"
        actions = proofqueue.plan([unit], [open_bead], FakeBr())
        self.assertEqual([(a["op"], a["id"]) for a in actions], [("close", "k1")])
        self.assertIn("banked.json", actions[0]["reason"])

    def test_exact_title_wins_over_prefix_match(self):
        title, body = proofqueue.render("f2", "judge:nimble", "decision tier", "STALE", "new",
                                        {"profiles": []})
        unit = self._adopt_unit()
        open_beads = [{"id": "owned", "title": title, "description": body},
                      {"id": "k1", "title": "prove f2 on judge:nimble / judge:tev1",
                       "description": "hand-filed"}]
        actions = proofqueue.plan([unit], open_beads, FakeBr())
        self.assertEqual([(a["op"], a["id"]) for a in actions], [("comment", "owned")])


if __name__ == "__main__":
    unittest.main()
