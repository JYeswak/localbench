"""Foundation proof specs plus one runner: validation, pre-reg, tiers, banking, beads. No inference."""

import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from localbench import decision, prove


def _answer(noul):
    return {"type": "noul", "noul": 0.9 if noul else 0.1}


QUESTIONS = {"unsafe": {"type": "noul", "instructions": "Is the command destructive?"}}


class FakeBr:
    def __init__(self, issues=()):
        self.calls = []
        self.issues = list(issues)

    def __call__(self, argv):
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv), argv
        self.calls.append(argv)
        if argv[0] == "list":
            return 0, json.dumps({"issues": self.issues}), ""
        return 0, "", ""


class Prove(unittest.TestCase):
    def _git(self, *argv):
        """git in the fixture repo with the hook env scrubbed, mirroring prove._git."""
        env = {key: value for key, value in os.environ.items() if key not in prove._HOOK_ENV}
        subprocess.run(["git", *argv], cwd=self.repo, env=env, capture_output=True, check=True)

    def setUp(self):
        tmp = Path(tempfile.mkdtemp(prefix="prove-"))
        self.tmp = tmp
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        self.repo = tmp / "repo"
        self.repo.mkdir()
        self._git("init", "-b", "main")
        self._git("config", "user.email", "t@t")
        self._git("config", "user.name", "t")
        (self.repo / "registries").mkdir()
        self.source = tmp / "corpus.jsonl"
        self.source.write_text('{"text": "synthetic"}\n')

    def _commit(self, *names):
        self._git("add", *names)
        self._git("commit", "-qm", "[test] prove fixture")

    def _spec(self, name="auto-thinking__screen.json", **over):
        doc = {"feature": "auto-thinking", "kind": "decision",
               "candidates": [{"route": "ollama:nimble:latest"}],
               "dataset": {"suite": "s", "items_sha256": "x" * 64},
               "assertions": [{"id": "a", "type": "metric",
                                "params": {"metric": "m", "min": 0.0}}],
               "stage": "screen"}
        doc.update(over)
        path = self.repo / name
        path.write_text(json.dumps(doc))
        return path

    def _suite(self, labels, ref_picks):
        questions = {"unsafe": copy.deepcopy(QUESTIONS["unsafe"])}
        items = [{"id": f"noul-{i}", "state": f"item {i}", "questions": questions,
                  "labels": {"unsafe": label}, "reference": {"unsafe": _answer(pick)},
                  "source": "synthetic:test_prove"} for i, (label, pick) in enumerate(zip(labels, ref_picks))]
        return decision.write_suite(self.tmp / f"suite-{len(labels)}", name="prove-suite",
                                    role="decision.noul", items=items, sources=[self.source],
                                    gate={"decision.noul.accuracy": {"min": 0.5}})

    def test_load_spec_accepts_and_rejects(self):
        path = self._spec()
        self.assertEqual(prove.load_spec(path)["feature"], "auto-thinking")
        bad = json.loads(path.read_text())
        bad["bogus"] = 1
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(prove.ProveError, "unknown fields"):
            prove.load_spec(path)
        with self.assertRaisesRegex(prove.ProveError, "missing fields"):
            missing = self.repo / "other.json"
            missing.write_text(json.dumps({"feature": "auto-thinking"}))
            prove.load_spec(missing)
        with self.assertRaisesRegex(prove.ProveError, "duplicated assertion id"):
            prove.load_spec(self._spec("dup.json", assertions=[
                {"id": "a", "type": "metric", "params": {}}, {"id": "a", "type": "metric", "params": {}}]))
        with self.assertRaisesRegex(prove.ProveError, "not in registries/features.tsv"):
            prove.load_spec(self._spec("nofeat.json", feature="nope"))
        with self.assertRaisesRegex(prove.ProveError, "memory specs need"):
            prove.load_spec(self._spec("mem.json", kind="memory"))

    def test_spec_slug(self):
        self.assertEqual(prove.spec_slug("x/auto-thinking__screen.json"), "auto-thinking__screen")
        with self.assertRaisesRegex(prove.ProveError, "filename"):
            prove.spec_slug("x/plain.json")

    def test_spec_commit_clean_dirty_and_untracked(self):
        path = self._spec()
        with self.assertRaisesRegex(prove.ProveError, "uncommitted"):
            prove.spec_commit(path, repo=self.repo)
        self._commit(path.name)
        sha = prove.spec_commit(path, repo=self.repo)
        self.assertRegex(sha, "^[0-9a-f]{40}$")
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(prove.ProveError, "modified since its commit"):
            prove.spec_commit(path, repo=self.repo)

    def test_git_calls_ignore_hook_env(self):
        poison = {"GIT_DIR": "/nonexistent-hook-dir", "GIT_INDEX_FILE": "/nonexistent-hook-index",
                  "GIT_WORK_TREE": "/nonexistent-hook-tree",
                  "GIT_OBJECT_DIRECTORY": "/nonexistent-hook-objects",
                  "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/nonexistent-hook-alt",
                  "GIT_PREFIX": "hook-prefix/"}
        previous = {key: os.environ.get(key) for key in poison}
        os.environ.update(poison)
        try:
            path = self._spec()
            with self.assertRaisesRegex(prove.ProveError, "uncommitted"):
                prove.spec_commit(path, repo=self.repo)
            self._commit(path.name)
            self.assertRegex(prove.spec_commit(path, repo=self.repo), "^[0-9a-f]{40}$")
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(prove.ProveError, "modified since its commit"):
                prove.spec_commit(path, repo=self.repo)
        finally:
            for key, value in previous.items():
                if value is None:
                    del os.environ[key]
                else:
                    os.environ[key] = value

    def test_resolve_candidate(self):
        self.assertEqual(prove.resolve_candidate({"route": "ollama:nimble:latest"}),
                         {"backend": "ollama", "model": "nimble:latest", "via": "route"})
        self.assertEqual(prove.resolve_candidate({"route": "laya:o/n"})["backend"], "laya")
        self.assertEqual(prove.resolve_candidate({"preset": "judge:nimble"})["model"], "nimble:latest")
        with self.assertRaisesRegex(prove.ProveError, "ollama: or laya: only"):
            prove.resolve_candidate({"route": "http://x"})
        with self.assertRaisesRegex(prove.ProveError, "candidate needs"):
            prove.resolve_candidate({"model": "x"})

    def test_check_assertion(self):
        table = {"verdict": "BETTER", "diff_pp": 1.0, "mcnemar_p": 0.01, "n": 10}
        ok, _ = prove.check_assertion({"id": "p", "type": "paired",
                                       "params": {"rule": "BETTER", "min_n": 5}}, {"paired": table})
        self.assertTrue(ok)
        ok, detail = prove.check_assertion({"id": "p", "type": "paired",
                                            "params": {"rule": "BETTER", "min_n": 50}}, {"paired": table})
        self.assertFalse(ok)
        self.assertIn("min_n", detail)
        ok, _ = prove.check_assertion({"id": "m", "type": "metric", "params": {"metric": "a", "min": 0.5}},
                                      {"metrics": {"a": 0.9}})
        self.assertTrue(ok)
        ok, _ = prove.check_assertion({"id": "m", "type": "metric", "params": {"metric": "a", "min": 0.5}},
                                      {"metrics": {}})
        self.assertFalse(ok)
        with self.assertRaisesRegex(prove.ProveError, "unknown type"):
            prove.check_assertion({"id": "x", "type": "nope", "params": {}}, {})

    def test_run_decision_candidate_end_to_end(self):
        suite = self._suite([True, False], [True, False])

        def run_suite(spec, candidate, resolved, suite_arg):
            outs = []
            for it in suite_arg.items:
                label = it["labels"]["unsafe"]
                outs.append({"id": it["id"], "repeat": 0, "ok": True,
                             "answers": {"unsafe": _answer(bool(label))}})
            arm = {"outcomes": outs, "metrics": {"decision.noul.accuracy": {"value": 1.0}},
                     "suite": suite_arg.pin()}
            return {"kind": "run", "run": {"decision": {"local": arm}}, "problems": []}

        spec = {"feature": "auto-thinking", "repeats": 1}
        evidence = prove.run_decision_candidate(spec, {"route": "ollama:nimble:latest"}, suite,
                                                run_suite=run_suite, corpora=self.tmp)
        self.assertEqual(evidence["resolved"]["model"], "nimble:latest")
        self.assertEqual(evidence["paired"]["noul"]["verdict"], "NOT_WORSE")
        self.assertEqual(evidence["metrics"], {"decision.noul.accuracy": 1.0})

    def test_run_memory_verdicts_splits_arms(self):
        import localbench.workloads as workloads
        seen = {}
        real = workloads.memory_verdict

        def fake(candidate_legs, baseline_legs, **kw):
            seen.setdefault("calls", []).append((candidate_legs, baseline_legs))
            return {"kind": "run", "candidate": [leg["label"] for leg in candidate_legs]}

        workloads.memory_verdict = fake
        try:
            spec = {"feature": "mnemopi-extraction", "kind": "memory",
                    "verdicts": [{"candidate": "ab_a", "baseline": "ab_b", "bank": "x"}]}
            legs = [{"label": f"ab_a{i}", "id": i} for i in (1, 2)] + [{"label": f"ab_b{i}", "id": i} for i in (1, 2)]
            out = prove.run_memory_verdicts(spec, legs)
        finally:
            workloads.memory_verdict = real
        self.assertEqual([v["candidate"] for v in out], ["ab_a"])
        (cand, base), = seen["calls"]
        self.assertEqual([leg["label"] for leg in cand], ["ab_a1", "ab_a2"])
        self.assertEqual([leg["label"] for leg in base], ["ab_b1", "ab_b2"])
        with self.assertRaisesRegex(prove.ProveError, "needs legs for arms"):
            prove.run_memory_verdicts(spec, [{"label": "ab_a1"}])

    def test_grade_proof_proven_path(self):
        import localbench.features as features
        sha = prove._module_sha("auto-thinking")
        proof = {"kind": "run", "run": {"label": "decision",
                                        "provenance": {"pins": {"model_digest": "d" * 64}}},
                 features.SHA_FIELD: sha, "problems": [],
                 "verdict": {"compare": "BETTER", "baseline": {"kind": "hosted", "id": "typesafe/proj-b-latest"}}}
        inc = {"selector": "typesafe/proj-b-latest", "local": False, "setting": None, "source": "test"}
        self.assertEqual(prove.grade_proof({"feature": "auto-thinking"}, proof, "ollama/nimble:latest",
                                           "d" * 64, inc)[0], "PROVEN")

    def test_unmeasured_win_is_a_problem(self):
        spec = {"wins": [{"metric": "decision.latency.warm_p95_s", "direction": "lower"}],
                "wins_needed": 1}
        evidence = {"metrics": {"decision.latency.warm_p95_s": 1.0}, "local": True}
        self.assertEqual(prove._check_wins(spec, evidence, None),
                         ["win decision.latency.warm_p95_s unmeasured (no hosted arm to compare)",
                          "only 0 measured wins, need 1"])
        hosted = {"metrics": {"decision.latency.warm_p95_s": 2.0},
                  "spreads": {"decision.latency.warm_p95_s": [1.9, 2.1]}}
        apart = {**evidence, "spreads": {"decision.latency.warm_p95_s": [0.9, 1.1]}}
        self.assertEqual(prove._check_wins(spec, apart, hosted), [])
        self.assertEqual(prove._check_wins({"wins": [{"metric": "privacy"}], "wins_needed": 1},
                                           {**evidence, "privacy_verified": True}, None), [])
        self.assertEqual(prove._check_wins({"wins": [{"metric": "privacy"}], "wins_needed": 1},
                                           evidence, None),
                         ["privacy win unverified (incumbent not hosted, or an arm not loopback)",
                          "only 0 measured wins, need 1"])

    def test_metric_win_within_noise_is_a_problem(self):
        spec = {"wins": [{"metric": "decision.latency.warm_p95_s", "direction": "lower"}],
                "wins_needed": 1}
        hosted = {"metrics": {"decision.latency.warm_p95_s": 2.0},
                  "spreads": {"decision.latency.warm_p95_s": [1.9, 2.1]}}
        overlapping = {"metrics": {"decision.latency.warm_p95_s": 1.8},
                       "spreads": {"decision.latency.warm_p95_s": [1.7, 2.0]}}
        problems = prove._check_wins(spec, overlapping, hosted)
        self.assertIn("overlaps hosted [1.9, 2.1]", " ".join(problems))
        self.assertIn("only 0 measured wins, need 1", problems)
        point = {"metrics": {"decision.latency.warm_p95_s": 1.0}}
        problems = prove._check_wins(spec, point, hosted)
        self.assertIn("no repeat spread on candidate arm", " ".join(problems))

    def test_accuracy_win_uses_the_paired_verdict(self):
        spec = {"wins": [{"metric": "unsafe", "direction": "higher"}], "wins_needed": 1}
        hosted = {"metrics": {"unsafe": 0.5}, "spreads": {"unsafe": [0.4, 0.6]}}
        better = {"metrics": {"unsafe": 0.9}, "spreads": {"unsafe": [0.85, 0.95]},
                  "paired": {"unsafe": {"verdict": "BETTER", "diff_pp": 40.0, "mcnemar_p": 0.001}}}
        self.assertEqual(prove._check_wins(spec, better, hosted), [])
        noisy = {**better, "paired": {"unsafe": {"verdict": "WITHIN-NOISE", "diff_pp": 5.0,
                                                 "mcnemar_p": 0.4}}}
        problems = prove._check_wins(spec, noisy, hosted)
        self.assertIn("paired WITHIN-NOISE", " ".join(problems))
        self.assertIn("only 0 measured wins, need 1", problems)

    def test_allow_errors_capped_at_load(self):
        with self.assertRaisesRegex(prove.ProveError, "allow_errors must be a number"):
            self._spec(allow_errors=0.06)
            prove.load_spec(self.repo / "auto-thinking__screen.json")
        ok = self._spec(name="auto-thinking__capped.json", allow_errors=0.05)
        self.assertEqual(prove.load_spec(ok)["allow_errors"], 0.05)

    def test_error_kinds_over_cap_are_problems(self):
        suite = self._suite([True, False], [True, False])
        outcomes = [{"id": "noul-0", "ok": True, "answers": {"unsafe": _answer(True)}},
                    {"id": "noul-1", "ok": True, "answers": {"unsafe": _answer(True)}}]
        kinds, problems = prove._check_error_kinds({"allow_errors": 0.05}, suite, outcomes)
        self.assertEqual(kinds["noul"]["error_rate"], 0.0)
        self.assertEqual(problems, [])
        bad_outcomes = [{"id": "noul-0", "ok": False, "answers": {}},
                        {"id": "noul-1", "ok": True, "answers": {"unsafe": _answer(True)}}]
        kinds, problems = prove._check_error_kinds({"allow_errors": 0.05}, suite, bad_outcomes)
        self.assertEqual(kinds["noul"]["error_rate"], 0.5)
        self.assertEqual(len(problems), 1)
        self.assertIn("exceeds allow_errors 0.05", problems[0])

    def test_judge_family_from_runtime_show(self):
        import io
        import urllib.request
        real = urllib.request.urlopen

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return io.BytesIO(self.payload)

            def __exit__(self, *exc):
                return False

        urllib.request.urlopen = lambda request, timeout=None: FakeResponse(
            b'{"details": {"family": "qwen3"}}')
        try:
            self.assertEqual(prove._judge_family("nimble:latest", "http://127.0.0.1:11434"), "qwen3")
        finally:
            urllib.request.urlopen = real
        urllib.request.urlopen = lambda request, timeout=None: FakeResponse(b'{}')
        try:
            with self.assertRaisesRegex(prove.ProveError, "no family"):
                prove._judge_family("nimble:latest", "http://127.0.0.1:11434")
        finally:
            urllib.request.urlopen = real

    def test_privacy_win_counts_only_when_verified(self):
        spec = {"wins": [{"metric": "privacy"}], "wins_needed": 1}
        bad = prove._check_wins(spec, {}, None)
        self.assertIn("privacy win unverified", " ".join(bad))
        self.assertIn("only 0 measured wins", " ".join(bad))
        self.assertEqual(prove._check_wins(spec, {"privacy_verified": True}, None), [])

    def test_bank_receipt_round_trip_and_name(self):
        out = prove.bank_receipt({"a": 1}, prove.receipt_name("decision", "f", "a/b", "T"), self.tmp / "r")
        self.assertTrue(out.is_file())
        self.assertIn("a_b", out.name)
        self.assertEqual(json.loads(out.read_text())["a"], 1)

    def test_prove_spec_dry_run_plans_without_inference(self):
        path = self._spec()
        self._commit(path.name)
        report = prove.prove_spec(path, dry_run=True, repo=self.repo)
        self.assertTrue(report["dry_run"])
        self.assertTrue(any("nimble" in action for action, _ in report["steps"]))
        self.assertTrue(any("bank" in action for action, _ in report["steps"]))

    def test_prove_spec_refuses_blocked_and_dirty(self):
        path = self._spec("auto-thinking__blocked.json", blocked={"reason": "no positives"})
        self._commit(path.name)
        with self.assertRaisesRegex(prove.ProveError, "blocked"):
            prove.prove_spec(path, repo=self.repo)
        dirty = self._spec()
        with self.assertRaisesRegex(prove.ProveError, "uncommitted"):
            prove.prove_spec(dirty, repo=self.repo)
        self._commit(dirty.name)
        dirty.write_text(dirty.read_text() + " ")
        with self.assertRaisesRegex(prove.ProveError, "modified since its commit"):
            prove.prove_spec(dirty, repo=self.repo)

    def test_prove_spec_generation_and_agent_refuse_live(self):
        path = self._spec("auto-thinking__gen.json", kind="generation")
        self._commit(path.name)
        with self.assertRaisesRegex(prove.ProveError, "%30"):
            prove.prove_spec(path, repo=self.repo)

    def test_prove_spec_decision_end_to_end(self):
        suite = self._suite([True, False], [True, False])

        def run_suite(spec, candidate, resolved, suite_arg):
            outs = [{"id": it["id"], "repeat": 0, "ok": True,
                     "answers": {"unsafe": _answer(bool(it["labels"]["unsafe"]))}} for it in suite_arg.items]
            arm = {"outcomes": outs, "metrics": {}, "suite": suite_arg.pin()}
            return {"kind": "run", "run": {"decision": {"local": arm}}, "problems": []}

        doc = {"feature": "auto-thinking", "kind": "decision",
               "candidates": [{"route": "ollama:nimble:latest"}],
               "dataset": {"suite": str(suite.manifest.parent),
                           "items_sha256": suite.items_sha256},
               "assertions": [{"id": "paired-vs-recorded",
                               "type": "paired",
                               "params": {"rule": ["BETTER", "NOT_WORSE"], "min_n": 1}}],
               "stage": "screen", "wins": [{"metric": "privacy"}], "wins_needed": 1}
        path = self.repo / "auto-thinking__t.json"
        path.write_text(json.dumps(doc))
        self._commit(path.name)
        fake = FakeBr()
        report = prove.prove_spec(path, run_suite_fn=run_suite, receipts_dir=self.tmp / "r",
                                  repo=self.repo, corpora=self.tmp,
                                  grade_fn=lambda receipt, model: ("STALE", "test"),
                                  br=fake)
        cand = report["candidates"][0]
        self.assertEqual(cand["grade"], "STALE")
        self.assertTrue(Path(cand["receipt"]).is_file())
        banked = json.loads(Path(cand["receipt"]).read_text())
        self.assertIn("verified", banked["privacy"])
        self.assertEqual(banked["privacy"]["endpoints"], [])
        self.assertEqual([c[0] for c in fake.calls], ["list"])

    def test_update_beads_and_due(self):
        fake = FakeBr([{"id": "k1", "title": "prove auto-thinking on judge:nimble / judge:tev1",
                        "description": "hand-filed"}])
        summary = prove.update_beads(
            {"feature": "auto-thinking"},
            [{"route": "judge:nimble", "grade": "STALE", "reason": "old", "receipt": "r.json",
              "spec": "s"}], fake)
        self.assertEqual(summary["commented"], ["prove auto-thinking on judge:nimble / judge:tev1"])
        pairs = [("s1", {"feature": "auto-thinking", "candidates": [{"preset": "judge:nimble"}]})]
        self.assertEqual(prove.due_specs(pairs, FakeBr([])), [])
        self.assertEqual(prove.due_specs(pairs, fake), [pairs[0]])
        blocked = [("s2", {"feature": "auto-thinking", "candidates": [{"preset": "judge:nimble"}],
                              "blocked": {"reason": "x"}})]
        self.assertEqual(prove.due_specs(blocked, fake), [])


    def _judge_ctx(self, thinking="auto", judge="typesafe/proj-b-latest", base="https://api.typesafe.ai"):
        cfg = {"modelRoles": {"judge": judge, "tiny": "localbench/tiny", "smol": "localbench/smol"},
               "defaultThinkingLevel": thinking}
        provs = {"typesafe": {"baseUrl": base, "api": "typesafe"},
                 "ollama": {"baseUrl": "http://127.0.0.1:11434", "api": "openai-completions"},
                 "localbench": {"baseUrl": "http://127.0.0.1:11300/v1", "api": "openai-completions"},
                 "localbench-sys1": {"baseUrl": "http://127.0.0.1:11300", "api": "typesafe"}}
        return cfg, provs

    def test_privacy_accepted_against_hosted_with_loopback_arms(self):
        cfg, provs = self._judge_ctx()
        ok, reason, evidence = prove._verify_privacy(
            "auto-thinking", ["http://127.0.0.1:11434/v1"], "default", cfg=cfg, provs=provs)
        self.assertTrue(ok, reason)
        self.assertEqual(evidence["incumbent"], "typesafe/proj-b-latest")
        self.assertEqual(evidence["endpoints"], ["http://127.0.0.1:11434/v1"])

    def test_privacy_refused_against_fixed_and_local_incumbents(self):
        cfg, provs = self._judge_ctx(thinking="high")
        ok, reason, _ = prove._verify_privacy("auto-thinking", ["http://127.0.0.1:11434/v1"],
                                               "default", cfg=cfg, provs=provs)
        self.assertFalse(ok)
        self.assertIn("not hosted", reason)
        cfg, provs = self._judge_ctx(judge="ollama/nimble:latest")
        ok, reason, _ = prove._verify_privacy("auto-thinking", ["http://127.0.0.1:11434/v1"],
                                               "default", cfg=cfg, provs=provs)
        self.assertFalse(ok)
        self.assertIn("not hosted", reason)

    def test_privacy_refused_for_remote_arm(self):
        cfg, provs = self._judge_ctx()
        ok, reason, _ = prove._verify_privacy("auto-thinking", ["https://example.invalid/v1"],
                                               "default", cfg=cfg, provs=provs)
        self.assertFalse(ok)
        self.assertIn("loopback", reason)

    def test_installed_digest_resolves_laya_snapshot(self):
        import localbench.features as features
        hub = self.tmp / "hub"
        snaps = hub / "models--o--n" / "snapshots" / ("a" * 40)
        snaps.mkdir(parents=True)
        previous = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = str(hub)
        try:
            self.assertEqual(features.installed_digest({}, "laya:o/n", {}), "a" * 40)
            (snaps.parent / ("b" * 40)).mkdir()
            self.assertIsNone(features.installed_digest({}, "laya:o/n", {}))
        finally:
            if previous is None:
                del os.environ["HF_HUB_CACHE"]
            else:
                os.environ["HF_HUB_CACHE"] = previous
        self.assertIsNone(features.installed_digest({}, "laya:o/n", {}))

if __name__ == "__main__":
    unittest.main()
