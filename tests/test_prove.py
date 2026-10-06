"""Foundation proof specs plus one runner: validation, pre-reg, tiers, banking, beads. No inference."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class FakeProofOllama:
    def __init__(self, resident: dict[str, int], shipped_contexts: dict[str, int] | None = None) -> None:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path != "/api/ps":
                    self._send(404, {"error": "not found"})
                    return
                with fake.lock:
                    rows = [{"name": name, "context_length": context}
                            for name, context in fake.resident.items()]
                self._send(200, {"models": rows})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    if self.path == "/api/show":
                        context = fake.shipped_contexts.get(body["model"])
                        parameters = f"num_ctx {context}" if context is not None else ""
                        status, payload = 200, {"parameters": parameters}
                    elif self.path == "/api/generate" and body.get("keep_alive") == 0:
                        fake.unloaded.append(body["model"])
                        fake.resident.pop(body["model"], None)
                        status, payload = 200, {"done": True}
                    elif self.path == "/api/generate":
                        fake.loaded.append(body["model"])
                        fake.resident[body["model"]] = body["options"]["num_ctx"]
                        status, payload = 200, {"done": True}
                    else:
                        status, payload = 404, {"error": "not found"}
                self._send(status, payload)

        self.resident = dict(resident)
        self.shipped_contexts = dict(shipped_contexts or {})
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def __enter__(self) -> "FakeProofOllama":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ProofPlanSteps(unittest.TestCase):
    def test_memory_dry_run_names_mem_sess_arms_and_run_design(self):
        spec = {"feature": "mnemopi-extraction", "kind": "memory", "stage": "proof",
                "dataset": {"suite": "mem-suite", "items_sha256": "a" * 64},
                "candidates": [{"route": "ollama:qwen3.6:35b-mlx"}],
                "arms": {"ab_a": {}, "ab_b": {}}, "tiers": ["mem", "sess"],
                "pairs": 12, "mem_rounds": 4, "repeats": 2}
        steps = prove.plan_steps(spec, "a" * 40, omp_frozen=True)
        action = next(action for action, _ in steps if action.startswith("launch mem/sess legs"))
        self.assertIn("ab_a", action)
        self.assertIn("ab_b", action)
        self.assertIn("25 interleaved A/B runs (12 pairs plus final A)", action)
        self.assertIn("4 rounds, 2 repeats", action)

    def test_generation_dry_run_names_output_producer(self):
        spec = {"feature": "skill-description-compression", "kind": "generation", "stage": "screen",
                "dataset": {"suite": "skilldesc-aux", "items_sha256": "b" * 64},
                "candidates": [{"route": "ollama:qwen3.8:27b-mlx"}]}
        steps = prove.plan_steps(spec, "b" * 40)
        action = next(action for action, _ in steps if action.startswith("run generation candidate"))
        reason = next(reason for step, reason in steps if step == action)
        self.assertIn("outputs_fn (replay_generation_outputs)", action)
        self.assertIn("per-candidate corpus outputs", reason)


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

    def test_generation_spec_pins_gold_set_and_refuses_invalid_references(self):
        spec = self._spec(
            "skill-description-compression__screen.json",
            feature="skill-description-compression", kind="generation",
            assertions=[{"id": "shape", "type": "deterministic",
                         "params": {"check": "check_skill_compression"}}],
            gold_set={"file": "gold-set.json", "sha256": "a" * 64})
        self.assertEqual(prove.load_spec(spec)["gold_set"]["file"], "gold-set.json")
        for reference in ({"file": "../gold-set.json", "sha256": "a" * 64},
                          {"file": "gold-set.json", "sha256": "short"}):
            with self.subTest(reference=reference):
                bad = json.loads(spec.read_text())
                bad["gold_set"] = reference
                spec.write_text(json.dumps(bad))
                with self.assertRaisesRegex(prove.ProveError, "gold_set"):
                    prove.load_spec(spec)

    def test_gold_set_file_is_pinned_and_has_blind_label_agreement(self):
        path = self.tmp / "gold-set.json"
        pair_ids = [f"p{i}" for i in range(10)]
        labels = {
            item_id: {"label": "candidate" if index < 5 else "tie", "source": "human",
                      "votes": ["candidate" if index < 5 else "tie"] * 2}
            for index, item_id in enumerate(pair_ids)}
        doc = {"version": 1, "pair_ids": pair_ids, "order_seed": "blind-seed",
               "labels": labels, "labeler": ["reviewer-a", "reviewer-b"],
               "kappa": 1.0, "agreement": 1.0}
        raw = json.dumps(doc, separators=(",", ":")).encode()
        path.write_bytes(raw)
        spec = {"dataset": {"suite": "screen"},
                "gold_set": {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}}
        result = prove.load_gold_set(spec, corpus_root=self.tmp)
        self.assertEqual(result["pair_ids"], pair_ids)
        self.assertEqual(result["labels"]["p0"]["label"], "candidate")
        self.assertEqual(result["kappa"], 1.0)
        bad = dict(doc, kappa=0.92)
        path.write_text(json.dumps(bad))
        spec["gold_set"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(prove.ProveError, "does not match"):
            prove.load_gold_set(spec, corpus_root=self.tmp)
        bad = dict(doc, agreement=0.89)
        path.write_text(json.dumps(bad))
        spec["gold_set"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(prove.ProveError, "90%"):
            prove.load_gold_set(spec, corpus_root=self.tmp)
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
        self.assertEqual(prove.resolve_candidate({"route": "mlx-serve:/models/x"}),
                         {"backend": "mlx-serve", "model": "/models/x", "via": "route"})
        self.assertEqual(prove.resolve_candidate({"preset": "judge:nimble"})["model"], "nimble:latest")
        with self.assertRaisesRegex(prove.ProveError, "mlx-serve"):
            prove.resolve_candidate({"route": "http://x"})
        with self.assertRaisesRegex(prove.ProveError, "candidate needs"):
            prove.resolve_candidate({"model": "x"})
        self.assertEqual(prove.resolve_candidate({"builtin": "previewSkillDescription"}),
                         {"backend": "builtin", "model": "previewSkillDescription", "via": "builtin"})

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

    def test_candidate_evidence_end_to_end(self):
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

        spec = {"feature": "auto-thinking", "repeats": 1, "stage": "screen", "assertions": []}
        evidence = prove.candidate_evidence(spec, {"route": "ollama:nimble:latest"}, suite,
                                                run_suite=run_suite, corpora=self.tmp)
        self.assertEqual(evidence["resolved"]["model"], "nimble:latest")
        self.assertEqual(evidence["paired"]["noul"]["verdict"], "NOT_WORSE")
        self.assertEqual(evidence["metrics"], {"decision.noul.accuracy": 1.0})

    def test_failed_paired_or_metric_gate_rejects_when_musts_pass(self):
        from unittest import mock

        suite = self._suite([True, False] * 20, [True, False] * 20)

        def run_suite(_spec, _candidate, _resolved, suite_arg):
            outcomes = [{"id": item["id"], "repeat": 0, "ok": True,
                         "answers": {"unsafe": _answer(not bool(item["labels"]["unsafe"]))}}
                        for item in suite_arg.items]
            arm = {"outcomes": outcomes, "metrics": {"decision.noul.accuracy": {"value": 0.0}},
                   "suite": suite_arg.pin()}
            return {"kind": "run",
                    "run": {"decision": {"local": arm},
                            "conformance": {"fixture": {"level": "MUST", "verdict": "PASS"}}},
                    "problems": []}

        assertions = (
            {"id": "paired", "type": "paired", "params": {"rule": "NOT_WORSE"}},
            {"id": "accuracy", "type": "metric",
             "params": {"metric": "decision.noul.accuracy", "min": 1.0}},
        )
        for assertion in assertions:
            with self.subTest(assertion=assertion["type"]):
                path = self._spec(
                    name=f"auto-thinking__screen-{assertion['type']}.json",
                    candidates=[{"route": "laya:org/model"}],
                    dataset={"suite": suite.name, "items_sha256": suite.items_sha256},
                    assertions=[assertion], wins=[{"metric": "privacy"}], wins_needed=1)
                self._commit(path.name)
                fake = FakeBr([{"id": "kit-auto",
                                "title": "prove auto-thinking on laya:org/model", "status": "open"}])
                with (mock.patch.object(decision, "resolve_suite", return_value=suite),
                      mock.patch.object(prove, "_verify_privacy", return_value=(True, "test", {}))):
                    report = prove.prove_spec(
                        path, run_suite_fn=run_suite, receipts_dir=self.tmp / f"receipts-{assertion['type']}",
                        repo=self.repo, corpora=self.tmp,
                        grade_fn=lambda _receipt, _model: ("PROVEN", "test"), br=fake)
                entry = report["candidates"][0]
                receipt = json.loads(Path(entry["receipt"]).read_text())
                self.assertEqual(receipt["run"]["conformance"]["fixture"]["verdict"], "PASS")
                self.assertEqual(entry["screen_verdict"], "REJECT")
                self.assertTrue(entry["problems"])

    def test_screen_verdict_is_reported_and_comments_instead_of_closing(self):
        from unittest import mock

        suite = self._suite([True, False], [True, False])
        path = self._spec(
            candidates=[{"route": "laya:org/model"}],
            dataset={"suite": suite.name, "items_sha256": suite.items_sha256},
            assertions=[{"id": "accuracy", "type": "metric",
                         "params": {"metric": "decision.noul.accuracy", "min": 0.0}}],
            wins=[{"metric": "privacy"}], wins_needed=1)
        self._commit(path.name)

        def run_suite(_spec, _candidate, _resolved, suite_arg):
            outcomes = [{"id": item["id"], "repeat": 0, "ok": True,
                         "answers": {"unsafe": _answer(bool(item["labels"]["unsafe"]))}}
                        for item in suite_arg.items]
            arm = {"outcomes": outcomes, "metrics": {"decision.noul.accuracy": {"value": 1.0}},
                   "suite": suite_arg.pin()}
            return {"kind": "run", "run": {"decision": {"local": arm}}, "problems": []}

        fake = FakeBr([{"id": "kit-auto", "title": "prove auto-thinking on laya:org/model", "status": "open"}])
        with (mock.patch.object(decision, "resolve_suite", return_value=suite),
              mock.patch.object(prove, "_verify_privacy", return_value=(True, "test", {}))):
            report = prove.prove_spec(
                path, run_suite_fn=run_suite, receipts_dir=self.tmp / "receipts",
                repo=self.repo, corpora=self.tmp,
                grade_fn=lambda _receipt, _model: ("PROVEN", "test"), br=fake)

        self.assertEqual(report["candidates"][0]["screen_verdict"], "ADVANCE")
        self.assertEqual(
            prove.screen_verdict({}, problems=[], error_kinds={}, allow_errors=0.0), "ADVANCE")
        comments = [call for call in fake.calls if call[:2] == ["comments", "add"]]
        self.assertEqual(len(comments), 1)
        self.assertIn("screen_verdict=ADVANCE", comments[0][-1])
        self.assertFalse(any(call[0] == "close" for call in fake.calls))

    def test_run_memory_verdicts_splits_arms(self):
        import localbench.workloads as workloads
        seen = {}
        real = workloads.memory_verdict
        real_sha, prove._module_sha = prove._module_sha, lambda feature: "0" * 64

        def fake(candidate_legs, baseline_legs, **kw):
            seen.setdefault("calls", []).append((candidate_legs, baseline_legs))
            return {"kind": "run", "candidate": [leg["label"] for leg in candidate_legs]}

        workloads.memory_verdict = fake
        try:
            spec = {"feature": "mnemopi-extraction", "kind": "memory",
                    "verdicts": [{"candidate": "ab_a", "baseline": "ab_b", "bank": "x"}]}
            legs = [{"label": f"ab_a{i}", "id": i} for i in (1, 2)] + [{"label": f"ab_b{i}", "id": i} for i in (1, 2)]
            out = prove.run_memory_verdicts(spec, legs)
            with self.assertRaisesRegex(prove.ProveError, "needs legs for arms"):
                prove.run_memory_verdicts(spec, [{"label": "ab_a1"}])
        finally:
            workloads.memory_verdict = real
            prove._module_sha = real_sha
        self.assertEqual([v["candidate"] for v in out], ["ab_a"])
        (cand, base), = seen["calls"]
        self.assertEqual([leg["label"] for leg in cand], ["ab_a1", "ab_a2"])
        self.assertEqual([leg["label"] for leg in base], ["ab_b1", "ab_b2"])

    def test_unknown_assertion_is_a_load_error(self):
        with self.assertRaisesRegex(prove.ProveError, "has no evaluator"):
            prove.load_spec(self._spec(assertions=[{"id": "x", "type": "nope"}]))
        mem = {"kind": "memory", "arms": {"ab_a": {}, "ab_b": {}},
               "verdicts": [{"candidate": "ab_a", "baseline": "ab_b"}]}
        with self.assertRaisesRegex(prove.ProveError, "unknown memory metric"):
            prove.load_spec(self._spec(**mem, assertions=[{"id": "m", "type": "metric",
                "params": {"metric": "nope", "test": "fisher-pooled", "direction": "not-worse"}}]))
        with self.assertRaisesRegex(prove.ProveError, "unknown memory test"):
            prove.load_spec(self._spec(**mem, assertions=[{"id": "m", "type": "metric",
                "params": {"metric": "mem.recall.hit_rate", "test": "bogus", "direction": "not-worse"}}]))
        gen = {"kind": "generation", "dataset": {"suite": "s", "items_sha256": "x" * 64}}
        with self.assertRaisesRegex(prove.ProveError, "has no evaluator"):
            prove.load_spec(self._spec(**gen, assertions=[{"id": "x", "type": "deterministic",
                "params": {"check": "check_nope"}}]))

    def test_memory_metric_assertions_read_verdict_rows(self):
        def assertion(direction, test="fisher-pooled", metric="mem.recall.hit_rate", margin=0.1):
            return {"id": "r", "type": "metric",
                    "params": {"metric": metric, "test": test, "direction": direction,
                               "margin": margin, "alpha": 0.05}}

        def row(judgement, delta=0.0, method="pooled-fisher", cls="quality"):
            return {"class": cls, "judgement": judgement, "delta": delta,
                    "test": {"method": method, "p": 0.2}}

        ok, _ = prove.evaluate_memory_metric(assertion("not-worse"), row("within_noise", -0.02))
        self.assertTrue(ok)
        bad, detail = prove.evaluate_memory_metric(assertion("not-worse"), row("loss", -0.2))
        self.assertFalse(bad)
        self.assertIn("margin", detail)
        ok, _ = prove.evaluate_memory_metric(
            assertion("lower", test="unpaired-t-log", metric="sess.turn.post_retain_wall_s", margin=0),
            row("gain", -0.5, method="welch-t-log", cls="win"))
        self.assertTrue(ok)
        bad, _ = prove.evaluate_memory_metric(
            assertion("lower", test="unpaired-t-log", metric="sess.turn.post_retain_wall_s", margin=0),
            row("within_noise", -0.1, method="welch-t-log", cls="win"))
        self.assertFalse(bad)
        bad, detail = prove.evaluate_memory_metric(assertion("not-worse"),
                                                   row("within_noise", 0.0, cls="win"))
        self.assertIn("needs a quality row", detail)
        bad, _ = prove.evaluate_memory_metric(assertion("not-worse"), {"class": "quality",
                                                                        "judgement": "unmeasured"})
        self.assertFalse(bad)
        bad, _ = prove.evaluate_memory_metric(assertion("not-worse"), None)
        self.assertFalse(bad)
        with self.assertRaisesRegex(prove.ProveError, "unknown memory direction"):
            prove.evaluate_memory_metric(assertion("sideways"), row("within_noise"))

    def test_sized_design_counts_legs_and_facts(self):
        def leg(n, m=10):
            return {"metrics": {"mem.recall.hit_rate": {"value": 0.9, "n": n},
                                "mem.derail.ok_rate": {"value": 0.9, "n": m}}}

        assertion = {"id": "sized-design", "type": "deterministic",
                     "params": {"min_pairs": 2, "min_recall_facts_b": 20, "min_derail_facts_b": 20}}
        ok, _ = prove.evaluate_sized_design(assertion, [leg(10), leg(10)], [leg(10), leg(10)])
        self.assertTrue(ok)
        bad, detail = prove.evaluate_sized_design(assertion, [leg(10)], [leg(10), leg(10)])
        self.assertIn("min_pairs", detail)
        bad, detail = prove.evaluate_sized_design(assertion, [leg(10), leg(10)], [leg(5), leg(5)])
        self.assertIn("min_recall_facts_b", detail)
        bad, detail = prove.evaluate_sized_design(assertion, [leg(10), leg(10)],
                                                  [{"metrics": {"mem.recall.hit_rate": {"value": 0.9}}},
                                                   leg(10)])
        self.assertIn("unrecorded", detail)

    def test_memory_entry_filters_by_candidate(self):
        row = {"class": "quality", "judgement": "within_noise", "delta": 0.0,
               "test": {"method": "pooled-fisher", "p": 0.5}}
        spec = {"assertions": [
            {"id": "q", "type": "metric",
             "params": {"metric": "mem.recall.hit_rate", "test": "fisher-pooled", "direction": "not-worse"}},
            {"id": "e", "type": "metric", "candidate": "ab_b",
             "params": {"metric": "mem.recall.hit_rate", "test": "fisher-pooled", "direction": "not-worse"}}]}
        verdict = {"candidate": "ab_a", "baseline": "ab_b",
                   "receipt": {"run": {"memory": {"compare": {"deltas": {"mem.recall.hit_rate": row}}}}}}
        problems, passed = prove.evaluate_memory_entry(spec, verdict, {"ab_a": [], "ab_b": []})
        self.assertEqual(len(passed), 1)
        self.assertEqual(problems, [])

    def test_generation_check_assertions_evaluate_outputs(self):
        description = "cobalt fixture marker session title generator output"
        good = {"item_id": "a", "text": "cobalt fixture marker session",
                "context": {"description": description}, "answerable": True}
        bad = {"item_id": "b", "text": "first line\nsecond line with far too many words to ever fit",
               "context": {"description": description}, "answerable": True}
        spec = {"feature": "skill-description-compression", "kind": "generation", "candidates": [
            {"route": "ollama:qwen3.6:35b-mlx"}, {"builtin": "previewSkillDescription"}]}
        shape = {"id": "compression-shape", "type": "deterministic",
                 "params": {"check": "check_skill_compression"}}
        problems, passed, rows = prove.evaluate_generation_assertions(
            {**spec, "assertions": [shape]}, "ollama:qwen3.6:35b-mlx", [good, good],
            {"ollama:qwen3.6:35b-mlx": [good, good]})
        self.assertEqual(problems, [])
        self.assertEqual(len(rows["compression-shape"]), 2)
        self.assertEqual(passed, ["compression-shape: all 2 answerable outputs pass"])
        problems, _, _ = prove.evaluate_generation_assertions(
            {**spec, "assertions": [shape]}, "ollama:qwen3.6:35b-mlx", [good, bad],
            {"ollama:qwen3.6:35b-mlx": [good, bad]})
        self.assertEqual(len(problems), 1)
        self.assertIn("b", problems[0])
        rate = {"id": "compression-pass-rate", "type": "deterministic",
                "params": {"check": "check_skill_compression", "min_pass_rate": 0.8}}
        problems, passed, _ = prove.evaluate_generation_assertions(
            {**spec, "assertions": [rate]}, "ollama:qwen3.6:35b-mlx",
            [good, good, good, good, {**bad, "item_id": "e"}],
            {"ollama:qwen3.6:35b-mlx": [good]})
        self.assertEqual(problems, [])
        self.assertIn("0.800", passed[0])
        problems, _, _ = prove.evaluate_generation_assertions(
            {**spec, "assertions": [dict(rate, params={**rate["params"], "min_pass_rate": 0.9})]},
            "ollama:qwen3.6:35b-mlx", [good, {**bad, "item_id": "e"}],
            {"ollama:qwen3.6:35b-mlx": [good]})
        self.assertEqual(len(problems), 1)
        beats = {"id": "beats-builtin", "type": "deterministic",
                 "params": {"baseline": "builtin", "compare": "violation_rate", "direction": "lower"}}
        everything = {"ollama:qwen3.6:35b-mlx": [good, good],
                      "builtin:previewSkillDescription": [bad, bad]}
        problems, passed, rows = prove.evaluate_generation_assertions(
            {**spec, "assertions": [beats]}, "ollama:qwen3.6:35b-mlx", [good, good], everything)
        self.assertEqual(problems, [])
        self.assertIn("0.000 < builtin 1.000", passed[0])
        tied = {"ollama:qwen3.6:35b-mlx": [good, bad], "builtin:previewSkillDescription": [good, bad]}
        problems, _, _ = prove.evaluate_generation_assertions(
            {**spec, "assertions": [beats]}, "ollama:qwen3.6:35b-mlx", [good, bad], tied)
        self.assertEqual(len(problems), 1)
        with self.assertRaisesRegex(prove.ProveError, "no builtin candidate"):
            prove.evaluate_generation_assertions(
                {"feature": "skill-description-compression", "kind": "generation",
                 "candidates": [{"route": "ollama:qwen3.6:35b-mlx"}], "assertions": [beats]},
                "ollama:qwen3.6:35b-mlx", [good], {"ollama:qwen3.6:35b-mlx": [good]})
    def test_generation_evidence_requires_calibrated_pairwise_receipt_fields(self):
        key = "ollama:qwen3.6:35b-mlx"
        raw = {key: {"outcomes": [], "raw_win_rate": 0.8}}
        with self.assertRaisesRegex(prove.ProveError, "raw win rate"):
            prove.normalize_generation_outputs(raw)
        paired = {key: {"outcomes": [], "judge_pairs": [
            {"item_id": "a", "candidate_text": "short", "baseline_text": "short",
             "winner": "candidate", "swap_winner": "candidate"},
            {"item_id": "b", "candidate_text": "short", "baseline_text": "short",
             "winner": "baseline", "swap_winner": "baseline"}]}}
        outcomes, calibration, backend_pins = prove.normalize_generation_outputs(paired)
        self.assertEqual(outcomes[key], [])
        self.assertIn("length_adjusted_win_rate", calibration[key])
        self.assertIn("verbosity", calibration[key])
        self.assertIn("swap_consistency", calibration[key])
        pinned = {key: {"outcomes": [], "backend_pins": {
            "backend": "mlx-serve", "model": "served-model", "model_dir": "/models/mock",
            "model_digest": "files:abc", "backend_version": "1.0", "backend_sha": "a" * 16}}}
        _, _, pins = prove.normalize_generation_outputs(pinned)
        self.assertEqual(pins[key]["backend"], "mlx-serve")

    def test_prove_spec_memory_evaluates_assertions(self):
        import localbench.workloads as workloads
        real = workloads.memory_verdict
        row = {"class": "quality", "judgement": "within_noise", "delta": 0.0,
               "test": {"method": "pooled-fisher", "p": 0.5}}

        def fake_verdict(candidate_legs, baseline_legs, **kw):
            return {"kind": "run", "run": {"memory": {"compare": {"deltas": {
                "mem.recall.hit_rate": dict(row), "mem.derail.ok_rate": dict(row)}}}}}

        def leg(n, m=70):
            return {"metrics": {"mem.recall.hit_rate": {"value": 0.9, "n": n},
                                "mem.derail.ok_rate": {"value": 0.9, "n": m}}}

        workloads.memory_verdict = fake_verdict
        real_sha, prove._module_sha = prove._module_sha, lambda feature: "0" * 64
        try:
            doc = {"feature": "mnemopi-extraction", "kind": "memory",
                   "candidates": [{"arm": "ab_a"}],
                   "dataset": {"suite": "s", "items_sha256": "x" * 64},
                   "arms": {"ab_a": {}, "ab_b": {}},
                   "verdicts": [{"candidate": "ab_a", "baseline": "ab_b"}],
                   "assertions": [
                       {"id": "recall-no-loss", "type": "metric",
                        "params": {"metric": "mem.recall.hit_rate", "test": "fisher-pooled",
                                   "direction": "not-worse", "margin": 0.1, "alpha": 0.05}},
                       {"id": "sized-design", "type": "deterministic",
                        "params": {"min_pairs": 2, "min_recall_facts_b": 72,
                                   "min_derail_facts_b": 120}}],
                   "stage": "screen"}
            path = self.repo / "mnemopi-extraction__t.json"
            path.write_text(json.dumps(doc))
            self._commit(path.name)
            legs = ([{**leg(40), "label": "ab_a1"}, {**leg(40), "label": "ab_a2"}]
                    + [{**leg(40), "label": "ab_b1"}, {**leg(40), "label": "ab_b2"}])
            fake = FakeBr()
            report = prove.prove_spec(path, legs_fn=lambda spec: legs, receipts_dir=self.tmp / "r",
                                      repo=self.repo,
                                      grade_fn=lambda receipt, model: ("STALE", "test"), br=fake)
        finally:
            workloads.memory_verdict = real
            prove._module_sha = real_sha
        (entry,) = report["candidates"]
        self.assertEqual(entry["passed"], ["recall-no-loss: no loss (delta +0.000, within_noise)",
                                           "sized-design: sized (2+2 legs)"])
        self.assertEqual(entry["problems"], [])
        self.assertTrue(Path(entry["receipt"]).is_file())

    def test_prove_spec_generation_evaluates_outputs(self):
        import localbench.decision as decision
        description = "cobalt fixture marker session title generator output"
        good = {"item_id": "a", "text": "cobalt fixture marker session",
                "context": {"description": description}, "answerable": True}
        items = [{"id": "a", "feature": "skill-description-compression", "body": {}, "source": "t"}]
        root = self.tmp / "gen" / "generation" / "t"
        root.mkdir(parents=True)
        (root / "items.jsonl").write_text(json.dumps(items[0]) + "\n")
        sha = hashlib.sha256((root / "items.jsonl").read_bytes()).hexdigest()
        (root / "manifest.json").write_text(json.dumps({"items_sha256": sha}))
        gold = {"version": 1, "pair_ids": ["a"], "order_seed": "blind-seed",
                "labels": {"a": {"label": "candidate", "source": "human",
                                 "votes": ["candidate", "candidate"]}},
                "labeler": ["reviewer-a", "reviewer-b"], "kappa": 1.0, "agreement": 1.0}
        gold_raw = json.dumps(gold, separators=(",", ":")).encode()
        (root / "gold-set.json").write_bytes(gold_raw)
        model_dir = "/models/mock"
        doc = {"feature": "skill-description-compression", "kind": "generation",
               "candidates": [{"route": f"mlx-serve:{model_dir}"}],
               "dataset": {"suite": "t", "items_sha256": sha},
               "gold_set": {"file": "gold-set.json",
                            "sha256": hashlib.sha256(gold_raw).hexdigest()},
               "assertions": [{"id": "compression-shape", "type": "deterministic",
                               "params": {"check": "check_skill_compression"}}],
               "stage": "screen"}
        path = self.repo / "skill-description-compression__t.json"
        path.write_text(json.dumps(doc))
        self._commit(path.name)
        real_corpora, fake = decision.CORPORA, FakeBr()
        decision.CORPORA = self.tmp / "gen"
        try:
            from unittest.mock import patch
            with patch.object(prove, "_module_sha", return_value="module-sha"):
                report = prove.prove_spec(
                    path, outputs_fn=lambda spec, items: {
                        f"mlx-serve:{model_dir}": {
                            "outcomes": [good],
                            "judge_pairs": [{"item_id": "a", "candidate_text": "good answer",
                                             "baseline_text": "also good", "winner": "candidate",
                                             "swap_winner": "candidate"}],
                            "backend_pins": {"backend": "mlx-serve", "model": "served-model",
                                             "model_dir": model_dir, "model_digest": "files:abc",
                                             "backend_version": "1.0", "backend_sha": "a" * 16}}},
                    receipts_dir=self.tmp / "r", repo=self.repo,
                    grade_fn=lambda receipt, model: ("STALE", "test"), br=fake)
        finally:
            decision.CORPORA = real_corpora
        (entry,) = report["candidates"]
        self.assertEqual(entry["passed"], ["compression-shape: all 1 answerable outputs pass"])
        self.assertTrue(Path(entry["receipt"]).is_file())
        receipt = json.loads(Path(entry["receipt"]).read_text())
        from localbench.__main__ import validate_doc
        self.assertEqual(validate_doc(receipt), [])
        self.assertEqual(receipt["omp_module_sha"], "module-sha")
        self.assertEqual(receipt["run"]["provenance"]["pins"]["model_digest"], "files:abc")
        calibration = receipt["run"]["judge_calibration"]
        self.assertIn("length_adjusted_win_rate", calibration)
        self.assertIn("swap_consistency", calibration)
        self.assertEqual(receipt["run"]["gold_set"]["kappa"], 1.0)
        self.assertEqual(receipt["run"]["backend_pins"]["backend"], "mlx-serve")

    def test_generation_default_grade_requires_complete_proof_contract(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        import localbench.features as features

        description = "cobalt fixture marker session title generator output"
        good = {"item_id": "a", "text": "cobalt fixture marker session",
                "context": {"description": description}, "answerable": True}
        route = "ollama:qwen3.6:35b-mlx"
        model = "qwen3.6:35b-mlx"
        cfg = {"modelRoles": {"smol": "ollama/qwen3.8:27b-mlx"}}
        provs = {"ollama": {"baseUrl": "http://127.0.0.1:11434", "api": "ollama"}}
        host = {"macos_build": "test-build", "host_id": "test-host"}
        real_corpora = decision.CORPORA
        decision.CORPORA = self.tmp / "gen"

        def run_case(label, *, digest="b" * 64, output=good, judged=True,
                     sealed=True, wrong_baseline=False):
            root = self.tmp / "gen" / "generation" / label
            root.mkdir(parents=True)
            item = {"id": "a", "feature": "skill-description-compression", "body": {}, "source": label}
            (root / "items.jsonl").write_text(json.dumps(item) + "\n")
            items_sha = hashlib.sha256((root / "items.jsonl").read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps({"items_sha256": items_sha}))
            gold = {"version": 1, "pair_ids": ["a"], "order_seed": "blind-seed",
                    "labels": {"a": {"label": "candidate", "source": "human",
                                     "votes": ["candidate", "candidate"]}},
                    "labeler": ["reviewer-a", "reviewer-b"], "kappa": 1.0, "agreement": 1.0}
            gold_raw = json.dumps(gold, separators=(",", ":")).encode()
            (root / "gold-set.json").write_bytes(gold_raw)
            gold_sha = hashlib.sha256(gold_raw).hexdigest()
            spec = {"feature": "skill-description-compression", "kind": "generation",
                    "candidates": [{"route": route}], "dataset": {"suite": label, "items_sha256": items_sha},
                    "assertions": [{"id": "compression-shape", "type": "deterministic",
                                    "params": {"check": "check_skill_compression"}}],
                    "analysis_sha": prove.analysis_sha(), "stage": "proof"}
            if sealed:
                spec["gold_set"] = {"file": "gold-set.json", "sha256": gold_sha}
            path = self.repo / f"skill-description-compression__{label}.json"
            path.write_text(json.dumps(spec))
            self._commit(path.name)
            candidate_output = {"outcomes": [output]}
            if judged:
                candidate_output["judge_pairs"] = [{
                    "item_id": "a", "candidate_text": "good answer", "baseline_text": "also good",
                    "winner": "candidate", "swap_winner": "candidate"}]
            with ExitStack() as stack:
                stack.enter_context(patch.object(prove, "_module_sha", return_value="module-sha"))
                stack.enter_context(patch.object(features, "omp_settings", return_value=cfg))
                stack.enter_context(patch.object(features, "providers", return_value=provs))
                stack.enter_context(patch.object(features, "ollama_digests",
                                                 return_value={model: digest} if digest else {}))
                stack.enter_context(patch("localbench.sysstats.host", return_value=host))
                stack.enter_context(patch("localbench.__main__._rev", return_value="test-rev"))
                if wrong_baseline:
                    stack.enter_context(patch.object(
                        prove, "_generation_baseline",
                        return_value={"kind": "route", "id": "ollama/qwen3.1:wrong"},
                    ))
                report = prove.prove_spec(
                    path, outputs_fn=lambda spec, items: {route: candidate_output},
                    receipts_dir=self.tmp / f"receipts-{label}", repo=self.repo, br=FakeBr())
            entry = report["candidates"][0]
            receipt = json.loads(Path(entry["receipt"]).read_text())
            return entry, receipt

        try:
            entry, receipt = run_case("proven")
            self.assertEqual(entry["grade"], "PROVEN")
            self.assertEqual(receipt["omp_module_sha"], "module-sha")
            self.assertEqual(receipt["run"]["provenance"]["pins"]["model_digest"], "b" * 64)
            self.assertEqual(receipt["verdict"]["baseline"],
                             {"kind": "route", "id": "ollama/qwen3.8:27b-mlx"})
            from localbench.__main__ import validate_doc
            self.assertEqual(validate_doc(receipt), [])

            missing_pins, _ = run_case("missing-pins", digest=None)
            self.assertNotEqual(missing_pins["grade"], "PROVEN")
            wrong_baseline, _ = run_case("wrong-baseline", wrong_baseline=True)
            self.assertNotEqual(wrong_baseline["grade"], "PROVEN")
            failed_assertion, failed_receipt = run_case(
                "failed-assertion",
                output={"item_id": "a", "context": {"description": description}, "answerable": True},
            )
            self.assertNotEqual(failed_assertion["grade"], "PROVEN")
            self.assertEqual(failed_receipt["verdict"]["compare"], "NOT_BETTER")
            missing_judge, judge_receipt = run_case("missing-judge", judged=False)
            self.assertNotEqual(missing_judge["grade"], "PROVEN")
            self.assertIn("proof generation requires calibrated judge pairs", judge_receipt["problems"])
            self.assertEqual(judge_receipt["verdict"]["compare"], "NOT_BETTER")
            missing_gold, gold_receipt = run_case("missing-gold", sealed=False)
            self.assertNotEqual(missing_gold["grade"], "PROVEN")
            self.assertIn("proof generation requires a sealed gold set", gold_receipt["problems"])
        finally:
            decision.CORPORA = real_corpora

    def test_replay_outputs_builtin_arm_needs_no_model(self):
        description = "cobalt fixture marker session title generator output"
        body = {"messages": [{"role": "user",
                              "content": "Compress into one routing hint of at most 12 words\n"
                                         "Description: " + description}]}
        spec = {"feature": "skill-description-compression", "kind": "generation",
                "candidates": [{"builtin": "previewSkillDescription"}]}
        out = prove.replay_generation_outputs(spec, [{"id": "a", "body": body, "profile": "default"}])
        (row,) = out["builtin:previewSkillDescription"]
        self.assertEqual(row["item_id"], "a")
        self.assertTrue(isinstance(row["text"], str) and row["text"])
        self.assertEqual(row["answerable"], True)
        self.assertIsInstance(row["violations"], list)

    def test_replay_outputs_route_arm_posts_through_loopback(self):
        seen = {}

        def post(url, body, timeout=120.0):
            seen["url"] = url
            seen["model"] = body.get("model")
            return {"status": 200, "body": {"choices": [{"message":
                    {"content": "cobalt fixture marker session"}}]}, "latency_s": 0.1}

        spec = {"feature": "skill-description-compression", "kind": "generation",
                "candidates": [{"route": "ollama:qwen3.6:35b-mlx"}]}
        out = prove.replay_generation_outputs(
            spec, [{"id": "a", "body": {"messages": [{"role": "user", "content": "x"}]},
                    "profile": "default"}], post=post)
        (row,) = out["ollama:qwen3.6:35b-mlx"]
        self.assertEqual((row["item_id"], row["text"]), ("a", "cobalt fixture marker session"))
        self.assertEqual(seen["model"], "qwen3.6:35b-mlx")
        self.assertIn("/omp-profile/default/chat/completions", seen["url"])
        self.assertTrue(row["violations"])   # replayed through the real check with no description
        key = "ollama:qwen3.6:35b-mlx"
        shape = {"id": "compression-shape", "type": "deterministic",
                 "params": {"check": "check_skill_compression"}}
        problems, _, _ = prove.evaluate_generation_assertions(
            {**spec, "assertions": [shape]}, key, [row], {key: [row]})
        self.assertEqual(len(problems), 1)   # producer violations are trusted, not re-checked

    def test_grade_proof_proven_path(self):
        import localbench.features as features
        real_sha, prove._module_sha = prove._module_sha, lambda feature: "d" * 64
        try:
            sha = prove._module_sha("auto-thinking")
            proof = {"kind": "run", "proof_spec": {"stage": "proof"},
                     "run": {"label": "decision",
                                            "provenance": {"pins": {"model_digest": "d" * 64}}},
                     features.SHA_FIELD: sha, "problems": [],
                     "verdict": {"compare": "BETTER", "baseline": {"kind": "hosted", "id": "typesafe/proj-b-latest"}}}
            inc = {"selector": "typesafe/proj-b-latest", "local": False, "setting": None, "source": "test"}
            self.assertEqual(prove.grade_proof({"feature": "auto-thinking"}, proof, "ollama/nimble:latest",
                                               "d" * 64, inc)[0], "PROVEN")
        finally:
            prove._module_sha = real_sha

    def test_module_sha_refuses_without_omp(self):
        import localbench.workloads as workloads
        real = workloads.omp_bin

        def gone(*args, **kwargs):
            raise FileNotFoundError("omp not found on PATH; set LOCALBENCH_OMP")

        workloads.omp_bin = gone
        try:
            with self.assertRaises(FileNotFoundError):
                prove._module_sha("auto-thinking")
        finally:
            workloads.omp_bin = real

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

    def test_screen_verdict_respects_infrastructure_error_budget(self):
        suite = self._suite([True, False] * 10, [True, False] * 10)
        receipt = {"run": {"details": {},
                           "conformance": {"fixture": {"level": "MUST", "verdict": "PASS"}}}}

        def verdict_for(errors):
            outcomes = [{"id": item["id"], "ok": index not in errors,
                         **({"error": errors[index]} if index in errors else {})}
                        for index, item in enumerate(suite.items)]
            kinds, problems = prove._check_error_kinds({"allow_errors": 0.05}, suite, outcomes)
            verdict = prove.screen_verdict(receipt, problems=problems, error_kinds=kinds, allow_errors=0.05)
            return verdict, kinds

        at_cap, kinds = verdict_for({0: "http"})
        self.assertEqual(kinds["noul"]["infra_error_rate"], 0.05)
        self.assertEqual(at_cap, "ADVANCE")
        above_cap, _ = verdict_for({0: "http", 1: "timeout"})
        self.assertEqual(above_cap, "VOID")
        quality_error, _ = verdict_for({0: "invalid", 1: "invalid"})
        self.assertEqual(quality_error, "REJECT")

    def test_prove_screen_verdict_uses_infrastructure_error_budget(self):
        from unittest import mock

        suite = self._suite([True, False] * 10, [True, False] * 10)
        assertions = [{"id": "accuracy", "type": "metric",
                       "params": {"metric": "decision.noul.accuracy", "min": 0.0}}]
        for error_count, expected in ((1, "ADVANCE"), (2, "VOID")):
            with self.subTest(error_count=error_count):
                path = self._spec(
                    name=f"auto-thinking__screen-errors-{error_count}.json",
                    candidates=[{"route": "laya:org/model"}],
                    dataset={"suite": suite.name, "items_sha256": suite.items_sha256},
                    assertions=assertions, wins=[{"metric": "privacy"}], wins_needed=1, allow_errors=0.05)
                self._commit(path.name)

                def run_suite(_spec, _candidate, _resolved, suite_arg):
                    outcomes = [
                        {"id": item["id"], "repeat": 0, "ok": index >= error_count,
                         **({"error": "http"} if index < error_count else
                            {"answers": {"unsafe": _answer(bool(item["labels"]["unsafe"]))}})}
                        for index, item in enumerate(suite_arg.items)
                    ]
                    arm = {"outcomes": outcomes, "metrics": {"decision.noul.accuracy": {"value": 1.0}},
                           "suite": suite_arg.pin()}
                    return {"kind": "run",
                            "run": {"decision": {"local": arm},
                                    "conformance": {"fixture": {"level": "MUST", "verdict": "PASS"}}},
                            "problems": []}

                fake = FakeBr([{"id": "kit-auto",
                                "title": "prove auto-thinking on laya:org/model", "status": "open"}])
                with (mock.patch.object(decision, "resolve_suite", return_value=suite),
                      mock.patch.object(prove, "_verify_privacy", return_value=(True, "test", {})),
                      mock.patch.object(prove, "bead_ids", return_value=["kit-auto"], create=True)):
                    report = prove.prove_spec(
                        path, run_suite_fn=run_suite, receipts_dir=self.tmp / f"receipts-errors-{error_count}",
                        repo=self.repo, corpora=self.tmp,
                        grade_fn=lambda _receipt, _model: ("PROVEN", "test"), br=fake)
                entry = report["candidates"][0]
                receipt = json.loads(Path(entry["receipt"]).read_text())
                self.assertEqual(entry["screen_verdict"], expected)
                self.assertEqual(receipt["error_kinds"]["noul"]["infra_error_rate"], error_count / 20)

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

    def _rejected_screen_fixture(self, *, verdict="REJECT (SCREEN)", digest="d" * 64, retry=False):
        suite = "decision.noul/unexpected-stop-jevlatest-20261001"
        items_sha256 = "a" * 64
        heading = "2026-10-02 — REJECT (SCREEN): unexpected-stop"
        receipt_rel = "docs/evidence/receipts/rejected.json"
        receipt_path = self.repo / receipt_rel
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({
            "run": {"provenance": {"pins": {
                "model": "historical-alias",
                "model_digest": digest,
                "suite_items_sha256": items_sha256,
            }}}
        }))
        ledger_path = self.repo / "docs/evidence/NEGATIVE_EVIDENCE.md"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            f"## {heading}\n- **Surface:** suite `{suite}`; receipt `{receipt_rel}`.\n"
            f"- **Verdict:** {verdict}\n")
        retry_fields = ({"retry_of": heading,
                         "new_hypothesis": "A model release changes the stop-judgment boundary."}
                        if retry else {})
        path = self._spec(
            "unexpected-stop__screen.json", feature="unexpected-stop",
            dataset={"suite": suite, "items_sha256": items_sha256}, **retry_fields)
        self._commit(path.name)
        return path, ledger_path, receipt_rel, digest

    def test_prove_spec_refuses_matching_rejected_screen_by_installed_digest(self):
        from unittest.mock import patch

        path, ledger, receipt_rel, digest = self._rejected_screen_fixture()
        with (patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              patch("localbench.features.ollama_digests", return_value={"nimble:latest": digest})):
            with self.assertRaises(prove.ProveError) as caught:
                prove.prove_spec(path, dry_run=True, repo=self.repo)
        self.assertIn("REJECT screen", str(caught.exception))
        self.assertIn(receipt_rel, str(caught.exception))
        self.assertIn(digest, str(caught.exception))

    def test_rejected_screen_identity_uses_items_hash_across_suite_renames(self):
        from unittest.mock import patch

        path, ledger, _, digest = self._rejected_screen_fixture()
        spec = json.loads(path.read_text())
        spec["dataset"]["suite"] = "decision.noul/renamed-suite"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        self._commit(path.name)
        with (patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              patch("localbench.features.ollama_digests", return_value={"nimble:latest": digest})):
            with self.assertRaisesRegex(prove.ProveError, "REJECT screen"):
                prove.prove_spec(path, dry_run=True, repo=self.repo)

    def test_rejected_screen_retry_of_ignores_surrounding_whitespace(self):
        from unittest.mock import patch

        path, ledger, _, digest = self._rejected_screen_fixture(retry=True)
        spec = json.loads(path.read_text())
        spec["retry_of"] = f"  {spec['retry_of']}  "
        path.write_text(json.dumps(spec, indent=2) + "\n")
        self._commit(path.name)
        with (patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              patch("localbench.features.ollama_digests", return_value={"nimble:latest": digest})):
            report = prove.prove_spec(path, dry_run=True, repo=self.repo)
        self.assertTrue(report["dry_run"])

    def test_prove_cli_dry_run_refuses_matching_rejected_screen(self):
        import argparse
        import contextlib
        import io
        from unittest import mock

        from localbench import __main__ as cli

        path, ledger, receipt_rel, digest = self._rejected_screen_fixture()
        args = argparse.Namespace(
            spec=str(path), due=False, dry_run=True, omp_frozen=False, json=False,
            mutation=cli.Mutation("prove", ["prove", str(path), "--dry-run"],
                                  dry_run=True, audited=False))
        stderr = io.StringIO()
        with (mock.patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              mock.patch("localbench.features.ollama_digests", return_value={"nimble:latest": digest}),
              mock.patch.object(prove, "spec_commit", return_value="a" * 40),
              mock.patch.object(prove, "plan_steps", return_value=[("safe plan", "fixture")]),
              contextlib.redirect_stderr(stderr)):
            rc = cli.cmd_prove(args)

        self.assertEqual(rc, 1)
        self.assertIn(receipt_rel, stderr.getvalue())
        self.assertIn(digest, stderr.getvalue())
        self.assertEqual(stderr.getvalue().count(str(path)), 1)

        changed_args = argparse.Namespace(
            spec=str(path), due=False, dry_run=True, omp_frozen=False, json=False,
            mutation=cli.Mutation("prove", ["prove", str(path), "--dry-run"],
                                  dry_run=True, audited=False))
        stdout = io.StringIO()
        with (mock.patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              mock.patch("localbench.features.ollama_digests",
                         return_value={"nimble:latest": "e" * 64}),
              mock.patch.object(prove, "spec_commit", return_value="a" * 40),
              mock.patch.object(prove, "plan_steps", return_value=[("safe plan", "fixture")]),
              contextlib.redirect_stdout(stdout)):
            self.assertEqual(cli.cmd_prove(changed_args), 0)
        self.assertIn("safe plan", stdout.getvalue())

    def test_prove_spec_allows_changed_digest_and_explicit_retry(self):
        from unittest.mock import patch

        changed, ledger, _, _ = self._rejected_screen_fixture()
        with (patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              patch("localbench.features.ollama_digests", return_value={"nimble:latest": "e" * 64})):
            self.assertTrue(prove.prove_spec(changed, dry_run=True, repo=self.repo)["dry_run"])

        retry, ledger, _, digest = self._rejected_screen_fixture(retry=True)
        with (patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              patch("localbench.features.ollama_digests", return_value={"nimble:latest": digest})):
            self.assertTrue(prove.prove_spec(retry, dry_run=True, repo=self.repo)["dry_run"])

    def test_prove_spec_does_not_block_a_void_screen(self):
        from unittest.mock import patch

        path, ledger, _, digest = self._rejected_screen_fixture(verdict="VOID")
        with (patch.object(prove, "NEGATIVE_EVIDENCE_PATH", ledger),
              patch("localbench.features.ollama_digests", return_value={"nimble:latest": digest})):
            self.assertTrue(prove.prove_spec(path, dry_run=True, repo=self.repo)["dry_run"])

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

    def test_plan_accepts_mlx_serve_generation_backend(self):
        spec = {"feature": "skill-description-compression", "kind": "generation", "dataset": {"suite": "s", "items_sha256": "x" * 64},
                "candidates": [{"route": "mlx-serve:/models/candidate"}]}
        steps = prove.plan_steps(spec, "c" * 40)
        actions = "\n".join(action for action, _ in steps)
        self.assertIn("run generation candidate", actions)
        self.assertIn("mlx-serve", actions)
        decision = {"feature": "auto-thinking", "kind": "decision", "dataset": {"suite": "s", "items_sha256": "x" * 64},
                    "candidates": [{"route": "ollama:nimble:latest"}]}
        self.assertTrue(prove.plan_steps(decision, "c" * 40))

    def test_plan_refuses_unsupported_generation_backend_with_typed_reason(self):
        spec = {"feature": "skill-description-compression", "kind": "generation", "dataset": {"suite": "s", "items_sha256": "x" * 64},
                "candidates": [{"route": "laya:o/n"}]}
        with self.assertRaisesRegex(prove.ProveError, "generation proof does not support backend 'laya'"):
            prove.plan_steps(spec, "c" * 40)

    def test_plan_refuses_blocked_and_unfrozen_memory_before_steps(self):
        blocked = {"kind": "decision", "blocked": {"reason": "no positives"},
                   "dataset": {"suite": "s", "items_sha256": "x" * 64}, "candidates": []}
        with self.assertRaisesRegex(prove.ProveError, "spec is blocked"):
            prove.plan_steps(blocked, "c" * 40)
        memory = {"kind": "memory", "dataset": {"suite": "s", "items_sha256": "x" * 64},
                  "candidates": [], "arms": {"ab_a": {}, "ab_b": {}}, "verdicts": []}
        with self.assertRaisesRegex(prove.ProveError, "--omp-frozen"):
            prove.plan_steps(memory, "c" * 40)
        self.assertEqual(prove.plan_steps(memory, "c" * 40, omp_frozen=True)[0][0],
                         "resolve dataset s pin xxxxxxxxxxxx")

    def test_prove_spec_generation_and_agent_refuse_live(self):
        path = self._spec("auto-thinking__gen.json", kind="generation", assertions=[
            {"id": "shape", "type": "deterministic", "params": {"check": "check_skill_compression"}}])
        self._commit(path.name)
        with self.assertRaisesRegex(prove.ProveError, "%pane"):
            prove.prove_spec(path, repo=self.repo)

    def test_prove_spec_decision_end_to_end(self):
        from unittest import mock

        from localbench import __main__ as cli
        from localbench import gateway

        suite = self._suite([True, False], [True, False])
        initial = {"nimble:latest": 1 << 15, "qwen3.8:27b-mlx": 1 << 15}
        with FakeProofOllama(initial) as ollama:
            run_calls = []

            def run_suite(spec, candidate, resolved, suite_arg):
                context = decision.ensure_context(
                    ollama.url, resolved["model"], decision.required_context(suite_arg)["required_tokens"],
                    in_use=lambda model: ["fake external client"] if model in initial else [])
                run_calls.append(resolved["model"])
                outs = [{"id": it["id"], "repeat": 0, "ok": True,
                         "answers": {"unsafe": _answer(bool(it["labels"]["unsafe"]))}}
                        for it in suite_arg.items]
                arm = {"outcomes": outs, "metrics": {}, "suite": suite_arg.pin(),
                       "endpoint": decision.endpoint(ollama.url)}
                return {"kind": "run", "run": {"decision": {"context": context, "local": arm}}, "problems": []}

            doc = {"feature": "auto-thinking", "kind": "decision",
                   "candidates": [{"route": f"ollama:{model}"} for model in
                                  ("nimble:latest", "tev1:latest", "tev1:0.8b")],
                   "dataset": {"suite": str(suite.manifest.parent),
                               "items_sha256": suite.items_sha256},
                   "assertions": [{"id": "paired-vs-recorded", "type": "paired",
                                   "params": {"rule": ["BETTER", "NOT_WORSE"], "min_n": 1}}],
                   "stage": "screen", "wins": [{"metric": "privacy"}], "wins_needed": 1}
            path = self.repo / "auto-thinking__t.json"
            path.write_text(json.dumps(doc))
            self._commit(path.name)
            fake = FakeBr([])
            with (mock.patch.object(cli, "DECISION_BASE", ollama.url),
                  mock.patch.object(decision, "loaded_model_cap", return_value={"cap": 3, "source": "fake"}),
                  mock.patch.object(gateway, "safe_to_unload", return_value=(True, None)) as safe_to_unload,
                  mock.patch.object(prove, "_verify_privacy", return_value=(True, "test", {}))):
                report = prove.prove_spec(path, run_suite_fn=run_suite, receipts_dir=self.tmp / "r",
                                          repo=self.repo, corpora=self.tmp,
                                          grade_fn=lambda receipt, model: ("STALE", "test"), br=fake)
            self.assertEqual([row["grade"] for row in report["candidates"]], ["STALE"] * 3)
            self.assertTrue(all(Path(row["receipt"]).is_file() for row in report["candidates"]))
            receipts = [json.loads(Path(row["receipt"]).read_text()) for row in report["candidates"]]
            self.assertEqual([receipt["privacy"] for receipt in receipts],
                             [{"verified": True, "reason": "test"}] * 3)
            self.assertEqual(run_calls, ["nimble:latest", "tev1:latest", "tev1:0.8b"])
            self.assertEqual(ollama.loaded, ["tev1:latest", "tev1:0.8b"])
            self.assertEqual(ollama.unloaded, ["tev1:latest", "tev1:0.8b"])
            self.assertEqual(ollama.resident, initial)
            self.assertEqual([call.args[0] for call in safe_to_unload.call_args_list],
                             ["tev1:latest", "tev1:0.8b"])

    def test_prove_refuses_unload_when_gateway_guard_blocks(self):
        from unittest import mock

        from localbench import __main__ as cli
        from localbench import gateway

        suite = self._suite([True, False], [True, False])
        initial = {"nimble:latest": 1 << 15, "qwen3.8:27b-mlx": 1 << 15}
        with FakeProofOllama(initial) as ollama:
            run_calls = []

            def run_suite(spec, candidate, resolved, suite_arg):
                context = decision.ensure_context(
                    ollama.url, resolved["model"], decision.required_context(suite_arg)["required_tokens"],
                    in_use=lambda model: ["fake external client"] if model in initial else [])
                run_calls.append(resolved["model"])
                outcomes = [{"id": item["id"], "repeat": 0, "ok": True,
                             "answers": {"unsafe": _answer(bool(item["labels"]["unsafe"]))}}
                            for item in suite_arg.items]
                local = {"outcomes": outcomes, "metrics": {}, "suite": suite_arg.pin(),
                         "endpoint": decision.endpoint(ollama.url)}
                return {"kind": "run", "run": {"decision": {"context": context, "local": local}},
                        "problems": []}

            doc = {"feature": "auto-thinking", "kind": "decision",
                   "candidates": [{"route": "ollama:tev1:latest"}],
                   "dataset": {"suite": str(suite.manifest.parent),
                               "items_sha256": suite.items_sha256},
                   "assertions": [{"id": "paired-vs-recorded", "type": "paired",
                                   "params": {"rule": ["BETTER", "NOT_WORSE"], "min_n": 1}}],
                   "stage": "screen"}
            path = self.repo / "auto-thinking__guard.json"
            path.write_text(json.dumps(doc))
            self._commit(path.name)
            fake = FakeBr([{"id": "kit-auto", "title": "auto-thinking"}])
            with (mock.patch.object(cli, "DECISION_BASE", ollama.url),
                  mock.patch.object(decision, "loaded_model_cap", return_value={"cap": 3, "source": "fake"}),
                  mock.patch.object(gateway, "safe_to_unload",
                                    return_value=(False, "gateway request is in flight")) as safe_to_unload,
                  mock.patch.object(prove, "_verify_privacy", return_value=(True, "test", {}))):
                with self.assertRaisesRegex(prove.ProveError, "guarded unload refused"):
                    prove.prove_spec(path, run_suite_fn=run_suite, receipts_dir=self.tmp / "blocked",
                                     repo=self.repo, corpora=self.tmp, br=fake)
            self.assertEqual(run_calls, ["tev1:latest"])
            safe_to_unload.assert_called_once_with("tev1:latest")
            self.assertEqual(ollama.unloaded, [])
            self.assertEqual(ollama.resident, {**initial, "tev1:latest": decision.required_context(suite)["required_tokens"]})
            self.assertEqual(fake.calls, [])

    def test_prove_spec_preflights_unadmittable_later_candidate(self):
        from unittest import mock

        from localbench import __main__ as cli
        from localbench import gateway

        suite = self._suite([True, False], [True, False])
        initial = {"nimble:latest": 1 << 15, "qwen3.8:27b-mlx": 1 << 15}
        with FakeProofOllama(initial, shipped_contexts={"tev1:0.8b": 1}) as ollama:
            run_calls = []

            def run_suite(spec, candidate, resolved, suite_arg):
                context = decision.ensure_context(
                    ollama.url, resolved["model"], decision.required_context(suite_arg)["required_tokens"],
                    in_use=lambda model: ["fake external client"] if model in initial else [])
                run_calls.append(resolved["model"])
                outs = [{"id": it["id"], "repeat": 0, "ok": True,
                         "answers": {"unsafe": _answer(bool(it["labels"]["unsafe"]))}}
                        for it in suite_arg.items]
                arm = {"outcomes": outs, "metrics": {}, "suite": suite_arg.pin(),
                       "endpoint": decision.endpoint(ollama.url)}
                return {"kind": "run", "run": {"decision": {"context": context, "local": arm}}, "problems": []}

            doc = {"feature": "auto-thinking", "kind": "decision",
                   "candidates": [{"route": f"ollama:{model}"} for model in
                                  ("nimble:latest", "tev1:latest", "tev1:0.8b")],
                   "dataset": {"suite": str(suite.manifest.parent),
                               "items_sha256": suite.items_sha256},
                   "assertions": [{"id": "paired-vs-recorded", "type": "paired",
                                   "params": {"rule": ["BETTER", "NOT_WORSE"], "min_n": 1}}],
                   "stage": "screen"}
            path = self.repo / "auto-thinking__unadmittable.json"
            path.write_text(json.dumps(doc))
            self._commit(path.name)
            fake = FakeBr([{"id": "kit-auto", "title": "auto-thinking"}])
            with (mock.patch.object(cli, "DECISION_BASE", ollama.url),
                  mock.patch.object(decision, "loaded_model_cap", return_value={"cap": 3, "source": "fake"}),
                  mock.patch.object(gateway, "safe_to_unload", return_value=(True, None)) as safe_to_unload,
                  mock.patch.object(prove, "_verify_privacy", return_value=(True, "test", {}))):
                with self.assertRaisesRegex(prove.ProveError, "admission preflight"):
                    prove.prove_spec(path, run_suite_fn=run_suite, receipts_dir=self.tmp / "refused",
                                     repo=self.repo, corpora=self.tmp, br=fake)
            self.assertEqual(run_calls, [])
            self.assertEqual(ollama.loaded, [])
            self.assertEqual(ollama.unloaded, [])
            self.assertEqual(list((self.tmp / "refused").glob("*.json")), [])
            self.assertEqual(fake.calls, [])
            safe_to_unload.assert_not_called()


    def test_due_specs_orders_cheaper_rounds_first(self):
        pairs = [("/z-expensive.json", {"feature": "z", "candidates": [], "repeats": 3, "pairs": 2}),
                 ("/a-cheap.json", {"feature": "a", "candidates": [], "repeats": 1, "pairs": 1}),
                 ("/m-mid.json", {"feature": "m", "candidates": [], "repeats": 2, "pairs": 1})]
        class FakeBr:
            def __call__(self, argv):
                return 0, json.dumps({"issues": [{"title": f"prove {x['feature']} on ollama:m"} for _, x in pairs]}), ""
        for _, spec in pairs:
            spec["candidates"] = [{"route": "ollama:m"}]
        self.assertEqual([path for path, _ in prove.due_specs(pairs, FakeBr())],
                         ["/a-cheap.json", "/m-mid.json", "/z-expensive.json"])

    def test_update_beads_and_due_match_exact_titles(self):
        fake = FakeBr([{"id": "k1", "title": "prove auto-thinking on judge:nimble"},
                       {"id": "k2", "title": "prove auto-thinking on judge:nimble / judge:tev1"},
                       {"id": "k3", "title": "prove auto-thinking on judge:nimble2"}])
        summary = prove.update_beads(
            {"feature": "auto-thinking"},
            [{"route": "judge:nimble", "grade": "STALE", "reason": "old", "receipt": "r.json",
              "spec": "s"},
             {"route": "judge:tev1", "grade": "STALE", "reason": "old", "receipt": "r.json",
              "spec": "s"}], fake)
        self.assertEqual(summary["commented"], ["prove auto-thinking on judge:nimble"])
        self.assertEqual(summary["missing"], ["prove auto-thinking on judge:tev1"])
        self.assertEqual([c for c in fake.calls if c[0] != "list"],
                         [["comments", "add", "k1", "-m", "proof s: STALE (old)"]])
        pairs = [("s1", {"feature": "auto-thinking", "candidates": [{"preset": "judge:nimble"}]})]
        self.assertEqual(prove.due_specs(pairs, FakeBr([])), [])
        self.assertEqual(prove.due_specs(pairs, fake), [pairs[0]])
        multi = FakeBr([{"id": "k2", "title": "prove auto-thinking on judge:nimble / judge:tev1"}])
        self.assertEqual(prove.due_specs(pairs, multi), [])
        blocked = [("s2", {"feature": "auto-thinking", "candidates": [{"preset": "judge:nimble"}],
                              "blocked": {"reason": "x"}})]
        self.assertEqual(prove.due_specs(blocked, fake), [])

    def test_wins_needed_must_be_a_satisfiable_int(self):
        for bad in ("1", True, 1.0):
            with self.assertRaisesRegex(prove.ProveError, "wins_needed", msg=repr(bad)):
                prove.load_spec(self._spec(wins_needed=bad, wins=[{"metric": "privacy"}]))
        for bad in (0, -1):
            with self.assertRaisesRegex(prove.ProveError, "wins_needed", msg=repr(bad)):
                prove.load_spec(self._spec(wins_needed=bad))
        with self.assertRaisesRegex(prove.ProveError, "wins_needed"):
            prove.load_spec(self._spec(wins_needed=2, wins=[{"metric": "privacy"}]))
        ok = self._spec(name="auto-thinking__w.json", wins=[{"metric": "privacy"}], wins_needed=1)
        self.assertEqual(prove.load_spec(ok)["wins_needed"], 1)


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


class AnalysisPin(unittest.TestCase):
    def test_current_analysis_hash_is_accepted(self):
        digest = prove.analysis_sha()
        prove.check_analysis_sha({"analysis_sha": digest})

    def test_changed_analysis_hash_is_refused(self):
        with self.assertRaises(prove.ProveError):
            prove.check_analysis_sha({"analysis_sha": "0" * 64})

    def test_proof_without_analysis_hash_is_refused(self):
        with self.assertRaisesRegex(prove.ProveError, "analysis_sha"):
            prove.check_analysis_sha({"stage": "proof"})
