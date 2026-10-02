"""System One decision tier against a fake /v1/systemone on loopback: label-echoing and wrong-answer models, every
malformed-response class as an ERROR, pinned suites refusing a changed label file, and compare() verdicts."""

import copy
import json
import math
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import decision

HOST = {"host_id": "test-host", "macos_build": "TEST"}
MODEL = "nimble:latest"
DIGEST = "24e550a16a70" + "0" * 52
ROUTE_MODEL = "qwen3.8:27b-mlx"

QUESTIONS = {
    "choice": {"intent": {"type": "choice", "instructions": "Which intent does the message ask about?",
                          "criteria": {"card": None, "loan": "a loan question", "fx": None}}},
    "noul": {"unsafe": {"type": "noul", "instructions": "Is the command destructive?"}},
    "score": {"quality": {"type": "score", "instructions": "Rate the answer.", "criteria": ["bad", "ok", "good"]}},
}
URGENCY = {"type": "choice", "instructions": "How urgent is it?", "criteria": {"low": None, "high": None}}


def label_for(kind: str, i: int):
    return {"choice": ["card", "loan", "fx"][i % 3], "noul": i % 2 == 0, "score": i % 3}[kind]


def answer(q: dict, pick) -> dict:
    """A well-formed Ollama 0.35 answer putting 0.9 on `pick` (decision/systemone.go Answer shape)."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": 0.9 if pick else 0.1}
    options = decision.question_options(q)
    probs = {o: 0.9 if o == str(pick) else 0.1 / (len(options) - 1) for o in options}
    entropy = -sum(p * math.log(p) for p in probs.values())
    confidence = max(0.0, min(1.0, 1 - entropy / math.log(len(options))))
    if q["type"] == "choice":
        return {"type": "choice", "choice": str(pick), "probabilities": probs, "confidence": confidence}
    return {"type": "score", "score": sum(j * probs[str(j)] for j in range(len(options))),
            "legend": {o: f"level {o}" for o in options}, "probabilities": probs, "confidence": confidence}


def wrong(q: dict, label):
    if q["type"] == "noul":
        return not label
    options = decision.question_options(q)
    nxt = options[(options.index(str(label)) + 1) % len(options)]
    return nxt if q["type"] == "choice" else int(nxt)


def make_items(kind: str, n: int, two: bool = False, pad: dict | None = None) -> list[dict]:
    """`pad` maps item index -> characters of filler in front of its state (a long prompt)."""
    items = []
    for i in range(n):
        questions = copy.deepcopy(QUESTIONS[kind])
        labels = {name: label_for(kind, i) for name in questions}
        if two:
            questions["urgency"] = copy.deepcopy(URGENCY)
            labels["urgency"] = ["low", "high"][i % 2]
        filler = (pad or {}).get(i, 0)
        state = ("." * filler + " " if filler else "") + f"synthetic {kind} item {i}"
        items.append({"id": f"{kind}-{i}", "state": state, "questions": questions,
                      "labels": labels, "reference": {name: answer(q, labels[name]) for name, q in questions.items()},
                      "source": "synthetic:test_decision"})
    return items


def fake_tokens(body: dict) -> int:
    """The fake's prompt token count: below decision's conservative estimate, as Ollama's real counts were."""
    return len(body["state"]) // 3 + 100


def tagged(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


class Fake:
    """A loopback Ollama: /v1/systemone answers via `behavior(body) -> (status, payload)` (payload bytes are sent
    verbatim) unless the prompt exceeds the resident runner's context, which answers Ollama 0.35's HTTP 400.
    `resident` maps tagged model -> context (/api/ps); /api/generate loads with options.num_ctx, capped at
    `max_ctx`, and evicts `evict_on_load`; a model asked while not resident loads at `default_ctx`. `models` is the
    installed set (/api/tags, /api/show): digest, weights blob, shipped num_ctx, other parameters. As Ollama 0.35
    was observed to (2026-10-01), /v1/systemone reloads a model that ships num_ctx at that value. /api/create FROM
    an installed model merges the parameters (create.ApplyModelfileLayers); `create_hook(entry)` may corrupt it."""

    def __init__(self, behavior, delay: float = 0.0):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/ps":
                    with fake.lock:
                        rows = [{"name": n, "context_length": c} for n, c in fake.resident.items()]
                    self._send(200, {"models": rows})
                    return
                if self.path == "/api/version":
                    self._send(200, {"version": "0.35.0"})
                elif self.path == "/api/tags":
                    with fake.lock:
                        rows = [{"name": n, "digest": e["digest"]} for n, e in fake.models.items()]
                    self._send(200, {"models": rows})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                status = None
                with fake.lock:
                    fake.log.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                    if self.path == "/api/show":
                        e = fake.models.get(tagged(body["model"]))
                        status, payload = (404, {"error": f"model '{body['model']}' not found"}) if e is None else (
                            200, {"parameters": "\n".join(
                                  [*([f"num_ctx{' ' * 24}{e['num_ctx']}"] if e["num_ctx"] else []), *e["params"]]),
                                  "modelfile": f"# Modelfile\nFROM /models/blobs/sha256-{e['weights']}\n"})
                    elif self.path == "/api/create":
                        base = fake.models.get(tagged(body["from"]))
                        if base is None:
                            status, payload = 404, {"error": f"model '{body['from']}' not found"}
                        else:
                            entry = {**base, "num_ctx": body["parameters"]["num_ctx"],
                                     "digest": "c" * 52 + str(body["parameters"]["num_ctx"]).rjust(12, "0")}
                            fake.create_hook(entry)
                            fake.models[tagged(body["model"])] = entry
                            status, payload = 200, {"status": "success"}
                    elif self.path == "/api/generate":
                        for name in fake.evict_on_load:
                            fake.resident.pop(name, None)
                        fake.resident[body["model"]] = min(body["options"]["num_ctx"], fake.max_ctx)
                        status, payload = 200, {"model": body["model"], "response": "", "done": True}
                    else:
                        shipped = (fake.models.get(tagged(body["model"])) or {}).get("num_ctx")
                        if shipped and fake.resident.get(body["model"]) != shipped:
                            fake.resident[body["model"]] = shipped    # /v1/systemone has no options
                        ctx = fake.resident.setdefault(body["model"], fake.default_ctx)
                        tokens = fake_tokens(body)
                        status = 400 if tokens > ctx else None
                        payload = {"error": f"prompt 0 has {tokens} tokens; expected 1–{ctx + 2} "
                                            "(input is never truncated)"}
                if status is None:
                    time.sleep(fake.delay)
                    status, payload = fake.behavior(body)
                self._send(status, payload)

        self.behavior, self.delay, self.log, self.lock = behavior, delay, [], threading.Lock()
        self.resident = {MODEL: 1 << 15, ROUTE_MODEL: 1 << 15}
        self.default_ctx, self.max_ctx, self.evict_on_load = 8192, 1 << 18, []
        self.models = {MODEL: {"digest": DIGEST, "weights": "a" * 64, "num_ctx": None,
                               "params": ['stop "<|im_end|>"', "temperature 0.6"]},
                       ROUTE_MODEL: {"digest": "f" * 64, "weights": "b" * 64, "num_ctx": None, "params": []}}
        self.create_hook = lambda entry: None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def respond(body: dict, picks: dict) -> dict:
    return {"model": body["model"],
            "answers": {name: answer(q, picks[name]) for name, q in body["questions"].items()},
            "usage": {"input_tokens": 100 + len(body["state"]), "output_tokens": len(body["questions"])}}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.source = self.tmp / "corpus.jsonl"
        self.source.write_text('{"text": "synthetic"}\n')
        self.truth: dict[str, dict] = {}
        self.users: list[str] = []          # what in_use() reports; the real probe reads lsof, GPU and the gateway
        self.users_by_model: dict[str, list[str]] = {}    # per-model override of `users`
        self.in_use_calls: list[str] = []
        # Ollama's loaded-model cap comes from launchd; tests pin it (None = Ollama's automatic default, 3).
        self.launchd_cap: str | None = None
        patcher = mock.patch.object(decision, "_launchctl_getenv", lambda name: self.launchd_cap)
        patcher.start()
        self.addCleanup(patcher.stop)

    def in_use(self, model: str) -> list[str]:
        self.in_use_calls.append(model)
        return list(self.users_by_model.get(model, self.users))

    def suite(self, kind: str, n: int, two: bool = False, name: str | None = None, gate: dict | None = None,
              pad: dict | None = None):
        items = make_items(kind, n, two, pad)
        self.truth.update({it["state"]: it["labels"] for it in items})
        role = f"decision.{kind}"
        return decision.write_suite(self.tmp / (name or f"{kind}{n}{int(two)}"), name=name or role, role=role,
                                    items=items, sources=[self.source],
                                    gate=gate or {f"decision.{kind}.accuracy": {"min": 0.9}})

    def fake(self, behavior, delay: float = 0.0) -> Fake:
        f = Fake(behavior, delay)
        self.addCleanup(f.close)
        return f

    def echo(self, body):
        return 200, respond(body, self.truth[body["state"]])

    def wrong_on(self, every: int):
        """Wrong on states whose item index is a multiple of `every`, echo otherwise."""
        def behavior(body):
            labels = self.truth[body["state"]]
            i = int(body["state"].rsplit(" ", 1)[1])
            picks = {n: wrong(q, labels[n]) if i % every == 0 else labels[n] for n, q in body["questions"].items()}
            return 200, respond(body, picks)
        return behavior

    def run_suite(self, url, suite, **kw):
        return decision.run_suite(url, MODEL, suite, host=HOST, rev="test", in_use=self.in_use, **kw)


class Quality(Base):
    def test_label_echoing_model_scores_perfectly_and_passes_its_gate(self):
        # Hand values for 0.9 on the label: choice Brier 0.1^2 + 2*0.05^2; noul Brier 0.1^2 and ECE |1 - 0.9|;
        # score labels 0,1,2 give expected scores 0.15, 1.0, 1.85 -> MAE (0.15 + 0 + 0.15) / 3.
        expect = {"choice": {"brier": 0.015, "ece": 0.1}, "noul": {"brier": 0.01, "ece": 0.1},
                  "score": {"mae": 0.1, "ece": 0.1}}
        for kind in ("choice", "noul", "score"):
            with self.subTest(kind=kind):
                receipt = self.run_suite(self.fake(self.echo).url, self.suite(kind, 6))
                m = receipt["run"]["metrics"]
                self.assertEqual(receipt["problems"], [])
                self.assertEqual(receipt["verdict"], {"compare": "NONE", "baseline": {"kind": "none", "id": None}})
                self.assertEqual(receipt["run"]["provenance"]["pins"]["model_digest"], DIGEST)
                self.assertEqual(m[f"decision.{kind}.accuracy"]["value"], 1.0)
                self.assertEqual(m[f"decision.{kind}.macro_f1"]["value"], 1.0)
                self.assertEqual(m[f"decision.{kind}.error_rate"]["value"], 0.0)
                self.assertEqual(m["decision.error_rate"]["value"], 0.0)
                for metric, value in expect[kind].items():
                    self.assertAlmostEqual(m[f"decision.{kind}.{metric}"]["value"], value, places=9)
                if kind == "score":
                    self.assertAlmostEqual(m["decision.score.agreement_mae"]["value"], 0.0, places=9)
                else:
                    self.assertEqual(m[f"decision.{kind}.agreement"]["value"], 1.0)
                gate = receipt["run"]["conformance"][f"gate:decision.{kind}.accuracy"]
                self.assertEqual((gate["level"], gate["verdict"]), ("MUST", "PASS"))
                self.assertEqual(cli.validate_doc(receipt), [])

    def test_wrong_answer_model_has_low_accuracy_and_fails_its_gate(self):
        receipt = self.run_suite(self.fake(self.wrong_on(1)).url, self.suite("choice", 6))
        m = receipt["run"]["metrics"]
        self.assertEqual(m["decision.choice.accuracy"]["value"], 0.0)
        self.assertEqual(m["decision.choice.agreement"]["value"], 0.0)
        self.assertEqual(receipt["run"]["verdicts"]["must_fail"], ["gate:decision.choice.accuracy"])
        self.assertTrue(any(p.startswith("MUST FAIL") for p in receipt["problems"]))

    def test_errors_count_as_wrong_in_the_denominator(self):
        def half_fail(body):
            i = int(body["state"].rsplit(" ", 1)[1])
            return (500, {"error": "runner crashed"}) if i % 2 else self.echo(body)
        receipt = self.run_suite(self.fake(half_fail).url, self.suite("choice", 6))
        m = receipt["run"]["metrics"]
        self.assertEqual(m["decision.choice.accuracy"]["value"], 0.5)
        self.assertEqual(m["decision.choice.accuracy"]["n"], 6)
        self.assertEqual(m["decision.choice.error_rate"]["value"], 0.5)
        self.assertAlmostEqual(m["decision.choice.brier"]["value"], (3 * 0.015 + 3 * 2.0) / 6, places=9)
        errors = [o for o in receipt["run"]["decision"]["local"]["outcomes"] if not o["ok"]]
        self.assertEqual({o["error"] for o in errors}, {"http"})
        # HTTP failures are availability (error_rate), not malformed answers: validity still holds.
        self.assertEqual(receipt["run"]["conformance"]["decision.responses_valid"]["verdict"], "PASS")

    def test_each_malformed_response_class_is_an_error_not_a_score(self):
        def missing_option(r):
            a = r["answers"]["intent"]
            dropped = next(k for k in a["probabilities"] if k != a["choice"])
            a["probabilities"][a["choice"]] += a["probabilities"].pop(dropped)

        def not_normalized(r):
            a = r["answers"]["intent"]
            for k in a["probabilities"]:
                if k != a["choice"]:
                    a["probabilities"][k] = 0.3

        def not_argmax(r):
            a = r["answers"]["intent"]
            a["choice"] = next(k for k in a["probabilities"] if k != a["choice"])

        def noul_out_of_range(r):
            r["answers"]["unsafe"]["noul"] = 1.5

        def noul_nan(r):
            r["answers"]["unsafe"]["noul"] = float("nan")

        def infinite_probability(r):
            r["answers"]["intent"]["confidence"] = float("inf")

        def score_not_expectation(r):
            r["answers"]["quality"]["score"] += 0.5

        def renamed(r):
            r["answers"]["purpose"] = r["answers"].pop("intent")

        def extra_answer(r):
            r["answers"]["extra"] = {"type": "noul", "noul": 0.5}

        def reordered(r):
            r["answers"] = dict(reversed(list(r["answers"].items())))

        def missing_usage(r):
            del r["usage"]

        cases = [("probabilities missing an option", "choice", False, missing_option),
                 ("probabilities not summing to 1", "choice", False, not_normalized),
                 ("choice not the argmax", "choice", False, not_argmax),
                 ("noul outside [0, 1]", "noul", False, noul_out_of_range),
                 ("NaN", "noul", False, noul_nan),
                 ("Infinity", "choice", False, infinite_probability),
                 ("score != sum(j * p_j)", "score", False, score_not_expectation),
                 ("answer renamed", "choice", False, renamed),
                 ("extra answer", "choice", False, extra_answer),
                 ("answers out of request order", "choice", True, reordered),
                 ("usage missing", "choice", False, missing_usage)]
        for label, kind, two, plant in cases:
            with self.subTest(label):
                def behavior(body, plant=plant):
                    r = respond(body, self.truth[body["state"]])
                    plant(r)
                    return 200, json.dumps(r).encode()      # json.dumps writes NaN/Infinity tokens verbatim
                receipt = self.run_suite(self.fake(behavior).url, self.suite(kind, 3, two=two))
                outcomes = receipt["run"]["decision"]["local"]["outcomes"]
                self.assertEqual([(o["ok"], o.get("error")) for o in outcomes], [(False, "invalid")] * 3)
                m = receipt["run"]["metrics"]
                self.assertEqual(m[f"decision.{kind}.accuracy"]["value"], 0.0)
                self.assertEqual(m["decision.error_rate"]["value"], 1.0)
                self.assertNotIn("answers", outcomes[0])
                valid = receipt["run"]["conformance"]["decision.responses_valid"]
                self.assertEqual((valid["level"], valid["verdict"], valid["invalid"]), ("MUST", "FAIL", 3))
                self.assertIn("decision.responses_valid", receipt["run"]["verdicts"]["must_fail"])
                self.assertTrue(any("decision.responses_valid" in p for p in receipt["problems"]))

    def test_an_answer_from_another_model_is_invalid(self):
        def impostor(body):
            return 200, respond(body | {"model": "tev1:0.8b"}, self.truth[body["state"]])
        receipt = self.run_suite(self.fake(impostor).url, self.suite("choice", 3))
        outcomes = receipt["run"]["decision"]["local"]["outcomes"]
        self.assertEqual([(o["ok"], o.get("error")) for o in outcomes], [(False, "invalid")] * 3)
        self.assertEqual(receipt["run"]["metrics"]["decision.choice.accuracy"]["value"], 0.0)

    def test_repeats_that_disagree_fail_the_determinism_check(self):
        calls: dict[str, int] = {}

        def drifting(body):
            calls[body["state"]] = calls.get(body["state"], 0) + 1
            r = respond(body, self.truth[body["state"]])
            if calls[body["state"]] == 2:
                a = r["answers"]["unsafe"]
                a["noul"] = 0.8 if a["noul"] > 0.5 else 0.2
            return 200, r
        suite = self.suite("noul", 4)
        steady = self.run_suite(self.fake(self.echo).url, suite, repeats=2)
        self.assertEqual(steady["run"]["conformance"]["decision.deterministic"]["verdict"], "PASS")
        self.assertEqual(steady["problems"], [])
        drift = self.run_suite(self.fake(drifting).url, suite, repeats=2)
        det = drift["run"]["conformance"]["decision.deterministic"]
        self.assertEqual((det["level"], det["verdict"], det["n_differs"]), ("MUST", "FAIL", 4))
        self.assertEqual(drift["run"]["verdicts"]["must_fail"], ["decision.deterministic"])
        self.assertEqual(drift["problems"], ["MUST FAIL: decision.deterministic"])


class Requests(unittest.TestCase):
    def test_limits_are_refused_before_sending_and_boundaries_accepted(self):
        def choice(n):
            return {"type": "choice", "instructions": "pick", "criteria": {f"o{i}": None for i in range(n)}}
        ok = {"q": choice(2)}
        decision.build_request(MODEL, "s", {f"q{i}": choice(26) for i in range(64)})
        refused = {"no questions": ("s", {}),
                   "65 questions": ("s", {f"q{i}": choice(2) for i in range(65)}),
                   "1 candidate": ("s", {"q": choice(1)}),
                   "27 candidates": ("s", {"q": choice(27)}),
                   "empty state": ("  ", ok),
                   "empty object state": ({}, ok),
                   "number state": (5, ok),
                   "body over 64 KiB": ("x" * (64 << 10), ok),
                   "noul null criteria": ("s", {"q": {"type": "noul", "instructions": "i", "criteria": None}}),
                   "unknown noul criterion": ("s", {"q": {"type": "noul", "instructions": "i",
                                                          "criteria": {"maybe": "?"}}}),
                   "empty instructions": ("s", {"q": {"type": "score", "instructions": "", "criteria": ["a", "b"]}}),
                   "unknown type": ("s", {"q": {"type": "rank", "instructions": "i", "criteria": ["a", "b"]}})}
        for label, (state, questions) in refused.items():
            with self.subTest(label), self.assertRaises(decision.RequestError):
                decision.build_request(MODEL, state, questions)
        overhead = len(decision.encode(decision.build_request(MODEL, "x", ok))) - 1
        exact = decision.build_request(MODEL, "x" * ((64 << 10) - overhead), ok)
        self.assertEqual(len(decision.encode(exact)), 64 << 10)
        with self.assertRaises(decision.RequestError):
            decision.build_request(MODEL, "x" * ((64 << 10) - overhead + 1), ok)

    def test_local_arm_is_loopback_only_and_remote_arm_https_only(self):
        self.assertEqual(decision.endpoint("http://127.0.0.1:11434/v1/"), "http://127.0.0.1:11434/v1/systemone")
        self.assertEqual(decision.endpoint("http://localhost:11434"), "http://localhost:11434/v1/systemone")
        self.assertEqual(decision.endpoint("https://api.typesafe.ai", allow_remote=True),
                         "https://api.typesafe.ai/v1/systemone")
        for url, remote in (("http://10.0.0.5:11434", False), ("https://api.typesafe.ai", False),
                            ("http://api.typesafe.ai", True)):
            with self.subTest(url), self.assertRaises(decision.RequestError):
                decision.endpoint(url, allow_remote=remote)


class Suites(Base):
    def test_written_suites_are_owner_only(self):
        target = self.tmp / "private"
        target.mkdir(mode=0o755)
        (target / "items.jsonl").write_text("stale\n")
        os.chmod(target / "items.jsonl", 0o644)
        suite = self.suite("choice", 2, name="private")
        self.assertEqual(stat.S_IMODE(suite.manifest.parent.stat().st_mode), 0o700)
        for f in (suite.items_path, suite.manifest):
            with self.subTest(f.name):
                self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)

    def test_changed_labels_or_source_fail_the_hash_check(self):
        suite = self.suite("choice", 3)
        items = suite.items_path.read_text()
        suite.items_path.write_text(items.replace('"labels":{"intent":"card"}', '"labels":{"intent":"loan"}', 1))
        with self.assertRaisesRegex(decision.SuiteError, "label-hash"):
            decision.load_suite(suite.manifest)
        suite.items_path.write_text(items)
        decision.load_suite(suite.manifest)
        self.source.write_text('{"text": "edited"}\n')
        with self.assertRaisesRegex(decision.SuiteError, "source"):
            decision.load_suite(suite.manifest)

    def test_items_must_match_their_role_and_label_every_question(self):
        noul = make_items("noul", 2)
        with self.assertRaisesRegex(decision.SuiteError, "decision.choice items ask"):
            decision.write_suite(self.tmp / "r", name="r", role="decision.choice", items=noul, sources=[self.source],
                                 gate={"decision.choice.accuracy": {"min": 0.5}})
        unlabeled = make_items("choice", 2)
        unlabeled[1]["labels"] = {}
        with self.assertRaisesRegex(decision.SuiteError, "labels"):
            decision.write_suite(self.tmp / "u", name="u", role="decision.choice", items=unlabeled,
                                 sources=[self.source], gate={"decision.choice.accuracy": {"min": 0.5}})
        partial = make_items("choice", 2)
        del partial[0]["reference"]
        with self.assertRaisesRegex(decision.SuiteError, "every item or none"):
            decision.write_suite(self.tmp / "p", name="p", role="decision.choice", items=partial,
                                 sources=[self.source], gate={"decision.choice.accuracy": {"min": 0.5}})


class Compare(Base):
    def test_faster_with_equal_quality_is_better_and_slower_is_not(self):
        suite = self.suite("choice", 8)
        fast = self.run_suite(self.fake(self.echo).url, suite)
        slow = self.run_suite(self.fake(self.echo, delay=0.05).url, suite)
        verdict = decision.compare(fast, slow)
        self.assertEqual(verdict["verdict"], "BETTER")
        self.assertEqual(verdict["wins"], ["p95_latency"])
        self.assertEqual(verdict["quality_losses"], [])
        self.assertEqual(decision.compare(slow, fast)["verdict"], "NOT_BETTER")

    def test_quality_drop_beyond_noise_is_worse_even_when_faster(self):
        suite = self.suite("choice", 8)
        candidate = self.run_suite(self.fake(self.wrong_on(2)).url, suite)
        baseline = self.run_suite(self.fake(self.echo, delay=0.05).url, suite)
        verdict = decision.compare(candidate, baseline)
        self.assertIn("p95_latency", verdict["wins"])
        self.assertIn("decision.choice.accuracy", verdict["quality_losses"])
        self.assertEqual(verdict["verdict"], "WORSE")

    def test_exact_tie_is_not_better(self):
        receipt = self.run_suite(self.fake(self.echo).url, self.suite("choice", 6))
        verdict = decision.compare(receipt, copy.deepcopy(receipt))
        self.assertEqual((verdict["verdict"], verdict["wins"], verdict["quality_losses"]), ("NOT_BETTER", [], []))

    def test_noise_is_the_aa_spread_when_repeats_exist_else_a_paired_bootstrap(self):
        # One wrong item in 20: with repeats the deterministic A/A spread is 0, so the loss is beyond noise; with one
        # repeat the seeded bootstrap CI over items still reaches 0 (the item is absent from ~36% of resamples).
        suite = self.suite("choice", 20)
        url_wrong, url_echo = self.fake(self.wrong_on(20)).url, self.fake(self.echo).url
        aa = decision.compare(self.run_suite(url_wrong, suite, repeats=2), self.run_suite(url_echo, suite, repeats=2))
        self.assertEqual(aa["noise"], "aa_spread")
        self.assertEqual(aa["deltas"]["decision.choice.accuracy"]["judgement"], "loss")
        self.assertEqual(aa["verdict"], "WORSE")
        candidate, baseline = self.run_suite(url_wrong, suite), self.run_suite(url_echo, suite)
        boot = decision.compare(candidate, baseline)
        self.assertEqual(boot["noise"]["bootstrap"], decision.BOOTSTRAP_RESAMPLES)
        self.assertEqual(boot["deltas"]["decision.choice.accuracy"]["judgement"], "within_noise")
        self.assertEqual(decision.compare(candidate, baseline)["deltas"], boot["deltas"])   # seeded: same CIs

    def test_arms_on_different_suites_are_not_compared(self):
        a = self.run_suite(self.fake(self.echo).url, self.suite("choice", 3, name="a"))
        b = self.run_suite(self.fake(self.echo).url, self.suite("choice", 4, name="b"))
        with self.assertRaises(ValueError):
            decision.compare(a, b)

    def test_a_run_worse_than_its_route_baseline_is_a_problem_and_feature_pins_are_written(self):
        suite = self.suite("choice", 8, gate={"decision.choice.accuracy": {"min": 0.5}})
        route = decision.run_suite(self.fake(self.echo, delay=0.05).url, ROUTE_MODEL, suite, host=HOST, rev="test")
        baseline = decision.Baseline("route", "ollama/qwen3.8:27b-mlx", route)
        receipt = self.run_suite(self.fake(self.wrong_on(2)).url, suite, baseline=baseline,
                                 feature="omp.judge.auto-thinking", omp_module_sha="ab" * 32)
        self.assertEqual(receipt["run"]["metrics"]["decision.choice.accuracy"]["value"], 0.5)   # the gate passes
        self.assertEqual(receipt["verdict"], {"compare": "WORSE",
                                              "baseline": {"kind": "route", "id": "ollama/qwen3.8:27b-mlx"}})
        self.assertEqual(receipt["run"]["decision"]["compare"]["verdict"], "WORSE")
        self.assertEqual(receipt["problems"], ["not better than route ollama/qwen3.8:27b-mlx: WORSE"])
        self.assertEqual((receipt["kind"], receipt["run"]["label"]), ("run", "decision"))
        self.assertEqual((receipt["feature"], receipt["omp_module_sha"]), ("omp.judge.auto-thinking", "ab" * 32))
        self.assertNotIn("feature", route)
        fixed = self.run_suite(self.fake(self.echo, delay=0.05).url, suite, setting="high")
        better = self.run_suite(self.fake(self.echo).url, suite, baseline=decision.Baseline("fixed", "high", fixed))
        self.assertEqual((better["verdict"]["compare"], better["problems"]), ("BETTER", []))

    def test_a_baseline_id_must_name_what_its_receipt_measured(self):
        suite = self.suite("choice", 3)
        route = decision.run_suite(self.fake(self.echo).url, ROUTE_MODEL, suite, host=HOST, rev="test")
        fixed = self.run_suite(self.fake(self.echo).url, suite, setting="high")
        candidate = self.fake(self.echo)
        for wrong_id in (decision.Baseline("route", "ollama/tev1:latest", route),
                         decision.Baseline("fixed", "low", fixed),
                         decision.Baseline("fixed", "high", route)):
            with self.subTest(wrong_id.id), self.assertRaisesRegex(ValueError, "baseline id"):
                self.run_suite(candidate.url, suite, baseline=wrong_id)
        self.assertEqual(candidate.log, [])

    def test_an_unsound_baseline_receipt_is_refused_before_any_request(self):
        suite = self.suite("choice", 4)
        broken = decision.run_suite(self.fake(self.wrong_on(1), delay=0.05).url, ROUTE_MODEL, suite, host=HOST,
                                    rev="test")
        self.assertEqual(broken["problems"], ["MUST FAIL: gate:decision.choice.accuracy"])
        candidate = self.fake(self.echo)
        with self.assertRaisesRegex(ValueError, "not a sound receipt"):
            self.run_suite(candidate.url, suite, baseline=decision.Baseline("route", ROUTE_MODEL, broken))
        self.assertEqual(candidate.log, [])
        # A bare arm doc carries no problems field and is taken as it is (the caller vouches for it).
        arm = broken["run"]["decision"]["local"]
        receipt = self.run_suite(candidate.url, suite, baseline=decision.Baseline("route", ROUTE_MODEL, arm))
        self.assertEqual(receipt["verdict"]["compare"], "BETTER")


class HostedArm(Base):
    def test_hosted_arm_runs_the_same_items_with_cost_and_never_records_the_key(self):
        suite = self.suite("choice", 4)
        local, hosted = self.fake(self.echo), self.fake(self.echo)
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "sk-test-secret"}):
            receipt = self.run_suite(local.url, suite, hosted=decision.Hosted(base_url=hosted.url))
        self.assertEqual({e["auth"] for e in hosted.log}, {"Bearer sk-test-secret"})
        self.assertEqual({e["auth"] for e in local.log}, {None})
        asked = [e["body"]["state"] for e in local.log if e["path"] == "/v1/systemone"]
        self.assertEqual([e["body"]["state"] for e in hosted.log], asked)
        self.assertEqual({e["body"]["model"] for e in hosted.log}, {"proj-b-latest"})
        self.assertNotIn("sk-test-secret", json.dumps(receipt))
        tokens = sum(len(e["body"]["state"]) + 100 for e in hosted.log)
        cost = receipt["run"]["decision"]["hosted"]["metrics"]["decision.cost_usd"]["value"]
        self.assertAlmostEqual(cost, tokens * 0.042e-6, places=15)
        verdict = receipt["run"]["decision"]["compare"]
        self.assertEqual((verdict["verdict"], verdict["wins"][-2:]), ("BETTER", ["privacy", "cost"]))
        self.assertEqual(receipt["verdict"], {"compare": "BETTER", "baseline": {"kind": "hosted", "id": "proj-b-latest"}})
        self.assertEqual(receipt["problems"], [])

    def test_hosted_arm_needs_its_key_and_never_stands_in_for_the_local_arm(self):
        suite = self.suite("choice", 3)
        local, hosted = self.fake(self.echo), self.fake(self.echo)
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), self.assertRaises(decision.RequestError):
            self.run_suite(local.url, suite, hosted=decision.Hosted(base_url=hosted.url))
        self.assertEqual((local.log, hosted.log), ([], []))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{s.getsockname()[1]}"
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "k"}):
            receipt = self.run_suite(dead, suite, hosted=decision.Hosted(base_url=hosted.url))
        d = receipt["run"]["decision"]
        self.assertEqual({o["error"] for o in d["local"]["outcomes"]}, {"unavailable"})
        self.assertEqual(receipt["run"]["metrics"]["decision.choice.accuracy"]["value"], 0.0)
        self.assertEqual(d["hosted"]["metrics"]["decision.choice.accuracy"]["value"], 1.0)
        self.assertEqual(d["compare"]["verdict"], "WORSE")
        self.assertIn("not better than hosted proj-b-latest: WORSE", receipt["problems"])


class Load(Base):
    def test_coresident_load_is_recorded_around_the_requests_and_never_voids_the_run(self):
        fake = self.fake(self.echo)

        class Sampler:
            contention = [{"t": 1.0, "foreign": {"ollama": ["qwen3.8:27b-mlx"]}}]
            load_spikes: list = []

            def __enter__(self):
                self.at_enter = sum(e["path"] == "/v1/systemone" for e in fake.log)
                return self

            def __exit__(self, *exc):
                self.at_exit = sum(e["path"] == "/v1/systemone" for e in fake.log)

            def summary(self):
                return {"samples": 3, "resident_unknown_samples": 0, "load": {"app_gpu_mean_pct": 40.0}}
        sampler = Sampler()
        receipt = self.run_suite(fake.url, self.suite("choice", 3), sampler=sampler)
        self.assertEqual((sampler.at_enter, sampler.at_exit), (0, 3))
        self.assertEqual(receipt["problems"], [])
        self.assertFalse(receipt["run"]["verdicts"]["contended"])
        self.assertEqual(receipt["run"]["system"]["contention"], Sampler.contention)
        self.assertEqual(receipt["run"]["system"]["during"]["load"]["app_gpu_mean_pct"], 40.0)


class Context(Base):
    """Ollama 0.35's /v1/systemone takes no num_ctx: the resident runner's context bounds every prompt. On
    2026-10-01 runners left at 8194 and 2050 tokens by an earlier client 400ed 23/100 and 76/100 find-judgment items."""

    def test_a_small_resident_context_is_reloaded_with_the_required_num_ctx_and_pinned(self):
        suite = self.suite("choice", 3, pad={2: 9000}, gate={"decision.choice.accuracy": {"min": 0.9}})
        need = decision.required_context(suite)
        self.assertEqual(need["max_prompt_item"], "choice-2")
        self.assertGreater(need["required_tokens"], fake_tokens({"state": suite.items[2]["state"]}))
        fake = self.fake(self.echo)
        fake.resident[MODEL] = 2048
        receipt = self.run_suite(fake.url, suite)
        loads = [e["body"] for e in fake.log if e["path"] == "/api/generate"]
        self.assertEqual(loads, [{"model": MODEL, "prompt": "", "options": {"num_ctx": need["required_tokens"]},
                                  "keep_alive": "30m"}])
        self.assertEqual([e["path"] for e in fake.log[:2]], ["/api/show", "/api/generate"])   # before any item
        self.assertEqual(self.in_use_calls, [MODEL])
        self.assertEqual(receipt["run"]["provenance"]["pins"]["loaded_context"], need["required_tokens"])
        ctx = receipt["run"]["decision"]["context"]
        self.assertEqual((ctx["action"], ctx["before"]), ("reloaded", 2048))
        self.assertEqual([o["ok"] for o in receipt["run"]["decision"]["local"]["outcomes"]], [True] * 3)
        self.assertEqual(receipt["run"]["conformance"]["decision.context_fits"]["verdict"], "PASS")
        self.assertEqual(receipt["problems"], [])
        items = receipt["run"]["decision"]["local"]["items"]
        self.assertEqual(max(it["state_bytes"] for it in items), need["max_state_bytes"])

    def test_a_large_enough_resident_runner_is_used_as_is(self):
        fake = self.fake(self.echo)
        receipt = self.run_suite(fake.url, self.suite("choice", 3, pad={1: 9000}))
        self.assertEqual([e["path"] for e in fake.log if e["path"] != "/v1/systemone"], ["/api/show"])
        self.assertEqual(self.in_use_calls, [])
        self.assertEqual(receipt["run"]["provenance"]["pins"]["loaded_context"], 1 << 15)
        self.assertEqual(receipt["run"]["decision"]["context"]["action"], "resident")

    def test_a_small_runner_in_use_by_another_client_is_refused_not_reloaded(self):
        fake = self.fake(self.echo)
        fake.resident[MODEL] = 2048
        self.users = ["established Ollama client pid 4242 (proj-b)"]
        with self.assertRaisesRegex(decision.ContextError, r"2048-token context.*in use.*pid 4242"):
            self.run_suite(fake.url, self.suite("choice", 3, pad={2: 9000}))
        self.assertEqual([e["path"] for e in fake.log], ["/api/show"])
        self.assertEqual(fake.resident[MODEL], 2048)

    def test_a_load_that_stays_below_the_requirement_is_refused(self):
        fake = self.fake(self.echo)
        del fake.resident[MODEL]
        fake.max_ctx = 1024                                           # the model's own maximum
        with self.assertRaisesRegex(decision.ContextError, "would not fit"):
            self.run_suite(fake.url, self.suite("choice", 3, pad={2: 9000}))
        self.assertEqual([e["path"] for e in fake.log], ["/api/show", "/api/generate"])
        self.assertEqual(self.in_use_calls, [])                       # not resident: nobody to take it from

    def test_a_load_that_evicts_another_resident_is_a_problem(self):
        fake = self.fake(self.echo)
        del fake.resident[MODEL]
        fake.evict_on_load = [ROUTE_MODEL]
        receipt = self.run_suite(fake.url, self.suite("choice", 2))
        self.assertEqual(receipt["run"]["decision"]["context"]["action"], "loaded")
        self.assertEqual(receipt["run"]["decision"]["context"]["evicted"], [ROUTE_MODEL])
        self.assertTrue(any(ROUTE_MODEL in p and "evicted" in p for p in receipt["problems"]))

    def test_context_400s_mid_run_are_their_own_error_class_and_fail_the_run(self):
        def reloaded_by_another_client(body):
            fake.resident[MODEL] = 2048                               # someone reloads it small after item 0
            return self.echo(body)
        fake = self.fake(reloaded_by_another_client)
        suite = self.suite("choice", 4, pad={2: 9000, 3: 9000}, gate={"decision.choice.accuracy": {"min": 0.5}})
        receipt = self.run_suite(fake.url, suite)
        outcomes = receipt["run"]["decision"]["local"]["outcomes"]
        self.assertEqual([o.get("error") for o in outcomes], [None, None, "context", "context"])
        m = receipt["run"]["metrics"]
        self.assertEqual((m["decision.choice.accuracy"]["value"], m["decision.choice.accuracy"]["n"]), (0.5, 4))
        fits = receipt["run"]["conformance"]["decision.context_fits"]
        self.assertEqual((fits["level"], fits["verdict"], fits["context_errors"]), ("MUST", "FAIL", 2))
        self.assertEqual(receipt["run"]["verdicts"]["must_fail"], ["decision.context_fits"])
        self.assertEqual(receipt["run"]["verdicts"]["pins_changed"]["loaded_context"], [1 << 15, 2048])
        self.assertEqual(receipt["run"]["conformance"]["decision.responses_valid"]["verdict"], "PASS")


class LoadedModelCap(Base):
    """2026-10-01 18:01Z: warm-loading tev1 with three models resident (Ollama's automatic cap is 3 per Metal GPU)
    evicted nimble (proj-b's live gate) and qwen3.8 (every profile's smol). A load at the cap must not evict a resident
    another client is using, unless --allow-evict."""

    def at_cap(self) -> Fake:
        fake = self.fake(self.echo)
        del fake.resident[MODEL]
        fake.resident.update({"qwen3.6:35b-mlx": 1 << 15, "tev1:latest": 2050})   # 3 residents with ROUTE_MODEL
        fake.evict_on_load = [ROUTE_MODEL]
        return fake

    def test_a_load_at_the_cap_with_an_in_use_resident_is_refused_before_any_load(self):
        fake = self.at_cap()
        self.users_by_model = {ROUTE_MODEL: ["established Ollama client pid 4242 (omp)"]}
        with self.assertRaisesRegex(decision.ContextError,
                                    rf"3 resident, cap 3.*{ROUTE_MODEL} \(established Ollama client pid 4242.*"
                                    r"--allow-evict"):
            self.run_suite(fake.url, self.suite("choice", 2))
        self.assertEqual([e["path"] for e in fake.log], ["/api/show"])
        self.assertIn(ROUTE_MODEL, fake.resident)
        self.assertEqual(sorted(self.in_use_calls), sorted(fake.resident))     # every resident was checked

    def test_the_same_load_with_idle_residents_proceeds_and_records_the_eviction(self):
        fake = self.at_cap()
        receipt = self.run_suite(fake.url, self.suite("choice", 2))
        ctx = receipt["run"]["decision"]["context"]
        self.assertEqual((ctx["action"], ctx["at_risk"], ctx["allow_evict"]), ("loaded", {}, False))
        self.assertEqual(ctx["loaded_model_cap"]["cap"], 3)
        self.assertEqual([e["path"] for e in fake.log].count("/api/generate"), 1)
        self.assertTrue(any("evicted" in p and ROUTE_MODEL in p for p in receipt["problems"]))

    def test_allow_evict_loads_past_an_in_use_resident_and_records_it(self):
        fake = self.at_cap()
        self.users_by_model = {ROUTE_MODEL: ["2 gateway request(s) in flight"]}
        receipt = self.run_suite(fake.url, self.suite("choice", 2), allow_evict=True)
        ctx = receipt["run"]["decision"]["context"]
        self.assertEqual((ctx["allow_evict"], ctx["evicted"]), (True, [ROUTE_MODEL]))
        self.assertEqual(ctx["at_risk"], {ROUTE_MODEL: ["2 gateway request(s) in flight"]})
        self.assertTrue(any("evicted" in p and ROUTE_MODEL in p for p in receipt["problems"]))

    def test_below_the_cap_no_resident_is_at_risk(self):
        fake = self.at_cap()
        self.launchd_cap = "4"                        # OLLAMA_MAX_LOADED_MODELS=4 in Ollama.app's launchd env
        fake.evict_on_load = []
        self.users = ["established Ollama client pid 4242 (omp)"]
        receipt = self.run_suite(fake.url, self.suite("choice", 2))
        self.assertEqual(self.in_use_calls, [])
        ctx = receipt["run"]["decision"]["context"]
        self.assertEqual((ctx["action"], ctx["loaded_model_cap"]["cap"]), ("loaded", 4))
        self.assertNotIn("at_risk", ctx)
        self.assertEqual(receipt["problems"], [])

    def test_the_cap_is_launchd_s_value_else_three_per_metal_gpu(self):
        for raw, cap in ((None, 3), ("0", 3), ("1", 1), ("6", 6)):
            with self.subTest(raw=raw):
                self.launchd_cap = raw
                self.assertEqual(decision.loaded_model_cap()["cap"], cap)
        self.launchd_cap = "three"
        with self.assertRaisesRegex(decision.ContextError, "cap is unknown"):
            decision.loaded_model_cap()


class ShippedContext(Base):
    """/v1/systemone reloads a runner at the num_ctx the model ships (2026-10-01: tev1 warm-loaded at 21507 tokens
    was back at its shipped 2050 by item 0, which 400ed): the shipped value decides, not the warm-load."""

    def test_a_suite_longer_than_the_shipped_num_ctx_is_refused_before_any_load_or_item(self):
        fake = self.fake(self.echo)
        fake.models[MODEL]["num_ctx"] = 2050
        suite = self.suite("choice", 3, pad={2: 9000})
        need = decision.required_context(suite)["required_tokens"]
        with self.assertRaisesRegex(decision.ContextError,
                                    rf"ships num_ctx 2050.*needs {need} tokens.*localbench decision derive "
                                    rf"ollama:{MODEL} --num-ctx {need}"):
            self.run_suite(fake.url, suite)
        self.assertEqual([e["path"] for e in fake.log], ["/api/show"])
        self.assertEqual(fake.resident[MODEL], 1 << 15)

    def test_a_model_shipping_enough_context_is_loaded_at_its_shipped_value_and_pinned(self):
        fake = self.fake(self.echo)
        fake.models[MODEL]["num_ctx"] = 40000
        fake.resident[MODEL] = 2048
        suite = self.suite("choice", 3, pad={2: 9000})
        self.assertLess(decision.required_context(suite)["required_tokens"], 40000)
        receipt = self.run_suite(fake.url, suite)
        loads = [e["body"]["options"]["num_ctx"] for e in fake.log if e["path"] == "/api/generate"]
        self.assertEqual(loads, [40000])      # at the required size, the first item would reload it
        pins = receipt["run"]["provenance"]["pins"]
        self.assertEqual((pins["model_num_ctx"], pins["loaded_context"]), (40000, 40000))
        self.assertEqual(receipt["problems"], [])
        self.assertEqual(receipt["run"]["verdicts"]["pins_changed"], {})

    def test_a_resident_runner_at_another_size_than_shipped_is_reloaded(self):
        fake = self.fake(self.echo)
        fake.models[MODEL]["num_ctx"] = 40000          # resident at 1 << 15: big enough, but not what items load
        receipt = self.run_suite(fake.url, self.suite("choice", 2))
        self.assertEqual(receipt["run"]["decision"]["context"]["action"], "reloaded")
        self.assertEqual(receipt["run"]["provenance"]["pins"]["loaded_context"], 40000)
        self.assertEqual(receipt["problems"], [])


class Derive(Base):
    """`decision derive`: FROM the base with parameters {num_ctx: N} via /api/create, read back, base untouched."""

    def test_default_names(self):
        self.assertEqual(decision.derived_name("nimble:latest", 32768), "nimble-ctx32768")
        self.assertEqual(decision.derived_name("tev1", 4096), "tev1-ctx4096")
        self.assertEqual(decision.derived_name("tev1:0.8b", 4096), "tev1:0.8b-ctx4096")

    def test_derive_creates_from_the_base_and_reads_back_the_context_and_the_same_weights(self):
        fake = self.fake(self.echo)
        base = dict(fake.models[MODEL])
        plan = decision.plan_derive(fake.url, MODEL, 32768)
        self.assertEqual((plan["refuse"], plan["noop"]), (None, None))
        self.assertEqual([e["path"] for e in fake.log], ["/api/show"])     # planning reads only
        out = decision.derive(fake.url, plan)
        creates = [e["body"] for e in fake.log if e["path"] == "/api/create"]
        self.assertEqual(creates, [{"model": "nimble-ctx32768", "from": MODEL, "parameters": {"num_ctx": 32768},
                                    "stream": False}])
        self.assertEqual((out["name"], out["num_ctx"], out["weights"]), ("nimble-ctx32768", 32768, "a" * 64))
        self.assertEqual(fake.models[MODEL], base)
        self.assertEqual(fake.models["nimble-ctx32768:latest"]["params"], base["params"])

    def test_a_readback_that_differs_from_what_was_asked_is_refused(self):
        def drop_num_ctx(entry):
            entry["num_ctx"] = None

        def other_weights(entry):
            entry["weights"] = "e" * 64

        def lost_parameters(entry):
            entry["params"] = []

        def touched_base(entry):
            fake.models[MODEL]["digest"] = "d" * 64
        for hook, why in ((drop_num_ctx, "num_ctx reads back None"), (other_weights, "weights eeeeeeeeeeee"),
                          (lost_parameters, "other parameters"), (touched_base, "the base nimble:latest changed")):
            with self.subTest(why):
                fake = self.fake(self.echo)
                fake.create_hook = hook
                plan = decision.plan_derive(fake.url, MODEL, 32768)
                with self.assertRaisesRegex(decision.DeriveError, why):
                    decision.derive(fake.url, plan)

    def test_an_existing_name_is_refused_unless_it_is_the_same_derivation(self):
        fake = self.fake(self.echo)
        decision.derive(fake.url, decision.plan_derive(fake.url, MODEL, 32768))
        again = decision.plan_derive(fake.url, MODEL, 32768)
        self.assertIsNone(again["refuse"])
        self.assertIn("already derives from", again["noop"])
        other_n = decision.plan_derive(fake.url, MODEL, 16384, name="nimble-ctx32768")
        self.assertIn("already exists and is not", other_n["refuse"])
        other_base = decision.plan_derive(fake.url, ROUTE_MODEL, 32768, name="nimble-ctx32768")
        self.assertIn("already exists and is not", other_base["refuse"])
        itself = decision.plan_derive(fake.url, MODEL, 32768, name="nimble")
        self.assertIn("is the base model itself", itself["refuse"])
        for plan in (again, other_n, other_base, itself):
            with self.assertRaises(decision.DeriveError):
                decision.derive(fake.url, plan)
        self.assertEqual(len([e for e in fake.log if e["path"] == "/api/create"]), 1)


# A stand-in for localbench/shims/laya_systemone.py: the same argv, /health and /v1/systemone shapes, stdlib only.
# Behavior comes from the JSON file named by $LAYA_FAKE_CONFIG: startup ok|exit|hang|wrong-model; die_after N POSTs
# (the process exits mid-request); max_len in characters of state (longer states are reported truncated); drift
# (answers move between identical requests); laya_file and model_dir for the pins.
FAKE_SHIM = r'''
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--port", type=int); ap.add_argument("--subfolder")
args = ap.parse_args()
cfg = json.load(open(os.environ["LAYA_FAKE_CONFIG"]))
name = args.model + (f"@{args.subfolder}" if args.subfolder else "")
print("fake laya loading", name, flush=True)
if cfg["startup"] == "exit":
    print("checkpoint not in the cache", flush=True)
    sys.exit(3)
if cfg["startup"] == "hang":
    time.sleep(600)
served = "someone/else" if cfg["startup"] == "wrong-model" else name
posts = 0

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, doc):
        data = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send({"ready": True, "model": served, "laya_version": "0.1.0", "laya_file": cfg["laya_file"],
                   "model_dir": cfg["model_dir"], "dtype": "float16", "batch_size": 16, "max_len": 512,
                   "head_max_len": 192})

    def do_POST(self):
        global posts
        posts += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if cfg["die_after"] is not None and posts > cfg["die_after"]:
            os._exit(4)
        # The label of make_items' noul item i (true for even i), moved toward 0.5 on every POST when drifting.
        true = int(body["state"].rsplit(" ", 1)[1]) % 2 == 0
        p = (0.9 if true else 0.1) + ((-0.01 if true else 0.01) * posts if cfg["drift"] else 0.0)
        long = len(body["state"]) > cfg["max_len"]
        self.send({"model": body["model"], "usage": {"input_tokens": 10, "output_tokens": 0},
                   "answers": {n: {"type": "noul", "noul": p, "laya": {"truncated": long, "head_truncated": False,
                                   "tokens": len(body["state"]), "max_len": cfg["max_len"]}}
                               for n in body["questions"]}})

HTTPServer(("127.0.0.1", args.port), H).serve_forever()
'''
SNAPSHOT = "20aed815fc6acde75733882e7ec0e3f28aeb9717"


class Laya(Base):
    """`laya:<hf repo>[@<subfolder>]`: a loopback shim started for the run and always stopped, pinned by the laya
    checkout's commit and the HF snapshot, truncation counted; here a fake shim run by a venv whose python3 is this
    interpreter."""

    def setUp(self):
        super().setUp()
        venv_bin = self.tmp / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python3").symlink_to(sys.executable)
        self.script = self.tmp / "fake_shim.py"
        self.script.write_text(FAKE_SHIM)
        checkout = self.tmp / "laya-mlx"
        (checkout / "laya_mlx").mkdir(parents=True)
        (checkout / "laya_mlx" / "__init__.py").write_text('__version__ = "0.1.0"\n')
        # A fixture checkout, not a commit of this project: the user's global hooks (core.hooksPath) do not apply.
        git = ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
               "-c", "core.hooksPath=/dev/null"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "--no-verify", "-m", "laya"]):
            subprocess.run(git + cmd, check=True, env=env, capture_output=True)
        self.commit = subprocess.run(git + ["rev-parse", "HEAD"], check=True, env=env, capture_output=True,
                                     text=True).stdout.strip()
        self.cfg = {"startup": "ok", "die_after": None, "max_len": 100, "drift": False,
                    "laya_file": str(checkout / "laya_mlx" / "__init__.py"),
                    "model_dir": str(self.tmp / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / SNAPSHOT)}
        self.spec = decision.parse_laya_spec("laya:aac6fef/laya-mlx")

    def shim(self, ready_timeout: float = 30.0, **cfg) -> decision.LayaShim:
        path = self.tmp / "fake.json"
        path.write_text(json.dumps(self.cfg | cfg))
        patcher = mock.patch.dict(os.environ, {"LAYA_FAKE_CONFIG": str(path)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return decision.LayaShim(self.spec, venv=self.tmp / "venv", script=self.script, ready_timeout=ready_timeout)

    def run_laya(self, suite, shim, **kw):
        return decision.run_laya(self.spec, suite, shim=shim, host=HOST, rev="test", **kw)

    def test_specs_name_a_hub_repo_and_an_optional_relative_subfolder(self):
        self.assertEqual(decision.parse_laya_spec("laya:aac6fef/laya-mlx"), decision.LayaSpec("aac6fef/laya-mlx"))
        sub = decision.parse_laya_spec("laya:convaiinnovations/laya@checkpoints/rl")
        self.assertEqual((sub.repo, sub.subfolder, sub.name),
                         ("convaiinnovations/laya", "checkpoints/rl", "convaiinnovations/laya@checkpoints/rl"))
        for bad in ("laya:", "laya:laya-mlx", "laya:/abs/dir", "laya:~/models/laya", "laya:org/../x",
                    "laya:org/repo@", "laya:org/repo@/abs", "laya:org/repo@a/../../b", "laya:org/repo@a//b",
                    "ollama:nimble:latest"):
            with self.subTest(bad), self.assertRaises(ValueError):
                decision.parse_laya_spec(bad)

    def test_a_run_goes_through_the_shim_pins_it_counts_truncation_and_stops_it(self):
        suite = self.suite("noul", 4, pad={1: 200, 3: 200})
        shim = self.shim()
        receipt = self.run_laya(suite, shim, repeats=2)
        run = receipt["run"]
        self.assertEqual(receipt["problems"], [])
        pins = run["provenance"]["pins"]
        self.assertEqual((pins["backend"], pins["backend_version"], pins["backend_sha"], pins["model"],
                          pins["model_digest"]), ("laya-mlx", "0.1.0", self.commit, "aac6fef/laya-mlx", SNAPSHOT))
        ctx = run["decision"]["context"]
        self.assertEqual((ctx["action"], ctx["max_len"], ctx["items_over_max_len"], ctx["items_truncated"]),
                         ("n/a", 512, 2, 2))
        self.assertEqual(run["metrics"]["decision.laya.truncated_items"]["value"], 2)
        self.assertEqual(run["metrics"][decision.ERROR_RATE]["value"], 0.0)
        self.assertEqual(run["conformance"]["decision.deterministic"]["verdict"], "PASS")
        self.assertEqual(cli.validate_doc(receipt), [])
        # Stopped: the process is reaped and nothing answers on its port any more.
        self.assertIsNotNone(shim.proc.returncode)
        with socket.socket() as s:
            self.assertNotEqual(s.connect_ex(("127.0.0.1", shim.port)), 0)

    def test_a_shim_whose_answers_drift_between_repeats_fails_the_determinism_must(self):
        receipt = self.run_laya(self.suite("noul", 2), self.shim(drift=True), repeats=2)
        self.assertEqual(receipt["run"]["conformance"]["decision.deterministic"]["verdict"], "FAIL")
        self.assertEqual(receipt["run"]["verdicts"]["must_fail"], ["decision.deterministic"])

    def test_a_shim_that_dies_mid_run_gives_availability_errors_not_invalid(self):
        shim = self.shim(die_after=2)
        receipt = self.run_laya(self.suite("noul", 5), shim)
        run = receipt["run"]
        errors = [o for o in run["decision"]["local"]["outcomes"] if not o["ok"]]
        self.assertEqual(len(errors), 3)
        self.assertEqual({o["error"] for o in errors}, {"unavailable"})
        self.assertEqual(run["conformance"]["decision.responses_valid"]["verdict"], "PASS")
        self.assertEqual(run["metrics"][decision.ERROR_RATE]["value"], 0.6)
        self.assertFalse(run["decision"]["context"]["shim_alive_after_run"])
        self.assertEqual(shim.proc.returncode, 4)

    def test_a_shim_that_cannot_become_ready_is_a_shim_error_and_is_reaped(self):
        for startup, timeout, why in (("exit", 30.0, r"exited with status 3 before it was ready:[\s\S]*checkpoint "
                                                     "not in the cache"),
                                      ("hang", 1.0, "not ready within 1s"),
                                      ("wrong-model", 30.0, "not a ready shim for aac6fef/laya-mlx")):
            with self.subTest(startup):
                shim = self.shim(ready_timeout=timeout, startup=startup)
                t0 = time.monotonic()
                with self.assertRaisesRegex(decision.ShimError, why):
                    self.run_laya(self.suite("noul", 1), shim)
                self.assertLess(time.monotonic() - t0, 15.0)
                self.assertIsNotNone(shim.proc.returncode)
        missing = decision.LayaShim(self.spec, venv=self.tmp / "no-venv", script=self.script)
        with self.assertRaisesRegex(decision.ShimError, "no python3 in the laya venv"):
            self.run_laya(self.suite("noul", 1), missing)

    def test_the_shim_is_stopped_when_the_run_raises(self):
        shim = self.shim()
        with mock.patch.object(decision, "run_suite", side_effect=RuntimeError("boom")), \
                self.assertRaisesRegex(RuntimeError, "boom"):
            self.run_laya(self.suite("noul", 1), shim)
        self.assertIsNotNone(shim.proc.returncode)

    def test_a_checkpoint_outside_the_hf_cache_or_a_dirty_checkout_is_not_pinned(self):
        receipt = self.run_laya(self.suite("noul", 1), self.shim(model_dir=str(self.tmp / "models" / "laya")))
        self.assertIsNone(receipt["run"]["provenance"]["pins"]["model_digest"])
        self.assertTrue(any("not a Hugging Face cache snapshot" in p for p in receipt["problems"]))
        Path(self.cfg["laya_file"]).write_text('__version__ = "0.1.1"\n')
        receipt = self.run_laya(self.suite("noul", 1), self.shim())
        self.assertEqual(receipt["run"]["provenance"]["pins"]["backend_sha"], self.commit + "-dirty")
        self.assertTrue(any("uncommitted changes" in p for p in receipt["problems"]))


class Paired(Base):
    def _items(self, kind, labels, refs):
        questions = copy.deepcopy(QUESTIONS[kind])
        name = next(iter(questions))
        return [{"id": f"{kind}-{i}", "state": f"synthetic {kind} item {i}", "questions": questions,
                 "labels": {name: label}, "reference": {name: ref}, "source": "synthetic:test_paired"}
                for i, (label, ref) in enumerate(zip(labels, refs))]

    def _suite(self, kind, labels, refs):
        return decision.write_suite(self.tmp / f"paired-{kind}", name=f"paired-{kind}",
                                    role=f"decision.{kind}", items=self._items(kind, labels, refs),
                                    sources=[self.source],
                                    gate={f"decision.{kind}.accuracy": {"min": 0.5}})

    def _arm(self, suite, answers):
        outcomes = []
        for it in suite.items:
            if it["id"] not in answers:
                continue
            name = next(iter(it["questions"]))
            pick = answers[it["id"]]
            if pick == "ERR":
                outcomes.append({"id": it["id"], "repeat": 0, "ok": False, "answers": {}})
            else:
                use = it["labels"][name] if pick == "label" else pick
                raw = use if isinstance(use, dict) else answer(it["questions"][name], use)
                outcomes.append({"id": it["id"], "repeat": 0, "ok": True,
                                 "answers": {name: raw}})
        return {"arm": "local", "model": "test", "endpoint": "http://127.0.0.1:9",
                "suite": suite.pin(), "repeats": 1, "has_reference": True,
                "outcomes": outcomes, "items": []}

    def _report(self, arm, **kw):
        kw.setdefault("resamples", 2000)
        return decision.paired(arm, corpora=self.tmp, **kw)

    def test_exact_mcnemar_p_is_known(self):
        table = decision._paired_table([(True, False)] * 8 + [(False, True)] * 2, 0.05, 1, 2000)
        self.assertEqual(table["mcnemar_p"], 0.109375)
        self.assertEqual((table["b"], table["c"], table["n"]), (8, 2, 10))
        self.assertEqual(table["verdict"], "NOT_WORSE")
        silent = decision._paired_table([(True, True)] * 3, 0.05, 1, 2000)
        self.assertEqual((silent["mcnemar_p"], silent["verdict"]), (1.0, "NOT_WORSE"))

    def test_better_and_worse_ends(self):
        better = decision._paired_table([(True, False)] * 6, 0.05, 1, 2000)
        self.assertEqual((better["mcnemar_p"], better["verdict"]), (0.03125, "BETTER"))
        worse = decision._paired_table([(False, True)] * 6, 0.05, 1, 2000)
        self.assertEqual((worse["mcnemar_p"], worse["verdict"]), (0.03125, "WORSE"))

    def test_noul_end_to_end_through_a_run_receipt(self):
        labels = [i % 2 == 0 for i in range(6)]
        suite = self._suite("noul", labels, [answer(QUESTIONS["noul"]["unsafe"], label) for label in labels])
        q = QUESTIONS["noul"]["unsafe"]
        arm = self._arm(suite, {f"noul-{i}": wrong(q, labels[i]) for i in range(6)})
        report = self._report({"kind": "run", "run": {"decision": {"local": arm}}})
        row = report["types"]["noul"]
        self.assertEqual((row["n"], row["local_accuracy"], row["hosted_accuracy"]), (6, 0.0, 1.0))
        self.assertEqual((row["b"], row["c"], row["mcnemar_p"], row["verdict"]), (0, 6, 0.03125, "WORSE"))
        self.assertEqual(report["suite"]["name"], "paired-noul")

    def test_noul_half_counts_as_non_destructive(self):
        suite = self._suite("noul", [False], [{"type": "noul", "noul": 0.5}])
        arm = self._arm(suite, {"noul-0": {"type": "noul", "noul": 0.5}})
        row = self._report(arm)["types"]["noul"]
        self.assertEqual((row["local_accuracy"], row["hosted_accuracy"], row["verdict"]),
                         (1.0, 1.0, "NOT_WORSE"))

    def test_score_and_choice_rules(self):
        suite = self._suite("score", [0, 2], [answer(QUESTIONS["score"]["quality"], 0),
                                              answer(QUESTIONS["score"]["quality"], 2)])
        arm = self._arm(suite, {"score-0": 0, "score-1": 0})
        row = self._report(arm)["types"]["score"]
        self.assertEqual((row["n"], row["local_accuracy"], row["hosted_accuracy"]), (2, 0.5, 1.0))
        suite = self._suite("choice", ["card", "loan"], [answer(QUESTIONS["choice"]["intent"], "card"),
                                                         answer(QUESTIONS["choice"]["intent"], "loan")])
        arm = self._arm(suite, {"choice-0": "card", "choice-1": "card"})
        row = self._report(arm)["types"]["choice"]
        self.assertEqual((row["n"], row["local_accuracy"], row["hosted_accuracy"]), (2, 0.5, 1.0))

    def test_error_outcome_scores_incorrect(self):
        suite = self._suite("noul", [True], [answer(QUESTIONS["noul"]["unsafe"], True)])
        arm = self._arm(suite, {"noul-0": "ERR"})
        row = self._report(arm)["types"]["noul"]
        self.assertEqual((row["local_accuracy"], row["hosted_accuracy"], row["verdict"]),
                         (0.0, 1.0, "NOT_WORSE"))

    def test_bootstrap_ci_is_deterministic_and_ordered(self):
        first = decision._paired_table([(True, False)] * 8 + [(False, True)] * 2, 0.05, 7, 10000)
        second = decision._paired_table([(True, False)] * 8 + [(False, True)] * 2, 0.05, 7, 10000)
        self.assertEqual(first["bootstrap"], second["bootstrap"])
        boot = first["bootstrap"]
        self.assertLessEqual(boot["lo_pp"], boot["mean_pp"])
        self.assertLessEqual(boot["mean_pp"], boot["hi_pp"])

    def test_refusals_name_the_gap(self):
        suite = self._suite("noul", [True, False], [answer(QUESTIONS["noul"]["unsafe"], True),
                                                   answer(QUESTIONS["noul"]["unsafe"], False)])
        arm = self._arm(suite, {"noul-0": True, "noul-1": False})
        arm["outcomes"] = [o for o in arm["outcomes"] if o["id"] != "noul-1"]
        with self.assertRaisesRegex(decision.PairedError, "no local outcome"):
            decision.paired({"kind": "run", "run": {"decision": {"local": arm}}},
                            corpora=self.tmp, resamples=10)
        bad = dict(arm["suite"])
        bad["items_sha256"] = "0" * 64
        with self.assertRaisesRegex(decision.PairedError, "not on disk"):
            decision.paired({**arm, "suite": bad}, corpora=self.tmp, resamples=10)
        with self.assertRaisesRegex(decision.PairedError, "not a decision run"):
            decision.paired({"kind": "run"}, corpora=self.tmp, resamples=10)
        stripped = []
        for line in suite.items_path.read_text().splitlines():
            if line.strip():
                stripped.append({k: v for k, v in json.loads(line).items() if k != "reference"})
        decision.write_suite(self.tmp / "paired-noref", name="paired-noref", role="decision.noul",
                             items=stripped, sources=[self.source],
                             gate={"decision.noul.accuracy": {"min": 0.5}})
        no_ref_suite = decision.load_suite(self.tmp / "paired-noref")
        with self.assertRaisesRegex(decision.PairedError, "no hosted reference"):
            decision.paired(self._arm(no_ref_suite, {}), corpora=self.tmp, resamples=10)

    def test_malformed_local_answer_refuses(self):
        suite = self._suite("choice", ["card"], [answer(QUESTIONS["choice"]["intent"], "card")])
        arm = self._arm(suite, {"choice-0": "card"})
        arm["outcomes"][0]["answers"]["intent"] = {"type": "choice", "choice": "zzz",
                                                   "probabilities": {"card": 0.5, "loan": 0.25, "fx": 0.25},
                                                   "confidence": 0.5}
        with self.assertRaisesRegex(decision.PairedError, "malformed local outcome"):
            self._report(arm)

    def test_cli_json_and_refusal(self):
        import argparse
        suite = self._suite("noul", [True, False], [answer(QUESTIONS["noul"]["unsafe"], True),
                                                   answer(QUESTIONS["noul"]["unsafe"], False)])
        arm = self._arm(suite, {"noul-0": True, "noul-1": False})
        receipt = self.tmp / "receipt.json"
        receipt.write_text(json.dumps({"kind": "run", "run": {"decision": {"local": arm}}}))
        args = argparse.Namespace(receipt=str(receipt), alpha=0.05, seed=7, json=True)
        real_corpora, decision.CORPORA = decision.CORPORA, self.tmp
        try:
            out = self._capture(cli.cmd_decision_paired, args)
        finally:
            decision.CORPORA = real_corpora
        self.assertEqual(out["rc"], 0)
        self.assertEqual(out["doc"]["types"]["noul"]["verdict"], "NOT_WORSE")
        bad = dict(arm["suite"])
        bad["items_sha256"] = "0" * 64
        receipt.write_text(json.dumps({"kind": "run", "run": {"decision": {"local": {**arm, "suite": bad}}}}))
        real_corpora, decision.CORPORA = decision.CORPORA, self.tmp
        try:
            out = self._capture(cli.cmd_decision_paired, args)
        finally:
            decision.CORPORA = real_corpora
        self.assertEqual(out["rc"], 1)

    def _capture(self, fn, args):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = fn(args)
        text = buf.getvalue()
        try:
            doc = json.loads(text)
        except ValueError:
            doc = None
        return {"rc": rc, "text": text, "doc": doc}

if __name__ == "__main__":
    unittest.main()
