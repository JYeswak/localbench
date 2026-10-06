"""Generation-corpus tests. All request bodies are synthetic shapes mirroring omp's call
sites; no captured user content may appear here."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as main_mod
from localbench import corpus, decision, generation, heavyslot, prove
from tests.test_heavyslot import spawn_holder

TITLE = {"model": "qwen3.8:27b-mlx",
         "messages": [{"role": "system", "content": "Write a ~5 word title."},
                      {"role": "user", "content": "<user>\nDo the thing\n</user>\n<title>"}]}
SKILL = {"model": "qwen3.8:27b-mlx",
         "messages": [{"role": "user",
                       "content": "Compress into one routing hint of at most 12 words:\n\nSkill: x\nDescription: Routes database migrations for deploy previews across staging and production"}]}
COMMIT = {"model": "qwen3.8:27b-mlx",
          "messages": [{"role": "system",
                        "content": "Senior engineer writing a conventional commit message."},
                       {"role": "user", "content": "diff --git a/b"}]}
OTHER = {"model": "qwen3.8:27b-mlx", "messages": [{"role": "user", "content": "hello"}]}


def capture_doc(body: dict) -> str:
    return json.dumps({"meta": {"purpose": "aux"}, "request": body})


class Classify(unittest.TestCase):
    def test_signatures_route_each_feature_and_leave_the_rest(self):
        self.assertEqual(generation.classify(TITLE), "titles")
        self.assertEqual(generation.classify(SKILL), "skill-description-compression")
        self.assertEqual(generation.classify(COMMIT), "commit-messages")
        self.assertIsNone(generation.classify(OTHER))
        self.assertIsNone(generation.classify(
            {"messages": [{"role": "user", "content": "a bare <title> without the envelope"}]}))

    def test_collect_groups_captures_and_parks_the_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(capture_doc(TITLE))
            (root / "b.json").write_text(capture_doc(SKILL))
            (root / "c.json").write_text(capture_doc(OTHER))
            (root / "d.response.json").write_text(capture_doc(TITLE))
            (root / "e.json").write_text("not json")
            (root / "f.json").write_text(json.dumps(
                {"meta": {"purpose": "aux", "profile": "codex"}, "request": TITLE}))
            grouped = generation.collect(root)
        self.assertEqual([r["body"] for r in grouped["titles"]], [TITLE, TITLE])
        self.assertEqual(grouped["titles"][1]["meta"]["profile"], "codex")
        self.assertEqual([r["body"] for r in grouped["skill-description-compression"]], [SKILL])
        self.assertEqual([r["body"] for r in grouped["other"]], [OTHER])
        self.assertNotIn("commit-messages", grouped)


class GatewayRoute(unittest.TestCase):
    def test_replay_url_follows_body_shape_under_the_profile(self):
        responses_body = {"model": "m", "input": [{"role": "user", "content": "hi"}]}
        self.assertEqual(generation.gateway_url("codex", responses_body),
                         "http://127.0.0.1:11300/omp-profile/codex/responses")
        chat_body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(generation.gateway_url("codex", chat_body),
                         "http://127.0.0.1:11300/omp-profile/codex/chat/completions")
        for bad in ("", "a/b", ".", "..", None, 42):
            with self.subTest(profile=bad), self.assertRaises(corpus.CorpusError):
                generation.gateway_url(bad, chat_body)


class Assemble(unittest.TestCase):
    def test_assemble_writes_pinned_manifest_and_content_addressed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "corpora" / "generation" / "titles-001"
            report = generation.assemble("titles", [{"body": TITLE, "source": "cap:1",
                                                      "meta": {"profile": "codex"}},
                                                    {"body": SKILL, "source": "cap:2"}],
                                         dest, seed=20261001)
            manifest = json.loads((dest / "manifest.json").read_text())
            self.assertEqual(manifest["feature"], "titles")
            self.assertEqual(manifest["gate"], {"min_items": generation.MIN_ITEMS})
            self.assertEqual(manifest["n_items"], 2)
            self.assertEqual(manifest["items_sha256"], report["items_sha256"])
            self.assertEqual(manifest["seed"], 20261001)
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(oct((dest / "items.jsonl").stat().st_mode & 0o777), "0o400")
            lines = (dest / "items.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["feature"], "titles")
            self.assertEqual(first["body"], TITLE)
            self.assertEqual(len(first["id"]), 64)
            self.assertEqual(first["profile"], "codex")
            self.assertIsNone(json.loads(lines[1])["profile"])

    def test_assembled_corpus_is_read_only_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "generation" / "skill-001"
            generation.assemble("skill-description-compression",
                               [{"body": SKILL, "source": "cap:1"}], dest, seed=20261003)
            items_path, manifest_path = dest / "items.jsonl", dest / "manifest.json"
            items_before, manifest_before = items_path.read_bytes(), manifest_path.read_bytes()
            try:
                for path in (dest, items_path, manifest_path):
                    self.assertEqual(path.stat().st_mode & 0o222, 0)
                with self.assertRaisesRegex(corpus.CorpusError, "already exists"):
                    generation.assemble("skill-description-compression",
                                        [{"body": SKILL, "source": "cap:2"}], dest, seed=20261004)
                self.assertEqual(items_path.read_bytes(), items_before)
                self.assertEqual(manifest_path.read_bytes(), manifest_before)
                self.assertEqual(dest.stat().st_mode & 0o222, 0)
            finally:
                items_path.chmod(0o600)
                manifest_path.chmod(0o600)
                dest.chmod(0o700)

    def test_unknown_feature_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "corpora" / "generation" / "nope-001"
            with self.assertRaises(corpus.CorpusError):
                generation.assemble("nope", [], dest, seed=1)
            self.assertFalse(dest.exists())

    def test_bad_seed_or_version_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kwargs in ({"seed": "20261001"}, {"seed": 20261001, "version": 0}):
                dest = Path(tmp) / "corpora" / "generation" / "bad-001"
                with self.assertRaises(corpus.CorpusError):
                    generation.assemble("titles", [], dest, **kwargs)
                self.assertFalse(dest.exists())


class ProofDryRun(unittest.TestCase):
    def test_generation_dry_run_refuses_a_stale_corpus_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "generation" / "dryrun-001"
            built = generation.assemble("skill-description-compression",
                                        [{"body": SKILL, "source": "cap:1"}], dest, seed=1)
            spec = {"kind": "generation", "feature": "skill-description-compression",
                    "dataset": {"suite": dest.name, "items_sha256": built["items_sha256"]},
                    "candidates": [{"builtin": "previewSkillDescription"}]}
            spec_path = "registries/proofs/skill-description-compression__screen.json"

            def run_dry_run(loaded_spec):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (mock.patch.object(prove, "load_spec", return_value=loaded_spec),
                      mock.patch.object(prove, "spec_commit", return_value="c" * 40),
                      mock.patch.object(decision, "CORPORA", root),
                      contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                    rc = main_mod.main(["prove", spec_path, "--dry-run"])
                return rc, stdout.getvalue(), stderr.getvalue()

            try:
                rc, stdout, _ = run_dry_run(spec)
                self.assertEqual(rc, 0)
                self.assertIn("resolve dataset dryrun-001 pin", stdout)

                stale = {**spec, "dataset": {**spec["dataset"], "items_sha256": "0" * 64}}
                rc, stdout, stderr = run_dry_run(stale)
                self.assertEqual(rc, 1)
                self.assertEqual(stdout, "")
                self.assertIn("register a new corpus id", stderr)
            finally:
                (dest / "items.jsonl").chmod(0o600)
                (dest / "manifest.json").chmod(0o600)
                dest.chmod(0o700)


class CorpusProgress(unittest.TestCase):
    def _roots(self, tmp):
        cap = Path(tmp) / "captured" / "aux"
        cap.mkdir(parents=True)
        gen = Path(tmp) / "generation"
        return cap, gen

    def _doc(self, body, stamp):
        return json.dumps({"meta": {"purpose": "aux", "t": stamp}, "request": body})

    def test_progress_counts_corpus_captured_and_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, gen = self._roots(tmp)
            (cap / "t1.json").write_text(self._doc(TITLE, 1759276800))
            (cap / "t2.json").write_text(self._doc(TITLE, 1759363200))
            dest = gen / "titles-aux-001"
            generation.assemble("titles", [{"body": TITLE, "source": "cap:t1", "meta": {}},
                                           {"body": COMMIT, "source": "cap:t2", "meta": {}}],
                                dest, seed=1)
            rows = generation.corpus_progress(cap, gen)
        titles = next(r for r in rows if r["feature"] == "titles")
        self.assertEqual(titles["corpus"], 2)
        self.assertEqual(titles["gate"], generation.MIN_ITEMS)
        self.assertFalse(titles["ready"])
        self.assertEqual(titles["captured"], 2)
        self.assertEqual((titles["oldest"], titles["newest"]), ("2025-10-01", "2025-10-02"))
        commits = next(r for r in rows if r["feature"] == "commit-messages")
        self.assertEqual((commits["corpus"], commits["captured"],
                          commits["oldest"], commits["newest"]), (0, 0, None, None))
        self.assertEqual([r["feature"] for r in rows],
                         [name for name, _ in generation.FEATURE_SIGNATURES])

    def test_progress_ready_at_gate_and_lines_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, gen = self._roots(tmp)
            bodies = [dict(TITLE, seed=i) for i in range(3)]
            for i, body in enumerate(bodies):
                (cap / f"t{i}.json").write_text(self._doc(body, 1759276800))
            generation.assemble("titles", [{"body": bodies[0], "source": "cap:t0", "meta": {}},
                                           {"body": bodies[1], "source": "cap:t1", "meta": {}}],
                                gen / "titles-aux-001", seed=1)
            with mock.patch.object(generation, "MIN_ITEMS", 2):
                rows = generation.corpus_progress(cap, gen)
                lines = generation.corpus_progress_lines(rows)
        titles = next(r for r in rows if r["feature"] == "titles")
        self.assertEqual((titles["corpus"], titles["captured"]), (2, 3))
        self.assertTrue(titles["ready"])
        self.assertIn("corpus titles: 2/2 (captured 3, 2025-10-01..2025-10-01) READY", lines)

    def test_progress_ignores_non_generation_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap, gen = self._roots(tmp)
            other = gen / "other-001"
            other.mkdir(parents=True)
            (other / "manifest.json").write_text(json.dumps({"kind": "suite"}))
            rows = generation.corpus_progress(cap, gen)
        self.assertTrue(all(r["corpus"] == 0 for r in rows))

class BuiltinArms(unittest.TestCase):
    def test_title_falls_back_to_cwd_basename_or_stays_unnamed(self):
        self.assertEqual(generation.builtin_title(None, "~/Dev"), "Dev")
        self.assertEqual(generation.builtin_title("My Session", "/x"), "My Session")
        self.assertIsNone(generation.builtin_title(None, "/"))
        self.assertIsNone(generation.builtin_title(None, ""))
        self.assertEqual(generation.builtin_title("   ", "/a"), "a")

    def test_skill_preview_truncates_like_omp(self):
        self.assertEqual(generation.builtin_skill_description("Short desc"), "Short desc")
        self.assertEqual(generation.builtin_skill_description("x" * 100), "x" * 100)
        long_text = ("This is a fairly long skill description that definitely exceeds one hundred "
                     "characters by a good margin and keeps going.")
        self.assertEqual(generation.builtin_skill_description(long_text),
                         "This is a fairly long skill description that definitely exceeds one hundred "
                         "characters by a good\u2026")

    def test_commit_without_a_model_is_a_typed_error(self):
        with self.assertRaises(generation.BuiltinUnavailable):
            generation.builtin_commit()


class DeterministicChecks(unittest.TestCase):
    def test_title_shape_violations(self):
        self.assertEqual(generation.check_title("<title>Do the thing now please</title>"), [])
        self.assertIn("exactly one", generation.check_title("no markers")[0])
        self.assertIn("3-8", generation.check_title("<title>Hi</title>")[0])
        self.assertIn("quoted", generation.check_title('<title>"Do the thing now"</title>')[0])
        self.assertIn("prefix", generation.check_title("<title>Title: do the thing now</title>")[0])
        self.assertIn("single line",
                      generation.check_title("<title>Do the\nthing now please</title>")[0])

    def test_commit_shape_violations(self):
        good = "feat(auth): add token refresh on expiry"
        self.assertEqual(generation.check_commit_message(good, ["src/auth/token.ts"]), [])
        self.assertIn("not type", generation.check_commit_message("WIP stuff", [])[0])
        self.assertIn("72", generation.check_commit_message("feat: " + "x" * 73, [])[0])
        self.assertIn("period", generation.check_commit_message("feat: add thing.", [])[0])
        self.assertIn("fences", generation.check_commit_message("```\nfeat: add thing\n```", [])[0])
        self.assertIn("scope", generation.check_commit_message("feat(qq): add thing",
                                                               ["src/auth/token.ts"])[0])

    def test_skill_compression_violations(self):
        source = "Compresses database migration files for deploy previews in CI pipelines daily"
        self.assertEqual(generation.check_skill_compression("Database migration deploy previews",
                                                            source), [])
        self.assertIn("single line",
                      generation.check_skill_compression("line one\nline two", source)[0])
        self.assertIn("12", generation.check_skill_compression("w " * 13, source)[0])
        self.assertIn("0.4", generation.check_skill_compression("totally unrelated words here",
                                                               source)[0])
        self.assertIn("empty", generation.check_skill_compression("   ", source)[0])


class JudgeHook(unittest.TestCase):
    def test_order_and_swap_are_deterministic_and_cover_both(self):
        ids = [f"item-{i:03d}" for i in range(20)]
        first = generation.assign_order(ids, "seed-1")
        self.assertEqual(generation.assign_order(ids, "seed-1"), first)
        self.assertEqual(generation.assign_order(ids, "seed-2") == first, False)
        self.assertIn("AB", set(first.values()))
        self.assertIn("BA", set(first.values()))
        swapped = generation.swap_subset(ids, "seed-1", 0.3)
        self.assertEqual(swapped, generation.swap_subset(ids, "seed-1", 0.3))
        self.assertEqual(len(swapped), 6)

    def test_win_lower_bound_math(self):
        self.assertAlmostEqual(generation.win_lower_bound(60, 10, 100), 0.65 - 1.96 * (0.65 * 0.35 / 100) ** 0.5)
        self.assertEqual(generation.win_lower_bound(0, 0, 0), 0.0)
        self.assertLess(generation.win_lower_bound(45, 10, 100), 0.45)

class JudgeCalibration(unittest.TestCase):
    def test_reports_arm_verbosity_adjusted_rate_and_swap_consistency(self):
        pairs = [
            {"item_id": "long-win", "candidate_text": "a b c d e f g h",
             "baseline_text": "a b", "winner": "candidate", "swap_winner": "candidate"},
            {"item_id": "short-loss", "candidate_text": "a b", "baseline_text": "a b c d e f g h",
             "winner": "baseline", "swap_winner": "baseline"},
            {"item_id": "equal-tie", "candidate_text": "a b c d e", "baseline_text": "a b c d e",
             "winner": "tie", "swap_winner": "tie"},
            {"item_id": "long-inconsistent", "candidate_text": "a b c d e f g h",
             "baseline_text": "a b", "winner": "candidate", "swap_winner": "baseline"},
        ]
        report = generation.pairwise_calibration(pairs)
        self.assertAlmostEqual(report["raw_win_rate"], 0.625)
        self.assertAlmostEqual(report["verbosity"]["candidate"]["mean_tokens"], 5.75)
        self.assertAlmostEqual(report["verbosity"]["baseline"]["mean_tokens"], 4.25)
        self.assertEqual(report["swap_consistency"], {"n": 4, "consistent": 3, "rate": 0.75})
        self.assertIn("length_adjusted_win_rate", report)
        self.assertGreaterEqual(report["length_adjusted_win_rate"], 0.0)
        self.assertLessEqual(report["length_adjusted_win_rate"], 1.0)

    def test_calibrated_receipt_rejects_raw_win_rate_only(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            generation.require_pairwise_calibration({"raw_win_rate": 0.8})
        report = generation.pairwise_calibration([
            {"item_id": "one", "candidate_text": "short", "baseline_text": "short",
             "winner": "candidate", "swap_winner": "candidate"},
            {"item_id": "two", "candidate_text": "short", "baseline_text": "short",
             "winner": "baseline", "swap_winner": "baseline"},
        ])
        generation.require_pairwise_calibration({"judge_calibration": report})

    def test_missing_or_invalid_swap_outcomes_are_refused(self):
        pair = {"item_id": "one", "candidate_text": "a", "baseline_text": "b",
                "winner": "candidate"}
        with self.assertRaisesRegex(ValueError, "swap"):
            generation.pairwise_calibration([pair])
        with self.assertRaisesRegex(ValueError, "winner"):
            generation.pairwise_calibration([{**pair, "swap_winner": "maybe"}])



class Replay(unittest.TestCase):
    def test_replay_body_sets_only_the_model(self):
        out = generation.replay_body({"messages": [], "temperature": 0}, "m")
        self.assertEqual(out, {"messages": [], "temperature": 0, "model": "m"})

    def test_non_loopback_targets_are_refused(self):
        with self.assertRaises(corpus.CorpusError):
            generation.post_loopback("https://example.com/v1", {}, timeout=1)

    def test_loopback_round_trip_records_latency(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _Echo(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                raw = self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = generation.post_loopback(f"http://127.0.0.1:{server.server_port}/x",
                                              {"model": "m"})
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["body"], {"model": "m"})
        self.assertGreaterEqual(result["latency_s"], 0.0)


class ResponseText(unittest.TestCase):
    def test_each_route_shape_resolves_its_text_path(self):
        self.assertEqual(generation.response_text({"choices": [{"message": {"content": "hi"}}]}),
                         "hi")
        blocks = {"choices": [{"message": {"content": [{"text": "a"}, {"nope": 1},
                                                      {"text": "b"}]}}]}
        self.assertEqual(generation.response_text(blocks), "ab")
        self.assertEqual(generation.response_text({"choices": [{"text": "done"}]}), "done")
        self.assertEqual(generation.response_text({"output": [{"content": [{"text": "out"}]}]}),
                         "out")
        self.assertEqual(generation.response_text({"response": "gen"}), "gen")

    def test_missing_text_is_none_never_empty(self):
        self.assertIsNone(generation.response_text({}))
        self.assertIsNone(generation.response_text({"choices": []}))
        self.assertIsNone(generation.response_text({"choices": [{"message": {}}]}))
        self.assertIsNone(generation.response_text({"output": [{"content": []}]}))
        self.assertIsNone(generation.response_text("not-a-body"))
        self.assertIsNone(generation.response_text(None))


class FeatureChecksMap(unittest.TestCase):
    def test_checks_live_in_generation_with_a_uniform_call_shape(self):
        self.assertEqual(set(generation.FEATURE_CHECKS),
                         {"titles", "commit-messages", "skill-description-compression"})
        self.assertEqual(generation.FEATURE_CHECKS["titles"]("<title>Do the thing now</title>",
                                                             {}), [])
        self.assertEqual(generation.FEATURE_CHECKS["commit-messages"](
            "feat(auth): add token refresh", {"diff_files": ["src/auth/x.ts"]}), [])
        self.assertEqual(generation.FEATURE_CHECKS["skill-description-compression"](
            "Database migration deploy previews",
            {"description": "Compresses database migration files for deploy previews daily"}), [])

class JudgeView(unittest.TestCase):
    def test_view_is_the_checked_core_span_only(self):
        self.assertEqual(generation.judge_view(
            "titles", "Here is your title:\n<title>Do the thing now please</title>\nHope it helps"),
            "Do the thing now please")
        self.assertEqual(generation.judge_view(
            "commit-messages", "feat(auth): add token refresh\n\nA long body nobody judges here",
            {"diff_files": []}), "feat(auth): add token refresh")
        self.assertEqual(generation.judge_view("skill-description-compression",
                                               "  Migration file compressor  "),
                         "Migration file compressor")
        with self.assertRaises(corpus.CorpusError):
            generation.judge_view("nope", "text")


class JudgeIsolation(unittest.TestCase):
    FAMS = {"ollama/qwen3.8:27b-mlx": "qwen3",
            "ollama/qwen3.6:35b-mlx": "qwen3", "nimble:latest": "qwen3",
            "ollama/llama3.1:8b": "llama", "z-ai/glm-5.3-flash": "glm"}

    def test_isolated_judge_differs_from_both_arms(self):
        arms = ["ollama/qwen3.8:27b-mlx", "ollama/qwen3.6:35b-mlx"]
        self.assertTrue(generation.judge_isolated("ollama/llama3.1:8b", arms, self.FAMS))
        self.assertFalse(generation.judge_isolated("nimble:latest", arms, self.FAMS))
        self.assertFalse(generation.judge_isolated("ollama/qwen3.8:27b-mlx", arms, self.FAMS))
        self.assertFalse(generation.judge_isolated("ollama/llama3.1:8b", arms, {}))
        self.assertFalse(generation.judge_isolated(
            "ollama/llama3.1:8b", ["ollama/qwen3.8:27b-mlx", "mystery:model"], self.FAMS))


class Answerable(unittest.TestCase):
    def test_empty_descriptions_are_unanswerable(self):
        good = {"messages": [{"role": "user", "content": "Compress now\nDescription: Routes database migrations"}]}
        self.assertTrue(generation.answerable("skill-description-compression", good))
        self.assertFalse(generation.answerable(
            "skill-description-compression",
            {"messages": [{"role": "user", "content": "Compress now\nDescription: y"}]}))
        self.assertFalse(generation.answerable("skill-description-compression", {"messages": []}))
        self.assertTrue(generation.answerable("titles", {"messages": []}))

    def test_assemble_skips_unanswerable_and_counts_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "corpora" / "generation" / "skill-001"
            report = generation.assemble(
                "skill-description-compression",
                [{"body": {"messages": [{"role": "user", "content": "Compress now\nDescription: y"}]},
                  "source": "cap:thin"}],
                dest, seed=7)
            self.assertEqual(report["items"], 0)
            self.assertEqual(report["skipped_unanswerable"], 1)


class CommitBodyRule(unittest.TestCase):
    def test_long_body_lines_fail_shape(self):
        self.assertEqual(generation.check_commit_message(
            "feat(auth): add token refresh\n\nShort body line.", ["src/auth/x.ts"]), [])
        violations = generation.check_commit_message(
            "feat(auth): add token refresh\n\n" + "x" * 73, ["src/auth/x.ts"])
        self.assertEqual(len(violations), 1)
        self.assertIn("body line 3", violations[0])


class EmptyResponse(unittest.TestCase):
    def test_empty_string_is_none(self):
        self.assertIsNone(generation.response_text({"choices": [{"message": {"content": ""}}]}))

class RunCandidate(unittest.TestCase):
    SKILL_ITEM = {"id": "s1", "feature": "skill-description-compression",
                  "body": {"model": "qwen3.8:27b-mlx",
                           "messages": [{"role": "user", "content":
                                         "Compress into one routing hint of at most 12 words:\n\n"
                                         "Skill: x\nDescription: Routes database migrations for deploy previews "
                                         "across staging and production"}]},
                  "source": "cap:1", "profile": "codex"}
    SPEC = {"kind": "generation", "feature": "skill-description-compression"}

    def _post(self, calls):
        def post(url, body, timeout=120.0):
            calls.append((url, body))
            return {"status": 200,
                    "body": {"choices": [{"message": {"content": "Database migrations deploy previews"}}]},
                    "latency_s": 0.5}
        return post

    def test_model_arm_replays_through_its_gateway_profile(self):
        calls = []
        out = generation.run_candidate(self.SPEC, {"route": "ollama:qwen3.6:35b-mlx"},
                                       [self.SKILL_ITEM], post=self._post(calls))
        self.assertEqual(out["model"], "qwen3.6:35b-mlx")
        self.assertIsNone(out["builtin"])
        self.assertEqual(len(out["outcomes"]), 1)
        (url, body), = calls
        self.assertEqual(url, "http://127.0.0.1:11300/omp-profile/codex/chat/completions")
        self.assertEqual(body["model"], "qwen3.6:35b-mlx")
        outcome = out["outcomes"][0]
        self.assertEqual(outcome["text"], "Database migrations deploy previews")
        self.assertEqual(outcome["violations"], [])
        self.assertEqual(outcome["latency_s"], 0.5)

    def test_mlx_serve_arm_starts_stops_and_replays_only_to_loopback(self):
        calls, lifecycle = [], []
        model_dir = "/models/mock"
        with mock.patch("localbench.backends.MlxServe.start",
                        new=lambda server: lifecycle.append(("start", server.root))), \
                mock.patch("localbench.backends.MlxServe.stop",
                           new=lambda server: lifecycle.append(("stop", server.root))), \
                mock.patch("localbench.backends.MlxServe.model_id", return_value="served-model"), \
                mock.patch("localbench.backends.MlxServe.fingerprint",
                           return_value={"backend": "mlx-serve", "model": "served-model",
                                         "model_digest": "files:abc", "model_dir": model_dir}):
            out = generation.run_candidate(
                self.SPEC, {"route": f"mlx-serve:{model_dir}"}, [self.SKILL_ITEM],
                post=self._post(calls))
        self.assertEqual(lifecycle, [("start", "http://127.0.0.1:11234"),
                                     ("stop", "http://127.0.0.1:11234")])
        self.assertEqual(out["pins"]["backend"], "mlx-serve")
        (url, body), = calls
        self.assertEqual(url, "http://127.0.0.1:11234/v1/chat/completions")
        self.assertEqual(body["model"], "served-model")
    def test_builtin_arm_needs_no_model_call(self):
        def refused(url, body, timeout=120.0):
            raise AssertionError("builtin must not call any model")
        out = generation.run_candidate(self.SPEC, {"builtin": "preview"},
                                       [self.SKILL_ITEM], post=refused)
        self.assertIsNone(out["model"])
        outcome = out["outcomes"][0]
        self.assertTrue(outcome["text"])
        self.assertEqual(outcome["latency_s"], 0.0)

    def test_missing_profile_is_a_named_error(self):
        item = dict(self.SKILL_ITEM, profile=None)
        out = generation.run_candidate(self.SPEC, {"route": "ollama:qwen3.6:35b-mlx"},
                                       [item], post=self._post([]))
        self.assertEqual(out["outcomes"][0]["error"], "no-profile")

    def test_bad_candidate_shapes_are_refused(self):
        with self.assertRaises(corpus.CorpusError):
            generation.run_candidate(self.SPEC, {"route": "openai:gpt-4"}, [], post=self._post([]))
        with self.assertRaises(corpus.CorpusError):
            generation.run_candidate(self.SPEC, {"model": "x"}, [], post=self._post([]))
        with self.assertRaises(corpus.CorpusError):
            generation.run_candidate({"kind": "generation", "feature": "nope"},
                                     {"route": "ollama:qwen3.6:35b-mlx"}, [], post=self._post([]))

class OutcomeContext(unittest.TestCase):
    """%pane evaluator seam: every replay outcome carries the check context the
    asserted check re-runs against (skill description, commit diff files, {})."""

    def test_all_arms_carry_their_check_context(self):
        item, spec = RunCandidate.SKILL_ITEM, RunCandidate.SPEC

        def post(url, body, timeout=120.0):
            return {"status": 200,
                    "body": {"choices": [{"message": {"content": "Database migrations deploy previews"}}]},
                    "latency_s": 0.5}
        route = generation.run_candidate(spec, {"route": "ollama:qwen3.6:35b-mlx"},
                                         [item], post=post)["outcomes"][0]
        self.assertIn("Routes database migrations", route["context"]["description"])
        builtin = generation.run_candidate(spec, {"builtin": "preview"},
                                           [item], post=post)["outcomes"][0]
        self.assertIn("Routes database migrations", builtin["context"]["description"])
        naked = generation.run_candidate(spec, {"route": "ollama:qwen3.6:35b-mlx"},
                                         [dict(item, profile=None)], post=post)["outcomes"][0]
        self.assertIn("Routes database migrations", naked["context"]["description"])
        self.assertEqual(naked["error"], "no-profile")

if __name__ == "__main__":
    unittest.main()
