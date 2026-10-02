"""Generation-corpus tests. All request bodies are synthetic shapes mirroring omp's call
sites; no captured user content may appear here."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from localbench import corpus, generation

TITLE = {"model": "qwen3.8:27b-mlx",
         "messages": [{"role": "system", "content": "Write a ~5 word title."},
                      {"role": "user", "content": "<user>\nDo the thing\n</user>\n<title>"}]}
SKILL = {"model": "qwen3.8:27b-mlx",
         "messages": [{"role": "user",
                       "content": "Compress into one routing hint of at most 12 words:\n\nSkill: x\nDescription: Routes database migrations for deploy previews"}]}
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
            grouped = generation.collect(root)
        self.assertEqual([r["body"] for r in grouped["titles"]], [TITLE])
        self.assertEqual([r["body"] for r in grouped["skill-description-compression"]], [SKILL])
        self.assertEqual([r["body"] for r in grouped["other"]], [OTHER])
        self.assertNotIn("commit-messages", grouped)


class Assemble(unittest.TestCase):
    def test_assemble_writes_pinned_manifest_and_content_addressed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "corpora" / "generation" / "titles-001"
            report = generation.assemble("titles", [{"body": TITLE, "source": "cap:1"},
                                                    {"body": SKILL, "source": "cap:2"}],
                                         dest, seed=20261001)
            manifest = json.loads((dest / "manifest.json").read_text())
            self.assertEqual(manifest["feature"], "titles")
            self.assertEqual(manifest["gate"], {"min_items": generation.MIN_ITEMS})
            self.assertEqual(manifest["n_items"], 2)
            self.assertEqual(manifest["items_sha256"], report["items_sha256"])
            self.assertEqual(manifest["seed"], 20261001)
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(oct((dest / "items.jsonl").stat().st_mode & 0o777), "0o600")
            lines = (dest / "items.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["feature"], "titles")
            self.assertEqual(first["body"], TITLE)
            self.assertEqual(len(first["id"]), 64)

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

if __name__ == "__main__":
    unittest.main()
