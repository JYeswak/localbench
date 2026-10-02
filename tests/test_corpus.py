"""Private-corpus importer tests. Every fixture database is synthetic and tiny, built in a
tmp dir; no real judgment-cache content may appear here (those states embed user code)."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from localbench import corpus, decision

MODEL = "typesafe/proj-b-latest"
OTHER_MODEL = "typesafe/other"
GATE_NOUL = {"decision.noul.accuracy": {"min": 0.8}}
GATE_CHOICE = {"decision.choice.accuracy": {"min": 0.8}}


def json_dumps(value) -> str:
    """omp's JSON.stringify for these values: compact, non-ASCII kept, insertion order."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fixture_db(path: Path) -> Path:
    """A synthetic judgment cache: s1 two valid noul questions, s2 one valid choice plus one
    noul with an out-of-range probability, s3 only the invalid question, s1/p00 also answered by
    a second model."""
    db_path = path / "judgment-cache.db"
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE states (id TEXT PRIMARY KEY, state TEXT NOT NULL, created_at INTEGER NOT NULL)")
    db.execute("""CREATE TABLE oracle (id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT NOT NULL,
                model TEXT NOT NULL, name TEXT NOT NULL, type TEXT NOT NULL, instruction TEXT NOT NULL,
                criteria TEXT NOT NULL, answer TEXT NOT NULL, created_at INTEGER NOT NULL)""")
    db.execute("""CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT NOT NULL,
                results TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
                price REAL NOT NULL, created_at INTEGER NOT NULL)""")
    db.executemany("INSERT INTO states VALUES (?,?,?)",
                   [("s1", "state one", 100), ("s2", "state two", 200), ("s3", "state three", 300)])
    db.executemany("INSERT INTO oracle(state,model,name,type,instruction,criteria,answer,created_at)"
                   " VALUES (?,?,?,?,?,?,?,?)", [
        ("s1", MODEL, "p00", "noul", "is it relevant?", '{"true":"yes","false":"no"}',
         '{"type":"noul","noul":0.9}', 101),
        ("s1", MODEL, "p01", "noul", "is it relevant?", "null",
         '{"type":"noul","noul":0.2}', 102),
        ("s1", OTHER_MODEL, "p00", "noul", "is it relevant?", '{"true":"yes","false":"no"}',
         '{"type":"noul","noul":0.1}', 103),
        ("s2", MODEL, "e00", "choice", "pick one", '{"high":"h","low":"l","medium":"m"}',
         '{"type":"choice","choice":"high","probabilities":{"high":0.7,"low":0.2,"medium":0.1},'
         '"confidence":0.8}', 201),
        ("s2", MODEL, "p02", "noul", "is it relevant?", "null",
         '{"type":"noul","noul":2.0}', 202),
        ("s3", MODEL, "p03", "noul", "is it relevant?", "null",
         '{"type":"noul","noul":-0.5}', 301),
    ])
    db.commit()
    db.close()
    return db_path


def profiles_root(tmp: str) -> Path:
    root = Path(tmp) / "profiles"
    cache = root / "codex" / "cache"
    cache.mkdir(parents=True)
    fixture_db(cache)
    return root


class ReadCache(unittest.TestCase):
    def test_states_come_back_with_their_oracle_rows_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = corpus.read_cache(fixture_db(Path(tmp)))
        self.assertEqual([row["state_id"] for row in rows], ["s1", "s2", "s3"])
        self.assertEqual([row["name"] for row in rows[0]["oracle"]], ["p00", "p01", "p00"])
        self.assertEqual(rows[0]["oracle"][2]["model"], OTHER_MODEL)

    def test_missing_database_and_missing_tables_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(corpus.CorpusError):
                corpus.read_cache(Path(tmp) / "absent.db")
            broken = Path(tmp) / "broken.db"
            with closing(sqlite3.connect(broken)) as db:   # the connection, not just its cursor: a GC'd one warns
                db.execute("CREATE TABLE states (id TEXT PRIMARY KEY)")
            with self.assertRaises(corpus.CorpusError):
                corpus.read_cache(broken)


class ToItems(unittest.TestCase):
    def test_noul_items_carry_bool_labels_and_validated_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = corpus.read_cache(fixture_db(Path(tmp)))
        items, skipped = corpus.to_items(rows, profile="codex", role="decision.noul", model=MODEL)
        self.assertEqual([item["id"] for item in items], ["s1"])
        item = items[0]
        self.assertEqual(item["labels"], {"p00": True, "p01": False})
        self.assertEqual(item["reference"]["p00"], {"type": "noul", "noul": 0.9})
        self.assertEqual(item["source"], "codex:s1")
        self.assertEqual(item["questions"]["p00"]["criteria"], {"true": "yes", "false": "no"})
        self.assertNotIn("criteria", item["questions"]["p01"])
        self.assertEqual(skipped, {"invalid_answer": 2, "empty_state": 2, "oversize_state": 0,
                                   "drifted_question": 0})

    def test_choice_items_reduce_to_the_argmax_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = corpus.read_cache(fixture_db(Path(tmp)))
        items, skipped = corpus.to_items(rows, profile="codex", role="decision.choice", model=MODEL)
        self.assertEqual([item["id"] for item in items], ["s2"])
        self.assertEqual(items[0]["labels"], {"e00": "high"})
        self.assertEqual(items[0]["reference"]["e00"]["choice"], "high")
        self.assertEqual(skipped["invalid_answer"], 0)

    def test_ttsr_state_and_level_criteria_round_trip_in_omp_order(self):
        level = corpus._LEVEL_QUESTION
        rows = [
            {"state_id": "t1", "state": '{"content":"c","output":"o"}', "created_at": 1,
             "oracle": [{"id": 1, "model": MODEL, "name": "q0", "type": "noul",
                         "instruction": corpus._TTSR_INSTRUCTIONS, "criteria": "null",
                         "answer": '{"type":"noul","noul":0.85}', "created_at": 2}]},
            {"state_id": "t2", "state": '{"request":"do x"}', "created_at": 3,
             "oracle": [{"id": 2, "model": MODEL, "name": "level", "type": "choice",
                         "instruction": level["instructions"],
                         "criteria": json_dumps(
                             {k: level["criteria"][k] for k in sorted(level["criteria"])}),
                         "answer": '{"type":"choice","choice":"low","confidence":0.2,'
                                   '"probabilities":{"low":0.5,"medium":0.3,"high":0.1,"xhigh":0.09}}',
                         "created_at": 4}]},
        ]
        noul_items, _ = corpus.to_items(rows, profile="codex", role="decision.noul", model=MODEL)
        self.assertEqual([item["id"] for item in noul_items], ["t1"])
        self.assertEqual(list(noul_items[0]["state"]), ["output", "content"])
        self.assertEqual(json_dumps(noul_items[0]["state"]), '{"output":"o","content":"c"}')
        choice_items, skipped = corpus.to_items(rows, profile="codex", role="decision.choice",
                                                model=MODEL)
        self.assertEqual([item["id"] for item in choice_items], ["t2"])
        question = choice_items[0]["questions"]["level"]
        self.assertEqual(list(question["criteria"]), ["low", "medium", "high", "xhigh"])
        self.assertEqual(json_dumps(question["criteria"]), json_dumps(level["criteria"]))
        self.assertEqual(choice_items[0]["labels"], {"level": "low"})
        rescaled = choice_items[0]["reference"]["level"]["probabilities"]
        self.assertAlmostEqual(sum(rescaled.values()), 1.0)
        self.assertEqual(list(rescaled), ["low", "medium", "high", "xhigh"])
        self.assertEqual(skipped["drifted_question"], 0)

    def test_known_name_with_drifted_text_is_skipped_not_reordered(self):
        rows = [{"state_id": "d1", "state": '{"request":"do x"}', "created_at": 1,
                 "oracle": [{"id": 1, "model": MODEL, "name": "level", "type": "choice",
                             "instruction": "a reworded difficulty question",
                             "criteria": '{"high":"h","low":"l","medium":"m","xhigh":"x"}',
                             "answer": '{"type":"choice","choice":"low","confidence":0.9,'
                                       '"probabilities":{"low":0.7,"medium":0.1,"high":0.1,"xhigh":0.1}}',
                             "created_at": 2}]}]
        items, skipped = corpus.to_items(rows, profile="codex", role="decision.choice", model=MODEL)
        self.assertEqual(items, [])
        self.assertEqual(skipped["drifted_question"], 1)
        self.assertEqual(skipped["empty_state"], 1)

    def test_two_answering_models_without_a_pin_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = corpus.read_cache(fixture_db(Path(tmp)))
        with self.assertRaisesRegex(corpus.CorpusError, "2 models"):
            corpus.to_items(rows, profile="codex", role="decision.noul")

    def test_unknown_role_is_refused(self):
        with self.assertRaises(corpus.CorpusError):
            corpus.to_items([], profile="codex", role="decision.rank")


class ImportProfile(unittest.TestCase):
    def test_import_writes_a_loadable_suite_and_reimport_lands_on_the_same_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                profiles_root(tmp)
                dest = Path(tmp) / "corpora"
                first = corpus.import_profile("codex", dest_root=dest, model=MODEL, gate=GATE_NOUL)
                second = corpus.import_profile("codex", dest_root=dest, model=MODEL, gate=GATE_NOUL)
            self.assertEqual(first["decision.noul"]["items"], 1)
            self.assertEqual(first, second)
            suite = decision.load_suite(first["decision.noul"]["directory"])
            self.assertEqual(suite.role, "decision.noul")
            self.assertTrue(suite.has_reference)
            self.assertEqual(suite.items[0]["labels"], {"p00": True, "p01": False})

    def test_suite_pins_a_snapshot_and_survives_later_cache_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                root = profiles_root(tmp)
                dest = Path(tmp) / "corpora"
                out = corpus.import_profile("codex", dest_root=dest, model=MODEL, gate=GATE_NOUL)
                directory = Path(out["decision.noul"]["directory"])
                snapshots = list((directory / "sources").glob("*.sqlite"))
                self.assertEqual(len(snapshots), 1)
                suite = decision.load_suite(directory)
                self.assertEqual(suite.sources[0]["path"], str(snapshots[0]))
                live = root / "codex" / "cache" / "judgment-cache.db"
                db = sqlite3.connect(live)
                db.execute("INSERT INTO states VALUES (?,?,?)", ("s9", "late state", 900))
                db.execute("INSERT INTO oracle(state,model,name,type,instruction,criteria,answer,created_at)"
                           " VALUES (?,?,?,?,?,?,?,?)",
                           ("s9", MODEL, "p09", "noul", "late?", "null",
                            '{"type":"noul","noul":0.4}', 901))
                db.commit()
                db.close()
                reloaded = decision.load_suite(directory)
                self.assertEqual(len(reloaded.items), 1)

    def test_import_without_an_explicit_gate_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                profiles_root(tmp)
                dest = Path(tmp) / "corpora"
                with self.assertRaisesRegex(decision.SuiteError, "explicit gate"):
                    corpus.import_profile("codex", dest_root=dest, model=MODEL)
            self.assertFalse(dest.exists())

    def test_output_inside_the_repo_is_refused_before_anything_is_written(self):
        probe = corpus.REPO_ROOT / "runs" / "corpus-probe"
        shutil.rmtree(probe, ignore_errors=True)  # a planted mutation run may have left one behind
        self.addCleanup(shutil.rmtree, probe, True)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                profiles_root(tmp)
                with self.assertRaisesRegex(corpus.CorpusError, "inside the repo"):
                    corpus.import_profile("codex", dest_root=probe, model=MODEL, gate=GATE_NOUL)
        self.assertFalse(probe.exists())

    def test_missing_profile_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                (Path(tmp) / "profiles").mkdir()
                with self.assertRaises(corpus.CorpusError):
                    corpus.import_profile("codex", dest_root=Path(tmp) / "corpora",
                                          model=MODEL, gate=GATE_NOUL)

    def test_nested_symlink_escape_into_a_git_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                profiles_root(tmp)
                dest = Path(tmp) / "corpora"
                git_tree = Path(tmp) / "evil-tree"
                (git_tree / ".git").mkdir(parents=True)
                dest.mkdir(parents=True)
                (dest / "decision.noul").symlink_to(git_tree, target_is_directory=True)
                with self.assertRaisesRegex(corpus.CorpusError, "git work tree"):
                    corpus.import_profile("codex", dest_root=dest, model=MODEL, gate=GATE_NOUL)
            self.assertEqual(list(git_tree.rglob("*.jsonl")), [])
            self.assertEqual(list(git_tree.rglob("*.sqlite")), [])

    def test_symlink_swapped_in_after_the_upfront_check_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                profiles_root(tmp)
                dest = Path(tmp) / "corpora"
                git_tree = Path(tmp) / "evil-tree"
                (git_tree / ".git").mkdir(parents=True)
                swapped = []
                real_mkdir = Path.mkdir

                def racing_mkdir(self, *args, **kwargs):
                    real_mkdir(self, *args, **kwargs)
                    role_dir = dest / "decision.noul"
                    if not swapped and role_dir.exists() and not role_dir.is_symlink():
                        swapped.append(True)
                        shutil.rmtree(role_dir)
                        role_dir.symlink_to(git_tree, target_is_directory=True)

                with mock.patch.object(Path, "mkdir", racing_mkdir):
                    with self.assertRaisesRegex(corpus.CorpusError, "git work tree"):
                        corpus.import_profile("codex", dest_root=dest, model=MODEL, gate=GATE_NOUL)
                self.assertTrue(swapped)
            self.assertEqual(list(git_tree.rglob("*.jsonl")), [])
            self.assertEqual(list(git_tree.rglob("*.sqlite")), [])


    def test_suite_dirs_and_files_are_owner_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                profiles_root(tmp)
                dest = Path(tmp) / "corpora"
                out = corpus.import_profile("codex", dest_root=dest, model=MODEL, gate=GATE_NOUL)
                directory = Path(out["decision.noul"]["directory"])
                self.assertEqual(oct((directory / "sources").stat().st_mode & 0o777), "0o700")
                snapshots = list((directory / "sources").glob("*.sqlite"))
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(oct(snapshots[0].stat().st_mode & 0o777), "0o600")
                spec = corpus.capture_on(["decision"], 5, 60, root=dest, now=1_000_000.0)
                spec_path = dest / "capture.json"
                self.assertTrue(spec_path.is_file())
                self.assertEqual(oct(spec_path.stat().st_mode & 0o777), "0o600")
                self.assertEqual(oct(dest.stat().st_mode & 0o777), "0o700")


class ListStats(unittest.TestCase):
    def test_list_and_stats_count_suites_without_validating_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(corpus, "PROFILES", Path(tmp) / "profiles"):
                profiles_root(tmp)
                dest = Path(tmp) / "corpora"
                corpus.import_profile("codex", dest_root=dest, model=MODEL, gate=GATE_NOUL)
                corpus.import_profile("codex", dest_root=dest, roles=("decision.choice",),
                                      model=MODEL, gate=GATE_CHOICE, name="codex-choice")
            suites = corpus.list_corpora(dest)
            self.assertEqual([(s["role"], s["items"], s["has_reference"]) for s in suites],
                             [("decision.choice", 1, True), ("decision.noul", 1, True)])
            self.assertEqual(corpus.stats(dest),
                             {"roles": {"decision.choice": {"suites": 1, "items": 1, "names": ["codex-choice"]},
                                        "decision.noul": {"suites": 1, "items": 1,
                                                          "names": [suites[1]["name"]]}},
                              "suites": 2, "items": 2,
                              "capture": {"items": 0, "active": False}})

    def test_missing_root_lists_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(corpus.list_corpora(Path(tmp) / "absent"), [])
            self.assertEqual(corpus.stats(Path(tmp) / "absent"),
                             {"roles": {}, "suites": 0, "items": 0,
                              "capture": {"items": 0, "active": False}})

class CaptureSpec(unittest.TestCase):
    def test_on_off_status_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpora"
            spec = corpus.capture_on(["decision", "main"], max_items=5, minutes=60,
                                     root=root, now=1_000_000.0)
            self.assertEqual(spec, {"enabled": True, "purposes": ["decision", "main"],
                                   "max_items": 5, "until": 1_003_600.0})
            status = corpus.capture_status(root, now=1_000_100.0)
            self.assertEqual(status["active"], True)
            self.assertEqual((status["enabled"], status["items"]), (True, 0))
            off = corpus.capture_off(root)
            self.assertEqual(off["enabled"], False)
            self.assertEqual(off["purposes"], ["decision", "main"])
            self.assertFalse(corpus.capture_status(root, now=1_000_100.0)["active"])

    def test_expiry_deactivates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpora"
            corpus.capture_on(["decision"], max_items=5, minutes=60, root=root, now=1_000_000.0)
            self.assertFalse(corpus.capture_status(root, now=1_003_601.0)["active"])

    def test_invalid_specs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpora"
            for purposes, max_items, minutes in (([], 5, 60), ("decision", 5, 60),
                                                (["decision"], 0, 60), (["decision"], True, 60),
                                                (["decision"], 5, 0), (["decision"], 5, float("inf"))):
                with self.subTest(purposes=purposes, max_items=max_items, minutes=minutes):
                    with self.assertRaises(corpus.CorpusError):
                        corpus.capture_on(purposes, max_items, minutes, root=root)
            self.assertFalse((root / "capture.json").exists())

    def test_spec_inside_the_repo_is_refused(self):
        with self.assertRaisesRegex(corpus.CorpusError, "inside the repo"):
            corpus.capture_on(["decision"], 5, 60, root=corpus.REPO_ROOT / "runs" / "corpus-probe")
    def test_stats_counts_captured_pairs_and_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpora"
            pair = root / "captured" / "decision"
            pair.mkdir(parents=True)
            (pair / ("a" * 64 + ".json")).write_text("{}", encoding="utf-8")
            (pair / ("a" * 64 + ".response.json")).write_text("{}", encoding="utf-8")
            (pair / ("b" * 64 + ".json")).write_text("{}", encoding="utf-8")
            self.assertEqual(corpus.stats(root)["capture"], {"items": 2, "active": False})
            corpus.capture_on(["decision"], 10, 60, root=root)
            self.assertEqual(corpus.stats(root)["capture"], {"items": 2, "active": True})


if __name__ == "__main__":
    unittest.main()
