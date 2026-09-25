"""The model cleanup must never offer to delete a model something still uses, and must delete only what it is told."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

SPEC = importlib.util.spec_from_file_location("prune_models",
                                              Path(__file__).resolve().parent.parent / "scripts" / "prune_models.py")
assert SPEC and SPEC.loader, "scripts/prune_models.py not found"
pm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pm
SPEC.loader.exec_module(pm)

DENSE = {"server": "ollama", "name": "qwen3.8:27b-mlx", "digest": "5642e97495e1", "gb": 18.0,
         "source": "qwen3.8:27b-mlx", "parked": False}
PARKED_DENSE = {**DENSE, "name": "localbench-parked:5642e97495e1", "parked": True}
SIBLING = {"server": "ollama", "name": "localbench-parked:23da7bcdf4d1", "digest": "23da7bcdf4d1", "gb": 19.0,
           "source": "qwen3.8-uncensored:latest", "parked": True}
NEMOTRON = {"server": "ollama", "name": "nemotron-3.5-lightning:30b-mlx", "digest": "8b1474be6e54", "gb": 22.0,
            "source": "nemotron-3.5-lightning:30b-mlx", "parked": False}
CLOUD = {"server": "ollama", "name": "minimax-m2.5:cloud", "digest": "c0d5751c800f", "gb": 0.0,
         "source": "minimax-m2.5:cloud", "parked": False, "freshness": "cloud model (runs remotely)"}
MOE_DIR = {"server": "mlx-serve", "name": "ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit", "gb": 20.0}
GOLDENS = [{"backend": "ollama", "model": "localbench-parked:5642e97495e1", "digest": "5642e97495e1",
            "golden": "ollama__localbench-parked_5642e97495e1.json"},
           {"backend": "mlx-serve", "model": "Qwen3.6-35B-A3B-MLX-Serve-4bit", "digest": "files:68dede",
            "golden": "mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit.json"}]
ROUTES = {"qwen3.8:27b-mlx": {"session titles": ["default", "grok"]}}


def planned(rows, resident=None, last_seen=None):
    return {r["name"]: r for r in pm.plan(rows, GOLDENS, ROUTES, resident or {}, last_seen or {})}


class Keep(unittest.TestCase):
    def test_a_golden_pins_its_model_by_digest_whatever_the_tag_is_called_now(self):
        self.assertIn("golden ollama__localbench-parked_5642e97495e1.json", planned([DENSE])[DENSE["name"]]["keep"])

    def test_a_parked_copy_is_kept_for_what_its_original_name_serves(self):
        keep = planned([PARKED_DENSE])[PARKED_DENSE["name"]]["keep"]
        self.assertTrue(any(k.startswith("omp routes session titles (default, grok)") for k in keep), keep)

    def test_resident_and_golden_match_an_mlx_dir_by_its_last_segment(self):
        keep = planned([MOE_DIR], resident={"mlx-serve": ["Qwen3.6-35B-A3B-MLX-Serve-4bit"]})[MOE_DIR["name"]]["keep"]
        self.assertIn("resident now", keep)
        self.assertIn("golden mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit.json", keep)

    def test_a_cloud_stub_is_kept(self):
        self.assertEqual(planned([CLOUD])[CLOUD["name"]]["keep"], ["cloud model (no local weights)"])

    def test_an_unused_model_is_a_candidate_with_when_it_was_last_loaded(self):
        p = planned([NEMOTRON, SIBLING], last_seen={("ollama", "nemotron-3.5-lightning:30b-mlx"): 100.0,
                                                    ("ollama", "qwen3.8-uncensored:latest"): 200.0})
        self.assertEqual((p[NEMOTRON["name"]]["keep"], p[NEMOTRON["name"]]["last_seen"]), ([], 100.0))
        self.assertEqual((p[SIBLING["name"]]["keep"], p[SIBLING["name"]]["last_seen"]), ([], 200.0))


class NamedElsewhere(unittest.TestCase):
    """Models other tools use without omp: Codex embeds with nomic-embed-text; the splash skill requires its model."""

    EMBED: ClassVar[dict] = {"server": "ollama", "name": "nomic-embed-text:latest", "digest": "0a109f422b47", "gb": 0.3,
                             "source": "nomic-embed-text:latest", "parked": False}
    SPLASH: ClassVar[dict] = {"server": "splash", "name": "incoai/Qwen3.8-27B-Splash", "gb": 17.4}
    CONFIGS: ClassVar[dict] = {"~/.codex/config.toml": 'EMBEDDING_MODEL = "nomic-embed-text"\n',
                               "~/.agents/skills/splash/SKILL.md": "require `incoai/Qwen3.8-27B-Splash`",
                               "~/.agents/skills/x/SKILL.md": "try nemotron-3.5-lightning-pro someday"}

    def keep(self, row):
        return pm.plan([row], GOLDENS, ROUTES, {}, {}, self.CONFIGS)[0]["keep"]

    def test_a_model_another_tool_names_is_kept(self):
        self.assertEqual(self.keep(self.EMBED), ["named in ~/.codex/config.toml"])
        self.assertEqual(self.keep(self.SPLASH), ["named in ~/.agents/skills/splash/SKILL.md"])

    def test_a_longer_name_does_not_keep_a_shorter_one(self):
        self.assertEqual(self.keep(NEMOTRON), [])


class RealInputs(unittest.TestCase):
    """The readers plan() depends on, against the files they read: a golden as golden.py writes it, a skills tree,
    and gather()'s wiring. A misread here offers a model that is in use for deletion."""

    def test_a_golden_as_golden_py_writes_it_pins_its_model(self):
        from localbench import golden
        pins = {"backend": "mlx-serve", "model": "Qwen3.6-35B-A3B-MLX-Serve-4bit", "model_digest": "files:68dedceb5da0",
                "backend_version": "26.9.2"}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "host" / f"{golden.slug('mlx-serve')}__{golden.slug(pins['model'])}.json"
            golden.write(path, {"pins": golden.stored_pin_block(pins), "metrics": {}, "conformance": {}})
            goldens = pm.golden_pins(Path(d))
        self.assertEqual(goldens, [{"backend": "mlx-serve", "model": "Qwen3.6-35B-A3B-MLX-Serve-4bit",
                                    "digest": "files:68dedc", "golden": path.name}])
        keep = pm.plan([MOE_DIR], goldens, {}, {}, {})[0]["keep"]
        self.assertEqual(keep, [f"golden {path.name}"])

    def test_skill_docs_are_read_by_their_real_file_name(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "splash").mkdir()
            (Path(d) / "splash" / "SKILL.md").write_text("require `incoai/Qwen3.8-27B-Splash`")
            texts = pm.config_texts(paths=(), skills=Path(d))
        self.assertEqual(list(texts.values()), ["require `incoai/Qwen3.8-27B-Splash`"])

    def test_gather_keeps_a_model_only_a_config_names(self):
        splash = {"server": "splash", "name": "incoai/Qwen3.8-27B-Splash", "gb": 17.4}
        with mock.patch.object(pm.models, "ollama_models", return_value=[]), \
                mock.patch.object(pm.models, "mlx_models", return_value=[splash]), \
                mock.patch.object(pm.models, "routes_by_model", return_value={}), \
                mock.patch.object(pm.models, "profiles", return_value={}), \
                mock.patch.object(pm.sysstats, "resident_models", return_value={}), \
                mock.patch.object(pm, "golden_pins", return_value=[]), \
                mock.patch.object(pm, "last_resident", return_value={}), \
                mock.patch.object(pm, "config_texts", return_value={"~/.agents/skills/splash/SKILL.md":
                                                                    "require `incoai/Qwen3.8-27B-Splash`"}):
            (row,) = pm.gather()
        self.assertEqual(row["keep"], ["named in ~/.agents/skills/splash/SKILL.md"])

    def test_a_parked_copy_is_named_by_its_original_tag(self):
        parked = {"server": "ollama", "name": "localbench-parked:0a109f422b47", "digest": "0a109f422b47", "gb": 0.3,
                  "source": "nomic-embed-text:latest", "parked": True}
        keep = pm.plan([parked], [], {}, {}, {}, {"~/.codex/config.toml": 'EMBEDDING_MODEL = "nomic-embed-text"'})
        self.assertEqual(keep[0]["keep"], ["named in ~/.codex/config.toml"])

    def test_last_resident_is_each_models_latest_sighting_in_the_watch_db(self):
        from localbench import observe
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "observe.db"
            con = observe.connect(db)
            con.executemany("insert into resident (t, server, model) values (?, ?, ?)",
                            [(100.0, "ollama", "m:1"), (300.0, "ollama", "m:1"), (200.0, "ollama", "m:1"),
                             (50.0, "mlx-serve", "org/x")])
            con.commit()
            con.close()
            self.assertEqual(pm.last_resident(db), {("ollama", "m:1"): 300.0, ("mlx-serve", "org/x"): 50.0})

    def test_last_seen_is_the_latest_sighting_under_either_name(self):
        p = planned([SIBLING], last_seen={("ollama", "localbench-parked:23da7bcdf4d1"): 300.0,
                                          ("ollama", "qwen3.8-uncensored:latest"): 100.0})
        self.assertEqual(p[SIBLING["name"]]["last_seen"], 300.0)
        p = planned([SIBLING], last_seen={("ollama", "localbench-parked:23da7bcdf4d1"): 100.0,
                                          ("ollama", "qwen3.8-uncensored:latest"): 300.0})
        self.assertEqual(p[SIBLING["name"]]["last_seen"], 300.0)


class CheckDelete(unittest.TestCase):
    def test_only_named_candidates_are_selected(self):
        p = pm.plan([DENSE, NEMOTRON, CLOUD], GOLDENS, ROUTES, {}, {})
        targets, refusals = pm.check_delete(p, [NEMOTRON["name"], DENSE["name"], CLOUD["name"], "nope:1"])
        self.assertEqual([t["name"] for t in targets], [NEMOTRON["name"]])
        self.assertEqual([r.split(": ")[0] for r in refusals], ["qwen3.8:27b-mlx", "minimax-m2.5:cloud", "nope:1"])


class Delete(unittest.TestCase):
    def test_deleting_a_parked_copy_leaves_parked_json_restoring_only_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "PARKED.json"
            state.write_text(json.dumps([
                {"name": "qwen3.8:27b-mlx", "parked_as": "localbench-parked:5642e97495e1", "digest": "d1", "role": "smol"},
                {"name": "qwen3.8-uncensored:latest", "parked_as": SIBLING["name"], "digest": "d2", "role": "fallback"}]))
            with mock.patch.object(pm.park, "_delete") as gone:
                pm.delete(SIBLING, state)
            gone.assert_called_once_with(SIBLING["name"])
            self.assertEqual([p["name"] for p in json.loads(state.read_text())], ["qwen3.8:27b-mlx"])

    def test_an_unparked_tag_leaves_parked_json_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "PARKED.json"
            state.write_text("[]")
            with mock.patch.object(pm.park, "_delete"):
                pm.delete(NEMOTRON, state)
            self.assertEqual(state.read_text(), "[]")

    def test_a_directory_outside_the_models_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            (root / "org" / "m").mkdir(parents=True)
            outside = Path(tmp) / "keep-me"
            outside.mkdir()
            with mock.patch.object(pm.models, "MLX_MODELS", root):
                with self.assertRaises(RuntimeError):
                    pm.delete({"server": "mlx-serve", "name": "../keep-me", "gb": 0}, Path(tmp) / "PARKED.json")
                pm.delete({"server": "mlx-serve", "name": "org/m", "gb": 0}, Path(tmp) / "PARKED.json")
            self.assertTrue(outside.is_dir())
            self.assertFalse((root / "org" / "m").exists())


class Main(unittest.TestCase):
    def run_main(self, argv, alive=False):
        out = io.StringIO()
        with mock.patch.object(pm, "gather", return_value=pm.plan([NEMOTRON, DENSE], GOLDENS, ROUTES, {}, {})), \
                mock.patch.object(pm, "run_alive", return_value=alive), \
                mock.patch.object(pm, "delete") as delete, \
                mock.patch.object(pm, "LOG", Path(tempfile.mkdtemp()) / "prune-log.jsonl"), \
                contextlib.redirect_stdout(out):
            rc = pm.main(argv)
        return rc, delete, out.getvalue()

    def test_listing_deletes_nothing(self):
        rc, delete, out = self.run_main([])
        self.assertEqual(rc, 0)
        delete.assert_not_called()
        self.assertIn("CANDIDATES (22.0 GB)", out)

    def test_nothing_is_deleted_while_a_run_is_alive(self):
        rc, delete, _ = self.run_main(["--delete", NEMOTRON["name"]], alive=True)
        self.assertEqual(rc, 1)
        delete.assert_not_called()

    def test_one_refused_name_deletes_nothing(self):
        rc, delete, _ = self.run_main(["--delete", NEMOTRON["name"], DENSE["name"]])
        self.assertEqual(rc, 1)
        delete.assert_not_called()

    def test_a_named_candidate_is_deleted(self):
        rc, delete, _ = self.run_main(["--delete", NEMOTRON["name"]])
        self.assertEqual(rc, 0)
        self.assertEqual(delete.call_args[0][0]["name"], NEMOTRON["name"])


if __name__ == "__main__":
    unittest.main()
