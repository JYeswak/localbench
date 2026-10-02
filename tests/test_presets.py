"""Feature presets (localbench/presets.py): plan, apply with backup and readback, rollback, drift, proof gate.

Every test runs against a temp HOME and a fake `omp` (FAKE_OMP below) that keeps each profile's config.yml as
`key: <json>` lines and rewrites the whole file on `config set`, so a restore is only byte-identical when it came
from the backup. No real omp, profile, model or server is touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import features, presets

FAKE_OMP = """import json, os, sys
from pathlib import Path
# find.enabled=on keeps the find-judgments route live, so its incumbent is the judge route, not "off".
DEFAULTS = {"modelRoles": {}, "task.disabledAgents": [], "defaultThinkingLevel": "high", "mnemopi.llmMode": "smol",
            "find.enabled": "on"}
args = sys.argv[1:]
profile = os.environ.get("OMP_PROFILE")
if args[:1] == ["--profile"]:
    profile, args = args[1], args[2:]
root = Path(os.environ["HOME"]) / ".omp"
path = (root / "agent" if not profile else root / "profiles" / profile / "agent") / "config.yml"
if not path.is_file():
    sys.exit(f"fake omp: no profile at {path}")
cfg = {}
for line in path.read_text().splitlines():
    if line.strip() and not line.lstrip().startswith("#"):
        k, _, v = line.partition(": ")
        cfg[k] = json.loads(v)
if args == ["config", "list", "--json"]:
    print(json.dumps({k: {"value": v} for k, v in {**DEFAULTS, **cfg}.items()}))
    sys.exit(0)
if args[:1] != ["config"] or len(args) < 3 or args[2] not in DEFAULTS:
    print(f"fake omp: unsupported {args}", file=sys.stderr)
    sys.exit(1)
action, key = args[1], args[2]
if action == "get" and args[3:] == ["--json"]:
    print(json.dumps({"key": key, "value": cfg.get(key, DEFAULTS[key]), "type": "x", "description": "x"}, indent=2))
elif action == "set" and len(args) == 4:
    if (profile or "default") in os.environ.get("FAKE_OMP_FAIL", "").split(","):
        print("fake omp: write failed", file=sys.stderr)
        sys.exit(1)
    if key not in os.environ.get("FAKE_OMP_IGNORE", "").split(","):
        cfg[key] = json.loads(args[3])
        path.write_text("".join(f"{k}: {json.dumps(v)}\\n" for k, v in sorted(cfg.items())))
    print(json.dumps({"key": key, "value": cfg.get(key)}))
else:
    print(f"fake omp: unsupported {args}", file=sys.stderr)
    sys.exit(1)
"""

# Hand-written bytes the fake would never produce: a comment, trailing spaces, no final newline.
CONFIG = ('# hand-edited, keep me\nmodelRoles: {"default": "muse/x", "smol": "ollama/qwen3.8:27b-mlx", '
          '"judge": "typesafe/proj-b-latest"}\ndefaultThinkingLevel: "auto"   \ntask.disabledAgents: ["sonic"]')
MODELS = "# local providers\nproviders:\n  # hosted judge\n  typesafe:\n    apiKey: secret-cmd\n  ollama-sys1:\n" \
         "    baseUrl: http://127.0.0.1:11434\n    api: typesafe\n"
FEATURE_ROWS = [
    ("auto-thinking", "src/auto-thinking/classifier.ts", "judge>tiny>smol", "setting", "defaultThinkingLevel=auto",
     "decision", "judge"),
    ("find-judgments", "src/tools/jfind/index.ts", "judge>tiny>smol", "setting", "find.enabled=on", "decision",
     "judge"),
    ("scout-agent", "src/prompts/agents/scout.md", "smol", "agent", "scout", "-", "scout"),
]


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# Injected installed-model digests (features.ollama_digests shape), so no test asks Ollama on :11434.
DIGESTS = {"nimble:latest": "24e550a16a70" + "1" * 52, "tev1:latest": "cef45ef93cf6" + "2" * 52}


class Env(unittest.TestCase):
    """Temp HOME with profiles default, claude, omp-test, omp-test2; a fake omp package; a features registry with
    two judge features and scout, and an (initially empty) receipts dir."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="presets-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"
        self.pkg = self.tmp / "node_modules" / "@oh-my-pi" / "pi-coding-agent"
        (self.pkg / "src" / "config").mkdir(parents=True)
        (self.pkg / "src" / "config" / "model-resolver.ts").write_text("// resolver\n")
        (self.tmp / "fake_omp.py").write_text(FAKE_OMP)
        omp = self.pkg / "bin" / "omp"
        omp.parent.mkdir()
        omp.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{self.tmp / "fake_omp.py"}" "$@"\n')
        omp.chmod(omp.stat().st_mode | stat.S_IXUSR)
        for name in ("default", "claude", "omp-test", "omp-test2"):
            self.config(name).parent.mkdir(parents=True)
            self.config(name).write_bytes(CONFIG.encode())
        self.models("omp-test").write_text(MODELS)
        self.registry = self.tmp / "features.tsv"
        self.registry.write_text("\t".join(features.COLUMNS) + "\n" + "".join(
            "\t".join(("omp", f, "pi-coding-agent", mod, "sym", role, kind, key, suite, fam, "-")) + "\n"
            for f, mod, role, kind, key, suite, fam in FEATURE_ROWS))
        for _f, mod, *_ in FEATURE_ROWS:
            (self.pkg / mod).parent.mkdir(parents=True, exist_ok=True)
            (self.pkg / mod).write_text(f"// {mod}\n")
        self.receipts = self.tmp / "receipts"
        self.receipts.mkdir()
        env = mock.patch.dict(os.environ, {"HOME": str(self.home), "LOCALBENCH_OMP": str(omp)})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("FAKE_OMP_IGNORE", None)
        os.environ.pop("FAKE_OMP_FAIL", None)
        self.originals = {p: self.snapshot(p) for p in ("default", "claude", "omp-test", "omp-test2")}

    def config(self, profile: str) -> Path:
        return self.home / ".omp" / ("agent" if profile == "default" else f"profiles/{profile}/agent") / "config.yml"

    def models(self, profile: str) -> Path:
        return self.config(profile).parent / "models.yml"

    def snapshot(self, profile: str) -> tuple:
        return sha(self.config(profile)), sha(self.models(profile))

    def receipt(self, feature: str, *, sha: str | None = None, pinned: str = DIGESTS["nimble:latest"][:12],
                baseline: dict | None = None):
        """A receipt in the shape features.grade's proof contract accepts: a decision run, BETTER than `baseline`
        (default: the target profiles' current route, the hosted TypeSafe judge proj-b-latest), measured on the installed
        module (unless `sha`) with model digest `pinned`."""
        mod = next(r[1] for r in FEATURE_ROWS if r[0] == feature)
        digest = sha or hashlib.sha256((self.pkg / mod).read_bytes()).hexdigest()
        (self.receipts / f"{feature}.json").write_text(json.dumps({
            "kind": "run", "feature": feature, "omp_module_sha": digest, "problems": [],
            "verdict": {"compare": "BETTER", "baseline": baseline or {"kind": "hosted", "id": "proj-b-latest"}},
            "run": {"label": "decision", "provenance": {"pins": {"model_digest": pinned}}}}))

    def kw(self, **extra) -> dict:
        return {"features_registry": self.registry, "receipts_dir": self.receipts, "package": self.pkg,
                "sessions": [], "digests": DIGESTS, **extra}


class Registry(unittest.TestCase):
    def test_every_seed_local_preset_is_gated_by_at_least_one_registered_feature(self):
        reg = presets.load()
        fams = {r["preset"] for r in features.load()}
        ungated = [p["name"] for p in reg["presets"] if p["local"] and presets.family(p["name"]) not in fams]
        self.assertEqual(ungated, [])

    def test_preset_writing_a_key_localbench_does_not_own_does_not_load(self):
        tmp = Path(tempfile.mkdtemp(prefix="presets-reg-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for op in ({"op": "set", "key": "ttsr.disabledRules", "value": []},
                   {"op": "role", "role": "default", "selector": "ollama/x"},
                   {"op": "set", "key": "modelRoles", "value": {"judge": "x/y"}},
                   {"op": "list_add", "key": "ttsr.disabledRules", "item": "r"}):
            path = tmp / "presets.json"
            path.write_text(json.dumps({"presets": [{"name": "bad:one", "local": False, "ops": [op]}]}))
            with self.subTest(op=op), self.assertRaises(ValueError):
                presets.load(path)


class Planning(Env):
    def test_plan_changes_only_owned_entries_and_writes_nothing(self):
        p = presets.plan("judge:nimble", ["omp-test"], "test", **self.kw())
        role = next(s for s in p["steps"] if s["key"] == "modelRoles")
        self.assertEqual(role["after"], {"default": "muse/x", "smol": "ollama/qwen3.8:27b-mlx",
                                         "judge": "localbench-sys1/nimble:latest"})
        prov = next(s for s in p["steps"] if s["kind"] == "provider")
        self.assertIn("    baseUrl: http://127.0.0.1:11300/omp-profile/omp-test\n", prov["after"])
        self.assertIn("    api: typesafe\n", prov["after"])
        off = presets.plan("scout:off", ["omp-test"], "test", **self.kw())["steps"][0]
        on = presets.plan("scout:on", ["omp-test"], "test", **self.kw())["steps"][0]
        self.assertEqual((off["before"], off["after"], on["after"], on["changed"]),
                         (["sonic"], ["sonic", "scout"], ["sonic"], False))
        self.assertEqual(self.snapshot("omp-test"), self.originals["omp-test"])

    def test_only_running_sessions_on_a_changed_profile_are_named_as_keeping_the_old_thinking_level(self):
        self.config("claude").write_text('defaultThinkingLevel: "high"\n')
        sessions = [{"pid": 101, "profile": "default"}, {"pid": 102, "profile": "claude"},
                    {"pid": 103, "profile": "grok"}]
        p = presets.plan("thinking:fixed-high", ["default", "claude"], "live", **self.kw(sessions=sessions))
        self.assertEqual([(k["pid"], k["key"], k["old"], k["new"]) for k in p["keeps_old"]],
                         [(101, "defaultThinkingLevel", "auto", "high")])
        self.assertTrue(any("101" in n and "'auto'" in n for n in p["notes"]))
        quiet = presets.plan("scout:off", ["default"], "live", **self.kw(sessions=sessions))
        self.assertEqual(quiet["keeps_old"], [])

    def test_test_target_refuses_a_non_test_profile_even_when_forced(self):
        with self.assertRaises(presets.PresetRefused):
            presets.apply("thinking:fixed-high", ["omp-test", "default"], "test", force=True, **self.kw())
        self.assertEqual(self.snapshot("default"), self.originals["default"])
        self.assertEqual(self.snapshot("omp-test"), self.originals["omp-test"])


class Applying(Env):
    def test_apply_backs_up_writes_reads_back_and_rollback_restores_identical_bytes(self):
        m = presets.apply("judge:nimble", ["omp-test", "omp-test2"], "test", **self.kw())
        bdir = self.home / ".localbench" / "rollback" / f"preset-{m['id']}"
        self.assertEqual((bdir / "omp-test.config.yml").read_bytes(), CONFIG.encode())
        self.assertEqual((bdir / "omp-test.models.yml").read_text(), MODELS)
        self.assertFalse((bdir / "omp-test2.models.yml").exists())
        for prof in ("omp-test", "omp-test2"):
            self.assertEqual(presets.omp_get(prof, "modelRoles")["judge"], "localbench-sys1/nimble:latest")
            self.assertEqual(features.providers(prof)["localbench-sys1"],
                             {"baseUrl": f"http://127.0.0.1:11300/omp-profile/{prof}", "api": "typesafe"})
        self.assertEqual(features.providers("omp-test")["ollama-sys1"]["api"], "typesafe")
        self.assertEqual(presets.applied()["omp-test"]["judge"]["id"], m["id"])
        presets.rollback(m["id"])
        for prof in ("omp-test", "omp-test2"):
            self.assertEqual(self.snapshot(prof), self.originals[prof])
        self.assertEqual(presets.applied()["omp-test"], {})

    def test_readback_mismatch_restores_every_profile_file_and_raises(self):
        os.environ["FAKE_OMP_IGNORE"] = "modelRoles"
        with self.assertRaises(presets.PresetError) as caught:
            presets.apply("judge:tev1", ["omp-test", "omp-test2"], "test", **self.kw())
        self.assertIn("readback mismatch", str(caught.exception))
        for prof in ("omp-test", "omp-test2"):
            self.assertEqual(self.snapshot(prof), self.originals[prof])
        self.assertEqual(presets.applied(), {})
        (manifest,) = (self.home / ".localbench" / "rollback").glob("preset-*/manifest.json")
        self.assertEqual(json.loads(manifest.read_text())["status"], "rolled-back")

    def test_live_writes_each_profile_through_its_own_omp_profile_whatever_the_caller_profile(self):
        os.environ["OMP_PROFILE"] = "claude"
        presets.apply("thinking:fixed-high", ["default", "omp-test"], "live", **self.kw())
        self.assertEqual(presets.omp_get("default", "defaultThinkingLevel"), "high")
        self.assertEqual(presets.omp_get("omp-test", "defaultThinkingLevel"), "high")
        self.assertEqual(self.snapshot("claude"), self.originals["claude"])

    def test_live_local_preset_needs_every_family_feature_proven_unless_forced(self):
        with self.assertRaises(presets.PresetRefused):
            presets.apply("judge:nimble", ["omp-test"], "live", **self.kw())
        self.receipt("auto-thinking")
        self.receipt("find-judgments", sha="0" * 64)
        with self.assertRaises(presets.PresetRefused) as caught:
            presets.apply("judge:nimble", ["omp-test"], "live", **self.kw())
        self.assertRegex(str(caught.exception), r"find-judgments@omp-test['\"]: ['\"]STALE")
        self.assertNotIn("auto-thinking", str(caught.exception))
        self.receipt("find-judgments", pinned=DIGESTS["tev1:latest"][:12])
        with self.assertRaises(presets.PresetRefused) as caught:
            presets.apply("judge:nimble", ["omp-test"], "live", **self.kw())
        self.assertRegex(str(caught.exception), r"find-judgments@omp-test['\"]: ['\"]UNPROVEN: proves model digest")
        self.assertEqual(self.snapshot("omp-test"), self.originals["omp-test"])
        forced = presets.apply("judge:nimble", ["omp-test"], "live", force=True, **self.kw())
        self.assertIn("find-judgments", forced["forced"])
        audit = [json.loads(x) for x in (self.home / ".localbench/presets/audit.jsonl").read_text().splitlines()]
        self.assertTrue(audit[-1]["forced"])
        presets.rollback(forced["id"])
        self.receipt("find-judgments")
        proven = presets.apply("judge:nimble", ["omp-test"], "live", **self.kw())
        self.assertIsNone(proven["forced"])
        self.assertEqual({q["model"] for per in proven["plan"]["proof"].values() for q in per.values()},
                         {"nimble:latest"})

    def test_live_proof_is_graded_against_each_target_profiles_own_current_route(self):
        self.config("omp-test2").write_text(CONFIG.replace("typesafe/proj-b-latest", "typesafe/proj-b-0.9"))
        self.receipt("auto-thinking")
        self.receipt("find-judgments")
        with self.assertRaises(presets.PresetRefused) as caught:
            presets.apply("judge:nimble", ["omp-test", "omp-test2"], "live", **self.kw())
        refused = str(caught.exception)
        for feature in ("auto-thinking", "find-judgments"):
            self.assertRegex(refused, rf"{feature}@omp-test2['\"]: ['\"]UNPROVEN: baseline hosted proj-b-latest is not "
                                      r"this profile's route typesafe/proj-b-0.9")
        self.assertNotRegex(refused, r"@omp-test['\"]")
        self.assertEqual(self.snapshot("omp-test"), self.originals["omp-test"])


class DriftAndRollback(Env):
    def test_drift_reports_owned_values_changed_after_apply_and_ignores_unowned_ones(self):
        presets.apply("scout:off", ["omp-test"], "test", **self.kw())
        presets.apply("judge:tev1", ["omp-test"], "test", **self.kw())
        self.assertEqual(presets.drift(), [])
        roles = presets.omp_get("omp-test", "modelRoles")
        presets.omp_set("omp-test", "modelRoles", {**roles, "default": "other/model"})
        self.assertEqual(presets.drift(), [])
        presets.omp_set("omp-test", "task.disabledAgents", ["sonic"])
        presets.omp_set("omp-test", "modelRoles", {**roles, "judge": "typesafe/proj-b-latest"})
        self.models("omp-test").write_text(MODELS)
        found = {(d["preset"], d["key"]): d["actual"] for d in presets.drift(["omp-test"])}
        self.assertEqual(found, {("scout:off", "task.disabledAgents"): ["sonic"],
                                 ("judge:tev1", "modelRoles.judge"): "typesafe/proj-b-latest",
                                 ("judge:tev1", "providers.localbench-sys1"): None})
        self.assertEqual(presets.drift(["omp-test2"]), [])

    def test_rollback_refuses_to_undo_a_later_change_unless_forced(self):
        m = presets.apply("thinking:fixed-high", ["omp-test"], "test", **self.kw())
        presets.omp_set("omp-test", "defaultThinkingLevel", "low")
        with self.assertRaises(presets.PresetError):
            presets.rollback(m["id"])
        self.assertEqual(presets.omp_get("omp-test", "defaultThinkingLevel"), "low")
        presets.rollback(m["id"], force=True)
        self.assertEqual(self.snapshot("omp-test"), self.originals["omp-test"])


class Confinement(Env):
    def test_profile_aliases_escapes_and_symlinked_agent_dirs_are_refused_before_any_write(self):
        escaped = self.home / ".omp" / "claude-escape" / "agent"  # where profiles/../claude-escape lands
        escaped.mkdir(parents=True)
        (escaped / "config.yml").write_bytes(CONFIG.encode())
        outside = self.tmp / "outside" / "agent"
        outside.mkdir(parents=True)
        (outside / "config.yml").write_bytes(CONFIG.encode())
        linked = self.home / ".omp" / "profiles" / "linked"
        linked.mkdir(parents=True)
        (linked / "agent").symlink_to(outside, target_is_directory=True)
        for name in ("../claude-escape", "./claude", "linked"):
            with self.subTest(profile=name), self.assertRaises(presets.PresetError):
                presets.apply("thinking:fixed-high", [name], "live", **self.kw())
        self.assertEqual((escaped / "config.yml").read_bytes(), CONFIG.encode())
        self.assertEqual((outside / "config.yml").read_bytes(), CONFIG.encode())
        self.assertEqual(self.snapshot("claude"), self.originals["claude"])
        self.assertFalse((self.home / ".localbench" / "rollback").exists())

    def test_failed_restore_leaves_a_manifest_that_rollback_finishes(self):
        os.environ["FAKE_OMP_FAIL"] = "omp-test2"
        real = presets._atomic_write

        def disk_full_on_config(path, data, mode=0o600):
            if Path(path).name == "config.yml":
                raise OSError(28, "No space left on device")
            return real(path, data, mode)

        with mock.patch.object(presets, "_atomic_write", disk_full_on_config), \
                self.assertRaises(presets.PresetError) as caught:
            presets.apply("thinking:fixed-high", ["omp-test", "omp-test2"], "test", **self.kw())
        (path,) = (self.home / ".localbench" / "rollback").glob("preset-*/manifest.json")
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["status"], "restore_failed")
        self.assertIn(manifest["id"], str(caught.exception))
        self.assertNotEqual(self.snapshot("omp-test"), self.originals["omp-test"])
        presets.rollback(manifest["id"])
        for prof in ("omp-test", "omp-test2"):
            self.assertEqual(self.snapshot(prof), self.originals[prof])
        self.assertEqual(json.loads(path.read_text())["status"], "rolled-back")


if __name__ == "__main__":
    unittest.main()
