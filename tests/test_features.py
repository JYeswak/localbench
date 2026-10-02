"""The omp feature map (localbench/features.py): live route per profile from omp's resolved settings, honoring
task.disabledAgents, and proof status from receipts bound to the sha256 of the installed omp module. Runs a fake omp
package (bin/omp answering `config list --json` from fixture settings) under a temp HOME; no model, no network."""

import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import features

FAKE_OMP = """import json, sys
from pathlib import Path
args = sys.argv[1:]
if args[-3:] != ["config", "list", "--json"]:
    sys.exit(f"fake omp: unsupported {args}")
profile = args[args.index("--profile") + 1] if "--profile" in args else "default"
settings = json.loads((Path(sys.argv[0]).resolve().parent / "settings" / f"{profile}.json").read_text())
print(json.dumps({k: {"value": v} for k, v in settings.items()}))
"""

QWEN = "ollama/qwen3.8:27b-mlx"
NIMBLE = "ollama/nimble:latest"
HOSTED = "typesafe/proj-b-latest"
# Installed ollama digests (full, as /api/tags reports them) and the 12-hex pins a run records (backends.py).
DIGESTS = {"nimble:latest": "24e550a16a70" + "a" * 52, "qwen3.8:27b-mlx": "5642e97495e1" + "b" * 52}
NIMBLE_PIN, QWEN_PIN = DIGESTS["nimble:latest"][:12], DIGESTS["qwen3.8:27b-mlx"][:12]
DROP = object()
BYPASS = "bypasses the localbench gateway"


class OmpHome:
    """A temp HOME with omp profile configs, and a fake omp package whose bin/omp prints each profile's settings."""

    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.pkg = root / "node_modules" / "@oh-my-pi" / "pi-coding-agent"
        (self.pkg / "src" / "config").mkdir(parents=True)
        (self.pkg / "src" / "config" / "model-resolver.ts").write_text("// resolver\n")
        (root / "settings").mkdir()
        (root / "fake_omp.py").write_text(FAKE_OMP)
        self.bin = self.pkg / "bin" / "omp"
        self.bin.parent.mkdir()
        self.bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{root / "fake_omp.py"}" "$@"\n')
        self.bin.chmod(self.bin.stat().st_mode | stat.S_IXUSR)

    def profile(self, name: str, roles: dict, disabled: tuple = (), flow: bool = False, settings: dict | None = None,
                models_yml: str | None = None):
        agent = self.home / ".omp" / ("agent" if name == "default" else f"profiles/{name}/agent")
        agent.mkdir(parents=True, exist_ok=True)
        text = "modelRoles:\n" + "".join(f"  {k}: {v}\n" for k, v in roles.items())
        if disabled:
            text += ("task:\n  enableLsp: true\n  disabledAgents: [" + ", ".join(disabled) + "]\n" if flow else
                     "task:\n  disabledAgents:\n" + "".join(f"    - {a}\n" for a in disabled) + "  enableLsp: true\n")
        (agent / "config.yml").write_text(text)
        if models_yml is not None:
            (agent / "models.yml").write_text(models_yml)
        (self.root / "settings" / f"{name}.json").write_text(json.dumps(
            {"modelRoles": roles, "task.disabledAgents": list(disabled), **(settings or {})}))

    def module(self, package: str, rel: str, text: str) -> Path:
        path = self.pkg.parent / package / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def env(self):
        return mock.patch.dict(os.environ, {"HOME": str(self.home), "LOCALBENCH_OMP": str(self.bin)})


HEADER = "\t".join(features.COLUMNS)
ROWS = [
    ("omp", "scout-agent", "pi-coding-agent", "src/prompts/agents/scout.md", "scout", "smol", "agent", "scout", "-",
     "scout", "-"),
    ("omp", "auto-thinking", "pi-coding-agent", "src/auto-thinking/classifier.ts", "classifyDifficulty",
     "judge>tiny>smol", "setting", "defaultThinkingLevel=auto", "decision", "judge", "-"),
    ("omp", "titles", "pi-coding-agent", "src/utils/title-generator.ts", "generateSessionTitle", "tiny>commit>smol",
     "model_role", "tiny", "-", "-", "-"),
    ("omp", "mnemopi-extraction", "pi-mnemopi", "src/core/extraction.ts", "extractFactCategories", "memory>tiny>smol",
     "setting", "memory.backend=mnemopi&mnemopi.llmMode=smol", "mem", "memory", "-"),
    ("omp", "find-judgments", "pi-coding-agent", "src/tools/jfind/index.ts", "FindTool", "judge>tiny>smol", "setting",
     "find.enabled=on|find.enabled=auto&native-judge", "decision", "judge", "-"),
]

# The local System One provider as ~/.omp/agent/models.yml declares it (2026-09-30).
SYS1_MODELS = """providers:
  # Local System One via Ollama 0.35
  ollama-sys1:
    baseUrl: http://127.0.0.1:11434
    api: typesafe
    apiKey: ollama-local-no-key
    models:
      - id: nimble:latest
  remote-sys1:
    baseUrl: "https://judge.example.com"
    api: typesafe
"""

# The localbench-managed ollama provider: through the gateway, not ollama's own port.
GATEWAY_MODELS = """providers:
  # >>> localbench ollama residency (managed)
  ollama:
    baseUrl: http://127.0.0.1:11300/omp-profile/gw
    api: openai-responses
"""


def tsv(rows) -> str:
    return "# comment\n" + HEADER + "\n" + "".join("\t".join(r) + "\n" for r in rows)


class FeatureMap(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="features-"))
        self.omp = OmpHome(self.tmp)
        self.registry = self.tmp / "features.tsv"
        self.registry.write_text(tsv(ROWS))
        self.receipts = self.tmp / "receipts"
        self.receipts.mkdir()
        for package, rel in {(r[2], r[3]) for r in ROWS}:
            self.omp.module(package, rel, f"// {rel}\n")
        local = {"defaultThinkingLevel": "auto", "memory.backend": "mnemopi", "mnemopi.llmMode": "smol",
                 "find.enabled": "auto"}
        self.omp.profile("default", {"smol": QWEN, "judge": NIMBLE}, disabled=("scout",), settings=local)
        self.omp.profile("lab", {"smol": QWEN, "judge": NIMBLE}, settings=local)
        self.omp.profile("qjudge", {"smol": QWEN}, settings=local)
        self.omp.profile("gw", {"smol": QWEN, "judge": NIMBLE}, models_yml=GATEWAY_MODELS, settings=local)
        self.omp.profile("hosted", {"smol": "anthropic/claude-haiku-4-5", "judge": "typesafe/proj-b-latest"},
                         settings={"defaultThinkingLevel": "auto", "memory.backend": "off"})
        self.omp.profile("fixed", {"smol": QWEN}, settings={"defaultThinkingLevel": "high"})
        self.omp.profile("sys1", {"smol": QWEN, "judge": "ollama-sys1/nimble:latest"}, models_yml=SYS1_MODELS,
                         settings={"find.enabled": "auto"})
        self.omp.profile("chain", {"smol": QWEN, "judge": ["typesafe/proj-b-latest", "@smol"]},
                         settings={"find.enabled": "auto"})
        # judge:nimble was applied over the hosted TypeSafe judge on these profiles; memory:qwen38 over llmMode none
        # on lab. lab's judge route was set by hand (no apply record), so its incumbent is its current route.
        hosted_judge = [{"profile": None, "kind": "set", "key": "modelRoles",
                         "before": {"smol": QWEN, "judge": HOSTED}, "after": None}]
        for prof in ("default", "qjudge", "gw", "sys1"):
            self.apply(prof, "judge:nimble", f"20261001T0300Z-{prof[:4]}", hosted_judge)
        self.apply("lab", "memory:qwen38", "20261001T0400Z-mem0",
                   [{"profile": None, "kind": "set", "key": "mnemopi.llmMode", "before": "none", "after": "smol"}])

    def apply(self, profile: str, preset: str, rid: str, steps: list[dict], manifest: bool = True):
        """Record a presets apply the way presets.apply() does: the applied.json entry, and the rollback manifest
        whose plan steps carry each written key's `before` value."""
        root = self.omp.home / ".localbench"
        state_path = root / "presets" / "applied.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        state.setdefault(profile, {})[preset.split(":")[0]] = {"preset": preset, "id": rid, "at": rid, "expect": []}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state))
        if manifest:
            path = root / "rollback" / f"preset-{rid}" / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"id": rid, "preset": preset, "status": "applied",
                                        "plan": {"steps": [{**s, "profile": profile} for s in steps]}}))

    def sha(self, package: str, rel: str) -> str:
        return hashlib.sha256((self.omp.pkg.parent / package / rel).read_bytes()).hexdigest()

    def proof(self, name: str, feature: str = "auto-thinking", rel: str = "src/auto-thinking/classifier.ts",
              package: str = "pi-coding-agent", label: str = "decision", digest: str = NIMBLE_PIN,
              baseline: dict | None = None, **changes):
        """Bank a receipt of the real decision/memory run shape (decision.run_suite with a hosted arm: baseline
        {kind hosted, id <hosted model>}) that meets the proof contract, then apply `changes` (dotted path -> value,
        DROP deletes) to plant one missing or contradicting field."""
        doc = {"kind": "run", "feature": feature, "omp_module_sha": self.sha(package, rel), "problems": [],
               "verdict": {"compare": "BETTER", "baseline": baseline or {"kind": "hosted", "id": "proj-b-latest"}},
               "run": {"label": label, "provenance": {"pins": {"model": "nimble:latest", "model_digest": digest}}}}
        for path, value in changes.items():
            *parents, last = path.split(".")
            node = doc
            for key in parents:
                node = node[key]
            if value is DROP:
                del node[last]
            else:
                node[last] = value
        (self.receipts / f"{name}.json").write_text(json.dumps(doc))

    def report(self, profiles):
        with self.omp.env():
            rows = features.report(profiles, registry=self.registry, receipts_dir=self.receipts, digests=DIGESTS)
        return {r["feature"]: r for r in rows}

    def all_findings(self, profiles):
        with self.omp.env():
            return features.doctor_findings(profiles, registry=self.registry, receipts_dir=self.receipts,
                                            digests=DIGESTS)

    def findings(self, profiles):
        """feature -> its proof finding (the gateway-bypass WARNs are bypass())."""
        names = [r[1] for r in ROWS]
        return {next(n for n in names if message.startswith((n + " ", n + ":"))): (level, message, fix)
                for level, message, fix in self.all_findings(profiles) if BYPASS not in message}

    def bypass(self, profiles):
        return {message.split(" ", 1)[0]: (level, message, fix)
                for level, message, fix in self.all_findings(profiles) if BYPASS in message}

    def test_disabled_agent_is_disabled_where_listed_and_a_local_route_where_not(self):
        scout = self.report(["default", "lab"])["scout-agent"]
        self.assertEqual(scout["routes"]["default"],
                         {"target": None, "local": False, "disabled": "task.disabledAgents lists scout"})
        self.assertEqual(scout["routes"]["lab"], {"target": QWEN, "local": True, "disabled": None})
        self.assertEqual(scout["local_profiles"], ["lab"])
        self.assertEqual(scout["status"], "UNPROVEN")
        level, message, _ = self.findings(["default", "lab"])["scout-agent"]
        self.assertEqual(level, "FAIL")
        self.assertIn("lab", message)
        self.assertNotIn("default", message)

    def test_agent_disabled_in_every_profile_is_disabled_and_not_a_failure(self):
        scout = self.report(["default"])["scout-agent"]
        self.assertEqual(scout["status"], "DISABLED")
        self.assertEqual(scout["local_profiles"], [])
        self.assertEqual(self.findings(["default"])["scout-agent"][0], "PASS")

    def test_disabled_agents_gate_only_the_agent_they_list(self):
        """scout:off writes task.disabledAgents [scout]: scout-agent goes DISABLED with that reason, sonic-agent on
        the same @smol keeps its local route; a profile without the entry keeps scout's local route."""
        sonic = ("omp", "sonic-agent", "pi-coding-agent", "src/task/agents.ts", "sonic", "smol", "agent", "sonic", "-",
                 "sonic", "-")
        self.registry.write_text(tsv([*ROWS, sonic]))
        self.omp.module("pi-coding-agent", "src/task/agents.ts", "// src/task/agents.ts\n")
        off = self.report(["default"])
        self.assertEqual((off["scout-agent"]["status"], off["scout-agent"]["routes"]["default"]["disabled"]),
                         ("DISABLED", "task.disabledAgents lists scout"))
        self.assertEqual(off["sonic-agent"]["routes"]["default"], {"target": QWEN, "local": True, "disabled": None})
        self.assertEqual(off["sonic-agent"]["status"], "UNPROVEN")
        on = self.report(["lab"])
        self.assertEqual(on["scout-agent"]["routes"]["lab"], {"target": QWEN, "local": True, "disabled": None})
        self.assertEqual(on["scout-agent"]["status"], "UNPROVEN")
        self.assertEqual(on["sonic-agent"]["routes"]["lab"], off["sonic-agent"]["routes"]["default"])

    def test_local_route_without_receipt_is_unproven_and_a_doctor_failure(self):
        row = self.report(["default"])["auto-thinking"]
        self.assertEqual(row["routes"]["default"]["target"], NIMBLE)
        self.assertEqual((row["proof"], row["status"], row["receipt"]), ("UNPROVEN", "UNPROVEN", None))
        level, message, fix = self.findings(["default"])["auto-thinking"]
        self.assertEqual(level, "FAIL")
        self.assertIn("UNPROVEN", message)
        sha = self.sha("pi-coding-agent", "src/auto-thinking/classifier.ts")
        self.assertIn(f"omp_module_sha={sha}", fix)

    def test_contract_receipt_for_module_and_routed_model_proves_the_feature(self):
        self.proof("decision-auto-thinking")
        row = self.report(["default"])["auto-thinking"]
        self.assertEqual((row["status"], row["receipt"], row["proofs"]["default"]["digest"]),
                         ("PROVEN", "decision-auto-thinking.json", DIGESTS["nimble:latest"]))
        self.assertEqual(self.findings(["default"])["auto-thinking"][0], "PASS")

    def test_generation_label_receipt_proves_the_feature_like_decision_and_memory(self):
        for label in ("decision", "memory", "generation"):
            with self.subTest(label=label):
                for f in self.receipts.glob("*.json"):
                    f.unlink()
                self.proof(f"gen-{label}", label=label)
                row = self.report(["default"])["auto-thinking"]
                self.assertEqual((row["status"], row["receipt"]), ("PROVEN", f"gen-{label}.json"))

    def test_each_missing_or_contradicting_contract_field_denies_proof_with_its_reason(self):
        cases = {
            "not a run receipt": ({"kind": "aa"}, "UNPROVEN", "not a decision/memory/generation run receipt"),
            "kind missing": ({"kind": DROP}, "UNPROVEN", "not a decision/memory/generation run receipt"),
            "run label is another tier": ({"run.label": "rel"}, "UNPROVEN", "run.label 'rel'"),
            "run label missing": ({"run.label": DROP}, "UNPROVEN", "run.label None"),
            "module sha of another build": ({"omp_module_sha": "0" * 64}, "STALE", "installed is"),
            "module sha missing": ({"omp_module_sha": DROP}, "STALE", "proved omp_module_sha None"),
            "unsound run": ({"problems": ["unknown residency"]}, "BAD", "unknown residency"),
            "not better than the baseline": ({"verdict.compare": "NOT_BETTER"}, "BAD", "NOT_BETTER, not BETTER"),
            "worse than the baseline": ({"verdict.compare": "WORSE"}, "BAD", "WORSE, not BETTER"),
            "no comparison": ({"verdict.compare": DROP}, "UNPROVEN", "no verdict.compare"),
            "no verdict at all": ({"verdict": DROP}, "UNPROVEN", "no verdict.compare"),
            "no baseline": ({"verdict.baseline": DROP}, "UNPROVEN", "no baseline"),
            "baseline without an id": ({"verdict.baseline.id": None}, "UNPROVEN", "no baseline"),
            "baseline is another hosted model": ({"verdict.baseline.id": "claude-haiku-4-5"}, "UNPROVEN",
                                                 f"is not this profile's route {HOSTED}"),
            "baseline kind misnames the incumbent": ({"verdict.baseline": {"kind": "route", "id": QWEN}}, "UNPROVEN",
                                                     f"baseline route {QWEN} is not this profile's route {HOSTED}"),
            "no model digest pinned": ({"run.provenance.pins.model_digest": DROP}, "UNPROVEN", "no model_digest"),
            "another model's digest": ({"run.provenance.pins.model_digest": QWEN_PIN}, "UNPROVEN",
                                       f"proves model digest {QWEN_PIN}"),
            "digest shorter than a pin": ({"run.provenance.pins.model_digest": NIMBLE_PIN[:6]}, "UNPROVEN",
                                          "proves model digest"),
        }
        for label, (changes, status, reason) in cases.items():
            with self.subTest(label):
                for f in self.receipts.glob("*.json"):
                    f.unlink()
                self.proof("decision-auto-thinking", **changes)
                row = self.report(["default"])["auto-thinking"]
                self.assertEqual(row["status"], status)
                self.assertIn(reason, row["reason"])
                level, message, _ = self.findings(["default"])["auto-thinking"]
                self.assertEqual(level, "FAIL")
                self.assertIn(reason, message)

    def test_baseline_must_be_the_route_the_applied_local_preset_replaced(self):
        self.proof("nimble-vs-qwen", baseline={"kind": "route", "id": QWEN})
        row = self.report(["default"])["auto-thinking"]
        self.assertEqual(row["status"], "UNPROVEN")
        self.assertIn(f"baseline route {QWEN} is not this profile's route {HOSTED} "
                      "(before preset judge:nimble apply 20261001T0300Z-defa)", row["reason"])
        self.assertEqual(self.findings(["default"])["auto-thinking"][0], "FAIL")
        self.proof("nimble-vs-hosted", baseline={"kind": "hosted", "id": HOSTED})
        self.assertEqual(self.report(["default"])["auto-thinking"]["status"], "PROVEN")

    def test_without_a_local_preset_apply_the_baseline_is_the_current_route_never_the_model_itself(self):
        # lab routes auto-thinking to local nimble and no judge preset apply is recorded: the incumbent is that route.
        # It is proven only against a declared non-local alternative of its family (judge:hosted -> proj-b-latest).
        self.proof("nimble-vs-hosted")
        row = self.report(["lab"])["auto-thinking"]
        self.assertEqual((row["status"], row["reason"]), ("PROVEN", ""))
        self.assertEqual([a["source"] for a in row["proofs"]["lab"]["alternatives"]], ["alternative judge:hosted"])
        self.proof("nimble-vs-hosted", baseline={"kind": "route", "id": NIMBLE})
        row = self.report(["lab"])["auto-thinking"]
        self.assertEqual(row["status"], "UNPROVEN")
        self.assertIn("is the routed model itself", row["reason"])
        # Another local model route is never a declared alternative.
        self.proof("nimble-vs-hosted", baseline={"kind": "route", "id": "localbench-sys1/tev1:latest"})
        row = self.report(["lab"])["auto-thinking"]
        self.assertEqual(row["status"], "UNPROVEN")
        self.assertIn("nor a declared alternative: typesafe/proj-b-latest (alternative judge:hosted)", row["reason"])

    def test_with_a_local_apply_record_the_recorded_route_stays_the_only_baseline(self):
        # judge:nimble applied over local tev1 on lab: the baseline must be tev1, not the declared hosted alternative.
        self.apply("lab", "judge:nimble", "20261001T0600Z-lab0",
                   [{"profile": None, "kind": "set", "key": "modelRoles",
                     "before": {"smol": QWEN, "judge": "ollama/tev1:latest"}, "after": None}])
        self.proof("nimble-vs-hosted")
        proof = self.report(["lab"])["auto-thinking"]["proofs"]["lab"]
        self.assertIsNone(proof["alternatives"])
        self.assertEqual(proof["proof"], "UNPROVEN")
        self.assertIn("is not this profile's route ollama/tev1:latest (before preset judge:nimble", proof["reason"])

    def test_pre_flip_incumbent_is_the_current_route_by_kind(self):
        row = dict(zip(features.COLUMNS, ROWS[1], strict=True))
        provs: dict = {}
        hosted = features.incumbent(row, {"defaultThinkingLevel": "auto", "modelRoles": {"judge": HOSTED}}, provs, "")
        fixed = features.incumbent(row, {"defaultThinkingLevel": "high", "modelRoles": {"judge": NIMBLE}}, provs, "")
        self.assertTrue(features.baseline_matches({"kind": "hosted", "id": "proj-b-latest"}, hosted))
        self.assertFalse(features.baseline_matches({"kind": "hosted", "id": "proj-b-latest"}, fixed))
        self.assertTrue(features.baseline_matches({"kind": "fixed", "id": "high"}, fixed))
        self.assertFalse(features.baseline_matches({"kind": "fixed", "id": "auto"}, fixed))
        self.assertFalse(features.baseline_matches({"kind": "hosted", "id": "nimble:latest"},
                                                   features.incumbent(row, {"defaultThinkingLevel": "auto",
                                                                            "modelRoles": {"judge": NIMBLE}},
                                                                      provs, "now")))

    def test_unreadable_apply_manifest_denies_proof(self):
        self.apply("default", "judge:nimble", "20261001T0500Z-gone", [], manifest=False)
        self.proof("nimble-vs-hosted")
        row = self.report(["default"])["auto-thinking"]
        self.assertEqual(row["status"], "UNPROVEN")
        self.assertIn("20261001T0500Z-gone of judge:nimble has no readable manifest", row["reason"])

    def test_receipt_for_another_feature_proves_nothing_here(self):
        self.proof("titles-on-classifier", feature="titles")
        self.assertEqual(self.report(["default"])["auto-thinking"]["status"], "UNPROVEN")

    def test_proof_applies_only_to_profiles_routing_the_measured_model(self):
        self.proof("decision-auto-thinking")
        row = self.report(["default", "qjudge"])["auto-thinking"]
        self.assertEqual(row["routes"]["qjudge"]["target"], QWEN)
        self.assertEqual((row["proofs"]["default"]["proof"], row["proofs"]["qjudge"]["proof"]), ("PROVEN", "UNPROVEN"))
        self.assertEqual(row["status"], "UNPROVEN")
        level, message, _ = self.findings(["default", "qjudge"])["auto-thinking"]
        self.assertEqual(level, "FAIL")
        self.assertIn("qjudge", message)
        self.assertNotIn("default", message)

    def test_route_model_not_installed_is_unproven(self):
        self.proof("decision-auto-thinking")
        with self.omp.env():
            rows = features.report(["default"], registry=self.registry, receipts_dir=self.receipts, digests={})
        row = next(r for r in rows if r["feature"] == "auto-thinking")
        self.assertEqual(row["status"], "UNPROVEN")
        self.assertIn("no installed digest", row["reason"])

    def test_module_edited_after_its_receipt_is_stale_and_a_doctor_failure(self):
        old = self.sha("pi-coding-agent", "src/auto-thinking/classifier.ts")
        self.proof("decision-auto-thinking")
        self.omp.module("pi-coding-agent", "src/auto-thinking/classifier.ts", "// omp update changed the classifier\n")
        row = self.report(["default"])["auto-thinking"]
        self.assertEqual((row["status"], row["receipt_sha"]), ("STALE", old))
        self.assertNotEqual(row["omp_module_sha"], old)
        level, message, _ = self.findings(["default"])["auto-thinking"]
        self.assertEqual(level, "FAIL")
        self.assertIn("STALE", message)

    def test_memory_receipt_proves_the_sibling_package_module(self):
        self.proof("mem-extraction", feature="mnemopi-extraction", rel="src/core/extraction.ts",
                   package="pi-mnemopi", label="memory", digest=QWEN_PIN, baseline={"kind": "fixed", "id": "none"})
        row = self.report(["lab"])["mnemopi-extraction"]
        self.assertEqual((row["status"], row["routes"]["lab"]["target"]), ("PROVEN", QWEN))

    def test_carried_forward_receipt_is_carried_and_only_a_warning(self):
        self.proof("carried", carried_forward={"from_sha": "0" * 64, "at": "2026-10-01T00:00:00Z"})
        self.assertEqual(self.report(["default"])["auto-thinking"]["status"], "CARRIED")
        self.assertEqual(self.findings(["default"])["auto-thinking"][0], "WARN")

    def test_hosted_route_without_proof_is_not_a_failure(self):
        row = self.report(["hosted"])["auto-thinking"]
        self.assertEqual(row["routes"]["hosted"], {"target": "typesafe/proj-b-latest", "local": False, "disabled": None})
        self.assertEqual(row["status"], "UNPROVEN")
        self.assertEqual(self.findings(["hosted"])["auto-thinking"][0], "PASS")

    def test_setting_gate_disables_the_route_and_role_chain_falls_through_to_smol(self):
        rows = self.report(["fixed"])
        self.assertEqual(rows["auto-thinking"]["status"], "DISABLED")
        self.assertIn("defaultThinkingLevel=high", rows["auto-thinking"]["routes"]["fixed"]["disabled"])
        self.assertEqual(rows["titles"]["routes"]["fixed"], {"target": QWEN, "local": True, "disabled": None})

    def test_local_system_one_provider_is_a_native_local_judge_and_an_unproven_failure(self):
        row = self.report(["sys1"])["find-judgments"]
        self.assertEqual(row["routes"]["sys1"],
                         {"target": "ollama-sys1/nimble:latest", "local": True, "disabled": None})
        self.assertEqual(self.findings(["sys1"])["find-judgments"][0], "FAIL")
        with self.omp.env():
            provs = features.providers("sys1")
        self.assertFalse(features.is_local("remote-sys1/proj-b", provs))

    def test_local_route_straight_to_ollama_warns_it_bypasses_the_gateway(self):
        self.proof("find-sys1", feature="find-judgments", rel="src/tools/jfind/index.ts")
        self.assertEqual(self.findings(["sys1"])["find-judgments"][0], "PASS")
        level, message, fix = self.bypass(["sys1"])["find-judgments"]
        self.assertEqual(level, "WARN")
        self.assertIn("sys1: ollama-sys1/nimble:latest via http://127.0.0.1:11434", message)
        self.assertIn("http://127.0.0.1:11300/omp-profile/sys1", fix)
        self.assertIn("auto-thinking", self.bypass(["default"]))  # built-in ollama, undeclared: ollama's own port
        self.assertEqual(self.bypass(["gw"]), {})

    def test_find_under_auto_is_off_while_the_judge_is_a_prompted_model(self):
        row = self.report(["lab"])["find-judgments"]
        self.assertEqual(row["status"], "DISABLED")
        self.assertIn("not native System One", row["routes"]["lab"]["disabled"])

    def test_judge_chain_never_falls_from_a_native_judge_to_a_prompted_local_model(self):
        row = self.report(["chain"])["find-judgments"]
        self.assertEqual(row["routes"]["chain"], {"target": "typesafe/proj-b-latest", "local": False, "disabled": None})
        self.assertEqual(self.findings(["chain"])["find-judgments"][0], "PASS")

    def test_unreachable_omp_is_one_failure_not_a_crash(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.omp.home), "LOCALBENCH_OMP": str(self.tmp / "no-omp")}):
            found = features.doctor_findings(["default"], registry=self.registry, receipts_dir=self.receipts,
                                             digests=DIGESTS)
        self.assertEqual([f[0] for f in found], ["FAIL"])


class Registry(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="features-reg-")) / "features.tsv"

    def test_malformed_rows_are_rejected_with_their_line(self):
        good = list(ROWS[1])
        cases = {
            "missing field": [good[:-1]],
            "unknown route kind": [good[:6] + ["role"] + good[7:]],
            "empty field": [good[:9] + [""] + good[10:]],
            "duplicate feature": [good, good],
            "model_role key is not the chain head": [ROWS[2][:7] + ("smol",) + ROWS[2][8:]],
            "setting key without a value": [good[:7] + ["defaultThinkingLevel"] + good[8:]],
            "module escapes the package": [good[:3] + ["../outside.ts"] + good[4:]],
            "role is not a chain": [good[:5] + ["judge,tiny"] + good[6:]],
        }
        for label, rows in cases.items():
            with self.subTest(label):
                self.path.write_text(tsv(rows))
                with self.assertRaisesRegex(ValueError, r"features\.tsv:\d+: "):
                    features.load(self.path)

    def test_wrong_header_is_rejected(self):
        self.path.write_text("\t".join(features.COLUMNS[:-1]) + "\n" + "\t".join(ROWS[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "header"):
            features.load(self.path)

    def test_shipped_registry_covers_every_audited_feature(self):
        rows = features.load(features.REGISTRY)
        self.assertLessEqual(
            {"auto-thinking", "find-judgments", "ttsr-judge", "unexpected-stop", "mnemopi-extraction",
             "recall-embeddings", "titles", "skill-description-compression", "commit-messages", "edit-auto-repair",
             "scout-agent", "sonic-agent"},
            {r["feature"] for r in rows})


if __name__ == "__main__":
    unittest.main()
