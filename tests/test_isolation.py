"""Measured omp children run in localbench's own agent dir (workloads.child_env): nothing of the user's agent dir (MCP
servers, plugins, settings, memory banks) may reach them. 2026-09-23: with the user's dir, mem controls read other
agents' databases and called the user's Agent Mail MCP server (76 side-effecting calls in one A/B)."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import golden, workloads
from localbench.workloads import AGENT_CONFIG, AGENT_DIR, child_env, ensure_localbench_model, omp_env

OMP = shutil.which("omp") or os.environ.get("LOCALBENCH_OMP") or ""


class ChildEnv(unittest.TestCase):
    def test_children_get_the_harness_agent_dir_and_no_profile_or_claude_overrides(self):
        leaked = {"OMP_PROFILE": "grok", "PI_PROFILE": "grok", "PI_CODING_AGENT_DIR": "/Users/x/.omp/agent",
                  "CLAUDE_CONFIG_DIR": "/Users/x/.claude"}
        with mock.patch.dict(os.environ, leaked):
            env = child_env()
        self.assertEqual(env["PI_CODING_AGENT_DIR"], str(AGENT_DIR))
        for k in ("OMP_PROFILE", "PI_PROFILE", "CLAUDE_CONFIG_DIR"):
            self.assertNotIn(k, env)

    def test_the_harness_dir_holds_the_tracked_base_config_and_no_mcp_config(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(workloads, "AGENT_DIR", Path(tmp)):
            (Path(tmp) / "config.yml").write_text("defaultThinkingLevel: high\n")   # left by an older fixture
            child_env()
            self.assertEqual((Path(tmp) / "config.yml").read_bytes(), AGENT_CONFIG.read_bytes())
            self.assertFalse((Path(tmp) / "mcp.json").exists())

    def test_reads_of_the_users_setup_keep_the_users_dir(self):
        with mock.patch.dict(os.environ, {"PI_CODING_AGENT_DIR": "/elsewhere"}):
            self.assertNotIn("PI_CODING_AGENT_DIR", omp_env())

    def test_the_provider_is_written_to_the_harness_dir(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(workloads, "AGENT_DIR", Path(tmp)):
            ensure_localbench_model("m:1", 262144, extra=("side:2",))
            text = (Path(tmp) / "models.yml").read_text()
        self.assertTrue(text.startswith("providers:\n  localbench:\n    baseUrl: http://127.0.0.1:11299/v1\n"))
        self.assertIn('- id: "m:1"\n        contextWindow: 262144', text)
        self.assertIn('- id: "side:2"', text)


class Binding(unittest.TestCase):
    def test_omp_bound_rows_bind_to_the_childrens_base_config(self):
        for tier in ("e2e", "rel", "mem", "sess"):
            self.assertIn("omp_agent_config", golden.tier_keys(tier), tier)
        for tier in ("conf", "micro", "replay"):
            self.assertNotIn("omp_agent_config", golden.tier_keys(tier), tier)


@unittest.skipUnless(OMP, "omp not installed")
class OmpHonoursTheHarnessDir(unittest.TestCase):
    """omp ships most days: if it stopped honouring PI_CODING_AGENT_DIR, children would silently read the user's
    config (roles, MCP servers) again."""

    def test_settings_resolve_from_the_harness_dir(self):
        out = subprocess.run([OMP, "config", "list", "--json"], capture_output=True, text=True, timeout=60,
                             env=child_env(), check=True).stdout
        cfg = {k: (v or {}).get("value") for k, v in json.loads(out).items()}
        self.assertEqual(cfg.get("modelRoles") or {}, {})       # the user's config routes smol and default
        self.assertEqual(cfg.get("defaultThinkingLevel"), "auto")   # from fixtures/omp/agent/config.yml


if __name__ == "__main__":
    unittest.main()
