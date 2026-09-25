"""After an unpark, every omp profile must be able to see the restored tags. omp caches its ollama discovery per
profile for 24 h, so a catalog taken during a park lacks the parked tag until refreshed (2026-09-24: 6 of 9 profiles).
The fake omp below keeps a stale catalog per profile; `models refresh ollama` replaces it with the server's tags."""

import contextlib
import io
import json
import subprocess
import unittest
from unittest import mock

from localbench import __main__ as cli
from localbench import park

SERVER = {"qwen3.8:27b-mlx", "qwen3.6:35b-mlx", "nomic-embed-text:latest"}
STALE = {"qwen3.6:35b-mlx", "localbench-parked:5642e97495e1"}


class FakeOmp:
    def __init__(self, profiles, broken=()):
        self.catalog = {p: set(STALE) for p in profiles}
        self.broken = set(broken)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        profile = argv[argv.index("--profile") + 1] if "--profile" in argv else "default"
        action = argv[argv.index("models") + 1]
        if action == "refresh":
            if profile not in self.broken:
                self.catalog[profile] = set(SERVER)
            return subprocess.CompletedProcess(argv, 0, "", "")
        rows = [{"provider": "ollama", "id": m} for m in sorted(self.catalog[profile])]
        rows.append({"provider": "openai", "id": "qwen3.8:27b-mlx"})      # another provider's id never counts
        return subprocess.CompletedProcess(argv, 0, json.dumps({"models": rows}), "")


class RefreshCatalogs(unittest.TestCase):
    def refresh(self, names, profiles, broken=()):
        omp = FakeOmp(profiles, broken)
        with mock.patch.object(park, "omp_bin", return_value="omp"):
            return park.refresh_catalogs(names, profiles, run=omp), omp

    def test_every_profile_is_refreshed_then_read_back(self):
        missing, omp = self.refresh(["qwen3.8:27b-mlx"], ["default", "grok"])
        self.assertEqual(missing, {"default": [], "grok": []})
        self.assertEqual(omp.calls, [["omp", "models", "refresh", "ollama"], ["omp", "models", "ollama", "--json"],
                                     ["omp", "--profile", "grok", "models", "refresh", "ollama"],
                                     ["omp", "--profile", "grok", "models", "ollama", "--json"]])

    def test_a_profile_that_still_cannot_see_the_tag_is_named(self):
        missing, _ = self.refresh(["qwen3.8:27b-mlx"], ["default", "muse"], broken={"muse"})
        self.assertEqual(missing, {"default": [], "muse": ["qwen3.8:27b-mlx"]})

    def test_an_untagged_name_is_found_as_latest(self):
        missing, _ = self.refresh(["nomic-embed-text"], ["default"])
        self.assertEqual(missing, {"default": []})


class UnparkExit(unittest.TestCase):
    def test_unpark_fails_while_a_profile_is_blind_to_a_restored_tag(self):
        restored = [{"name": "qwen3.8:27b-mlx", "digest": "5642e97495e1aa", "parked_as": "localbench-parked:5642e97495e1"}]
        with mock.patch.object(park, "parked_now", return_value=restored), \
                mock.patch.object(park, "unpark", return_value=restored), \
                mock.patch.object(cli.models, "profiles", return_value=["default", "grok"]), \
                mock.patch.object(park, "refresh_catalogs", return_value={"default": [], "grok": ["qwen3.8:27b-mlx"]}), \
                mock.patch.object(cli.sysstats, "omp_processes", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = cli.main(["unpark"])
        self.assertEqual(rc, 1)
        self.assertIn("omp catalog grok: still missing qwen3.8:27b-mlx", out.getvalue())


if __name__ == "__main__":
    unittest.main()
