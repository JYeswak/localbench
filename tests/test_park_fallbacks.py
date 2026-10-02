"""Parking a smol model must also park what omp hands its role instead: on 2026-09-23 omp's resolver gave sessions
started while qwen3.8:27b-mlx was parked qwen3.8-uncensored:latest, which they kept for hours after unpark."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

from localbench import park

TARGET = "qwen3.8:27b-mlx"
SIBLING = "qwen3.8-uncensored:latest"


def omp_like(order: list[str]):
    """A resolver that returns the first of `order` still installed, like omp's for its smol selector."""
    return lambda selector, ids: next((n for n in order if n in ids), None)


class Fallbacks(unittest.TestCase):
    def fallbacks(self, installed: list[str], order: list[str]) -> list[str]:
        with mock.patch.object(park, "smol_targets", return_value=[TARGET]):
            return park.fallbacks(omp_like(order), {n: "sha256:x" for n in installed})

    def test_the_sibling_omp_falls_back_to_is_a_fallback(self):
        self.assertEqual(self.fallbacks([TARGET, "qwen3.6:35b-mlx", SIBLING], [TARGET, SIBLING]), [SIBLING])

    def test_every_hop_is_followed_until_omp_resolves_to_nothing(self):
        self.assertEqual(self.fallbacks([TARGET, "a:1", "b:1", "c:1"], [TARGET, "b:1", "a:1"]), ["b:1", "a:1"])

    def test_nothing_is_a_fallback_once_the_sibling_is_parked(self):
        self.assertEqual(self.fallbacks(["qwen3.6:35b-mlx", "localbench-parked:23da7bcdf4d1"], [TARGET, SIBLING]), [])

    def test_a_parked_copy_omp_would_pick_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.fallbacks(["localbench-parked:5642e97495e1"], [TARGET, "localbench-parked:5642e97495e1"])


class FakeOllama:
    """Tags, copy and delete over a dict: enough of ollama's API for park()/unpark()."""

    def __init__(self, tags: dict[str, str]):
        self.tags = dict(tags)

    def install(self, stack: list) -> None:
        for name, fn in (("_tags", lambda: dict(self.tags)), ("_copy", self.copy), ("_delete", self.tags.pop),
                         ("_post", lambda *a, **k: {}), ("_get", lambda url, timeout=10: {"models": []})):
            p = mock.patch.object(park, name, fn)
            p.start()
            stack.append(p)

    def copy(self, src: str, dst: str) -> None:
        self.tags[dst] = self.tags[src]


class ParkRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = []
        self.ollama = FakeOllama({TARGET: "sha256:5642e97495e1aa", SIBLING: "sha256:23da7bcdf4d1bb",
                                  "qwen3.6:35b-mlx": "sha256:cc"})
        self.ollama.install(self.patches)
        p = mock.patch.object(park.gateway, "database_path", return_value=self.tmp / "gateway" / "leases.sqlite")
        p.start()
        self.patches.append(p)
        p = mock.patch.object(park.gateway, "safe_to_unload", return_value=(True, None))
        p.start()
        self.patches.append(p)
        for name, value in (("STATE", self.tmp / "PARKED.json"), ("HISTORY", self.tmp / "park-history.jsonl"),
                            ("smol_targets", lambda: [TARGET])):
            p = mock.patch.object(park, name, value)
            p.start()
            self.patches.append(p)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.tmp)

    def test_park_hides_target_and_sibling_and_unpark_restores_both(self):
        parked = park.park(resolve=omp_like([TARGET, SIBLING]))
        self.assertEqual([(p["name"], p["role"]) for p in parked], [(TARGET, "smol"), (SIBLING, "fallback")])
        self.assertEqual(sorted(self.ollama.tags), ["localbench-parked:23da7bcdf4d1", "localbench-parked:5642e97495e1",
                                                    "qwen3.6:35b-mlx"])
        self.assertIsNone(omp_like([TARGET, SIBLING])("ollama/" + TARGET, list(self.ollama.tags)),
                          "with both parked, omp has nothing to hand a smol role")
        self.assertEqual(json.loads(park.HISTORY.read_text().splitlines()[0]),
                         {"event": "park", "t": mock.ANY, "names": [TARGET, SIBLING], "sealed": True})
        park.unpark()
        self.assertEqual(sorted(self.ollama.tags), ["qwen3.6:35b-mlx", SIBLING, TARGET])

    def test_same_digest_alias_restores_every_original_before_cleanup(self):
        self.ollama.tags[SIBLING] = self.ollama.tags[TARGET]
        parked = park.park(resolve=omp_like([TARGET, SIBLING]))
        alias = parked[0]["parked_as"]
        self.assertEqual(parked[1]["parked_as"], alias)
        self.assertEqual(self.ollama.tags[alias], "sha256:5642e97495e1aa")

        park.unpark()

        self.assertEqual(self.ollama.tags[TARGET], "sha256:5642e97495e1aa")
        self.assertEqual(self.ollama.tags[SIBLING], "sha256:5642e97495e1aa")
        self.assertNotIn(alias, self.ollama.tags)
        self.assertFalse(park.STATE.exists())

    def test_same_digest_alias_survives_later_restore_failure_and_retry(self):
        self.ollama.tags[SIBLING] = self.ollama.tags[TARGET]
        parked = park.park(resolve=omp_like([TARGET, SIBLING]))
        alias = parked[0]["parked_as"]
        self.ollama.tags[SIBLING] = "sha256:changed-by-another-client"

        with self.assertRaisesRegex(RuntimeError, "restored .* digest differs"):
            park.unpark()
        self.assertEqual(self.ollama.tags[alias], "sha256:5642e97495e1aa")
        self.assertTrue(park.STATE.exists())

        self.ollama.tags.pop(SIBLING)
        park.unpark()
        self.assertEqual(self.ollama.tags[TARGET], "sha256:5642e97495e1aa")
        self.assertEqual(self.ollama.tags[SIBLING], "sha256:5642e97495e1aa")
        self.assertNotIn(alias, self.ollama.tags)
        self.assertFalse(park.STATE.exists())

    def test_unpark_refuses_active_gateway_request_on_parked_alias(self):
        parked = park.park(resolve=omp_like([TARGET]))
        alias = parked[0]["parked_as"]
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")
        store.set_profiles({"default": "/test-profile"})
        request_id = store.start_request("default", alias, "active-parked-alias")
        journal_before = park.STATE.read_bytes()
        tags_before = dict(self.ollama.tags)

        with self.assertRaisesRegex(RuntimeError, "gateway request is in flight"):
            park.unpark()

        self.assertEqual(park.STATE.read_bytes(), journal_before)
        self.assertEqual(self.ollama.tags, tags_before)
        self.assertIsNone(store.park_fence(alias))
        store.finish_request(request_id, completed=True, outcome="test")

    def test_unpark_refuses_external_activity_on_parked_alias(self):
        parked = park.park(resolve=omp_like([TARGET]))
        alias = parked[0]["parked_as"]
        journal_before = park.STATE.read_bytes()
        tags_before = dict(self.ollama.tags)
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")

        with mock.patch.object(park.gateway, "safe_to_unload",
                               return_value=(False, "external Ollama client activity is unknown")):
            with self.assertRaisesRegex(RuntimeError, "external Ollama client activity is unknown"):
                park.unpark()

        self.assertEqual(park.STATE.read_bytes(), journal_before)
        self.assertEqual(self.ollama.tags, tags_before)
        self.assertIsNone(store.park_fence(alias))

    def test_planned_entry_does_not_claim_matching_alias_created_by_another_client(self):
        with mock.patch.object(park, "_copy", side_effect=RuntimeError("simulated copy failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated copy failure"):
                park.park(resolve=omp_like([TARGET]))

        entry = park.parked_now()[0]
        self.assertEqual(entry["_park_phase"], "planned")
        alias = entry["parked_as"]
        self.ollama.tags[alias] = entry["digest"]
        journal_before = park.STATE.read_bytes()
        tags_before = dict(self.ollama.tags)
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")

        with self.assertRaisesRegex(RuntimeError, "journal does not own"):
            park.unpark()

        self.assertEqual(park.STATE.read_bytes(), journal_before)
        self.assertEqual(self.ollama.tags, tags_before)
        self.assertIsNone(store.park_fence(alias))

    def test_digest_prefix_collision_is_refused_before_any_tag_moves(self):
        self.ollama.tags[SIBLING] = "sha256:5642e97495e1bb"
        original = dict(self.ollama.tags)

        with self.assertRaisesRegex(RuntimeError, "parked alias collision"):
            park.park(resolve=omp_like([TARGET, SIBLING]))

        self.assertEqual(self.ollama.tags, original)
        self.assertFalse(park.STATE.exists())
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")
        self.assertIsNone(store.park_fence(TARGET))
        self.assertIsNone(store.park_fence(SIBLING))


    def test_preexisting_same_digest_alias_is_refused_and_preserved(self):
        alias = "localbench-parked:5642e97495e1"
        self.ollama.tags[alias] = self.ollama.tags[TARGET]
        tags_before = dict(self.ollama.tags)

        with self.assertRaisesRegex(RuntimeError, "pre-existing parked alias"):
            park.park(resolve=omp_like([TARGET]))

        self.assertEqual(self.ollama.tags, tags_before)
        self.assertFalse(park.STATE.exists())
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")
        self.assertIsNone(store.park_fence(TARGET))


    def test_stale_cli_plan_is_refused_before_mutation(self):
        resolve = omp_like([TARGET])
        stale_plan = park.plan_park(resolve=resolve)
        self.ollama.tags[TARGET] = "sha256:5642e97495e1bb"

        with self.assertRaisesRegex(RuntimeError, "park plan changed before mutation"):
            park.park(resolve=resolve, plan=stale_plan)

        self.assertEqual(self.ollama.tags[TARGET], "sha256:5642e97495e1bb")
        self.assertFalse(park.STATE.exists())
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")
        self.assertIsNone(store.park_fence(TARGET))

    def test_legacy_partial_digest_prefix_collision_restores_both_tags(self):
        self.ollama.tags[SIBLING] = "sha256:5642e97495e1bb"
        planned = park.plan_park(resolve=omp_like([TARGET, SIBLING]))
        alias = planned[0]["parked_as"]
        self.assertEqual(planned[1]["parked_as"], alias)
        fence_id = "legacy-collision"
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")
        self.assertIsNone(store.acquire_park_fence([TARGET, SIBLING], fence_id))
        self.ollama.tags[alias] = self.ollama.tags.pop(TARGET)
        park.STATE.write_text(json.dumps([{**entry, "_park_fence_id": fence_id,
                                           "_park_phase": "parked" if entry["name"] == TARGET else "planned"}
                                          for entry in planned]))

        park.unpark()

        self.assertEqual(self.ollama.tags[TARGET], "sha256:5642e97495e1aa")
        self.assertEqual(self.ollama.tags[SIBLING], "sha256:5642e97495e1bb")
        self.assertNotIn(alias, self.ollama.tags)
        self.assertFalse(park.STATE.exists())
        self.assertIsNone(store.park_fence(TARGET))
        self.assertIsNone(store.park_fence(SIBLING))

    def test_foreign_alias_digest_is_kept_even_after_both_names_exist(self):
        self.ollama.tags[SIBLING] = self.ollama.tags[TARGET]
        parked = park.park(resolve=omp_like([TARGET, SIBLING]))
        alias = parked[0]["parked_as"]
        for entry in parked:
            self.ollama.tags[entry["name"]] = entry["digest"]
        self.ollama.tags[alias] = "sha256:aaaaaaaaaaaaaa"

        with self.assertRaisesRegex(RuntimeError, "differs from all journaled digests"):
            park.unpark()

        self.assertEqual(self.ollama.tags[alias], "sha256:aaaaaaaaaaaaaa")
        self.assertTrue(park.STATE.exists())
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")
        self.assertIsNotNone(store.park_fence(TARGET))
        self.assertIsNotNone(store.park_fence(SIBLING))

    def test_new_park_refuses_collision_with_existing_partial_journal(self):
        self.ollama.tags[SIBLING] = "sha256:5642e97495e1bb"
        planned = park.plan_park(resolve=omp_like([TARGET, SIBLING]))
        original = {**planned[0], "_park_phase": "parked", "_park_fence_id": "prior-fence"}
        alias = original["parked_as"]
        self.ollama.tags[alias] = self.ollama.tags.pop(TARGET)
        tags_before = dict(self.ollama.tags)
        park.STATE.write_text(json.dumps([original]))
        store = park.gateway.GatewayStore(self.tmp / "gateway" / "leases.sqlite")
        self.assertIsNone(store.acquire_park_fence([TARGET], "prior-fence"))

        with mock.patch.object(park, "plan_park", return_value=[planned[1]]):
            with self.assertRaisesRegex(RuntimeError, "parked alias collision"):
                park.park(plan=[planned[1]])

        self.assertEqual(self.ollama.tags, tags_before)
        self.assertEqual(park.parked_now(), [original])
        self.assertEqual(store.park_fence(TARGET), "prior-fence")
        self.assertIsNone(store.park_fence(SIBLING))
        park.unpark()
        self.assertEqual(self.ollama.tags[TARGET], original["digest"])
        self.assertEqual(self.ollama.tags[SIBLING], planned[1]["digest"])
        self.assertNotIn(alias, self.ollama.tags)
        self.assertIsNone(store.park_fence(TARGET))

    def test_a_second_park_parks_nothing_new(self):
        park.park(resolve=omp_like([TARGET, SIBLING]))
        again = park.park(resolve=omp_like([TARGET, SIBLING]))
        self.assertEqual([p["name"] for p in again], [TARGET, SIBLING])


class SealedWindows(unittest.TestCase):
    """Only a park that left omp a fallback strands the sessions started inside it."""

    CLIENT: ClassVar[list] = [{"pid": 7, "cwd": "/cp", "cmd": "bun /x/omp --auto-approve"}]

    def stuck(self, rows: list[dict], started: float) -> list[dict]:
        with mock.patch.object(park, "read_history", return_value=rows), \
                mock.patch.object(park, "process_started", return_value=started):
            return park.stuck_sessions(self.CLIENT)

    def test_a_sealed_park_strands_nobody(self):
        rows = [{"event": "park", "t": 100.0, "names": [TARGET, SIBLING], "sealed": True},
                {"event": "unpark", "t": 200.0, "names": [TARGET, SIBLING]}]
        self.assertEqual(self.stuck(rows, 150.0), [])

    def test_a_sealing_park_ends_an_open_leaky_span(self):
        rows = [{"event": "park", "t": 100.0, "names": [TARGET]},
                {"event": "park", "t": 150.0, "names": [SIBLING], "sealed": True},
                {"event": "unpark", "t": 200.0, "names": [TARGET, SIBLING]}]
        self.assertEqual([s["window"] for s in self.stuck(rows, 120.0)], [[100.0, 150.0]])
        self.assertEqual(self.stuck(rows, 170.0), [])

    def test_the_whole_window_still_counts_when_leaky_spans_are_not_asked_for(self):
        rows = [{"event": "park", "t": 100.0, "names": [TARGET, SIBLING], "sealed": True},
                {"event": "unpark", "t": 200.0, "names": [TARGET, SIBLING]}]
        self.assertEqual(park.park_windows(rows, now=300.0), [[100.0, 200.0]])
        self.assertEqual(park.park_windows(rows, now=300.0, leaky_only=True), [])



class NoFallbackSeal(unittest.TestCase):
    """A park that finds no sibling still seals. sealed=bool(fallbacks()) would leave that row open, and preflight
    would then name every session started in the window."""

    def test_a_park_with_no_fallback_still_seals(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        ollama = FakeOllama({TARGET: "sha256:5642e97495e1aa", "qwen3.6:35b-mlx": "sha256:cc"})
        stack = []
        ollama.install(stack)
        for name, value in (("STATE", tmp / "PARKED.json"), ("HISTORY", tmp / "park-history.jsonl"),
                            ("smol_targets", lambda: [TARGET])):
            p = mock.patch.object(park, name, value)
            p.start()
            stack.append(p)
        p = mock.patch.object(park.gateway, "database_path", return_value=tmp / "gateway" / "leases.sqlite")
        p.start()
        stack.append(p)
        p = mock.patch.object(park.gateway, "safe_to_unload", return_value=(True, None))
        p.start()
        stack.append(p)
        self.addCleanup(lambda: [p.stop() for p in reversed(stack)])
        park.park(resolve=omp_like([TARGET]))
        row = json.loads((tmp / "park-history.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["names"], [TARGET])
        self.assertIs(row.get("sealed"), True)


class OmpPackage(unittest.TestCase):
    """omp's resolver runs from the package behind `omp`. proj-c's trust guard (~/.local/bin/omp, a compiled
    wrapper) has no package beside it: follow its OMP_REAL_BIN; with nothing to follow, refuse (naming
    LOCALBENCH_OMP) before bun runs, never answer with another install's resolver."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "guard" / "bin").mkdir(parents=True)
        self.guard = self.tmp / "guard" / "bin" / "omp"
        self.guard.write_text("")
        self.config = self.tmp / "omp-trust-config"
        patcher = mock.patch.object(park, "OMP_TRUST_CONFIG", self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def package(self) -> Path:
        pkg = self.tmp / "pkg"
        (pkg / "src" / "config").mkdir(parents=True)
        (pkg / "src" / "config" / "model-resolver.ts").write_text("")
        (pkg / "dist").mkdir()
        (pkg / "dist" / "cli.js").write_text("")
        return pkg

    def test_the_trust_guard_is_followed_to_its_real_package(self):
        pkg = self.package()
        self.config.write_text(f"OMP_OWN_ROOT=/x\nOMP_REAL_BIN={pkg / 'dist' / 'cli.js'}\n")
        self.assertEqual(park.omp_package(str(self.guard)), Path(os.path.realpath(pkg)))

    def test_a_binary_without_package_or_guard_is_refused_before_bun_runs(self):
        with mock.patch.object(park, "omp_bin", return_value=str(self.guard)), \
                mock.patch.object(park.subprocess, "run") as run:
            with self.assertRaises(RuntimeError) as err:
                park.omp_resolves("ollama/" + TARGET, [TARGET])
        run.assert_not_called()
        self.assertIn("LOCALBENCH_OMP", str(err.exception))


@unittest.skipUnless(shutil.which("bun") and shutil.which("omp"), "needs bun and omp")
class RealResolver(unittest.TestCase):
    """The bridge to omp's own resolver: an installed exact id resolves to itself; an empty set to nothing."""

    def test_exact_id_and_empty_set(self):
        self.assertEqual(park.omp_resolves("ollama/" + TARGET, [TARGET, "qwen3.6:35b-mlx"]), TARGET)
        self.assertIsNone(park.omp_resolves("ollama/" + TARGET, []))


if __name__ == "__main__":
    unittest.main()
