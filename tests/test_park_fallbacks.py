"""Parking a smol model must also park what omp hands its role instead: on 2026-09-23 omp's resolver gave sessions
started while qwen3.8:27b-mlx was parked qwen3.8-uncensored:latest, which they kept for hours after unpark."""

import json
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
                         ("_post", lambda *a, **k: {}), ("_get", lambda url: {"models": []})):
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
        self.addCleanup(lambda: [p.stop() for p in reversed(stack)])
        park.park(resolve=omp_like([TARGET]))
        row = json.loads((tmp / "park-history.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["names"], [TARGET])
        self.assertIs(row.get("sealed"), True)


@unittest.skipUnless(shutil.which("bun") and shutil.which("omp"), "needs bun and omp")
class RealResolver(unittest.TestCase):
    """The bridge to omp's own resolver: an installed exact id resolves to itself; an empty set to nothing."""

    def test_exact_id_and_empty_set(self):
        self.assertEqual(park.omp_resolves("ollama/" + TARGET, [TARGET, "qwen3.6:35b-mlx"]), TARGET)
        self.assertIsNone(park.omp_resolves("ollama/" + TARGET, []))


if __name__ == "__main__":
    unittest.main()
