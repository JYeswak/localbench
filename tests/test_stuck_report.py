"""Sessions that started while a smol model was parked kept whatever local model omp's fuzzy match gave them
(2026-09-23: qwen3.8-uncensored, for hours). `unpark` and `status` name them, preflight refuses while they live,
and `gpu` marks the connected ones."""

import argparse
import contextlib
import io
import json
import time
import unittest
from typing import ClassVar
from unittest import mock

from localbench import park, sysstats
from localbench.__main__ import _busy_check, cmd_gpu, cmd_unpark

RESTORED = [{"name": "qwen3.8:27b-mlx", "digest": "5642e97495e1a088", "parked_as": "localbench-parked:5642e97495e1"}]
HIT = {"pid": 2779, "cwd": "~/Developer/proj-c", "started": 1790200507.0,
       "window": [1790200326.0, 1790201009.0]}


def unpark_output(stuck: list[dict]) -> str:
    buf = io.StringIO()
    with mock.patch.object(park, "parked_now", return_value=RESTORED), \
            mock.patch.object(park, "unpark", return_value=RESTORED), \
            mock.patch.object(park, "refresh_catalogs", return_value={}), \
            mock.patch.object(park, "stuck_sessions", return_value=stuck), \
            mock.patch.object(sysstats, "omp_processes", return_value=[]), \
            contextlib.redirect_stdout(buf):
        cmd_unpark(None)
    return buf.getvalue()


class Unpark(unittest.TestCase):
    def test_a_session_started_while_parked_is_named_with_what_to_do(self):
        out = unpark_output([HIT])
        self.assertIn("restored qwen3.8:27b-mlx", out)
        self.assertIn("pid 2779 cwd=~/Developer/proj-c", out)
        self.assertIn("restart that session", out)

    def test_a_clean_unpark_prints_only_the_restore(self):
        self.assertEqual(unpark_output([]).splitlines(), ["restored qwen3.8:27b-mlx (digest 5642e97495e1)"])

    def test_an_idle_session_with_no_open_connection_is_still_named(self):
        # pid 2586 started inside the window but held no connection at the check (2026-09-23 live smoke).
        history = [{"event": "park", "t": 100.0, "names": ["qwen3.8:27b-mlx"]},
                   {"event": "unpark", "t": 200.0, "names": ["qwen3.8:27b-mlx"]}]
        idle = {"pid": 2586, "cmd": "bun ~/.bun/bin/omp --auto-approve", "cwd": "/cp"}
        buf = io.StringIO()
        with mock.patch.object(park, "parked_now", return_value=RESTORED), \
            mock.patch.object(park, "unpark", return_value=RESTORED), \
                mock.patch.object(park, "refresh_catalogs", return_value={}), \
                mock.patch.object(park, "read_history", return_value=history), \
                mock.patch.object(park, "process_started", return_value=150.0), \
                mock.patch.object(sysstats, "omp_processes", return_value=[idle]), \
                mock.patch.object(sysstats, "inference_clients", return_value=[]), \
                contextlib.redirect_stdout(buf):
            cmd_unpark(None)
        self.assertIn("pid 2586 cwd=/cp", buf.getvalue())


class _QuietSampler:
    """Sampler stand-in for an idle machine: GPU 3% busy, no process above the veto."""
    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def summary(self):
        return {"gpu_device_pct": {"mean": 3.0}, "gpu_by_process": [], "swap_used_mb": {"max": 0}}


class Preflight(unittest.TestCase):
    def problems(self, stuck: list[dict], loadable: tuple[str, ...] = ("qwen3.8-uncensored:latest",)) -> list[str]:
        with mock.patch.object(sysstats, "Sampler", _QuietSampler), \
                mock.patch.object(sysstats, "cpu_busy_pct", return_value=4.0), \
                mock.patch.object(park, "reachable_smol", return_value=[]), \
                mock.patch.object(sysstats, "omp_processes", return_value=[]), \
                mock.patch.object(park, "stuck_sessions", return_value=stuck), \
                mock.patch.object(park, "fallbacks", return_value=list(loadable)):
            return _busy_check()[2]

    def test_an_idle_machine_with_a_stuck_session_is_refused_naming_it(self):
        (problem,) = self.problems([HIT, {**HIT, "pid": 2586}])
        self.assertIn("pid 2779 (~/Developer/proj-c)", problem)
        self.assertIn("pid 2586", problem)
        self.assertIn("CONTENDED", problem)

    def test_stuck_sessions_with_their_fallback_parked_pass(self):
        # After `localbench park` parks the fallback, a stuck session's calls fail instead of loading a model.
        self.assertEqual(self.problems([HIT], loadable=()), [])

    def test_the_refusal_names_the_loadable_fallback(self):
        (problem,) = self.problems([HIT])
        self.assertIn("qwen3.8-uncensored:latest", problem)

    def test_an_idle_machine_with_no_stuck_session_passes(self):
        self.assertEqual(self.problems([]), [])


class Gpu(unittest.TestCase):
    """`localbench gpu` marks a connected client that started inside a park window, and only that one."""

    CLIENTS: ClassVar[list] = [
        {"pid": 2779, "name": "bun", "cmd": "bun omp", "cwd": "/cp", "servers": {"ollama": 2}, "conns": [],
         "omp_profile": "default"},
        {"pid": 26326, "name": "bun", "cmd": "bun omp", "cwd": "/cp", "servers": {"ollama": 1}, "conns": [],
         "omp_profile": "default"}]

    def gpu(self, *, as_json: bool) -> str:
        clients = [dict(c) for c in self.CLIENTS]
        buf = io.StringIO()
        with mock.patch.object(time, "sleep"), \
                mock.patch.object(sysstats, "gpu_time_by_pid", return_value={}), \
                mock.patch.object(sysstats, "connection_bytes", return_value={}), \
                mock.patch.object(sysstats, "gpu_utilization", return_value={"device_pct": 90}), \
                mock.patch.object(sysstats, "resident_models", return_value={"ollama": ["qwen3.8-uncensored:latest"]}), \
                mock.patch.object(sysstats, "inference_clients", return_value=clients), \
                mock.patch.object(park, "local_routes", return_value={}), \
                mock.patch.object(park, "stuck_sessions", return_value=[HIT]), \
                contextlib.redirect_stdout(buf):
            cmd_gpu(argparse.Namespace(seconds=0, json=as_json))
        return buf.getvalue()

    def test_only_the_stuck_client_is_marked(self):
        lines = {int(ln.split()[1]): ln for ln in self.gpu(as_json=False).splitlines() if ln.startswith("  pid ")}
        self.assertIn("STARTED WHILE PARKED", lines[2779])
        self.assertNotIn("STARTED WHILE PARKED", lines[26326])

    def test_json_carries_the_window_for_the_stuck_client_only(self):
        got = {c["pid"]: c.get("started_while_parked") for c in json.loads(self.gpu(as_json=True))["clients"]}
        self.assertEqual(got, {2779: HIT["window"], 26326: None})


if __name__ == "__main__":
    unittest.main()
