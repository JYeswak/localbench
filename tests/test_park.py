"""Park history and stuck sessions. Does not call ollama."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import park


class History(unittest.TestCase):
    def test_a_start_inside_a_window_is_stuck_and_the_unpark_instant_is_not(self):
        rows = [
            {"event": "park", "t": 100.0, "names": ["qwen3.8:27b-mlx"]},
            {"event": "unpark", "t": 200.0, "names": ["qwen3.8:27b-mlx"]},
        ]
        clients = [{"pid": 7, "cwd": "/tmp/cp", "cmd": "omp --mode rpc", "omp_profile": "default"}]
        with mock.patch.object(park, "read_history", return_value=rows), \
                mock.patch.object(park, "process_started", return_value=150.0):
            hit = park.stuck_sessions(clients)
        self.assertEqual(hit, [{"pid": 7, "cwd": "/tmp/cp", "started": 150.0, "window": [100.0, 200.0]}])
        with mock.patch.object(park, "read_history", return_value=rows), \
                mock.patch.object(park, "process_started", return_value=200.0):
            self.assertEqual(park.stuck_sessions(clients), [])

    def test_an_empty_park_does_not_open_a_window(self):
        rows = [{"event": "park", "t": 100.0, "names": []}]
        clients = [{"pid": 7, "cwd": "/tmp/cp", "cmd": "omp", "omp_profile": "default"}]
        with mock.patch.object(park, "read_history", return_value=rows), \
                mock.patch.object(park, "process_started", return_value=150.0):
            self.assertEqual(park.stuck_sessions(clients), [])

    def test_a_non_omp_client_inside_a_real_window_is_not_stuck(self):
        rows = [
            {"event": "park", "t": 100.0, "names": ["qwen3.8:27b-mlx"]},
            {"event": "unpark", "t": 200.0, "names": ["qwen3.8:27b-mlx"]},
        ]
        clients = [{"pid": 9, "cwd": "/tmp", "cmd": "Google Chrome"}]
        with mock.patch.object(park, "read_history", return_value=rows), \
                mock.patch.object(park, "process_started", return_value=150.0):
            self.assertEqual(park.stuck_sessions(clients), [])

    def test_a_second_park_does_not_move_the_start(self):
        rows = [
            {"event": "park", "t": 100.0, "names": ["qwen3.8:27b-mlx"]},
            {"event": "park", "t": 150.0, "names": ["qwen3.8:27b-mlx"]},
            {"event": "unpark", "t": 200.0, "names": ["qwen3.8:27b-mlx"]},
        ]
        clients = [{"pid": 7, "cwd": "/tmp/cp", "cmd": "omp"}]
        with mock.patch.object(park, "read_history", return_value=rows), \
                mock.patch.object(park, "process_started", return_value=120.0):
            hit = park.stuck_sessions(clients)
        self.assertEqual(hit[0]["window"], [100.0, 200.0])

    def test_an_open_park_is_a_window_until_now(self):
        rows = [{"event": "park", "t": 100.0, "names": ["qwen3.8:27b-mlx"]}]
        clients = [{"pid": 7, "cwd": "/tmp/cp", "cmd": "omp"}]
        with mock.patch.object(park, "read_history", return_value=rows), \
                mock.patch.object(park, "process_started", return_value=180.0), \
                mock.patch.object(park.time, "time", return_value=250.0):
            hit = park.stuck_sessions(clients)
        self.assertEqual(hit, [{"pid": 7, "cwd": "/tmp/cp", "started": 180.0, "window": [100.0, 250.0]}])

    def test_append_writes_the_row_and_does_not_print(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "park-history.jsonl"
            with mock.patch.object(park, "HISTORY", path):
                row = park.append_history("park", ["qwen3.8:27b-mlx"], t=100.0)
            self.assertEqual(json.loads(path.read_text()), row)
            self.assertEqual(row["event"], "park")
            self.assertEqual(row["names"], ["qwen3.8:27b-mlx"])
