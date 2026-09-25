"""GPU attribution: which model an ollama runner serves, whether its GPU time is the run's own or a contender, and which
client moved bytes to which server (connected is not used)."""

import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

from localbench import observe, sysstats

WEIGHTS = "a" * 12 + "6eaa8b1a8d3489403e44ee8002ab7dd15cfd7855b434fabed43df3f587"  # 64 hex
PARAMS = "b" * 64
OTHER = "c" * 64


def manifest(root: Path, rel: str, weights: str, params: str = PARAMS) -> None:
    path = root / "manifests" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layers": [
        {"mediaType": sysstats.OLLAMA_WEIGHTS, "digest": f"sha256:{weights}"},
        {"mediaType": "application/vnd.ollama.image.params", "digest": f"sha256:{params}"}]}))


class Models(unittest.TestCase):
    """A throwaway ollama models dir: manifests only (blobs need not exist to be named)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        manifest(self.root, "registry.ollama.ai/library/qwen3.8-uncensored/latest", WEIGHTS)
        manifest(self.root, "registry.ollama.ai/jo/tool/v1", OTHER)
        manifest(self.root, "hf.co/org/repo/q4", OTHER)

    def tearDown(self):
        self.tmp.cleanup()

    def blob(self, hexpart: str) -> str:
        return str(self.root / "blobs" / f"sha256-{hexpart}")


class OllamaBlobNames(Models):
    def test_a_weights_blob_is_named_as_ollama_lists_it(self):
        self.assertEqual(sysstats.ollama_blob_names(self.blob(WEIGHTS)), ["qwen3.8-uncensored:latest"])

    def test_another_namespace_or_registry_keeps_its_prefix(self):
        self.assertEqual(sysstats.ollama_blob_names(self.blob(OTHER)), ["hf.co/org/repo:q4", "jo/tool:v1"])

    def test_a_cut_hex_still_names_the_model_but_a_short_one_does_not(self):
        # observe.db rows before the fix hold 58 of 64 hex digits; 8 digits are too few to trust.
        self.assertEqual(sysstats.ollama_blob_names(self.blob(WEIGHTS[:58])), ["qwen3.8-uncensored:latest"])
        self.assertEqual(sysstats.ollama_blob_names(self.blob(WEIGHTS[:8])), [])

    def test_a_non_weights_layer_names_no_model(self):
        self.assertEqual(sysstats.ollama_blob_names(self.blob(PARAMS)), [])

    def test_a_path_that_is_not_an_ollama_blob_names_no_model(self):
        self.assertEqual(sysstats.ollama_blob_names(str(self.root / "models" / f"sha256-{WEIGHTS}")), [])
        self.assertEqual(sysstats.ollama_blob_names("qwen3.6:35b-mlx"), [])


class RunnerAttribution(Models):
    def share(self, pid: int, cmd: str, name: str = "llama-server") -> dict:
        (row,) = sysstats.gpu_share({pid: {"name": name, "ns": 0}}, {pid: {"name": name, "ns": 10**9}}, 1.0,
                                    commands={pid: cmd})
        return row

    def test_a_runner_command_longer_than_the_stored_cut_is_still_named(self):
        cmd = ("/Applications/Ollama.app/Contents/Resources/llama-server --model " + self.blob(WEIGHTS)
               + " --port 60123 --ctx-size 131072 --batch-size 2048 --ubatch-size 2048 --context-shift --keep 4")
        row = self.share(214, cmd)
        self.assertEqual(row["model"], "qwen3.8-uncensored:latest")
        self.assertEqual(len(row["cmd"]), 160)
        self.assertEqual(sysstats.proc_label(row), "llama-server pid 214 (qwen3.8-uncensored:latest) 100.0%")

    def test_the_backends_own_gguf_runner_is_not_a_contender(self):
        row = self.share(214, "llama-server --model " + self.blob(WEIGHTS))
        self.assertTrue(sysstats.gpu_is_ours(row, "ollama", "qwen3.8-uncensored:latest"))
        self.assertFalse(sysstats.gpu_is_ours(row, "ollama", "qwen3.6:35b-mlx"))
        self.assertFalse(sysstats.gpu_is_ours(row, "mlx-serve", "qwen3.8-uncensored:latest"))

    def test_an_unnamed_blob_runner_stays_a_contender_and_keeps_its_whole_path(self):
        # The stored path is what a later lookup (after a re-pull, or in `report`) resolves; a cut one cannot be trusted.
        blob = self.blob("d" * 64)
        row = self.share(215, "/Applications/Ollama.app/Contents/Resources/llama-server --model " + blob + " --port 1")
        self.assertFalse(sysstats.gpu_is_ours(row, "ollama", "qwen3.8-uncensored:latest"))
        self.assertEqual(row["model"], blob)

    def test_ollama_serve_and_an_mlx_runner_by_name_stay_ours(self):
        self.assertTrue(sysstats.gpu_is_ours(self.share(1, "/x/ollama serve", "ollama"), "ollama", "any"))
        mlx = self.share(2, "/x/ollama runner --model qwen3.6:35b-mlx --port 1", "ollama")
        self.assertTrue(sysstats.gpu_is_ours(mlx, "ollama", "qwen3.6:35b-mlx"))
        self.assertFalse(sysstats.gpu_is_ours(mlx, "ollama", "qwen3.8:27b-mlx"))


class ReportMergesOldAndNewRows(Models):
    def test_cut_path_rows_and_named_rows_of_one_model_add_up(self):
        db = self.root / "observe.db"
        con = observe.connect(db)
        t = time.time()
        with con:
            con.executemany("insert into samples values (?, ?, ?, ?, ?)", [(t - 120, 60, 90, 50, 0), (t - 60, 60, 90, 50, 0)])
            con.executemany("insert into gpu values (?, ?, ?, ?, ?)", [
                (t - 120, 214, "llama-server", self.blob(WEIGHTS[:58]), 50.0),   # recorded before the lookup
                (t - 60, 214, "llama-server", "qwen3.8-uncensored:latest", 100.0)])
        con.close()
        gpu = observe.report(3600, db)["gpu"]
        self.assertEqual(gpu, [{"process": "llama-server", "model": "qwen3.8-uncensored:latest", "gpu_s": 90.0}])

# Real `nettop -m tcp -L 1 -n -J bytes_in,bytes_out` lines (2026-09-23), plus a non-loopback one.
NETTOP = """,bytes_in,bytes_out,
ollama.78482,1908713,2628608,
tcp4 127.0.0.1:11434<->*:*,,,
tcp4 127.0.0.1:11434<->127.0.0.1:49228,1505093,2881815,
tcp4 127.0.0.1:11434<->127.0.0.1:52604,11829,0,
tcp4 10.0.0.5:11434<->10.0.0.9:55001,999,999,
bun.2779,2881815,1505093,
tcp4 127.0.0.1:49228<->127.0.0.1:11434,2881815,1505093,
tcp4 127.0.0.1:53615<->127.0.0.1:53613,300977672,28775344,
"""


class OllamaAutoUpdate(unittest.TestCase):
    def db(self, tmp, schema, row=None):
        path = Path(tmp) / "db.sqlite"
        con = sqlite3.connect(path)
        con.execute(schema)
        if row is not None:
            con.execute("INSERT INTO settings VALUES (?, ?)", row)
        con.commit()
        con.close()
        return path

    def test_on_off_and_unknown(self):
        schema = "CREATE TABLE settings (id INTEGER PRIMARY KEY, auto_update_enabled BOOLEAN NOT NULL DEFAULT 1)"
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIs(sysstats.ollama_auto_update(self.db(tmp, schema, (1, 1))), True)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIs(sysstats.ollama_auto_update(self.db(tmp, schema, (1, 0))), False)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(sysstats.ollama_auto_update(Path(tmp) / "absent.sqlite"))
            old_app = self.db(tmp, "CREATE TABLE settings (id INTEGER PRIMARY KEY, models TEXT)", (1, "/x"))
            self.assertIsNone(sysstats.ollama_auto_update(old_app))       # app older than the setting

    def test_models_dir_follows_the_apps_order(self):
        schema = "CREATE TABLE settings (id INTEGER PRIMARY KEY, models TEXT NOT NULL DEFAULT '')"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OLLAMA_MODELS": "/env/models"}):
            self.assertEqual(sysstats.ollama_models_dir(self.db(tmp, schema, (1, "/Volumes/X/m"))), Path("/Volumes/X/m"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OLLAMA_MODELS": "/env/models"}):
            self.assertEqual(sysstats.ollama_models_dir(self.db(tmp, schema, (1, ""))), Path("/env/models"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, clear=True):
            os.environ["HOME"] = tmp
            self.assertEqual(sysstats.ollama_models_dir(Path(tmp) / "absent.sqlite"), Path(tmp) / ".ollama" / "models")


class ConnectionBytes(unittest.TestCase):
    def test_only_the_server_side_of_loopback_inference_connections_is_kept(self):
        with mock.patch.object(sysstats, "_run", return_value=NETTOP):
            self.assertEqual(sysstats.connection_bytes(),
                             {(11434, 49228): (1505093, 2881815), (11434, 52604): (11829, 0)})


class Traffic(unittest.TestCase):
    CONNS: ClassVar[list] = [[11434, 49228], [11434, 52604]]

    def test_an_idle_keep_alive_connection_is_not_use(self):
        before = {(11434, 49228): (100, 200), (11434, 52604): (5, 0)}
        self.assertEqual(sysstats.traffic(before, dict(before), self.CONNS), {"ollama": {"up": 0, "down": 0}})

    def test_only_bytes_inside_the_window_count(self):
        before = {(11434, 49228): (100, 200)}
        after = {(11434, 49228): (150, 1200), (11434, 52604): (40, 7)}   # 52604 opened inside the window
        self.assertEqual(sysstats.traffic(before, after, self.CONNS), {"ollama": {"up": 90, "down": 1007}})

    def test_a_reused_port_counts_from_zero_not_negative(self):
        before = {(11434, 49228): (5000, 9000)}
        after = {(11434, 49228): (300, 700)}
        self.assertEqual(sysstats.traffic(before, after, [[11434, 49228]]), {"ollama": {"up": 300, "down": 700}})

    def test_a_connection_gone_by_the_end_is_skipped(self):
        self.assertEqual(sysstats.traffic({(11434, 49228): (1, 1)}, {}, [[11434, 49228]]), {})

# `lsof -nP -iTCP -sTCP:ESTABLISHED -Fpcn` field output: two omp clients, and ollama's own side of one connection.
LSOF = """p2779
cbun
f23
n127.0.0.1:49228->127.0.0.1:11434
f31
n127.0.0.1:52198->127.0.0.1:11434
p65918
cbun
f19
n127.0.0.1:52604->127.0.0.1:11434
p78482
collama
f12
n127.0.0.1:11434->127.0.0.1:49228
"""


def fake_run(*cmd, timeout=10):
    if cmd[0] == "lsof" and "-Fpcn" in cmd:
        return LSOF
    if cmd[0] == "lsof":
        return f"p{cmd[3]}\nfcwd\nn/work/{cmd[3]}"
    if cmd[0] == "nettop":
        return NETTOP
    return "bun ~/.bun/bin/omp --auto-approve"


class ClientTraffic(unittest.TestCase):
    def test_each_client_is_charged_only_for_its_own_connections(self):
        with mock.patch.object(sysstats, "_run", side_effect=fake_run):
            clients = sysstats.inference_clients()
            after = sysstats.connection_bytes()
        self.assertEqual([(c["pid"], c["conns"]) for c in clients],
                         [(2779, [[11434, 49228], [11434, 52198]]), (65918, [[11434, 52604]])])
        used = {c["pid"]: sysstats.traffic({}, after, c["conns"]) for c in clients}
        self.assertEqual(used, {2779: {"ollama": {"up": 1505093, "down": 2881815}},
                                65918: {"ollama": {"up": 11829, "down": 0}}})


class OmpProcesses(unittest.TestCase):
    def test_every_omp_session_is_listed_with_its_cwd_connected_or_not(self):
        ps = "\n".join([
            " 2586 bun ~/.bun/bin/omp --auto-approve",
            " 2844 bun ~/.bun/bin/omp --profile grok",
            " 3001 bun scripts/omp-fuzzy-probe.ts ollama/qwen3.8:27b-mlx",
            " 3002 /usr/libexec/compose --omp-like-flag",
            f" {os.getpid()} python -m unittest omp",
        ])
        lsof = "p2586\nfcwd\nn~/Developer/proj-c\np2844\nfcwd\nn~/Developer/proj-b\n"
        with mock.patch.object(sysstats, "_run", side_effect=lambda *cmd, **_: ps if cmd[0] == "ps" else lsof):
            got = [(p["pid"], p["cwd"]) for p in sysstats.omp_processes()]
        self.assertEqual(got, [(2586, "~/Developer/proj-c"), (2844, "~/Developer/proj-b")])


class ReportTraffic(Models):
    def test_traffic_is_attributed_to_the_client_and_the_only_resident_model(self):
        db = self.root / "observe.db"
        con = observe.connect(db)
        t = time.time()
        with con:
            con.executemany("insert into samples values (?, ?, ?, ?, ?)", [(t - 120, 60, 90, 50, 0), (t - 60, 60, 90, 50, 0)])
            con.executemany("insert into clients values (?, ?, ?, ?, ?, ?)", [
                (t - 120, 2779, "bun", "default", "/cp", '{"ollama": 2}'),
                (t - 60, 2779, "bun", "default", "/cp", '{"ollama": 2}'),
                (t - 60, 65918, "bun", "default", "/lb", '{"ollama": 1}')])
            con.executemany("insert into traffic values (?, ?, ?, ?, ?, ?)", [
                (t - 120, 2779, "ollama", 1000, 50000, '["qwen3.8-uncensored:latest"]'),
                (t - 60, 2779, "ollama", 500, 25000, '["qwen3.8-uncensored:latest"]'),
                (t - 60, 65918, "ollama", 10, 700, '["qwen3.8-uncensored:latest", "qwen3.8:27b-mlx"]')])
        con.close()
        self.assertEqual(observe.report(3600, db)["traffic"], [
            {"who": "default", "cwd": "/cp", "server": "ollama", "while_resident": "qwen3.8-uncensored:latest",
             "up": 1500, "down": 75000},
            {"who": "default", "cwd": "/lb", "server": "ollama",
             "while_resident": "several: qwen3.8-uncensored:latest, qwen3.8:27b-mlx", "up": 10, "down": 700}])



if __name__ == "__main__":
    unittest.main()
