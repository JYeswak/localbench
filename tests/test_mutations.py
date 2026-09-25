"""Mutation discipline: a dry run changes nothing and writes no audit row; a real run executes the plan the dry run
printed and appends exactly one row (done, refused or failed); a second run of an applied change is a no-op that
exits 0; `why` and `validate` read, never write. Every live-state path is redirected (tests/__init__.py)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import audit, park, quiet, smol
from tests import test_keep_pull as keep_pull
from tests.test_park_fallbacks import SIBLING, TARGET, FakeOllama, omp_like

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = ROOT / "docs" / "evidence" / "receipts" / "ab-incumbent-vs-moe.json"
GOLDEN = min((ROOT / "goldens").glob("*/*.json"))


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def snapshot(*roots: Path) -> dict[str, bytes]:
    return {str(f): f.read_bytes() for r in roots if r.exists() for f in sorted(r.rglob("*")) if f.is_file()}


class Ledgered(unittest.TestCase):
    """Each test gets an empty ledger of its own."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="localbench-test-mut-"))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(audit, "AUDIT_PATH", self.tmp / "audit.jsonl"))
        self.stack.enter_context(mock.patch.object(cli, "_run_alive", return_value=False))

    def patch(self, target, name, value):
        self.stack.enter_context(mock.patch.object(target, name, value))


class SuiteIsolation(unittest.TestCase):
    """No test may write the user's ledger or smol state: on 2026-09-25 the aa refusal tests (a CLI subprocess, which
    tests/__init__.py cannot redirect) appended six rows to the live ~/.localbench/audit.jsonl."""

    def test_in_process_state_is_redirected_away_from_home(self):
        live = Path.home() / ".localbench"
        for path in (audit.path(), smol.state_path(), smol.plist_path()):
            self.assertFalse(path.resolve().is_relative_to(live.resolve()), path)
        self.assertNotEqual(smol.LAUNCH_AGENTS, Path.home() / "Library" / "LaunchAgents")

    def test_a_cli_subprocess_writes_its_rows_under_the_scratch_home(self):
        from tests.test_cli import SCRATCH_HOME
        from tests.test_cli import cli as subprocess_cli

        ledger = Path(SCRATCH_HOME) / ".localbench" / "audit.jsonl"
        before = len(ledger.read_text().splitlines()) if ledger.exists() else 0
        p = subprocess_cli("aa", "ollama:no-such-model", "--mem-rounds", "9", "--write-golden")
        self.assertEqual(p.returncode, 1, p.stderr)
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        self.assertEqual(len(rows), before + 1)
        self.assertEqual((rows[-1]["verb"], rows[-1]["outcome"]), ("aa --write-golden", "refused"))


class ParkPlan(Ledgered):
    def setUp(self):
        super().setUp()
        self.ollama = FakeOllama({TARGET: "sha256:5642e97495e1aa", SIBLING: "sha256:23da7bcdf4d1bb",
                                  "qwen3.6:35b-mlx": "sha256:cc"})
        patches: list = []
        self.ollama.install(patches)
        for p in patches:
            self.addCleanup(p.stop)
        self.patch(park, "STATE", self.tmp / "PARKED.json")
        self.patch(park, "HISTORY", self.tmp / "park-history.jsonl")
        self.patch(park, "smol_targets", lambda: [TARGET])
        self.patch(park, "omp_resolves", omp_like([TARGET, SIBLING]))

    def test_a_dry_run_changes_nothing_and_the_real_run_does_what_it_printed(self):
        tags = dict(self.ollama.tags)
        rc, out, _ = run("park", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertEqual(self.ollama.tags, tags)
        self.assertEqual((park.STATE.exists(), park.HISTORY.exists(), audit.rows()), (False, False, []))
        planned = out.splitlines()
        self.assertEqual(len(planned), 2)
        rc, _, _ = run("park", "--dry-run", "--json")
        self.assertEqual(rc, 0)
        rc, _, _ = run("park")
        self.assertEqual(rc, 0)
        self.assertNotIn(TARGET, self.ollama.tags)
        [row] = audit.rows()
        self.assertEqual((row["verb"], row["outcome"], row["actions"], row["argv"]), ("park", "done", planned, ["park"]))

    def test_the_json_plan_names_the_verb_and_every_action(self):
        rc, out, _ = run("park", "--dry-run", "--json")
        plan = json.loads(out)
        self.assertEqual((rc, plan["verb"], len(plan["actions"]), plan["would_refuse"]), (0, "park", 2, None))

    def test_a_second_park_is_a_recorded_no_op(self):
        run("park")
        tags, state = dict(self.ollama.tags), park.STATE.read_bytes()
        history = park.HISTORY.read_bytes()
        rc, _, _ = run("park")
        self.assertEqual(rc, 0)
        self.assertEqual((self.ollama.tags, park.STATE.read_bytes(), park.HISTORY.read_bytes()), (tags, state, history))
        second = audit.rows()[1]
        self.assertEqual((second["outcome"], second["actions"]), ("done", []))
        self.assertIn("noop", second["detail"])

    def test_unpark_with_nothing_parked_is_a_no_op(self):
        with mock.patch.object(park, "refresh_catalogs") as refresh:
            rc, _, _ = run("unpark")
        refresh.assert_not_called()
        [row] = audit.rows()
        self.assertEqual((rc, row["verb"], row["actions"]), (0, "unpark", []))
        self.assertIn("noop", row["detail"])

    def test_a_dry_run_whose_planning_raises_writes_no_row(self):
        def broken(selector, ids):
            raise RuntimeError("omp resolver failed")
        self.patch(park, "omp_resolves", broken)
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            cli.main(["park", "--dry-run"])
        self.assertEqual(audit.rows(), [])

    def test_a_park_that_raises_is_recorded_as_failed(self):
        self.patch(park, "_copy", lambda src, dst: self.ollama.tags.__setitem__(dst, "sha256:wrong"))
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            cli.main(["park"])
        [row] = audit.rows()
        self.assertEqual(row["outcome"], "failed")
        self.assertIn("digest", row["detail"]["error"])
        self.assertIn(TARGET, self.ollama.tags, "the name is not deleted when its copy does not match")


class SmolPlan(Ledgered):
    def setUp(self):
        super().setUp()
        self.home = self.tmp / "home"
        for name in ("default", "claude"):
            d = self.home / name
            d.mkdir(parents=True)
            (d / "config.yml").write_text("modelRoles:\n  smol: ollama/qwen3.8:27b-mlx\n")
        dirs = {n: self.home / n for n in ("default", "claude")}
        state_dir = self.tmp / "state"
        self.patch(smol, "STATE_DIR", state_dir)
        self.patch(smol, "profile_dirs", lambda home=None: dirs)
        self.patch(smol, "server_up", lambda port=smol.PORT: False)
        selector = "mlx-smol/M"
        record = smol.set_profiles(selector, smol.provider_block("M", 1024), dirs, self.tmp / "backup")
        smol._save_state({"selector": selector, "model_id": "M", "port": smol.PORT, "binary": "/x/mlx-serve",
                          "model_dir": "/m/M", "server_args": [], "profiles": record, "backup": str(self.tmp / "backup")},
                         state_dir)
        self.state_dir = state_dir

    def test_revert_dry_run_leaves_every_file_byte_identical_and_writes_no_row(self):
        before = snapshot(self.home, self.state_dir, smol.LAUNCH_AGENTS)
        with mock.patch.object(smol, "stop_server") as stop, mock.patch.object(smol, "remove_autostart") as rm:
            rc, out, _ = run("smol", "revert", "--dry-run")
        self.assertEqual(rc, 0)
        stop.assert_not_called()
        rm.assert_not_called()
        self.assertEqual(snapshot(self.home, self.state_dir, smol.LAUNCH_AGENTS), before)
        self.assertEqual(audit.rows(), [])
        planned = out.splitlines()
        with mock.patch.object(smol, "stop_server"), mock.patch.object(smol, "remove_autostart"), \
                mock.patch.object(cli, "_smol_verify", return_value=True):
            rc, _, _ = run("smol", "revert")
        self.assertEqual(rc, 0)
        self.assertEqual((self.home / "claude" / "config.yml").read_text(), "modelRoles:\n  smol: ollama/qwen3.8:27b-mlx\n")
        self.assertIsNone(smol.load_state(self.state_dir))
        [row] = audit.rows()
        self.assertEqual((row["verb"], row["outcome"], row["actions"]), ("smol revert", "done", planned))

    def test_a_refusal_is_recorded_and_its_dry_run_is_not(self):
        with mock.patch.object(cli, "_run_alive", return_value=True), mock.patch.object(smol, "start_server") as start:
            dry, _, _ = run("smol", "start", "--dry-run")
            self.assertEqual(audit.rows(), [])
            rc, out, _ = run("smol", "start")
        start.assert_not_called()
        self.assertEqual((dry, rc, out), (1, 1, ""))
        [row] = audit.rows()
        self.assertEqual((row["verb"], row["outcome"]), ("smol start", "refused"))
        self.assertIn("run is alive", row["detail"]["reason"])

    def test_stop_when_down_does_not_stop(self):
        with mock.patch.object(smol, "stop_server") as stop:
            rc, _, _ = run("smol", "stop")
        stop.assert_not_called()
        self.assertEqual((rc, audit.rows()[0]["actions"]), (0, []))

    def test_autostart_on_when_on_does_not_reload(self):
        st = smol.load_state(self.state_dir)
        smol.plist_path().write_bytes(smol.launchd_plist(st))
        self.addCleanup(smol.plist_path().unlink, missing_ok=True)
        with mock.patch.object(smol, "server_up", return_value=True), \
                mock.patch.object(smol, "stop_server") as stop, mock.patch.object(smol, "install_autostart") as install:
            rc, _, _ = run("smol", "autostart", "on")
        stop.assert_not_called()
        install.assert_not_called()
        self.assertEqual((rc, audit.rows()[0]["actions"]), (0, []))


class KeepPlan(keep_pull.CliAgainstFake):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(audit, "AUDIT_PATH", Path(tempfile.mkdtemp()) / "audit.jsonl")
        p.start()
        self.addCleanup(p.stop)

    def test_keeping_forever_what_is_kept_forever_sends_nothing(self):
        keep_pull.FakeOllama.loaded = {"m:1": keep_pull.FOREVER}
        rc, _ = self.run_cli("keep", "ollama:m:1", "forever")
        self.assertEqual((rc, keep_pull.FakeOllama.requests, audit.rows()[0]["actions"]), (0, [], []))

    def test_unloading_what_is_not_loaded_sends_nothing(self):
        rc, _ = self.run_cli("keep", "ollama:m:1", "0")
        self.assertEqual((rc, keep_pull.FakeOllama.requests), (0, []))

    def test_a_dry_run_sends_nothing_and_writes_no_row(self):
        rc, out = self.run_cli("keep", "ollama:m:1", "forever", "--dry-run")
        self.assertEqual((rc, keep_pull.FakeOllama.requests, audit.rows()), (0, [], []))
        self.assertEqual(len(out.splitlines()), 1)

    def test_a_real_keep_records_one_done_row(self):
        rc, _ = self.run_cli("keep", "ollama:m:1", "forever")
        [row] = audit.rows()
        self.assertEqual((rc, row["verb"], row["outcome"], len(row["actions"])), (0, "keep", "done", 1))


class QuietPlan(Ledgered):
    def test_pausing_with_nothing_to_pause_is_a_recorded_no_op(self):
        self.patch(quiet, "STATE", self.tmp / "quiet.json")
        self.patch(quiet, "processes", dict)
        rc, _, _ = run("quiet")
        [row] = audit.rows()
        self.assertEqual((rc, quiet.STATE.exists(), row["actions"]), (0, False, []))
        self.assertIn("noop", row["detail"])


class Why(Ledgered):
    def test_an_unknown_id_is_a_usage_error_naming_where_ids_come_from(self):
        with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(io.StringIO()) as err:
            cli.main(["why", "20990101T000000Z-000000"])
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("localbench audit", err.getvalue())

    def test_a_row_is_printed_whole(self):
        rid = audit.record("park", ["park"], ["a", "b"], "done", {"k": 1})
        rc, out, _ = run("why", rid, "--json")
        self.assertEqual((rc, json.loads(out)), (0, audit.row(rid)))
        rc, out, _ = run("audit", "--since", "1h", "--json")
        self.assertEqual([r["id"] for r in json.loads(out)], [rid])


class Validate(unittest.TestCase):
    def check(self, doc) -> tuple[int, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write(doc if isinstance(doc, str) else json.dumps(doc))
            f.flush()
            rc, _, err = run("validate", f.name)
        return rc, err

    def test_a_banked_receipt_and_a_golden_are_valid(self):
        for path in (RECEIPT, GOLDEN):
            rc, _, err = run("validate", str(path))
            self.assertEqual((rc, err), (0, ""), path)

    def test_a_leg_without_a_pin_is_invalid_and_named(self):
        doc = json.loads(RECEIPT.read_text())
        del doc["legs"][1]["provenance"]["pins"]["model_digest"]
        rc, err = self.check(doc)
        self.assertEqual(rc, 1)
        self.assertIn("model_digest", err)

    def test_a_golden_without_metrics_is_invalid(self):
        doc = json.loads(GOLDEN.read_text())
        doc["metrics"] = {}
        self.assertEqual(self.check(doc)[0], 1)

    def test_text_that_is_not_json_is_invalid_not_a_crash(self):
        rc, err = self.check("# a markdown receipt\n")
        self.assertEqual(rc, 1)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
