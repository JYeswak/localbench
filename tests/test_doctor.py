"""`localbench doctor`: one dead subsystem is one FAIL row and the rest still report; `--fix` removes a mutation lock
only when its owner pid is provably dead, records that in the audit ledger, and without --fix nothing changes."""

import json
import os
import plistlib
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import audit, doctor, gateway


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()  # reaped: no process has this pid now
    return p.pid


class Isolation(unittest.TestCase):
    def test_a_raising_probe_is_one_fail_row_and_the_later_probes_still_run(self):
        ran = []

        def ok(_fix):
            ran.append("before")
            return doctor._row("PASS", "fine")

        def dead(_fix):
            raise OSError("subsystem gone")

        def later(_fix):
            ran.append("after")
            return doctor._row("WARN", "gap", "some command")

        with mock.patch.object(doctor, "CHECKS", [("a", ok), ("b", dead), ("c", later)]):
            rows = doctor.checks()
        self.assertEqual([(r["subsystem"], r["status"]) for r in rows], [("a", "PASS"), ("b", "FAIL"), ("c", "WARN")])
        self.assertEqual(ran, ["before", "after"])


class MutationLock(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lock = Path(tmp.name) / ".mutation.lock"
        for p in (mock.patch.object(doctor, "MUTATION_LOCK", self.lock),
                  mock.patch.object(audit, "AUDIT_PATH", Path(tmp.name) / "audit.jsonl"),
                  mock.patch.object(doctor, "CHECKS", [("mutation lock", doctor.check_mutation_lock)])):
            p.start()
            self.addCleanup(p.stop)

    def _hold(self, pid: int | None) -> None:
        self.lock.mkdir()
        if pid is not None:
            (self.lock / "owner").write_text(f"pid {pid} since {time.ctime()}\n")

    def test_fix_removes_a_lock_whose_owner_is_dead_once_and_records_it(self):
        pid = _dead_pid()
        self._hold(pid)
        [row] = doctor.checks(fix=True)
        self.assertFalse(self.lock.exists())
        self.assertEqual((row["status"], row["fixed"]), ("PASS", True))
        [entry] = audit.rows()
        self.assertEqual((entry["verb"], entry["outcome"], entry["detail"]["pid"]), ("doctor --fix", "done", pid))
        # idempotent: nothing left to repair, nothing more recorded
        [again] = doctor.checks(fix=True)
        self.assertEqual((again["status"], again["fixed"]), ("PASS", False))
        self.assertEqual(len(audit.rows()), 1)

    def test_fix_keeps_a_lock_whose_owner_is_alive(self):
        for pid in (os.getpid(), 1):  # ours, and launchd's (another user's: kill(1, 0) is EPERM, not ESRCH)
            with self.subTest(pid=pid):
                self._hold(pid)
                [row] = doctor.checks(fix=True)
                self.assertTrue((self.lock / "owner").is_file())
                self.assertEqual((row["status"], row["fixed"]), ("PASS", False))
                self.assertEqual(audit.rows(), [])
                (self.lock / "owner").unlink()
                self.lock.rmdir()

    def test_fix_keeps_a_lock_with_no_owner_pid(self):
        self._hold(None)
        [row] = doctor.checks(fix=True)
        self.assertTrue(self.lock.is_dir())
        self.assertEqual((row["status"], row["fixed"]), ("WARN", False))
        self.assertEqual(audit.rows(), [])

    def test_without_fix_a_dead_owners_lock_stays_and_the_row_names_the_repair(self):
        self._hold(_dead_pid())
        [row] = doctor.checks(fix=False)
        self.assertTrue((self.lock / "owner").is_file())
        self.assertEqual((row["status"], row["fixed"]), ("WARN", False))
        self.assertIsNotNone(row["fix"])
        self.assertFalse(audit.path().exists())


class Coverage(unittest.TestCase):
    def test_named_checks_resolve_from_repo_when_cwd_differs(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(doctor, "MUTATION_LOCK", Path(tmp) / "runs" / ".mutation.lock"):
            old = Path.cwd()
            os.chdir(tmp)
            try:
                self.assertEqual(doctor.check_tick_driver(False)["status"], "PASS")
                self.assertEqual(doctor.check_public_export(False)["status"], "PASS")
                self.assertEqual(doctor.check_hooks(False)["status"], "PASS")
            finally:
                os.chdir(old)
        heavy_path = str(__import__("localbench.heavyslot", fromlist=["lock_path"]).lock_path())
        self.assertIn(heavy_path, doctor.check_heavyslot(False)["detail"])

    def test_required_foundational_subsystems_have_probes(self):
        self.assertEqual(doctor.missing_required_subsystems(), [])
        names = {name for name, _ in doctor.CHECKS}
        self.assertTrue(set(doctor.REQUIRED_SUBSYSTEMS) <= names)


class GatewayPin(unittest.TestCase):
    def test_matching_export_and_sha_pass_but_deliberate_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            document = gateway.launchd_plist(home=home)
            path = gateway.plist_path(home)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(plistlib.dumps(document))
            with mock.patch.object(gateway, "service_status", return_value={"health": True}):
                passed = doctor.check_gateway(False, home=home)
            self.assertEqual(passed["status"], "PASS")
            details = json.loads(passed["detail"])
            self.assertEqual(details["code"]["code_sha"], details["code"]["promoted_sha"])
            self.assertTrue(Path(details["code"]["interpreter"]).is_file())
            self.assertTrue(Path(details["code"]["export"]).is_dir())

            document["EnvironmentVariables"]["LOCALBENCH_GATEWAY_SHA"] = "0" * 64
            path.write_bytes(plistlib.dumps(document))
            with mock.patch.object(gateway, "service_status", return_value={"health": True}):
                failed = doctor.check_gateway(False, home=home)
            self.assertEqual(failed["status"], "FAIL")
            self.assertIn("does not match", failed["detail"])


class Features(unittest.TestCase):
    def row(self, found):
        from localbench import features
        with mock.patch.object(features, "doctor_findings", return_value=found):
            return doctor.check_features(False)

    def test_the_worst_finding_sets_the_row_and_its_fix(self):
        row = self.row([("PASS", "a ok", None), ("FAIL", "b routes local and is UNPROVEN", "prove b"),
                        ("WARN", "c carried", "re-prove c")])
        self.assertEqual((row["status"], row["fix"]), ("FAIL", "prove b"))
        self.assertIn("b routes local", row["detail"])

    def test_a_warning_is_not_hidden_by_passes(self):
        row = self.row([("PASS", "a ok", None), ("WARN", "c carried", "re-prove c")])
        self.assertEqual((row["status"], row["fix"]), ("WARN", "re-prove c"))

    def test_all_pass_is_pass(self):
        self.assertEqual(self.row([("PASS", "a ok", None)])["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
