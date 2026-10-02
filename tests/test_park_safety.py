"""CPU-only park admission-fence and durable recovery regressions."""

import concurrent.futures
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import audit, gateway, park

MODEL = "qwen3.8:27b-mlx"
DIGEST = "sha256:5642e97495e1aa"
MODEL2 = "qwen3.8-uncensored:latest"
DIGEST2 = "sha256:23da7bcdf4d1bb"
PLAN = {"name": MODEL, "parked_as": "localbench-parked:5642e97495e1", "digest": DIGEST, "role": "smol"}
PLAN2 = {"name": MODEL2, "parked_as": "localbench-parked:23da7bcdf4d1", "digest": DIGEST2, "role": "fallback"}


class ParkSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="localbench-test-park-safety-"))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.db_path = self.tmp / "gateway" / "leases.sqlite"
        self.stack.enter_context(mock.patch.object(gateway, "database_path", return_value=self.db_path))
        self.store = gateway.GatewayStore(self.db_path)
        self.store.set_profiles({"default": "/omp-profile/default"})
        self.stack.enter_context(mock.patch.object(audit, "AUDIT_PATH", self.tmp / "audit.jsonl"))
        self.stack.enter_context(mock.patch.object(cli, "_run_alive", return_value=False))
        self.stack.enter_context(mock.patch.object(park, "STATE", self.tmp / "PARKED.json"))
        self.stack.enter_context(mock.patch.object(park, "HISTORY", self.tmp / "park-history.jsonl"))
        self.stack.enter_context(mock.patch.object(park, "plan_park", return_value=[dict(PLAN)]))
        self.tags = {MODEL: DIGEST, MODEL2: DIGEST2}
        self.mutations = []
        self.delete_fail = None
        self.copy_fail = None

        def copy(src, dst):
            self.mutations.append(("copy", src, dst))
            self.tags[dst] = self.tags[src]
            if self.copy_fail == dst:
                self.copy_fail = None
                raise RuntimeError(f"simulated copy response failure for {dst}")

        def unload(_url, body):
            self.mutations.append(("unload", body["model"]))
            return {}

        def delete(name):
            if self.delete_fail == name:
                self.delete_fail = None
                raise RuntimeError(f"simulated delete failure for {name}")
            self.mutations.append(("delete", name))
            del self.tags[name]

        self.stack.enter_context(mock.patch.object(park, "_tags", side_effect=lambda: dict(self.tags)))
        self.stack.enter_context(mock.patch.object(park, "_copy", side_effect=copy))
        self.stack.enter_context(mock.patch.object(park, "_post", side_effect=unload))
        self.stack.enter_context(mock.patch.object(park, "_delete", side_effect=delete))
        self.stack.enter_context(mock.patch.object(park, "_get", return_value={"models": []}))
        self.guard = self.stack.enter_context(mock.patch.object(gateway, "safe_to_unload", return_value=(True, None)))
        park.STATE.write_text("[]\n")
        self.initial_state = park.STATE.read_bytes()

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = cli.main(["park", *args])
        return rc, stdout.getvalue(), stderr.getvalue()

    def assert_no_mutations(self):
        self.assertEqual(self.mutations, [])
        self.assertEqual(park.STATE.read_bytes(), self.initial_state)
        self.assertFalse(park.HISTORY.exists())

    def test_clean_dry_run_keeps_the_plan_available(self):
        rc, out, err = self.run_cli("--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertIn("park ollama " + MODEL, out)
        self.assert_no_mutations()

    def test_active_gateway_request_refuses_dry_run(self):
        self.guard.return_value = (False, "gateway request is in flight for this model")
        rc, _, err = self.run_cli("--dry-run")
        self.assertEqual(rc, 1)
        self.assertIn("gateway request is in flight", err)
        self.assert_no_mutations()

    def test_active_gateway_request_refuses_apply_before_park_mutation(self):
        self.guard.return_value = (False, "gateway request is in flight for this model")
        rc, _, err = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertIn("gateway request is in flight", err)
        self.assert_no_mutations()

    def test_established_external_socket_refuses_apply_and_dry_run(self):
        for args in ((), ("--dry-run",)):
            with self.subTest(args=args):
                self.guard.return_value = (False, "established non-gateway Ollama client connection remains")
                rc, _, err = self.run_cli(*args)
                self.assertEqual(rc, 1)
                self.assertIn("established non-gateway Ollama client", err)
                self.assert_no_mutations()

    def test_unknown_external_activity_refuses_apply_and_dry_run(self):
        for args in ((), ("--dry-run",)):
            with self.subTest(args=args):
                self.guard.return_value = (None, "external-client activity telemetry is unavailable")
                rc, _, err = self.run_cli(*args)
                self.assertEqual(rc, 1)
                self.assertIn("external-client activity telemetry is unavailable", err)
                self.assert_no_mutations()

    def test_activity_appearing_after_cli_admission_is_caught_at_mutation_boundary(self):
        self.guard.side_effect = [(True, None), (False, "gateway request is in flight for this model")]
        rc, _, err = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertIn("gateway request is in flight", err)
        self.assert_no_mutations()
        self.assertIsNone(self.store.park_fence(MODEL))

    def test_active_gateway_request_racing_the_fence_is_refused_before_model_mutation(self):
        request_id = self.store.start_request("default", MODEL, "racing-client")
        rc, _, err = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertIn("gateway request is in flight", err)
        self.assertEqual(self.store.active_requests(MODEL), 1)
        self.assert_no_mutations()
        self.store.finish_request(request_id, completed=False, outcome="test")

    def test_clean_unpark_restores_model_and_reopens_gateway_admission(self):
        rc, _, err = self.run_cli()
        self.assertEqual(rc, 0, err)
        self.assertIsNotNone(self.store.park_fence(MODEL))
        park.unpark()
        self.assertEqual(self.tags, {MODEL: DIGEST, MODEL2: DIGEST2})
        self.assertIsNone(self.store.park_fence(MODEL))
        request_id = self.store.start_request("default", MODEL, "after-unpark")
        self.assertEqual(self.store.active_requests(MODEL), 1)
        self.store.finish_request(request_id, completed=True, outcome="test")

    def test_two_tag_partial_failure_is_durable_and_unpark_recovers_both(self):
        self.delete_fail = MODEL2
        with mock.patch.object(park, "plan_park", return_value=[dict(PLAN), dict(PLAN2)]):
            with self.assertRaisesRegex(RuntimeError, "simulated delete failure"):
                park.park(plan=[dict(PLAN), dict(PLAN2)])

        entries = park.parked_now()
        self.assertEqual([entry["name"] for entry in entries], [MODEL, MODEL2])
        self.assertEqual([entry["_park_phase"] for entry in entries], ["parked", "unloaded"])
        self.assertEqual(len({entry["_park_fence_id"] for entry in entries}), 1)
        self.assertIsNotNone(self.store.park_fence(MODEL))
        self.assertEqual(self.store.park_fence(MODEL2), self.store.park_fence(MODEL))
        self.assertNotIn(MODEL, self.tags)
        self.assertIn(PLAN["parked_as"], self.tags)
        self.assertIn(MODEL2, self.tags)
        self.assertIn(PLAN2["parked_as"], self.tags)

        park.unpark()
        self.assertEqual(self.tags, {MODEL: DIGEST, MODEL2: DIGEST2})
        self.assertFalse(park.STATE.exists())
        self.assertIsNone(self.store.park_fence(MODEL))
        self.assertIsNone(self.store.park_fence(MODEL2))
        request_id = self.store.start_request("default", MODEL2, "after-recovery")
        self.store.finish_request(request_id, completed=True, outcome="test")

    def test_copy_failure_with_uncheckpointed_alias_refuses_cleanup_until_alias_removed(self):
        self.copy_fail = PLAN["parked_as"]
        with mock.patch.object(park, "plan_park", return_value=[dict(PLAN), dict(PLAN2)]):
            with self.assertRaisesRegex(RuntimeError, "simulated copy response failure"):
                park.park(plan=[dict(PLAN), dict(PLAN2)])

        entries = park.parked_now()
        self.assertEqual([entry["name"] for entry in entries], [MODEL, MODEL2])
        self.assertEqual([entry["_park_phase"] for entry in entries], ["planned", "planned"])
        self.assertIn(PLAN["parked_as"], self.tags)
        self.assertIn(MODEL, self.tags)
        self.assertIsNotNone(self.store.park_fence(MODEL2))
        journal_before = park.STATE.read_bytes()
        tags_before = dict(self.tags)

        with self.assertRaisesRegex(RuntimeError, "journal does not own"):
            park.unpark()

        self.assertEqual(park.STATE.read_bytes(), journal_before)
        self.assertEqual(self.tags, tags_before)
        self.assertIsNotNone(self.store.park_fence(MODEL))

        self.tags.pop(PLAN["parked_as"])
        park.unpark()
        self.assertEqual(self.tags, {MODEL: DIGEST, MODEL2: DIGEST2})
        self.assertIsNone(self.store.park_fence(MODEL))
        self.assertIsNone(self.store.park_fence(MODEL2))



    def test_unpark_cannot_race_a_park_between_journal_and_fence(self):
        with tempfile.TemporaryDirectory(prefix="localbench-test-park-lock-") as tmp:
            root = Path(tmp)
            tags_path = root / "tags.json"
            tags_path.write_text(json.dumps({MODEL: DIGEST, MODEL2: DIGEST2}))
            (root / "plan.json").write_text(json.dumps([PLAN]))
            script = r'''
import json
import os
import sys
import time
from pathlib import Path
from localbench import gateway, park

root = Path(os.environ["PARK_TEST_ROOT"])
park.STATE = root / "PARKED.json"
park.HISTORY = root / "park-history.jsonl"
gateway.database_path = lambda home=None: root / "leases.sqlite"
tags_path = root / "tags.json"
plan = json.loads((root / "plan.json").read_text())
def tags():
    return json.loads(tags_path.read_text())
def save_tags(value):
    tags_path.write_text(json.dumps(value))
def copy(src, dst):
    value = tags()
    value[dst] = value[src]
    save_tags(value)
def delete(name):
    value = tags()
    del value[name]
    save_tags(value)
park._tags = tags
park._copy = copy
park._delete = delete
park._post = lambda *args, **kwargs: {}
park._get = lambda *args, **kwargs: {"models": []}
park.plan_park = lambda resolve=None: plan
gateway.safe_to_unload = lambda *args, **kwargs: (True, None)
write_state = park._write_state
def checkpoint(entries):
    write_state(entries)
    if sys.argv[1] == "park" and any(e.get("_park_phase") == "planned" for e in entries):
        (root / "journal-written").touch()
        while not (root / "continue-park").exists():
            time.sleep(0.01)
park._write_state = checkpoint
if sys.argv[1] == "park":
    park.park(plan=plan)
else:
    park.unpark()
'''
            env = {**os.environ, "PARK_TEST_ROOT": str(root)}
            command = [sys.executable, "-c", script]
            parked = subprocess.Popen([*command, "park"], cwd=Path(__file__).resolve().parents[1],
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            unparked = None
            try:
                deadline = time.monotonic() + 10
                while not (root / "journal-written").exists():
                    if parked.poll() is not None:
                        stdout, stderr = parked.communicate()
                        self.fail(f"park exited before its journal checkpoint: {stdout}\n{stderr}")
                    if time.monotonic() >= deadline:
                        self.fail("park did not reach its journal checkpoint")
                    time.sleep(0.01)

                journal_before = (root / "PARKED.json").read_bytes()
                tags_before = json.loads(tags_path.read_text())
                unparked = subprocess.run([*command, "unpark"], cwd=Path(__file__).resolve().parents[1],
                                          env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(unparked.returncode, 1, unparked.stderr)
                self.assertIn("park/unpark is already in progress", unparked.stderr)
                self.assertEqual((root / "PARKED.json").read_bytes(), journal_before)
                self.assertEqual(json.loads(tags_path.read_text()), tags_before)
            finally:
                (root / "continue-park").touch()
                stdout, stderr = parked.communicate(timeout=30)

            self.assertEqual(parked.returncode, 0, f"{stdout}\n{stderr}")
            self.assertNotIn(MODEL, json.loads(tags_path.read_text()))
            lock_path = root / "PARKED.json.lock"
            self.assertTrue(lock_path.is_file())
            lock_inode = lock_path.stat().st_ino
            self.assertEqual(lock_path.stat().st_mode & 0o777, 0o600)
            restored = subprocess.run([*command, "unpark"], cwd=Path(__file__).resolve().parents[1],
                                      env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(json.loads(tags_path.read_text()), {MODEL: DIGEST, MODEL2: DIGEST2})
            self.assertFalse((root / "PARKED.json").exists())
            self.assertEqual(lock_path.stat().st_ino, lock_inode)
            store = gateway.GatewayStore(root / "leases.sqlite")
            self.assertIsNone(store.park_fence(MODEL))


class GatewayParkFence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="localbench-test-gateway-fence-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "leases.sqlite"
        self.store = gateway.GatewayStore(self.path)
        self.store.set_profiles({"default": "/omp-profile/default"})

    def test_request_admission_racing_fence_is_serialized(self):
        model = "race-model"
        barrier = threading.Barrier(3)

        def admit():
            barrier.wait()
            try:
                return self.store.start_request("default", model, "race-client")
            except gateway.GatewayError:
                return None

        def fence():
            barrier.wait()
            return self.store.acquire_park_fence([model], "race-fence")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            request_future = pool.submit(admit)
            fence_future = pool.submit(fence)
            barrier.wait()
            request_id = request_future.result(timeout=5)
            refusal = fence_future.result(timeout=5)

        request_won = request_id is not None
        fence_won = refusal is None
        self.assertNotEqual(request_won, fence_won)
        if request_won:
            self.assertEqual(self.store.active_requests(model), 1)
            self.store.finish_request(request_id, completed=False, outcome="test")
        else:
            self.assertEqual(self.store.active_requests(model), 0)
            self.assertEqual(self.store.park_fence(model), "race-fence")
            self.store.release_park_fence("race-fence")

    def test_fence_survives_gateway_store_restart_and_blocks_admission(self):
        self.assertIsNone(self.store.acquire_park_fence(["parked-model"], "durable-fence"))
        restarted = gateway.GatewayStore(self.path)
        restarted.set_accepting(True)
        with self.assertRaisesRegex(gateway.GatewayError, "fenced for park"):
            restarted.start_request("default", "parked-model", "restarted-gateway")
        request_id = restarted.start_request("default", "unfenced-model", "restarted-gateway")
        restarted.finish_request(request_id, completed=True, outcome="test")
        restarted.release_park_fence("durable-fence")

    def test_unfenced_gateway_clients_keep_admission(self):
        request_id = self.store.start_request("default", "ordinary-model", "client")
        self.assertEqual(self.store.active_requests("ordinary-model"), 1)
        self.store.finish_request(request_id, completed=True, outcome="test")


if __name__ == "__main__":
    unittest.main()
