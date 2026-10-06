"""Heavy-job slot: one holder at a time, admission by CPU busy, memory pressure and GPU.
Tests use fake sensors and homes: nothing here measures the machine."""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import heavyslot


def gpu_report(pct):
    return {"device_pct": pct, "process_pct": 0.5 if pct < 5.0 else pct,
            "coverage": None if pct < 5.0 else 100.0, "unattributed_pct": 0.0,
            "status": "IDLE" if pct < 5.0 else "ATTRIBUTED"}


def calm(load=10.0, gpu=12):
    """Fake quiet-machine readings."""
    return lambda: (load,), lambda: gpu_report(gpu)


def spawn_holder(test, home, verb, seconds):
    """A live rival in a fresh interpreter (same-process flock is platform-defined; only a real
    second process proves the refusal). Ready when it prints HELD."""
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; sys.path.insert(0, sys.argv[1]); "
         "from localbench import heavyslot; "
         "s = heavyslot.acquire(sys.argv[2], home=sys.argv[3], load_fn=lambda: (1.0,), "
         "cpu_fn=lambda: 1.0, memory_fn=lambda: {'pressure_level': 'normal'}, "
         "gpu_fn=lambda: {'device_pct': 1, 'process_pct': 0.5, 'coverage': None, 'unattributed_pct': 0.0, 'status': 'IDLE'}); "
         "print('HELD', flush=True); time.sleep(float(sys.argv[4]))",
         str(root), verb, str(home), str(seconds)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    test.addCleanup(_stop_child, proc)
    seen = []
    while True:
        line = proc.stdout.readline()
        if line.strip() == "HELD":
            return proc
        seen.append(line)
        if not line or proc.poll() is not None:
            raise AssertionError("holder child failed: " + "".join(seen)[-2000:])


def _stop_child(proc):
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    if proc.stderr is not None:
        proc.stderr.close()
    if proc.stdout is not None:
        proc.stdout.close()


def _wait_for_ticket(home, pid):
    path = heavyslot.queue_dir(home) / f"{pid}.json"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"waiter {pid} did not enqueue")


def _spawn_queue_waiter(test, home, verb, needs_gpu, events, label, poll_s):
    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys,time; sys.path.insert(0,sys.argv[1]); "
        "from localbench import heavyslot; heavyslot.background_priority=lambda: True; "
        "slot=heavyslot.acquire(sys.argv[2],home=sys.argv[3],wait_s=10,"
        "needs_gpu=sys.argv[4]=='1',load_fn=lambda:(1.0,),cpu_fn=lambda:1.0,"
        "memory_fn=lambda:{'pressure_level':'normal'},gpu_fn=lambda:{'device_pct':1,'process_pct':0.5,'coverage':None,'unattributed_pct':0.0,'status':'IDLE'},"
        "poll_s=float(sys.argv[6])); "
        "f=open(sys.argv[5],'a'); f.write(sys.argv[7]+'\\n'); f.flush(); f.close(); "
        "time.sleep(0.01); slot.release()"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(root), verb, str(home), "1" if needs_gpu else "0",
         str(events), str(poll_s), label],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    test.addCleanup(_stop_child, proc)
    _wait_for_ticket(home, proc.pid)
    return proc


class Acquire(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.load, self.gpu = calm()

    def test_take_names_holder_and_release_frees(self):
        slot = heavyslot.acquire("ab", home=self.home, load_fn=self.load, gpu_fn=self.gpu)
        held = heavyslot.holder(self.home)
        assert held is not None
        self.assertEqual((held["verb"], held["pid"], slot.admission["forced"]),
                         ("ab", os.getpid(), False))
        self.assertTrue(held["started_at"] and held["repo"])
        slot.release()
        self.assertIsNone(heavyslot.holder(self.home))
        slot.release()  # idempotent

    def test_release_closes_the_lock_fd(self):
        slot = heavyslot.acquire("ab", home=self.home, load_fn=self.load, gpu_fn=self.gpu)
        fd = slot._fd
        slot.release()
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_live_rival_refuses_naming_the_holder(self):
        spawn_holder(self, self.home, "ab", 30)
        with self.assertRaisesRegex(heavyslot.SlotRefused, r"held by pid \d+ \(ab"):
            heavyslot.acquire("run", home=self.home, load_fn=self.load, gpu_fn=self.gpu)

    def test_stale_owner_without_a_lock_is_free(self):
        heavyslot.owner_path(self.home).parent.mkdir(parents=True, exist_ok=True)
        heavyslot.owner_path(self.home).write_text(json.dumps({"pid": 1, "verb": "dead"}) + "\n")
        self.assertIsNone(heavyslot.holder(self.home))
        slot = heavyslot.acquire("run", home=self.home, load_fn=self.load, gpu_fn=self.gpu)
        held = heavyslot.holder(self.home)
        assert held is not None
        self.assertEqual(held["verb"], "run")
        slot.release()

    def test_same_process_reenters_without_relocking(self):
        outer = heavyslot.acquire("prove", home=self.home, load_fn=self.load, gpu_fn=self.gpu)
        inner = heavyslot.acquire("prove", home=self.home, load_fn=lambda: (99.0,),
                                  gpu_fn=lambda: gpu_report(99))
        inner.release()
        self.assertIsNotNone(heavyslot.holder(self.home))  # outer still holds
        outer.release()
        self.assertIsNone(heavyslot.holder(self.home))

    def test_wait_slot_queues_until_release(self):
        spawn_holder(self, self.home, "aa", 0.5)
        got = heavyslot.acquire("run", home=self.home, wait_s=10.0, load_fn=self.load, gpu_fn=self.gpu,
                                poll_s=0.05)
        got.release()
        self.assertIsNone(heavyslot.holder(self.home))

    def test_wait_timeout_names_the_holder(self):
        spawn_holder(self, self.home, "aa", 30)
        with self.assertRaisesRegex(heavyslot.SlotRefused, r"held by pid \d+ \(aa"):
            heavyslot.acquire("run", home=self.home, wait_s=0.2, load_fn=self.load, gpu_fn=self.gpu,
                              poll_s=0.05)


class Admission(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def take(self, **kw):
        return heavyslot.acquire("run", home=self.home, **kw)


    def test_high_load_with_idle_cpu_and_normal_memory_is_admitted(self):
        slot = self.take(load_fn=lambda: (106.0,), cpu_fn=lambda: 36.0,
                         memory_fn=lambda: {"pressure_level": "normal", "free_pct": 64},
                         gpu_fn=lambda: gpu_report(1))
        try:
            self.assertEqual((slot.admission["load1"], slot.admission["cpu_busy"],
                              slot.admission["mem_pressure"]), (106.0, 36.0, "normal"))
        finally:
            slot.release()

    def test_unreadable_load_is_context_not_an_admission_gate(self):
        def unreadable():
            raise OSError("load unavailable")

        slot = self.take(load_fn=unreadable, cpu_fn=lambda: 36.0,
                         memory_fn=lambda: {"pressure_level": "normal"},
                         gpu_fn=lambda: gpu_report(1))
        try:
            self.assertIsNone(slot.admission["load1"])
        finally:
            slot.release()

    def test_95_percent_cpu_busy_refuses(self):
        with self.assertRaisesRegex(heavyslot.SlotRefused, r"CPU busy 95\.0% >= 80%"):
            self.take(load_fn=lambda: (1.0,), cpu_fn=lambda: 95.0,
                      memory_fn=lambda: {"pressure_level": "normal"},
                      gpu_fn=lambda: gpu_report(1))

    def test_cpu_busy_limit_boundary_refuses(self):
        with self.assertRaisesRegex(heavyslot.SlotRefused, r"CPU busy 80\.0% >= 80%"):
            self.take(load_fn=lambda: (1.0,), cpu_fn=lambda: 80.0,
                      memory_fn=lambda: {"pressure_level": "normal"},
                      gpu_fn=lambda: gpu_report(1))

    def test_non_normal_memory_pressure_refuses(self):
        with self.assertRaisesRegex(heavyslot.SlotRefused, "memory pressure warn"):
            self.take(load_fn=lambda: (1.0,), cpu_fn=lambda: 36.0,
                      memory_fn=lambda: {"pressure_level": "warn"},
                      gpu_fn=lambda: gpu_report(1))

    def test_unreadable_cpu_and_memory_refuse_closed(self):
        with self.assertRaisesRegex(heavyslot.SlotRefused, "CPU busy percentage unreadable"):
            self.take(load_fn=lambda: (1.0,), cpu_fn=lambda: None,
                      memory_fn=lambda: {"pressure_level": "normal"},
                      gpu_fn=lambda: gpu_report(1))
        with self.assertRaisesRegex(heavyslot.SlotRefused, "memory pressure unreadable"):
            self.take(load_fn=lambda: (1.0,), cpu_fn=lambda: 36.0,
                      memory_fn=lambda: {"pressure_level": None},
                      gpu_fn=lambda: gpu_report(1))

    def test_busy_gpu_refuses(self):
        with self.assertRaisesRegex(heavyslot.SlotRefused, r"GPU 85% >= 80% busy"):
            self.take(load_fn=lambda: (10.0,), gpu_fn=lambda: gpu_report(85))

    def test_unreadable_gpu_refuses_closed(self):
        with self.assertRaisesRegex(heavyslot.SlotRefused, "GPU utilization unreadable"):
            self.take(load_fn=lambda: (10.0,), gpu_fn=lambda: {})
        with self.assertRaisesRegex(heavyslot.SlotRefused, "GPU utilization unreadable"):
            self.take(load_fn=lambda: (10.0,), gpu_fn=lambda: {"device_pct": None})

    def test_idle_gpu_under_two_percent_passes_accounting_gate(self):
        idle = {"device_pct": 1.0, "process_pct": 0.5, "coverage": None,
                "unattributed_pct": 0.0, "status": "IDLE"}
        slot = self.take(load_fn=lambda: (10.0,), cpu_fn=lambda: 36.0,
                         memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=lambda: idle)
        try:
            self.assertEqual(slot.admission["gpu_status"], "IDLE")
        finally:
            slot.release()

    def test_near_idle_absolute_noise_passes_accounting_gate(self):
        near_idle = {"device_pct": 2.2, "process_pct": 4.1, "coverage": None,
                     "unattributed_pct": 0.0, "status": "IDLE"}
        slot = self.take(load_fn=lambda: (10.0,), cpu_fn=lambda: 36.0,
                         memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=lambda: near_idle)
        try:
            self.assertEqual((slot.admission["gpu_status"], slot.admission["gpu_pct"],
                              slot.admission["gpu_process_pct"]), ("IDLE", 2.2, 4.1))
        finally:
            slot.release()

    def test_attributed_terminal_window_passes_coverage_gate(self):
        terminal = {"device_pct": 8.0, "process_pct": 7.6, "coverage": 95.0,
                    "unattributed_pct": 5.0, "status": "ATTRIBUTED"}
        slot = self.take(load_fn=lambda: (10.0,), cpu_fn=lambda: 36.0,
                         memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=lambda: terminal)
        try:
            self.assertEqual((slot.admission["gpu_status"], slot.admission["gpu_coverage"]),
                             ("ATTRIBUTED", 95.0))
        finally:
            slot.release()

    def test_unattributed_gpu_refuses_even_below_busy_limit(self):
        unaccounted = {"device_pct": 10.0, "process_pct": 1.05, "coverage": 10.5,
                       "unattributed_pct": 89.5, "status": "UNATTRIBUTED"}
        with self.assertRaisesRegex(heavyslot.SlotRefused, r"GPU attribution UNATTRIBUTED.*10.5%"):
            self.take(load_fn=lambda: (10.0,), cpu_fn=lambda: 36.0,
                      memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=lambda: unaccounted)

    def test_unaligned_gpu_window_refuses_admission(self):
        unaligned = {"device_pct": 10.0, "process_pct": 11.52, "coverage": 100.0,
                     "unattributed_pct": 0.0, "status": "UNALIGNED"}
        with self.assertRaisesRegex(heavyslot.SlotRefused, r"GPU attribution UNALIGNED"):
            self.take(load_fn=lambda: (10.0,), cpu_fn=lambda: 36.0,
                      memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=lambda: unaligned)
    def test_force_load_takes_and_records(self):
        slot = self.take(load_fn=lambda: (70.0,), gpu_fn=lambda: gpu_report(99), force_load=True)
        try:
            self.assertEqual(slot.admission["forced"], True)
        finally:
            slot.release()

    def test_cpu_holder_skips_the_gpu_gate(self):
        slot = self.take(load_fn=lambda: (10.0,), cpu_fn=lambda: 36.0,
                         memory_fn=lambda: {"pressure_level": "normal"},
                         gpu_fn=lambda: gpu_report(95), needs_gpu=False)
        try:
            self.assertEqual((slot.admission["gpu_checked"], slot.admission["cpu_busy"],
                              slot.admission["mem_pressure"]), (False, 36.0, "normal"))
        finally:
            slot.release()

    def test_cpu_holder_still_checks_cpu_and_memory_gates(self):
        with self.assertRaisesRegex(heavyslot.SlotRefused, "CPU busy 95.0%"):
            self.take(load_fn=lambda: (1.0,), cpu_fn=lambda: 95.0,
                      memory_fn=lambda: {"pressure_level": "normal"},
                      gpu_fn=lambda: gpu_report(95), needs_gpu=False)
        with self.assertRaisesRegex(heavyslot.SlotRefused, "memory pressure critical"):
            self.take(load_fn=lambda: (1.0,), cpu_fn=lambda: 36.0,
                      memory_fn=lambda: {"pressure_level": "critical"},
                      gpu_fn=lambda: gpu_report(95), needs_gpu=False)

    def test_cpu_holder_with_high_load_and_idle_cpu_runs_at_background_priority(self):
        """A load of 106 with 36% CPU busy and normal pressure passes, but yields via nice 10."""
        demoted = []
        orig = heavyslot.background_priority
        heavyslot.background_priority = lambda: demoted.append(1) or True
        self.addCleanup(setattr, heavyslot, "background_priority", orig)
        slot = self.take(load_fn=lambda: (106.0,), cpu_fn=lambda: 36.0,
                         memory_fn=lambda: {"pressure_level": "normal"},
                         gpu_fn=lambda: gpu_report(95), needs_gpu=False)
        try:
            self.assertEqual((slot.admission["background"], demoted), (True, [1]))
        finally:
            slot.release()


class Decorator(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self._patch(heavyslot, "lock_dir", lambda home=None: self.home / ".localbench")
        self._patch(heavyslot, "LOAD_FN", lambda: (1.0, 1.0, 1.0))
        self._patch(heavyslot, "CPU_FN", lambda: 1.0)
        self._patch(heavyslot, "MEMORY_FN", lambda: {"pressure_level": "normal"})
        self._patch(heavyslot, "GPU_FN", lambda: gpu_report(1))

    def _patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, fn):
        from localbench import __main__ as main_mod

        args = argparse.Namespace(wait_slot=0, force_load=False,
                                  mutation=main_mod.Mutation("run", [], audited=False))
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            rc = fn(args)
        return rc, out.getvalue(), args

    def test_held_holder_refuses_naming_the_verb(self):
        from localbench import __main__ as main_mod

        spawn_holder(self, self.home, "ab", 30)
        calls = []

        @main_mod._held("run")
        def cmd(args):
            calls.append(True)
            return 0

        rc, err, _ = self._run(cmd)
        self.assertEqual(rc, 1)
        self.assertEqual(calls, [])
        self.assertIn("held by pid", err)
        self.assertIn("(ab", err)

    def test_free_slot_runs_releases_and_records_admission(self):
        from localbench import __main__ as main_mod

        @main_mod._held("run")
        def cmd(args):
            return 7

        rc, _, args = self._run(cmd)
        self.assertEqual(rc, 7)
        self.assertIsNone(heavyslot.holder())
        self.assertEqual(args.mutation.detail["heavy_slot"]["forced"], False)

    def test_eval_run_refuses_on_a_held_slot(self):
        from localbench import __main__ as main_mod

        spawn_holder(self, self.home, "eval run", 30)
        args = argparse.Namespace(wait_slot=0, force_load=False,
                                  mutation=main_mod.Mutation("eval run", [], audited=False))
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            rc = main_mod.cmd_eval_run(args)
        self.assertEqual(rc, 1)
        self.assertIn("held by pid", out.getvalue())

    def test_eval_run_parser_has_slot_flags(self):
        root = Path(__file__).resolve().parent.parent
        proc = subprocess.run([sys.executable, "-m", "localbench", "eval", "run", "--help"],
                              capture_output=True, text=True, cwd=root, timeout=60)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--wait-slot", proc.stdout)
        self.assertIn("--force-load", proc.stdout)


class Observability(unittest.TestCase):
    """Queue tickets, release history and the ETAs `localbench slot` derives from them."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.load, self.gpu = calm()

    def assert_acquisition_order(self, requests, expected):
        events = self.home / "acquired.txt"
        events.touch()
        holder = heavyslot.acquire(
            "holder", home=self.home, needs_gpu=False, load_fn=self.load, cpu_fn=lambda: 1.0,
            memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=self.gpu)
        procs = []
        try:
            for label, needs_gpu in requests:
                procs.append(_spawn_queue_waiter(
                    self, self.home, label, needs_gpu, events, label, poll_s=0.25))
        finally:
            holder.release()
        for proc in procs:
            self.assertEqual(proc.wait(timeout=10), 0, proc.stderr.read())
        self.assertEqual(events.read_text().splitlines(), expected)

    def test_gpu_ticket_precedes_five_earlier_cpu_tickets(self):
        cpu = [(f"cpu-{i}", False) for i in range(5)]
        self.assert_acquisition_order(cpu + [("gpu", True)], ["gpu"] + [name for name, _ in cpu])

    def test_gpu_class_precedence_preserves_fifo_within_each_class(self):
        requests = [("cpu-first", False), ("gpu-first", True),
                    ("cpu-second", False), ("gpu-second", True)]
        self.assert_acquisition_order(
            requests, ["gpu-first", "gpu-second", "cpu-first", "cpu-second"])

    def test_zero_wait_request_cannot_bypass_a_queued_ticket(self):
        events = self.home / "acquired.txt"
        events.touch()
        holder = heavyslot.acquire(
            "holder", home=self.home, needs_gpu=False, load_fn=self.load, cpu_fn=lambda: 1.0,
            memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=self.gpu)
        waiter = None
        try:
            waiter = _spawn_queue_waiter(
                self, self.home, "queued", True, events, "queued", poll_s=1.0)
            holder.release()
            try:
                slot = heavyslot.acquire(
                    "immediate", home=self.home, load_fn=self.load, cpu_fn=lambda: 1.0,
                    memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=self.gpu)
            except heavyslot.SlotRefused as exc:
                self.assertIn("earlier queue ticket", str(exc))
            else:
                slot.release()
                self.fail("zero-wait caller bypassed the queued GPU ticket")
        finally:
            holder.release()
        self.assertEqual(waiter.wait(timeout=5), 0, waiter.stderr.read())
        self.assertEqual(events.read_text().splitlines(), ["queued"])

    def live_pid(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        return proc.pid

    def ticket(self, pid, verb, enqueued_at):
        d = heavyslot.queue_dir(self.home)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{pid}.json").write_text(json.dumps({"pid": pid, "verb": verb, "repo": "/r", "needs_gpu": True,
                                                   "enqueued_at": enqueued_at}))

    def history(self, *rows):
        path = heavyslot.history_path(self.home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"verb": v, "wall_s": w}) + "\n" for v, w in rows))

    def test_waiters_listed_in_enqueue_order_not_pid_order(self):
        a, b = sorted([self.live_pid(), self.live_pid()])
        self.ticket(b, "first", 100.0)    # the higher pid enqueued first: order follows enqueued_at
        self.ticket(a, "second", 200.0)
        queue = heavyslot.report(self.home, now=260.0)["queue"]
        self.assertEqual([(t["pid"], t["position"], t["waited_s"]) for t in queue], [(b, 1, 160.0), (a, 2, 60.0)])

    def test_dead_waiter_is_pruned(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        live = self.live_pid()
        self.ticket(dead.pid, "gone", 1.0)
        self.ticket(live, "here", 2.0)
        self.assertEqual([t["pid"] for t in heavyslot.waiters(self.home)], [live])
        self.assertFalse((heavyslot.queue_dir(self.home) / f"{dead.pid}.json").exists())

    def test_eta_is_history_median_chained_through_waiters_ahead(self):
        a, b = self.live_pid(), self.live_pid()
        self.history(("ab", 10.0), ("ab", 30.0), ("ab", 1000.0), ("run", 50.0), ("ab", 20.0))
        self.ticket(a, "ab", 1.0)
        self.ticket(b, "run", 2.0)
        queue = heavyslot.report(self.home)["queue"]
        # no holder: the first starts now; the second after the first's median ab hold (10,30,1000,20 -> 25)
        self.assertEqual([t["estimated_start_s"] for t in queue], [0.0, 25.0])

    def test_holder_remaining_from_median_and_unknown_without_history(self):
        rival = spawn_holder(self, self.home, "ab", 30)
        self.history(("ab", 1000.0))
        rep = heavyslot.report(self.home)
        self.assertEqual(rep["holder"]["pid"], rival.pid)
        self.assertGreater(rep["holder"]["expected_remaining_s"], 900.0)
        heavyslot.history_path(self.home).unlink()
        self.assertIsNone(heavyslot.report(self.home)["holder"]["expected_remaining_s"])

    def test_last_release_appends_one_history_line(self):
        self.history(("old", 1.0))
        with heavyslot.acquire("ab", home=self.home, load_fn=self.load, gpu_fn=self.gpu, repo="/repo"):
            with heavyslot.acquire("ab", home=self.home, load_fn=self.load, gpu_fn=self.gpu):
                pass   # releasing a re-entry is not the last release: no row
        rows = heavyslot.history(self.home)
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[-1]["verb"], rows[-1]["repo"], rows[-1]["pid"]), ("ab", "/repo", os.getpid()))
        self.assertGreaterEqual(rows[-1]["wall_s"], 0.0)
        self.assertTrue(rows[-1]["acquired_at"] and rows[-1]["released_at"])
        self.assertTrue(rows[-1]["needs_gpu"])

    def test_history_records_cpu_class(self):
        with mock.patch.object(heavyslot, "background_priority", return_value=True):
            with heavyslot.acquire(
                    "suite", home=self.home, needs_gpu=False, load_fn=self.load, cpu_fn=lambda: 1.0,
                    memory_fn=lambda: {"pressure_level": "normal"}, gpu_fn=self.gpu):
                pass
        self.assertFalse(heavyslot.history(self.home)[-1]["needs_gpu"])

    def test_history_read_is_bounded_to_the_tail(self):
        self.history(*[("ab", float(i)) for i in range(50)])
        self.assertEqual([r["wall_s"] for r in heavyslot.history(self.home, tail=3)], [47.0, 48.0, 49.0])

    def test_refusal_names_queue_position_and_ticket_is_removed(self):
        spawn_holder(self, self.home, "ab", 30)
        self.history(("ab", 1000.0), ("run", 5.0))
        live = self.live_pid()
        self.ticket(live, "run", 0.0)   # queued before us
        with self.assertRaises(heavyslot.SlotRefused) as cm:
            heavyslot.acquire("aa", home=self.home, wait_s=0.3, poll_s=0.05, load_fn=self.load, gpu_fn=self.gpu)
        self.assertIn("queue position 2", str(cm.exception))
        self.assertNotIn("unknown", str(cm.exception))
        self.assertEqual([t["pid"] for t in heavyslot.waiters(self.home)], [live])

    def test_waiter_that_raises_removes_its_ticket(self):
        spawn_holder(self, self.home, "ab", 30)
        mine = heavyslot.queue_dir(self.home) / f"{os.getpid()}.json"
        seen = []

        def load():
            if mine.exists():
                seen.append(True)
                raise KeyboardInterrupt
            return (1.0,)

        with self.assertRaises(KeyboardInterrupt):
            heavyslot.acquire("aa", home=self.home, wait_s=30, poll_s=0.05, load_fn=load, gpu_fn=self.gpu)
        self.assertEqual(seen, [True])
        self.assertFalse(mine.exists())


if __name__ == "__main__":
    unittest.main()
