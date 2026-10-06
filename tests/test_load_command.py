"""Read-only load command: native parsers, per-owner roll-ups, and bounded sampling."""
from __future__ import annotations

import io
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from subprocess import CompletedProcess

from localbench import __main__ as main
from localbench import load

TEST_HOME = "/".join(("", "Users", "test"))

PS = f"""\
1 0 10-00:00:00 10:00:00 0.0 1024 launchd /sbin/launchd
10 1 02:00 00:01:00 0.0 2048 zsh -zsh
101 10 00:30 00:00:10 5.0 4096 bun {TEST_HOME}/.bun/bin/omp
102 101 00:10 00:00:01 1.0 1024 python python child.py
300 1 10-00:00:00 02:00:00 0.0 1024 launchd job
200 300 00:30 00:00:03 2.0 2048 python python /opt/launch/scheduler.py
201 200 00:02 00:00:01 1.0 512 bash bash /opt/launch/worker.sh
"""


def _top_table(load: str, cpu: str, rows: str) -> str:
    return (f"Load Avg: {load}\nCPU usage: {cpu}\n"
            "PID PPID %CPU CSW SYSBSD SYSMACH #TH COMMAND\n" + rows.rstrip() + "\n")


TOP_BASE_ROWS = """\
10 1 0.0 0 0 0 2 zsh
101 10 0.0 0 0 0 3 bun
102 101 0.0 0 0 0 2 python
200 300 0.0 0 0 0 6 python
201 200 0.0 0 0 0 2 bash
"""
TOP_DELTA_ROWS_1S = """\
10 1 2.0 2 4 1 2 zsh
101 10 10.0 10 20 4 3 bun
102 101 8.0 8 16 3 2 python
200 300 1.0 1 2 0.4 6 python
201 200 3.0 3 6 1.2 2 bash
999 200 30.0 6 12 2 1 transient-helper
"""
TOP_DELTA_ROWS_NO_SPAWN = """\
10 1 2.0 2 4 1 2 zsh
101 10 10.0 10 20 4 3 bun
102 101 8.0 8 16 3 2 python
200 300 1.0 1 2 0.4 6 python
201 200 3.0 3 6 1.2 2 bash
"""
TOP_BASE = _top_table("1.00, 2.00, 3.00", "10% user, 5% sys, 85% idle", TOP_BASE_ROWS)
TOP_INTERVAL_1S = _top_table("1.20, 2.10, 3.10", "15% user, 5% sys, 80% idle", TOP_DELTA_ROWS_1S)
TOP_INTERVAL_1S_NO_SPAWN = _top_table("1.20, 2.10, 3.10", "15% user, 5% sys, 80% idle",
                                      TOP_DELTA_ROWS_NO_SPAWN)
TOP_DELTA_ROWS_15S = """\
10 1 2.0 30 60 15 2 zsh
101 10 10.0 150 300 60 3 bun
102 101 8.0 120 240 45 2 python
200 300 1.0 15 30 6 6 python
201 200 3.0 45 90 18 2 bash
999 200 30.0 90 180 30 1 transient-helper
"""
TOP = TOP_BASE + _top_table("1.20, 2.10, 3.10", "15% user, 5% sys, 80% idle", TOP_DELTA_ROWS_15S)
TOP_TRANSIENT = TOP_BASE + TOP_INTERVAL_1S + TOP_INTERVAL_1S_NO_SPAWN


def _repeat_top_interval(sample_count: int) -> str:
    return TOP_BASE + TOP_INTERVAL_1S * (sample_count - 1)
IOSTAT = """\
disk0 disk6
KB/t tps MB/s KB/t tps MB/s
17.98 1236 21.70 44.26 445 19.23
7.84 2749 21.04 4.33 314 1.33
"""

VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
    free   active   specul inactive throttle    wired  prgable   faults     copy    0fill reactive   purged file-backed anonymous cmprssed cmprssor  dcomprs   comprs  pageins  pageout  swapins swapouts
 1000 2000 300 4000 0 500 60 1000 10 20 30 40 50 60 70 80 90 100 110 120 130 140
  900 2100 310 3900 0 490 70 1100 20 30 40 50 60 70 80 90 100 110 7 1 0 0
"""

PRESSURE = """\
The system has 549755813888 (33554432 pages with a page size of 16384).
System-wide memory free percentage: 92%
"""


class LoadParsing(unittest.TestCase):
    def test_ps_preserves_command_and_elapsed_time(self) -> None:
        rows = load.parse_ps(PS)
        self.assertEqual(rows[201].age_s, 2.0)
        self.assertEqual(rows[201].ppid, 200)
        self.assertEqual(rows[201].args, "bash /opt/launch/worker.sh")
        self.assertEqual(rows[1].age_s, 10 * 86400)

    def test_top_reads_the_second_interval_and_rates_counters(self) -> None:
        rows = load.parse_top(TOP, 15)
        self.assertEqual(rows[101].cpu_pct, 10.0)
        self.assertEqual(rows[101].csw_s, 10.0)
        self.assertEqual(rows[101].sysbsd_s, 20.0)
        self.assertEqual(rows[101].sysmach_s, 4.0)

    def test_iostat_uses_last_disk_sample(self) -> None:
        report = load.parse_iostat(IOSTAT)
        self.assertEqual(report["window_seconds"], 1)
        self.assertEqual(report["devices"], [
            {"device": "disk0", "kb_per_transfer": 7.84, "transfers_per_s": 2749.0, "mb_per_s": 21.04},
            {"device": "disk6", "kb_per_transfer": 4.33, "transfers_per_s": 314.0, "mb_per_s": 1.33},
        ])

    def test_vm_stat_separates_interval_events_from_current_pages(self) -> None:
        report = load.parse_vm_stat(VM_STAT)
        self.assertEqual(report["page_size_bytes"], 16384)
        self.assertEqual(report["window_seconds"], 1)
        self.assertEqual(report["current_pages"]["free"], 900)
        self.assertEqual(report["pages_per_s"],
                         {"pageins": 7.0, "pageouts": 1.0, "swapins": 0.0, "swapouts": 0.0})

    def test_memory_pressure_reports_available_memory_and_free_percentage(self) -> None:
        report = load.parse_memory_pressure(PRESSURE)
        self.assertEqual(report["available_bytes"], 549755813888)
        self.assertEqual(report["available_pages"], 33554432)
        self.assertEqual(report["page_size_bytes"], 16384)
        self.assertEqual(report["free_pct"], 92)

    def test_process_display_name_uses_full_argv_when_ps_comm_is_truncated(self) -> None:
        proc = load.Process(42, 1, 1.0, 0.0, 0.0, 128, "~/.loc",
                            "~/.local/bin/tier1-grade.py --worker")
        owner = load._owner_for(42, {42: proc}, {}, {}, {})
        rows, _ = load._process_rows({42: proc}, {}, {42: owner})
        self.assertEqual((owner, rows[0]["command"]),
                         ("unowned:tier1-grade.py", "tier1-grade.py"))

    def test_native_parsers_reject_large_near_misses_in_linear_time(self) -> None:
        malformed = "z" * 800_000
        parsers = (
            ("ps", lambda: load.parse_ps(malformed)),
            ("top", lambda: load.parse_top(malformed, 15)),
            ("iostat", lambda: load.parse_iostat(malformed)),
            ("vm_stat", lambda: load.parse_vm_stat(malformed)),
            ("memory_pressure", lambda: load.parse_memory_pressure(malformed)),
        )
        for name, parser in parsers:
            started = time.process_time_ns()
            self.assertEqual(parser(), {}, name)
            elapsed = (time.process_time_ns() - started) / 1_000_000_000
            self.assertLess(elapsed, 0.5, f"{name} near-miss parser cost {elapsed:.3f}s")


class LoadReport(unittest.TestCase):
    def fake_runner(self, calls: list[list[str]], *, top_text: str | None = None, no_pid_for: str | None = None):
        def run(argv, **kwargs):
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            self.assertGreaterEqual(kwargs["timeout"], 1)
            self.assertLessEqual(kwargs["timeout"], 3 * load.MAX_SECONDS + 10)
            calls.append(argv)
            if argv[0] == "/usr/bin/top":
                sample_count = int(argv[argv.index("-l") + 1])
                self.assertEqual(kwargs["timeout"], 3 * (sample_count - 1) + 10)
                stdout = top_text if top_text is not None else _repeat_top_interval(sample_count)
            else:
                stdout = {
                    "ps": PS,
                    "tmux": "10|bench|%pane\n",
                    "launchctl": "PID Status Label\n300 0 com.example.worker\n",
                    "lsof": f"p101\nn{TEST_HOME}/omp-test\n",
                    "iostat": IOSTAT,
                    "vm_stat": VM_STAT,
                    "memory_pressure": PRESSURE,
                }.get(argv[0], "")

            result = CompletedProcess(argv, 0, stdout, "")
            if argv[0] != no_pid_for:
                setattr(result, "pid", 10_000 + len(calls))
            return result
        return run
    def test_reports_process_owners_io_paging_pressure_and_probe_cost(self) -> None:
        calls: list[list[str]] = []
        report = load.collect(15, runner=self.fake_runner(calls))
        self.assertEqual(report["cpu_basis"], {
            "system_percent": "machine-wide (last /usr/bin/top table)",
            "process_percent": "one logical core = 100%",
            "process_window_seconds": 15,
            "process_interval_seconds": 1,
        })
        by_owner = {row["owner"]: row for row in report["owners"]}
        pane = by_owner["tmux:bench:%pane"]
        omp = by_owner[f"omp:{TEST_HOME}/omp-test (pid 101)"]
        job = by_owner["launchd:com.example.worker"]

        self.assertEqual((omp["cpu_pct"], omp["csw_s"], omp["sysbsd_s"],
                          omp["rss_mib"], omp["processes"]), (18.0, 18.0, 36.0, 5.0, 2))
        self.assertEqual((pane["cpu_pct"], pane["csw_s"], pane["sysbsd_s"],
                          pane["rss_mib"], pane["processes"]), (2.0, 2.0, 4.0, 2.0, 1))
        self.assertEqual((job["cpu_pct"], job["csw_s"], job["sysbsd_s"],
                          job["rss_mib"], job["processes"]), (34.0, 10.0, 20.0, 3.5, 4))
        self.assertFalse(report["warnings"])

        transient = next(row for row in report["top_cpu"] if row["pid"] == 999)
        self.assertEqual((transient["cpu_pct"], transient["csw_s"], transient["sysbsd_s"], transient["owner"]),
                         (30.0, 6.0, 12.0, "launchd:com.example.worker"))
        self.assertEqual(report["spawns"]["observed_in_window"], 1)
        self.assertEqual(report["spawns"]["by_parent_script"],
                         [{"script": "/opt/launch/scheduler.py", "owner": "launchd:com.example.worker", "count": 1}])

        omp_session = report["omp_sessions"][0]
        self.assertEqual((omp_session["pid"], omp_session["cpu_pct"], omp_session["rss_mib"],
                          omp_session["processes"]), (101, 18.0, 5.0, 2))
        top_csw = next(row for row in report["top_csw_per_s"] if row["pid"] == 101)
        top_sysbsd = next(row for row in report["top_sysbsd_per_s"] if row["pid"] == 101)
        self.assertEqual((top_csw["csw_s"], top_sysbsd["sysbsd_s"]), (10.0, 20.0))
        self.assertEqual(report["io"]["devices"][0]["device"], "disk0")
        self.assertEqual(report["io"]["devices"][0]["mb_per_s"], 21.04)
        self.assertEqual(report["paging"]["pages_per_s"]["pageins"], 7.0)
        self.assertEqual(report["paging"]["current_pages"]["free"], 900)
        self.assertEqual(report["pressure"]["free_pct"], 92)
        self.assertEqual(report["pressure"]["available_bytes"], 549755813888)
        self.assertEqual(report["probe_cost"]["budget_cpu_basis"], "machine")
        self.assertEqual(report["probe_cost"]["probe_invocations"], len(calls))
        self.assertEqual(report["probe_cost"]["spawns"], len(calls))
        self.assertLess(report["probe_cost"]["spawns"], 50)
        self.assertEqual(sum(call[0] == "ps" for call in calls), 1)
        self.assertEqual(sum(call[0] == "/usr/bin/top" for call in calls), 1)
        self.assertEqual(sum(call[0] == "lsof" for call in calls), 1)
        self.assertEqual(sum(call[0] == "iostat" for call in calls), 1)
        self.assertEqual(sum(call[0] == "vm_stat" for call in calls), 1)
        self.assertEqual(sum(call[0] == "memory_pressure" for call in calls), 1)
        view = load.render(report)
        self.assertIn("CPU basis: system=machine-wide (last /usr/bin/top table); process=one logical core = 100%", view)
        self.assertIn("process intervals=1s over 15s", view)
        self.assertIn("disk I/O", view)
        self.assertIn("paging", view)
        self.assertIn("memory pressure", view)
    def test_spawns_include_pid_seen_between_top_tables_even_if_absent_from_final_ps(self) -> None:
        calls: list[list[str]] = []
        report = load.collect(2, runner=self.fake_runner(calls, top_text=TOP_TRANSIENT))

        self.assertEqual(report["spawns"]["observed_in_window"], 1)
        row = report["spawns"]["by_parent_script"][0]
        self.assertEqual((row["script"], row["owner"], row["count"]),
                         ("/opt/launch/scheduler.py", "launchd:com.example.worker", 1))
        transient = next(row for row in report["top_cpu"] if row["pid"] == 999)
        self.assertEqual((transient["cpu_pct"], transient["owner"]),
                         (15.0, "launchd:com.example.worker"))
        job = next(row for row in report["owners"] if row["owner"] == "launchd:com.example.worker")
        self.assertEqual((job["cpu_pct"], job["csw_s"], job["sysbsd_s"], job["processes"]),
                         (19.0, 7.0, 14.0, 4))

    def test_probe_cost_counts_started_pids_not_runner_invocations(self) -> None:
        calls: list[list[str]] = []
        report = load.collect(1, runner=self.fake_runner(calls, no_pid_for="memory_pressure"))

        self.assertEqual(report["probe_cost"]["probe_invocations"], len(calls))
        self.assertEqual(report["probe_cost"]["spawns"], len(calls) - 1)
        self.assertTrue(report["top_cpu"])
        self.assertTrue(report["owners"])
    def test_sample_window_maximum_and_rejection_boundary(self) -> None:
        calls: list[list[str]] = []
        report = load.collect(32, runner=self.fake_runner(calls))
        top_call = next(call for call in calls if call[0] == "/usr/bin/top")
        self.assertEqual(report["sample_seconds"], 32)
        self.assertEqual(top_call[top_call.index("-s") + 1], "1")
        self.assertEqual(top_call[top_call.index("-l") + 1], "33")
        for seconds in (0, 33, True):
            with self.subTest(seconds=seconds), self.assertRaises(load.LoadError):
                load.collect(seconds, runner=self.fake_runner(calls))
        self.assertEqual(len(calls), 8)

    def test_cli_rejects_invalid_window_as_usage_error(self) -> None:
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main.main(["load", "--seconds", "33"])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("--seconds", stderr.getvalue())
        self.assertIn("32", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
