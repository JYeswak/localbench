"""Interval math behind the call splits (busy seconds, memory-LLM overlap with main calls) and the omp child flags."""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench.client import Sample
from localbench.render import CLIP
from localbench.sysstats import omp_client_identity
from localbench.workloads import (
    MEM_ROUNDS,
    REL_ATTEMPTS,
    THINK_MAX_TOKENS,
    THINK_TASKS,
    Ctx,
    _Rpc,
    _busy_s,
    _leaked,
    _overlap_s,
    answer_call,
    child_flags,
    e2e,
    final_answer,
    mem,
    prompt_token_verdict,
    recorded_answers,
    rel,
    replay,
    sess,
    think,
)


def r(a: float, b: float) -> dict:
    return {"t_start": a, "t": b}


class Intervals(unittest.TestCase):
    MAIN = [r(0, 2), r(1, 3), r(10, 12)]          # union [0,3] + [10,12]
    MEM = [r(2.5, 4), r(11, 20), r(11.5, 13)]     # union [2.5,4] + [11,20]

    def test_busy_counts_parallel_calls_once(self):
        self.assertEqual(_busy_s(self.MAIN), 5.0)
        self.assertIsNone(_busy_s([]))

    def test_overlap_is_the_intersection_of_unions(self):
        self.assertEqual(_overlap_s(self.MAIN, self.MEM), 1.5)   # [2.5,3] + [11,12]
        self.assertEqual(_overlap_s(self.MEM, self.MAIN), 1.5)

    def test_touching_and_empty_are_zero(self):
        self.assertEqual(_overlap_s([r(0, 1)], [r(1, 2)]), 0)
        self.assertEqual(_overlap_s(self.MAIN, []), 0)


class ChildFlags(unittest.TestCase):
    def test_mode_is_explicit_and_single(self):
        flags = child_flags("m", mode="rpc")
        self.assertEqual(flags[flags.index("--mode") + 1], "rpc")
        self.assertEqual(flags.count("--mode"), 1)
        self.assertIn("--no-session", flags)
        # smol goes to the model under test: one model on the machine
        self.assertEqual(flags[flags.index("--smol") + 1], "localbench/m")

    def test_mem_children_get_the_memory_tools_and_nothing_else(self):
        # --no-tools left the model calling the recall/retain its prompt describes (2026-09-27 ledger correction);
        # bash/grep let a mem control search the machine (2026-09-23). Only the tool list differs from a lean child.
        flags = child_flags("m", tools="memory")
        listed = [f for f in flags if f.startswith("--tools") or f == "--no-tools"]
        self.assertEqual(listed, ["--tools=memory_edit,recall,reflect,retain"])
        self.assertEqual([f for f in flags if f not in listed],
                         [f for f in child_flags("m") if not f.startswith("--tools")])

    def test_an_unknown_tool_set_is_refused_not_given_the_lean_tools(self):
        # A caller still passing the old tools=False must not get bash and read.
        with self.assertRaises(ValueError):
            child_flags("m", tools=False)


class MemLeaks(unittest.TestCase):
    """A control answer (fresh project, nothing planted) must not contain any value planted earlier in the run."""

    def test_this_round_and_earlier_rounds_both_count(self):
        planted = ["ZEBRA-5567", "53817", "Moreau", "ZEBRA-7836"]
        self.assertEqual(_leaked("ZEBRA-5567", planted), ["ZEBRA-5567"])       # 2026-09-23: missed by the old check
        self.assertEqual(_leaked("the code is zebra-7836.", planted), ["ZEBRA-7836"])

    def test_a_refusal_is_not_a_leak(self):
        self.assertEqual(_leaked("No project named Falcon exists in this workspace.", ["ZEBRA-5567", "53817"]), [])



class MemRounds(unittest.TestCase):
    """The loop must read ctx.mem_rounds. Leaving range(MEM_ROUNDS) makes --mem-rounds a no-op."""

    def test_one_round_plants_three_facts_not_the_constant(self):
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=1024, mem_rounds=1)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run",
                               return_value=subprocess.CompletedProcess([], 0, "", "")), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                results = mem(ctx)
        recall = next(r for r in results if r.case == "mem.recall")
        self.assertEqual(recall.detail["attempts"], 3)
        self.assertEqual({a["round"] for a in recall.detail["rounds"]}, {0})
        self.assertNotEqual(MEM_ROUNDS, 1)

    def test_a_turn_that_times_out_is_a_miss_and_the_tier_still_reports(self):
        # 2026-09-26: one omp turn ran 30 min on Bonsai 2 / mlxfast; the TimeoutExpired ended the whole A/B.
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

        def run(args, **kwargs):
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=1024, mem_rounds=1)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=run), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                results = mem(ctx)
        recall = next(r for r in results if r.case == "mem.recall")
        self.assertEqual(recall.detail["attempts"], 3)
        self.assertEqual(recall.metrics["hit_rate"]["value"], 0)
        self.assertTrue(all("timeout" in a["rcs"] for a in recall.detail["rounds"]))
        turn = next(r for r in results if r.case == "mem.turn")
        self.assertEqual(turn.metrics["max_tool_calls"]["value"], 0)
        self.assertEqual(turn.metrics["timeouts"]["value"], 12)
        self.assertEqual(len(turn.detail["turns"]), 12)
        self.assertTrue(all(t["timeout"] for t in turn.detail["turns"]))


class MemTurns(unittest.TestCase):
    """mem() times its recall turns from the proxy log it writes, and runs every child with the tool list the run pins
    as omp_mem_tools."""

    def run_mem(self, omp_turn, tmp: str) -> list:
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

        ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                  loaded_context=1024, mem_rounds=1)
        with mock.patch("localbench.proxy.Proxy", _Proxy), \
                mock.patch("localbench.workloads.ensure_localbench_model"), \
                mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                mock.patch("localbench.workloads.subprocess.run", side_effect=omp_turn), \
                mock.patch("localbench.workloads._mem_facts",
                           return_value=[(n, f"PLANT {n}", f"QUESTION {n}", f"ZEBRA-{n}") for n in "abc"]), \
                mock.patch("localbench.memory.banks", return_value=[]), \
                mock.patch("localbench.memory.remove_banks"):
            return mem(ctx)

    def oracle_results(self, *, plant_rc=0, control_text="unknown", control_rc=0,
                       control_main=True, control_status=200, recall_text=None, extraction=True,
                       extraction_status=200, first_main_failed=False):
        """Replay child outcomes and proxy rows without starting omp or using a live memory bank."""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mem_calls.jsonl"

            def omp_turn(argv, **kwargs):
                prompt = argv[2]
                kind = ("plant" if prompt.startswith("PLANT") else
                        "control" if "control-" in str(kwargs["cwd"]) else
                        "recall" if prompt.startswith("QUESTION") else "derail")
                name = prompt.rsplit(" ", 1)[-1] if kind != "derail" else ""
                now = time.time()
                row = {"t_start": now, "t": now, "status": 200, "aborted": False,
                       "response_complete": True}
                rows = ([{**row, "purpose": "main", "status": 502}, {**row, "purpose": "main"}]
                        if first_main_failed and (kind != "control" or control_main) else
                        [{**row, "purpose": "main"}] if kind != "control" or control_main else [])
                if kind == "control" and rows:
                    rows[-1]["status"] = control_status
                if kind == "plant" and extraction:
                    rows.append({**row, "purpose": "memory-extract", "status": extraction_status})
                with log.open("a") as fh:
                    for item in rows:
                        fh.write(json.dumps(item) + "\n")
                text = {"plant": "NOTED", "recall": recall_text or f"ZEBRA-{name}",
                        "control": control_text, "derail": "OK"}[kind]
                return subprocess.CompletedProcess(argv, plant_rc if kind == "plant" else
                                                   control_rc if kind == "control" else 0,
                                                   _omp_stdout(text), "")

            return self.run_mem(omp_turn, tmp)

    def test_recall_must_match_the_planted_value_exactly(self):
        recall = next(r for r in self.oracle_results(recall_text="ZEBRA-a-stale")
                      if r.case == "mem.recall")
        self.assertEqual(recall.detail["hits"], 0)
        self.assertIn("did not match planted fact", recall.detail["rounds"][0]["reason"])

    def test_nonblank_control_with_failed_proxy_response_is_not_valid_evidence(self):
        control = next(r for r in self.oracle_results(control_status=502)
                       if r.case == "mem.no_leak")
        self.assertEqual(control.verdict, "VOID")
        self.assertEqual(len(control.detail["invalid"]), 3)

    def test_completed_plant_recall_and_fresh_control_are_a_hit_and_no_leak(self):
        results = self.oracle_results()
        recall = next(r for r in results if r.case == "mem.recall")
        control = next(r for r in results if r.case == "mem.no_leak")
        self.assertEqual(recall.detail["hits"], 3)
        self.assertTrue(all(a["hit"] and not a["reason"] for a in recall.detail["rounds"]))
        self.assertTrue(all(a["retention_provenance"] == "extraction" for a in recall.detail["rounds"]))
        self.assertEqual(control.verdict, "PASS")

    def test_failed_plant_cannot_turn_a_matching_recall_into_a_hit_or_no_leak_pass(self):
        results = self.oracle_results(plant_rc=1)
        recall = next(r for r in results if r.case == "mem.recall")
        control = next(r for r in results if r.case == "mem.no_leak")
        self.assertEqual(recall.detail["hits"], 0)
        self.assertTrue(all("plant" in a["reason"] for a in recall.detail["rounds"]))
        self.assertEqual(control.verdict, "VOID")
        self.assertTrue(control.detail["invalid"])

    def test_failed_one_shot_extraction_cannot_prove_retention(self):
        results = self.oracle_results(extraction_status=502)
        recall = next(r for r in results if r.case == "mem.recall")
        self.assertEqual(recall.detail["hits"], 0)
        self.assertTrue(all("extraction" in a["reason"] for a in recall.detail["rounds"]))
        self.assertEqual(next(r for r in results if r.case == "mem.no_leak").verdict, "VOID")

    def test_one_shot_retention_is_proven_by_fresh_process_recall_without_extraction(self):
        results = self.oracle_results(extraction=False)
        recall = next(r for r in results if r.case == "mem.recall")
        self.assertEqual(recall.detail["hits"], 3)
        self.assertTrue(all(a["retention_provenance"] == "fresh-process-recall" for a in recall.detail["rounds"]))
        self.assertEqual(next(r for r in results if r.case == "mem.no_leak").verdict, "PASS")

    def test_fresh_control_exposing_attempted_value_is_a_leak_even_without_extraction(self):
        result = next(r for r in self.oracle_results(extraction=False, control_text="ZEBRA-a")
                      if r.case == "mem.no_leak")
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.detail["leaks"][0]["values"], ["ZEBRA-a"])

    def test_a_completed_retry_can_prove_a_main_call_after_a_failed_first_attempt(self):
        results = self.oracle_results(first_main_failed=True)
        recall = next(r for r in results if r.case == "mem.recall")
        self.assertEqual(recall.detail["hits"], 3)
        self.assertEqual(next(r for r in results if r.case == "mem.no_leak").verdict, "PASS")

    def test_blank_failed_or_unexecuted_control_voids_no_leak_and_recall(self):
        for options in ({"control_text": ""}, {"control_rc": 1},
                        {"control_main": False}):
            with self.subTest(options=options):
                results = self.oracle_results(**options)
                recall = next(r for r in results if r.case == "mem.recall")
                control = next(r for r in results if r.case == "mem.no_leak")
                self.assertEqual(recall.detail["hits"], 0)
                self.assertTrue(all("control" in a["reason"] for a in recall.detail["rounds"]))
                self.assertEqual(control.verdict, "VOID")
                self.assertEqual(len(control.detail["invalid"]), 3)

    def test_a_completed_control_with_a_planted_value_is_a_leak_not_an_invalid_control(self):
        control = next(r for r in self.oracle_results(control_text="ZEBRA-a")
                       if r.case == "mem.no_leak")
        self.assertEqual(control.verdict, "FAIL")
        self.assertEqual(control.detail["leaks"][0]["values"], ["ZEBRA-a"])
        self.assertEqual(control.detail["invalid"], [])

    def test_per_turn_body_tool_calls_feed_max_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mem_calls.jsonl"
            calls = 0

            def omp_turn(argv, **kwargs):
                nonlocal calls
                calls += 1
                now = time.time()
                body_dir = Path(tmp) / "bodies"
                body_dir.mkdir(exist_ok=True)
                body = f"mem-{calls:02d}.json"
                count = 5 if str(argv[2]).startswith("QUESTION") else 0
                (body_dir / body).write_text(json.dumps({
                    "messages": [{"role": "assistant", "tool_calls": [{} for _ in range(count)]}]
                }))
                with log.open("a") as fh:
                    fh.write(json.dumps({"t_start": now, "t": now, "purpose": "main",
                                         "body": body, "tools": 4, "prompt_tokens": 1}) + "\n")
                return subprocess.CompletedProcess([], 0, "", "")

            results = self.run_mem(omp_turn, tmp)
        turn = next(r for r in results if r.case == "mem.turn")
        self.assertEqual(turn.metrics["max_tool_calls"]["value"], 5)
        self.assertEqual(turn.metrics["timeouts"]["value"], 0)
        self.assertEqual(len(turn.detail["turns"]), 12)
        self.assertEqual(max(t["tool_calls"] for t in turn.detail["turns"]), 5)
    def test_mem_proxy_saves_request_bodies(self):
        seen = {}

        class _Proxy:
            def __init__(self, *args, **kwargs):
                seen.update(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=root, emit=lambda ev: None,
                      loaded_context=1024, mem_rounds=1)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run",
                               return_value=subprocess.CompletedProcess([], 0, "", "")), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                mem(ctx)
        self.assertEqual(seen["save_dir"], root / "bodies")

    def test_missing_main_body_voids_max_tool_call_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mem_calls.jsonl"

            def omp_turn(argv, **kwargs):
                now = time.time()
                with log.open("a") as fh:
                    fh.write(json.dumps({"t_start": now, "t": now, "purpose": "main",
                                         "body": "missing.json", "tools": 4}) + "\n")
                return subprocess.CompletedProcess([], 0, "", "")

            results = self.run_mem(omp_turn, tmp)
        turn = next(r for r in results if r.case == "mem.turn")
        metric = turn.metrics["max_tool_calls"]
        self.assertIsNone(metric["value"])
        self.assertIn("body", metric["void"])

    def test_pre_main_is_measured_to_the_first_main_call_of_the_recall_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mem_calls.jsonl"

            def omp_turn(argv, **kwargs):
                # What the proxy logs for one memory-tools turn: the classifier, a tool-free side call no marker
                # names (omp adds side calls; one it adds before the answer must not become the answer), then two
                # main calls (the model called a memory tool in between). Each turn of a fact waits differently
                # (plant 1.0 s, recall 0.2, control 3.0, derail 2.0), so pre_main taken from any turn but the recall
                # turn misses the band. The classifier starts 1 s into the turn: omp's own startup is part of the
                # wait a user sees, so pre_main counts from the launch.
                now, prompt, control = time.time(), argv[2], "control-" in str(kwargs.get("cwd"))
                first = (1.0 if prompt.startswith("PLANT") else 3.0 if control else
                         0.2 if prompt.startswith("QUESTION") else 2.0)
                rows = [{"t": now, "t_start": now + dt, "tools": tools, "purpose": p, "prompt_tokens": 2000}
                        for dt, p, tools in ((1.0, "auto-thinking", 0), (1.0 + first / 2, "aux", 0),
                                             (1.0 + first, "main", 4), (1.3 + first, "main", 4))]
                log.write_text("".join(json.dumps(r) + "\n" for r in rows))
                return subprocess.CompletedProcess([], 0, "", "")

            results = self.run_mem(omp_turn, tmp)
        pre = next(r for r in results if r.case == "mem.recall").metrics["pre_main_s"]
        self.assertEqual(pre.get("n"), 3, pre)
        self.assertTrue(1.15 < pre["value"] < 1.4, pre)

    def test_every_turn_runs_with_the_pinned_memory_tools(self):
        # A golden's mem rows are only as current as the tool list they ran with (golden.tier_keys): the pin must
        # name what the children got, or a changed list reads as CURRENT.
        argvs = []

        def omp_turn(argv, **kwargs):
            argvs.append(argv)
            return subprocess.CompletedProcess([], 0, "", "")

        with tempfile.TemporaryDirectory() as tmp:
            self.run_mem(omp_turn, tmp)
        with mock.patch.object(cli, "omp_bin", return_value="omp"), \
                mock.patch.object(cli, "_first_line", return_value="omp/18.3.1"), \
                mock.patch.object(cli, "sha16", return_value="s"):
            pin = cli.omp_pins()["omp_mem_tools"]
        self.assertEqual(len(argvs), 12)
        for argv in argvs:
            tools = [f.removeprefix("--tools=") for f in argv if f.startswith("--tools") or f == "--no-tools"]
            self.assertEqual(tools, [pin], argv)


class SessTurns(unittest.TestCase):
    def test_records_body_tool_calls_and_a_turn_timeout(self):
        proxy_args = {}

        class _Proxy:
            def __init__(self, *args, **kwargs):
                proxy_args.update(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "sess_calls.jsonl"
            bodies = root / "bodies"

            class _Rpc:
                def __init__(self, argv, cwd, stderr_path):
                    self.ready = False
                    self.turn = 0

                def send(self, command):
                    self.turn = int(command["id"].rsplit("t", 1)[1])

                def until(self, accept, timeout):
                    if not self.ready:
                        self.ready = True
                        return [json.dumps({"type": "ready"})]
                    if self.turn == 3:
                        raise TimeoutError("test turn timeout")
                    now = time.time()
                    bodies.mkdir(exist_ok=True)
                    counts = (2,) if self.turn == 1 else (3, 5)
                    for part, count in enumerate(counts):
                        body = f"sess-{self.turn:02d}-{part:02d}.json"
                        (bodies / body).write_text(json.dumps({
                            "messages": [{"role": "assistant", "tool_calls": [{} for _ in range(count)]}]
                        }))
                        with log.open("a") as fh:
                            fh.write(json.dumps({"t_start": now, "t": now, "purpose": "main",
                                                 "body": body, "tools": 4}) + "\n")
                    return [
                        json.dumps({"type": "message_end", "message": {
                            "role": "assistant", "content": [{"type": "text", "text": "NOTED"}],
                            "usage": {"input": 1, "output": 1}}}),
                        json.dumps({"type": "agent_end"}),
                    ]

                def close(self, timeout):
                    return 0.0, 0

            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=root, emit=lambda ev: None,
                      loaded_context=1024)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads._Rpc", _Rpc), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                results = sess(ctx)

        turn = next(r for r in results if r.case == "sess.turn")
        self.assertEqual(turn.metrics["max_tool_calls"]["value"], 3)
        self.assertEqual(turn.metrics["timeouts"]["value"], 1)
        self.assertEqual([t["turn"] for t in turn.detail["turns"]], [1, 2, 3])
        self.assertTrue(turn.detail["turns"][2]["timeout"])
        self.assertEqual(proxy_args["save_dir"], root / "bodies")

    def oracle_results(self, *, extraction="ok", overlap=True, stop_at=None, wrong_turn=None,
                       main_trace=True, repeats=1, missing_session=None):
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "sess_calls.jsonl"

            class _Rpc:
                def __init__(self, argv, cwd, stderr_path):
                    self.ready = False
                    self.turn = 0
                    self.session = int(cwd.name.rsplit("-", 1)[-1])

                def send(self, command):
                    self.turn = int(command["id"].rsplit("t", 1)[1])
                    self.sent = time.time()

                def until(self, accept, timeout):
                    if not self.ready:
                        self.ready = True
                        return [json.dumps({"type": "ready"})]
                    if self.turn == stop_at:
                        raise EOFError("child exited")
                    time.sleep(0.008)
                    row = {"t_start": self.sent + 0.0005, "t": self.sent + 0.004,
                           "status": 200, "aborted": False, "response_complete": True}
                    rows = [{**row, "purpose": "main"}] if main_trace else []
                    if (self.turn in (5, 9, 12) and extraction != "missing"
                            and not (self.turn == 12 and extraction == "missing_last")
                            and self.session != missing_session):
                        rows.append({**row, "purpose": "memory-extract",
                                     "t_start": self.sent + (0.001 if overlap else 0.0045),
                                     "t": self.sent + (0.0035 if overlap else 0.0055),
                                     "aborted": extraction == "aborted"})
                    with log.open("a") as fh:
                        for item in rows:
                            fh.write(json.dumps(item) + "\n")
                    return [_omp_stdout("WRONG" if self.turn == wrong_turn else "NOTED") + "\n",
                            json.dumps({"type": "agent_end"}) + "\n"]

                def close(self, timeout):
                    return 0.0, 0

            ctx = Ctx(backend=_Backend(), model="m", repeats=repeats, run_dir=root, emit=lambda ev: None,
                      loaded_context=1024)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads._Rpc", _Rpc), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                return {r.case: r for r in sess(ctx)}

    def test_all_12_acks_and_a_completed_overlapping_extraction_pass(self):
        results = self.oracle_results()
        self.assertEqual(results["sess.turns_complete"].verdict, "PASS")
        self.assertEqual(results["sess.turns_complete"].detail["turns"], 12)
        self.assertEqual(results["sess.turns_complete"].detail["expected"], 12)
        self.assertEqual([t["turn"] for t in results["sess.turn"].detail["turns"]], list(range(1, 13)))
        self.assertEqual(results["sess.memory_calls_ok"].detail["successful_extracts"], 3)
        self.assertEqual(results["sess.memory_calls_ok"].verdict, "PASS")
        self.assertGreater(results["sess.memory"].detail["overlap_with_main_s"], 0)

    def test_missing_or_aborted_extraction_cannot_pass(self):
        for extraction in ("missing", "aborted"):
            with self.subTest(extraction=extraction):
                result = self.oracle_results(extraction=extraction)["sess.memory_calls_ok"]
                self.assertNotEqual(result.verdict, "PASS")
                self.assertIn("extract", result.detail["reason"])


    def test_missing_one_of_three_retention_calls_cannot_pass(self):
        result = self.oracle_results(extraction="missing_last")["sess.memory_calls_ok"]
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("extract", result.detail["reason"])

    def test_one_good_session_cannot_mask_missing_extraction_in_another(self):
        results = self.oracle_results(repeats=2, missing_session=1)
        self.assertEqual(results["sess.turns_complete"].verdict, "PASS")
        self.assertEqual(results["sess.memory_calls_ok"].verdict, "FAIL")
        self.assertIn("session 1", results["sess.memory_calls_ok"].detail["reason"])

    def test_nonoverlapping_extraction_fails_the_concurrent_memory_verdict(self):
        result = self.oracle_results(overlap=False)["sess.memory_calls_ok"]
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("overlap", result.detail["reason"])

    def test_short_session_or_wrong_ack_cannot_pass_turns_complete(self):
        for options in ({"stop_at": 4}, {"wrong_turn": 12}):
            with self.subTest(options=options):
                result = self.oracle_results(**options)["sess.turns_complete"]
                self.assertEqual(result.verdict, "FAIL")
                self.assertTrue(result.detail["reason"])

    def test_missing_main_trace_cannot_prove_success_from_acknowledgements(self):
        results = self.oracle_results(main_trace=False)
        self.assertEqual(results["sess.turns_complete"].verdict, "VOID")
        self.assertEqual(results["sess.memory_calls_ok"].verdict, "VOID")

    def test_valid_memory_activity_in_a_short_session_is_not_a_successful_session(self):
        results = self.oracle_results(stop_at=6)
        self.assertEqual(results["sess.turns_complete"].verdict, "FAIL")
        self.assertNotEqual(results["sess.memory_calls_ok"].verdict, "PASS")


class Think(unittest.TestCase):
    """The think tier counts reasoning per round, scores only a parsed final answer, and a cut-off reply is wrong."""

    def test_the_last_answer_line_is_the_answer(self):
        self.assertEqual(final_answer("ANSWER: 1\nthinking again\n**ANSWER:** $18."), "18")
        self.assertEqual(final_answer("ANSWER: Bob"), "bob")
        self.assertEqual(final_answer("So the total is 18.\nAnswer: 18"), "18")     # models vary the case
        self.assertIsNone(final_answer("Let me count the multiples of 3 below"))

    def run_tier(self, reply, repeats=2, bodies=None):
        ctx = Ctx(backend=None, model="m", repeats=repeats, run_dir=Path("."), emit=lambda ev: None)

        def chat(_self, label, body):
            if bodies is not None:
                bodies.append(body)
            return reply(label.removeprefix("think."))
        with mock.patch.object(Ctx, "chat", chat):
            (r,) = think(ctx)
        return r

    def test_rounds_sum_reasoning_and_a_cut_off_reply_is_wrong(self):
        answers = {name: expected for name, _, expected in THINK_TASKS}

        def reply(task):
            if task == "digits":        # ran out of budget mid-thought: no ANSWER line
                return Sample("b", "m", task, completion_tokens=8192, reasoning="x" * 500, finish_reason="length")
            return Sample("b", "m", task, completion_tokens=100, reasoning="x" * 100,
                          text=f"ANSWER: {answers[task]}", finish_reason="stop", total_s=1.0)
        r = self.run_tier(reply)
        self.assertEqual(r.metrics["reasoning_chars"]["value"], 5 * 100 + 500)
        self.assertEqual(r.metrics["completion_tokens"]["value"], 5 * 100 + 8192)
        self.assertEqual((r.detail["hits"], r.detail["attempts"], r.detail["cut_off"]), (10, 12, 2))
        self.assertAlmostEqual(r.metrics["accuracy"]["value"], 10 / 12, places=4)

    def test_cut_off_counts_the_budget_not_a_missing_answer_and_scores_wrong(self):
        # arith and code reach the right ANSWER line only as the budget runs out; pens stops without one.
        answers = {name: expected for name, _, expected in THINK_TASKS}

        def reply(task):
            if task == "pens":
                return Sample("b", "m", task, completion_tokens=50, reasoning="x", text="about 18 dollars",
                              finish_reason="stop")
            finish = "length" if task in ("arith", "code") else "stop"
            return Sample("b", "m", task, completion_tokens=100, reasoning="x", text=f"ANSWER: {answers[task]}",
                          finish_reason=finish)
        r = self.run_tier(reply)
        self.assertEqual((r.detail["hits"], r.detail["attempts"], r.detail["cut_off"]), (6, 12, 4))

    def test_every_request_carries_the_token_budget(self):
        # Without it a runaway thought is never cut off, and reasoning_chars, wall and cut_off change meaning.
        bodies = []
        self.run_tier(lambda task: Sample("b", "m", task, reasoning="x", text="ANSWER: 1"), bodies=bodies)
        self.assertEqual(len(bodies), 2 * len(THINK_TASKS))
        self.assertEqual({b.get("max_tokens") for b in bodies}, {THINK_MAX_TOKENS})

    def test_a_model_that_does_not_think_voids_the_reasoning_metric(self):
        r = self.run_tier(lambda task: Sample("b", "m", task, completion_tokens=5, text="ANSWER: 0", total_s=0.1))
        self.assertEqual(r.metrics["reasoning_chars"].get("void"), "no reasoning streamed")
        self.assertEqual(r.detail["hits"], 0)


class ClientIdentity(unittest.TestCase):
    HARNESS = "~/Developer/localbench/runs/omp-agent"
    USER = "~/.omp/agent"

    def test_a_harness_child_is_not_the_default_profile(self):
        ident = omp_client_identity("bun ~/.bun/bin/omp --mode rpc",
                                    f"PI_CODING_AGENT_DIR={self.HARNESS}")
        self.assertEqual(ident["agent_dir"], self.HARNESS)
        self.assertNotIn("omp_profile", ident)

    def test_the_user_dir_is_not_the_harness_dir(self):
        ident = omp_client_identity("omp", f"PI_CODING_AGENT_DIR={self.USER}")
        self.assertEqual(ident["agent_dir"], self.USER)
        self.assertNotEqual(ident["agent_dir"], self.HARNESS)
        self.assertNotIn("omp_profile", ident)

    def test_an_explicit_profile_is_kept_beside_the_agent_dir(self):
        ident = omp_client_identity("omp --profile=claude", f"PI_CODING_AGENT_DIR={self.HARNESS}")
        self.assertEqual(ident["omp_profile"], "claude")
        self.assertEqual(ident["agent_dir"], self.HARNESS)

    def test_no_env_still_defaults(self):
        self.assertEqual(omp_client_identity("omp", ""), {"omp_profile": "default"})


class RecordedAnswers(unittest.TestCase):
    def test_a_pass_and_a_fail_both_keep_the_answer(self):
        short, long = "OK", "not ok " + ("x" * 400)
        got = recorded_answers([short, long, ""])
        self.assertEqual(got[0], "OK")
        self.assertEqual(got[2], "")
        self.assertTrue(got[1].endswith(f"…[+{len(long) - CLIP} chars]"), got[1][-40:])
        self.assertNotEqual(got[1], long[:80])


def _omp_stdout(text: str) -> str:
    return json.dumps({"type": "message_end", "message": {
        "role": "assistant", "content": [{"type": "text", "text": text}],
        "usage": {"input": 3, "output": 2}}})


class AnswerReachesTheVerdict(unittest.TestCase):

    def test_wrong_expected_answer_rejects_correct_e2e_output(self):
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

            def isolate(self, model, warm=True):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=8192, e2e_case="ok")
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run",
                               return_value=subprocess.CompletedProcess([], 0, _omp_stdout("OK"), "")):
                rows = e2e(ctx)

        correct = next(row for row in rows if row.case == "e2e.ok.correct")
        self.assertEqual(correct.verdict, "PASS")
        self.assertEqual(correct.detail["answers"], ["OK", "OK"])


    def test_campaign_case_runs_only_that_task_and_retains_each_exit_code(self):
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

            def isolate(self, model, warm=True):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=8192, e2e_case="ok")
            attempts = iter((0, 7))

            def run(args, **kwargs):
                return subprocess.CompletedProcess(args, next(attempts), _omp_stdout("OK"), "")

            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=run):
                rows = e2e(ctx)
            self.assertEqual([row.case for row in rows], ["e2e.ok", "e2e.ok.correct"])
            self.assertEqual(rows[1].verdict, "FAIL")
            self.assertEqual([attempt["returncode"] for attempt in rows[1].detail["attempts"]], [0, 7])
            repeat = json.loads((Path(tmp) / "e2e.ok.repeat.result.json").read_text())
            self.assertEqual(repeat["returncode"], 7)


    def test_pre_warmed_first_call_voids_first_timing_metrics(self):
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

            def isolate(self, model, warm=True):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=8192, e2e_case="ok")

            def run(args, **kwargs):
                now = time.time()
                with (Path(tmp) / "omp_calls.jsonl").open("a") as fh:
                    fh.write(json.dumps({"t": now, "t_start": now, "tools": 1, "cached_tokens": 12,
                                         "purpose": "main"}) + "\n")
                return subprocess.CompletedProcess(args, 0, _omp_stdout("OK"), "")

            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=run):
                rows = e2e(ctx)

        timing = next(row for row in rows if row.case == "e2e.ok")
        correct = next(row for row in rows if row.case == "e2e.ok.correct")
        self.assertEqual(timing.detail["first"]["first_call_cached_tokens"], 12)
        for name in ("first_wall_s", "first_llm_s"):
            with self.subTest(metric=name):
                self.assertIsNone(timing.metrics[name]["value"])
                self.assertIn("cached tokens", timing.metrics[name]["void"])
        self.assertEqual(correct.verdict, "PASS")

    FAIL = "WRONG " + ("x" * 400)

    def test_a_pass_and_a_fail_reach_e2e_and_rel_details(self):
        """Restored after 8b2ad07 dropped it: no other test reaches the rel tier, so a rel that kept only passing
        answers, or lost a failed attempt from `wrong`, would bank a receipt that hides the wrong answers it measured."""
        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

            def isolate(self, model, warm=True):
                return []

        read_events = (json.dumps({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read",
                                   "args": {"path": "/tmp/localbench-e2e/answer.txt"}}) + "\n"
                       + json.dumps({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "read",
                                     "isError": False,
                                     "result": {"details": {"displayContent": {"text": "4817"}}}}) + "\n")
        ok_attempt = {"n": 0}       # reset per tier: the first "ok" attempt of each tier is the FAIL

        def run(args, **kwargs):
            if "exactly: OK" in args[2]:
                text = self.FAIL if ok_attempt["n"] % 2 == 0 else "OK"
                ok_attempt["n"] += 1
                return subprocess.CompletedProcess(args, 0, _omp_stdout(text), "")
            return subprocess.CompletedProcess(args, 0, read_events + _omp_stdout("4817"), "")

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=8192)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=run):
                e2e_rows = e2e(ctx)
                ok_attempt["n"] = 0
                rel_rows = rel(ctx)
            first_fail_trace = (Path(tmp) / "rel.ok.0.omp.jsonl").read_text()

        clipped = recorded_answers([self.FAIL])[0]
        self.assertTrue(clipped.startswith("WRONG ") and clipped.endswith(f"…[+{len(self.FAIL) - CLIP} chars]"))

        correct = next(r for r in e2e_rows if r.case == "e2e.ok.correct")
        self.assertEqual(correct.verdict, "FAIL")
        self.assertEqual(correct.detail["answers"], [clipped, "OK"])
        read = next(r for r in e2e_rows if r.case == "e2e.tool_read.correct")
        self.assertEqual(read.verdict, "PASS")
        self.assertEqual(read.detail["answers"], ["4817", "4817"])

        self.assertEqual([r.case for r in rel_rows], ["rel.tool_read", "rel.ok"])
        rel_ok = rel_rows[1]
        expected = [clipped if i % 2 == 0 else "OK" for i in range(REL_ATTEMPTS)]
        self.assertEqual(rel_ok.detail["answers"], expected)
        self.assertEqual(rel_ok.detail["passed"], REL_ATTEMPTS // 2)
        self.assertEqual(rel_ok.metrics["pass_rate"]["value"], 0.5)
        self.assertEqual(rel_ok.detail["wrong"],
                         [{"attempt": i, "rc": 0, "answer": clipped, "bodies": []} for i in range(0, REL_ATTEMPTS, 2)])
        self.assertEqual(first_fail_trace, _omp_stdout(self.FAIL))
        rel_read = rel_rows[0]
        self.assertEqual(rel_read.detail["answers"], ["4817"] * REL_ATTEMPTS)
        self.assertEqual((rel_read.detail["passed"], rel_read.detail["wrong"]), (REL_ATTEMPTS, []))


class PromptTokens(unittest.TestCase):
    """kit-l7l: lean.meta.json's 11433 was counted by ollama. mlx-serve must not FAIL that comparison."""

    def test_the_recording_backend_fails_a_drift_past_two_percent(self):
        v = prompt_token_verdict("lean", 11724, 11433, "ollama", "ollama")
        self.assertEqual(v.verdict, "FAIL")
        self.assertGreater(v.detail["drift"], 0.02)

    def test_the_recording_backend_passes_inside_the_band(self):
        v = prompt_token_verdict("lean", 11433, 11433, "ollama", "ollama")
        self.assertEqual(v.verdict, "PASS")

    def test_a_different_backend_is_void_not_a_tokenizer_fail(self):
        v = prompt_token_verdict("lean", 11724, 11433, "ollama", "mlx-serve")
        self.assertEqual(v.verdict, "VOID")
        self.assertIn("recorded by ollama", v.detail["reason"])

    def test_a_sidecar_without_a_backend_still_compares(self):
        v = prompt_token_verdict("lean", 11724, 11433, None, "mlx-serve")
        self.assertEqual(v.verdict, "FAIL")

    def test_drift_at_exactly_two_percent_passes(self):
        v = prompt_token_verdict("lean", 10200, 10000, "ollama", "ollama")
        self.assertEqual(v.detail["drift"], 0.02)
        self.assertEqual(v.verdict, "PASS")


    def test_same_backend_with_a_different_reported_tokenizer_is_void(self):
        verdict = prompt_token_verdict("lean", 10853, 11433, "ollama", "ollama",
                                       recorded_tok="ollama-ggml-sha256:recorded",
                                       run_tok="ollama-ggml-sha256:other")
        self.assertEqual(verdict.verdict, "VOID")
        self.assertEqual(verdict.detail["reason"], "different tokenizer identity")

    def test_same_tokenizer_on_a_different_model_still_judges_drift(self):
        # The model name is intentionally absent from this boundary: same tokenizer, 2.55% drift, so it must FAIL.
        verdict = prompt_token_verdict("lean", 11724, 11433, "ollama", "ollama",
                                       recorded_tok="ollama-ggml-sha256:qwen",
                                       run_tok="ollama-ggml-sha256:qwen")
        self.assertEqual(verdict.verdict, "FAIL")

    def test_missing_tokenizer_identity_falls_back_to_the_backend_rule(self):
        verdict = prompt_token_verdict("lean", 11724, 11433, "ollama", "ollama",
                                       recorded_tok=None, run_tok=None)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertGreater(verdict.detail["drift"], 0.02)




class ReplayTokenWiring(unittest.TestCase):
    """replay() must pass the sidecar's backend into the verdict. Testing the helper alone missed that."""

    def case(self, backend: str, tokens: int, recorded_tok: str | None = None,
             run_tok: str | None = None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            omp = root / "omp"
            omp.mkdir()
            (omp / "lean.json").write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
            metadata = {"prompt_tokens": 10000, "backend_pins": {"backend": "ollama"}}
            if recorded_tok is not None:
                metadata["backend_pins"]["tokenizer_identity"] = recorded_tok
            (omp / "lean.meta.json").write_text(json.dumps(metadata))
            sample = Sample(backend=backend, model="m", label="x", prompt_tokens=tokens, ttft_s=0.1, text="OK")

            class Backend:
                name = backend

            ctx = Ctx(backend=Backend(), model="m", repeats=1, run_dir=root, emit=lambda ev: None,
                      loaded_context=200000)
            ctx.tok_identity = run_tok
            with mock.patch.object(ctx, "chat", return_value=sample), \
                    mock.patch("localbench.workloads.FIXTURES", root):
                rows = replay(ctx)
        return next(r for r in rows if r.case == "replay.lean.prompt_tokens")

    def test_mlx_serve_against_an_ollama_sidecar_is_void(self):
        self.assertEqual(self.case("mlx-serve", 11724).verdict, "VOID")

    def test_ollama_against_its_own_sidecar_fails_a_drift(self):
        self.assertEqual(self.case("ollama", 11724).verdict, "FAIL")

    def test_ollama_against_its_own_sidecar_passes_a_match(self):
        self.assertEqual(self.case("ollama", 10000).verdict, "PASS")

    def test_same_backend_tokenizer_change_is_void_through_replay(self):
        self.assertEqual(self.case("ollama", 10853, "ollama-ggml-sha256:qwen36",
                                   "ollama-ggml-sha256:nemotron").verdict, "VOID")


class AnswerCall(unittest.TestCase):
    """mem children declare the memory tools, so the answer is the first `main` call. The classifier's retry carries a
    forced tool (omp 18.3.1) and a tool-free call is side work: neither is the answer."""

    def test_the_answer_is_the_first_main_call_not_the_classifier_or_its_retry(self):
        rows = (
            {"t_start": 14.0, "tools": 4, "purpose": "main"},
            {"t_start": 10.0, "tools": 0, "purpose": "auto-thinking"},
            {"t_start": 11.0, "tools": 1, "purpose": "auto-thinking"},     # submit_judgment retry
            {"t_start": 12.5, "tools": 4, "purpose": "main", "prompt_tokens": 2079},
        )
        self.assertEqual(answer_call(rows)["t_start"], 12.5)

    def test_a_tool_free_call_is_never_the_answer(self):
        rows = [{"t_start": 1.0, "tools": 0, "purpose": "auto-thinking"},
                {"t_start": 2.0, "tools": 0, "purpose": "aux"}]
        self.assertIsNone(answer_call(rows))


class RpcCancellation(unittest.TestCase):
    def test_sess_closes_child_and_propagates_keyboard_interrupt(self):
        child = ("import json, sys; print(json.dumps({'type': 'ready'}), flush=True); "
                 "sys.stdin.readline(); sys.stdin.read()")
        instances = []

        class _Proxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _Backend:
            base_url = "http://127.0.0.1:9"

        def make_rpc(argv, cwd, stderr_path):
            rpc = _Rpc(argv, cwd, stderr_path)
            instances.append(rpc)
            return rpc

        until = _Rpc.until

        def interrupt_after_ready(rpc, accept, timeout):
            if getattr(rpc, "_ready_consumed", False):
                raise KeyboardInterrupt
            rpc._ready_consumed = True
            return until(rpc, accept, timeout)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=root, emit=lambda ev: None,
                      loaded_context=1024)
            try:
                with mock.patch("localbench.proxy.Proxy", _Proxy), \
                        mock.patch("localbench.workloads._Rpc", side_effect=make_rpc), \
                        mock.patch.object(_Rpc, "until", interrupt_after_ready), \
                        mock.patch("localbench.workloads.child_flags", return_value=["-c", child]), \
                        mock.patch("localbench.workloads.omp_bin", return_value=sys.executable), \
                        mock.patch("localbench.workloads.ensure_localbench_model"), \
                        mock.patch("localbench.memory.banks", return_value=[]), \
                        mock.patch("localbench.memory.remove_banks"):
                    with self.assertRaises(KeyboardInterrupt):
                        sess(ctx)
                self.assertEqual(len(instances), 1)
                self.assertIsNotNone(instances[0].proc.poll())
                self.assertTrue(instances[0]._stderr.closed)
            finally:
                for rpc in instances:
                    if rpc.proc.poll() is None:
                        rpc.proc.kill()
                        rpc.proc.wait()
                    if not rpc._stderr.closed:
                        rpc._stderr.close()

    def test_constructor_cancellation_kills_child_and_closes_pipes(self):
        children = []
        streams = []
        real_popen = subprocess.Popen

        def capture_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            children.append(proc)
            streams.extend((proc.stdin, proc.stdout, kwargs["stderr"]))
            return proc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                with mock.patch("localbench.workloads.subprocess.Popen", side_effect=capture_popen), \
                        mock.patch("localbench.workloads.threading.Thread.start", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        _Rpc([sys.executable, "-c", "import time; time.sleep(30)"], root,
                             root / "rpc.stderr.log")
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertTrue(all(stream.closed for stream in streams))
            finally:
                for proc in children:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                for stream in streams:
                    if not stream.closed:
                        stream.close()

    def test_close_cancellation_kills_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rpc = _Rpc([sys.executable, "-c", "import time; time.sleep(30)"], root,
                       root / "rpc.stderr.log")
            original_wait = rpc.proc.wait
            waits = 0

            def interrupt_once(timeout=None):
                nonlocal waits
                waits += 1
                if waits == 1:
                    raise KeyboardInterrupt
                return original_wait(timeout=timeout)

            try:
                with mock.patch.object(rpc.proc, "wait", side_effect=interrupt_once):
                    with self.assertRaises(KeyboardInterrupt):
                        rpc.close(120)
                self.assertIsNotNone(rpc.proc.poll())
                self.assertTrue(rpc._stderr.closed)
            finally:
                if rpc.proc.poll() is None:
                    rpc.proc.kill()
                    rpc.proc.wait()
                if not rpc._stderr.closed:
                    rpc._stderr.close()


if __name__ == "__main__":
    unittest.main()
