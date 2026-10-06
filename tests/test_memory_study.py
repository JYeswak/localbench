"""Memory study (kit-memory-study-vce): a separate memory (smol) model on the same Ollama, the llmMode none overlay and
its sess scoring, and the memory proof receipt (workloads.memory_verdict) graded by features.grade."""

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import features, presets, render
from localbench.backends import sha16
from localbench.workloads import (
    FIXTURES,
    MEM_CONFIG,
    Ctx,
    child_flags,
    embedding_pins,
    fastembed_digest,
    mem,
    mem_llm_mode,
    mem_overlay,
    memory_verdict,
    sess,
    smol_pins,
)

NONE_CONFIG = FIXTURES / "omp" / "child-config-mem-nollm.yml"
FTS_CONFIG = FIXTURES / "omp" / "child-config-mem-fts.yml"
QWEN = "qwen3.8:27b-mlx"
QWEN_DIGEST = "5642e97495e1"  # the incumbent's digest; MEMORY_BASELINE matches by it
# The incumbent's park alias as the r2 baseline legs pinned it (runs/20261001T094732Z__ab_a1__..., smol_model).
PARKED = "localbench-parked:5642e97495e1"
CANDIDATE = "qwen3.6:35b"
CANDIDATE_DIGEST = "bbbbbbbbbbbb"
EMBEDDER, EMBEDDER_DIGEST = "local/fast-bge-base-en-v1.5", "eeeeeeeeeeee"


class _Proxy:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Backend:
    name = "ollama"
    base_url = "http://127.0.0.1:9"


def _omp_stdout(text: str) -> str:
    return json.dumps({"type": "message_end", "message": {
        "role": "assistant", "content": [{"type": "text", "text": text}], "usage": {"input": 3, "output": 2}}})


def run_sess(*, smol=None, pins=None, mem_config=MEM_CONFIG, extract_model=None, extraction=True):
    """One 12-turn fake session: every turn acks NOTED with a completed main call; after turns 4, 8, 12 the next
    turn's window carries a completed memory-extract call that overlaps it (served by `extract_model`)."""
    seen = {"argv": None, "extra": None}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "sess_calls.jsonl"

        class _Rpc:
            def __init__(self, argv, cwd, stderr_path):
                seen["argv"] = argv
                self.ready = False
                self.turn = 0

            def send(self, command):
                self.turn = int(command["id"].rsplit("t", 1)[1])
                self.sent = time.time()

            def until(self, accept, timeout):
                if not self.ready:
                    self.ready = True
                    return [json.dumps({"type": "ready"})]
                time.sleep(0.008)
                row = {"t_start": self.sent + 0.0005, "t": self.sent + 0.004, "status": 200, "aborted": False,
                       "response_complete": True}
                rows = [{**row, "purpose": "main", "model": "m"}]
                if extraction and self.turn in (5, 9, 12):
                    rows.append({**row, "purpose": "memory-extract", "model": extract_model,
                                 "t_start": self.sent + 0.001, "t": self.sent + 0.0035})
                with log.open("a") as fh:
                    for item in rows:
                        fh.write(json.dumps(item) + "\n")
                return [_omp_stdout("NOTED") + "\n", json.dumps({"type": "agent_end"}) + "\n"]

            def close(self, timeout):
                return 0.0, 0

        def ensure(model_id, context_window, extra=()):
            seen["extra"] = extra

        ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=root, emit=lambda ev: None, loaded_context=1024,
                  smol_model=smol, pins=pins or {}, mem_config=mem_config)
        with mock.patch("localbench.proxy.Proxy", _Proxy), \
                mock.patch("localbench.workloads._Rpc", _Rpc), \
                mock.patch("localbench.workloads.ensure_localbench_model", side_effect=ensure), \
                mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                mock.patch("localbench.memory.banks", return_value=[]), \
                mock.patch("localbench.memory.remove_banks"):
            return {r.case: r for r in sess(ctx)}, seen


def _flag(argv, name):
    return argv[argv.index(name) + 1]


class SeparateMemoryModel(unittest.TestCase):
    def test_smol_model_flows_into_child_flags_and_models_yml(self):
        flags = child_flags("m", MEM_CONFIG, tools="memory", smol=CANDIDATE)
        self.assertEqual(_flag(flags, "--model"), "localbench/m")
        self.assertEqual(_flag(flags, "--smol"), f"localbench/{CANDIDATE}")
        results, seen = run_sess(smol=CANDIDATE, pins={"smol_model": CANDIDATE, "smol_digest": CANDIDATE_DIGEST},
                                 extract_model=CANDIDATE)
        self.assertEqual(_flag(seen["argv"], "--smol"), f"localbench/{CANDIDATE}")
        self.assertEqual(_flag(seen["argv"], "--model"), "localbench/m")
        self.assertEqual(seen["extra"], (CANDIDATE,))
        self.assertEqual(results["sess.memory_calls_ok"].verdict, "PASS")
        self.assertEqual(results["sess.memory"].detail["by_purpose"]["memory-extract"]["models"], [CANDIDATE])

    def test_default_stays_one_artifact(self):
        self.assertEqual(_flag(child_flags("m"), "--smol"), "localbench/m")
        results, seen = run_sess(extract_model="m")
        self.assertEqual(_flag(seen["argv"], "--smol"), "localbench/m")
        self.assertEqual(seen["extra"], ())
        self.assertEqual(results["sess.memory_calls_ok"].verdict, "PASS")
        self.assertEqual(smol_pins(_Backend(), None), {})

    def test_extraction_served_by_the_main_model_is_not_credited_to_the_declared_smol(self):
        results, _ = run_sess(smol=CANDIDATE, pins={"smol_model": CANDIDATE, "smol_digest": CANDIDATE_DIGEST},
                              extract_model="m")
        calls = results["sess.memory_calls_ok"]
        self.assertEqual(calls.verdict, "FAIL")
        self.assertIn("not the declared smol model", calls.detail["reason"])

    def test_unpinned_smol_model_refuses_before_any_child_runs(self):
        results, seen = run_sess(smol=CANDIDATE, pins={})
        self.assertEqual(list(results), ["sess"])
        self.assertEqual(results["sess"].verdict, "FAIL")
        self.assertIn("smol_digest", results["sess"].detail["reason"])
        self.assertIsNone(seen["argv"])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("localbench.workloads.subprocess.run") as run:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=1024, smol_model=CANDIDATE)
            (refusal,) = mem(ctx)
        self.assertEqual(refusal.verdict, "FAIL")
        run.assert_not_called()

    def test_mem_children_get_the_smol_model(self):
        argvs, extras = [], []
        with tempfile.TemporaryDirectory() as tmp:
            def omp_turn(argv, **kwargs):
                argvs.append(argv)
                return subprocess.CompletedProcess(argv, 0, _omp_stdout("NOTED"), "")

            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=1024, mem_rounds=1, smol_model=CANDIDATE,
                      pins={"smol_model": CANDIDATE, "smol_digest": CANDIDATE_DIGEST})
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model",
                               side_effect=lambda m, c, extra=(): extras.append(extra)), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=omp_turn), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                results = mem(ctx)
        self.assertEqual(len(argvs), 12)
        self.assertTrue(all(_flag(a, "--smol") == f"localbench/{CANDIDATE}" for a in argvs))
        self.assertEqual(extras, [(CANDIDATE,)])
        recall = next(r for r in results if r.case == "mem.recall")
        self.assertEqual((recall.detail["smol_model"], recall.detail["llm_mode"]), (CANDIDATE, "smol"))

    def test_smol_pins_record_the_smol_digest_and_refuse_a_one_model_server(self):
        class _Ollama(_Backend):
            def pins(self, model):
                return {"model": model, "model_digest": {CANDIDATE: CANDIDATE_DIGEST}.get(model)}

        self.assertEqual(smol_pins(_Ollama(), CANDIDATE), {"smol_model": CANDIDATE, "smol_digest": CANDIDATE_DIGEST})

        class _MlxServe(_Ollama):
            name = "mlx-serve"

        with self.assertRaises(ValueError):
            smol_pins(_MlxServe(), CANDIDATE)


# r3 leg a3 (runs/20261001T190835Z__ab_a3__ollama__qwen3.6_35b-mlx, session 0 turn 12), seconds after t_send: the main
# call started at +2.457, omp's agent_end (t_end) came at +3.4988, the proxy logged the completed row at +3.5379,
# 39.1 ms after t_end. r3 a2 session 1 turn 9: the row came +9.9 ms after t_end, the next send +0.6 ms after it.
A3_START, A3_END, A3_ROW = 2.457, 3.4988, 3.5379
A2_LATE = 0.0099
SEND_GAP = 0.0006   # the fake clock's step: each time.time() call is 0.6 ms after the previous one, as sess sends


class SessMainCallAttribution(unittest.TestCase):
    """A sess turn's main call is the one that started in [t_send, t_end]; its completion row may land after t_end and
    after the next send (the proxy's bookkeeping trails omp), but before the next turn's first main call starts (the
    session end for the last turn)."""

    def run_sess(self, late: dict[int, float]) -> dict:
        """One 12-turn session on a fake clock (workloads.time patched; perf_counter real). Every turn has a main call
        at the a3 offsets; `late` maps a turn to how far after t_end its completion row is logged (default: before)."""
        clock = {"now": 1790882232.0, "last": None}

        def now():
            clock["last"] = clock["now"]
            clock["now"] += SEND_GAP
            return clock["last"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "sess_calls.jsonl"

            class _Rpc:
                def __init__(self, argv, cwd, stderr_path):
                    self.ready, self.turn, self.sent = False, 0, None

                def send(self, command):
                    self.turn = int(command["id"].rsplit("t", 1)[1])
                    self.sent = clock["last"]   # the t_send sess just read

                def until(self, accept, timeout):
                    if not self.ready:
                        self.ready = True
                        return [json.dumps({"type": "ready"})]
                    t_end = self.sent + A3_END
                    row = {"purpose": "main", "model": "m", "status": 200, "aborted": False,
                           "response_complete": True, "t_start": round(self.sent + A3_START, 3),
                           "t": t_end + late.get(self.turn, -0.001)}
                    with log.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    clock["now"] = t_end          # the next time.time() is this turn's t_end
                    return [_omp_stdout("NOTED") + "\n", json.dumps({"type": "agent_end"}) + "\n"]

                def close(self, timeout):
                    clock["now"] += 3.839   # r3 a3 session 0's exit_s: the session ends after omp exits
                    return 3.839, 0

            fake_time = mock.Mock(time=now, perf_counter=time.perf_counter)
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=root, emit=lambda ev: None,
                      loaded_context=1024, mem_config=NONE_CONFIG)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads._Rpc", _Rpc), \
                    mock.patch("localbench.workloads.time", fake_time), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                return {r.case: r for r in sess(ctx)}

    def test_a_row_logged_after_agent_end_and_the_next_send_but_before_the_next_main_call_counts(self):
        for turn, late in ((9, A2_LATE), (12, A3_ROW - A3_END)):   # mid-session (a2) and the last turn (a3)
            with self.subTest(turn=turn):
                self.assertGreater(late, SEND_GAP)                  # after the next turn's send
                results = self.run_sess({turn: late})
                done = results["sess.turns_complete"]
                self.assertEqual((done.verdict, done.detail["reason"]), ("PASS", None))
                turns = {t["turn"]: t for t in results["sess.turn"].detail["turns"]}
                self.assertTrue(turns[turn]["main_call_ok"])

    def test_a_row_completing_after_the_next_turns_first_main_call_started_is_not_the_turns_main_call(self):
        late = SEND_GAP + A3_START + 0.01   # 10 ms after turn 6's main call started
        results = self.run_sess({5: late})
        turns = {t["turn"]: t for t in results["sess.turn"].detail["turns"]}
        self.assertFalse(turns[5]["main_call_ok"])
        self.assertTrue(turns[6]["main_call_ok"])
        done = results["sess.turns_complete"]
        self.assertEqual((done.verdict, done.detail["reason"]), ("VOID", "main call trace missing or incomplete"))


class NoneOverlay(unittest.TestCase):
    def test_none_overlay_differs_from_the_mem_overlay_only_in_llm_mode(self):
        self.assertEqual(mem_llm_mode(NONE_CONFIG), "none")
        self.assertEqual(mem_llm_mode(MEM_CONFIG), "smol")
        none, base = mem_overlay(NONE_CONFIG), mem_overlay(MEM_CONFIG)
        self.assertEqual(none["mnemopi"].pop("llmMode"), "none")
        self.assertEqual(base["mnemopi"].pop("llmMode"), "smol")
        self.assertEqual(none, base)

    def test_an_overlay_the_reader_cannot_parse_is_refused_not_defaulted(self):
        with tempfile.TemporaryDirectory() as tmp:
            for body in ("mnemopi:\n  llmMode: [none]\n", "mnemopi:\n  llmMode:\n    - none\n",
                         "mnemopi:\n  llmMode: off\n", "mnemopi:\n  llmMode: none\n  llmMode: smol\n"):
                with self.subTest(body=body):
                    path = Path(tmp) / "overlay.yml"
                    path.write_text(body)
                    with self.assertRaises(ValueError):
                        mem_llm_mode(path)

    def test_sess_under_none_expects_zero_memory_calls_and_says_so(self):
        results, _ = run_sess(mem_config=NONE_CONFIG, extraction=False)
        self.assertNotIn("sess.memory_calls_ok", results)
        calls = results["sess.memory_calls_none"]
        self.assertEqual(calls.verdict, "PASS")
        detail = calls.detail
        self.assertEqual((detail["llm_mode"], detail["memory_calls"], detail["expected_memory_calls"]), ("none", 0, 0))
        self.assertEqual(results["sess.turns_complete"].verdict, "PASS")

    def test_a_memory_llm_call_under_none_fails(self):
        results, _ = run_sess(mem_config=NONE_CONFIG, extract_model="m")
        calls = results["sess.memory_calls_none"]
        self.assertEqual(calls.verdict, "FAIL")
        self.assertIn("llmMode none", calls.detail["reason"])

    def test_zero_calls_under_the_extraction_overlay_still_cannot_pass(self):
        results, _ = run_sess(extraction=False)
        self.assertNotEqual(results["sess.memory_calls_ok"].verdict, "PASS")

    def run_mem_none(self, *, extraction: bool) -> dict:
        """mem under the none overlay: every child answers correctly with a completed main call; the plant also
        shows a memory-extract call when `extraction`."""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mem_calls.jsonl"

            def omp_turn(argv, **kwargs):
                prompt = argv[2]
                kind = ("plant" if prompt.startswith("PLANT") else
                        "control" if "control-" in str(kwargs["cwd"]) else
                        "recall" if prompt.startswith("QUESTION") else "derail")
                now = time.time()
                row = {"t_start": now, "t": now, "status": 200, "aborted": False, "response_complete": True}
                rows = [{**row, "purpose": "main"}]
                if kind == "plant" and extraction:
                    rows.append({**row, "purpose": "memory-extract"})
                with log.open("a") as fh:
                    for item in rows:
                        fh.write(json.dumps(item) + "\n")
                text = {"plant": "NOTED", "recall": f"ZEBRA-{prompt.rsplit(' ', 1)[-1]}", "control": "unknown",
                        "derail": "OK"}[kind]
                return subprocess.CompletedProcess(argv, 0, _omp_stdout(text), "")

            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=1024, mem_rounds=1, mem_config=NONE_CONFIG)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=omp_turn), \
                    mock.patch("localbench.workloads._mem_facts",
                               return_value=[(n, f"PLANT {n}", f"QUESTION {n}", f"ZEBRA-{n}") for n in "abc"]), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                return {r.case: r for r in mem(ctx)}

    def test_mem_under_none_recalls_by_fresh_process_and_rejects_an_extraction_call(self):
        recall = self.run_mem_none(extraction=False)["mem.recall"]
        self.assertEqual((recall.detail["hits"], recall.detail["llm_mode"]), (3, "none"))
        self.assertTrue(all(a["retention_provenance"] == "fresh-process-recall" for a in recall.detail["rounds"]))
        results = self.run_mem_none(extraction=True)
        self.assertEqual(results["mem.recall"].detail["hits"], 0)
        self.assertTrue(all("llmMode none" in a["reason"] for a in results["mem.recall"].detail["rounds"]))
        self.assertEqual(results["mem.no_leak"].verdict, "VOID")


def leg(label, *, smol=CANDIDATE, digest=CANDIDATE_DIGEST, mode="smol", hit=1.0, hit_n=9, derail=1.0,
        no_leak="PASS", calls="PASS", turns="PASS", pre_main=2.0, post=8.4, extract=5.0, gpu=60.0,
        contended=False, resident_unknown=0, by_purpose="served-by-pin", mem_tool_calls=9, sess_tool_calls=1,
        mem_timeouts=0, sess_timeouts=0, ack_slips=0, turns_detail=None, created="20261001T150000Z", config=None,
        embedder_digest=EMBEDDER_DIGEST, regime=None, contention=()):
    """An `execute` summary of a mem+sess leg, reduced to what memory_verdict reads. By default its memory calls
    were served by its pinned memory model (none under llmMode none). `ack_slips` > 0 is sess's own result when the
    main model missed that many NOTED replies and nothing else failed (ab_b1/ab_b2 of
    receipts/ab-memory-nollm-vs-qwen38-20261001.json): turns_complete FAIL "incorrect acknowledgement", failures [],
    turns == expected, and the memory-calls case FAIL with the same reason."""
    calls_case = "sess.memory_calls_none" if mode == "none" else "sess.memory_calls_ok"
    # The overlay the leg's children ran (the run pins hash it as omp_mem_config); mem/sess details name it.
    config = str(config or (NONE_CONFIG if mode == "none" else MEM_CONFIG))
    # embedding_pins' shape: the FTS-only overlay pins no embedder.
    embedder = ({"embedder": None, "embedder_digest": None} if config == str(FTS_CONFIG)
                else {"embedder": EMBEDDER, "embedder_digest": embedder_digest})
    turns_complete = {"reason": None, "failures": [], "turns": 24, "acks": 24, "expected": 24}
    calls_reason = None
    if ack_slips:
        turns, calls, calls_reason = "FAIL", "FAIL", "incorrect acknowledgement"
        turns_complete = {**turns_complete, "reason": "incorrect acknowledgement", "acks": 24 - ack_slips}
    elif turns != "PASS":
        turns_complete = {"reason": "incomplete turns or failed session", "failures": ["session 1 turn 12: EOFError"],
                          "turns": 23, "acks": 23, "expected": 24}
    turns_complete = {**turns_complete, **(turns_detail or {})}
    if by_purpose == "served-by-pin":
        by_purpose = {} if mode == "none" else {"memory-extract": {"calls": 3, "models": [smol or "main:1"]}}
    during = {"gpu_device_pct": {"mean": gpu}}
    if resident_unknown is not None:
        during["resident_unknown_samples"] = resident_unknown
    loop_values = (("mem", "max_tool_calls", mem_tool_calls), ("sess", "max_tool_calls", sess_tool_calls),
                   ("mem", "timeouts", mem_timeouts), ("sess", "timeouts", sess_timeouts))
    loop = {f"{tier}.turn.{name}": {"value": value, "better": "lower", "n": 12}
            for tier, name, value in loop_values if value is not None}
    return {
        # The pins shape of runs/20261001T094732Z__ab_a1__ollama__qwen3.6_35b-mlx/summary.json (values made up); created
        # defaults to after workloads.ACK_SLIP_RULE_FROM, i.e. a leg of the next run.
        "provenance": {"label": label, "created": created, "localbench_rev": "abc1234",
                       "pins": {"backend": "ollama", "backend_version": "0.35.0", "backend_sha": "1111111111111111",
                                "backend_args": "", "model": "main:1", "model_digest": "cccccccccccc",
                                "macos_build": "25F71", "host_id": "host0", "smol_model": smol, "smol_digest": digest,
                                "omp_mem_config": sha16(config), **embedder,
                                **({"regime": regime} if regime else {})}},
        "run_dir": f"runs/{label}",
        "verdicts": {"contended": contended, "must_fail": [], "pins_changed": {}},
        "metrics": {"mem.recall.hit_rate": {"value": hit, "better": "higher", "n": hit_n},
                    "mem.derail.ok_rate": {"value": derail, "better": "higher", "n": 9},
                    "mem.recall.pre_main_s": {"value": pre_main, "better": "lower", "n": 9},
                    "sess.turn.post_retain_wall_s": {"value": post, "better": "lower", "n": 2},
                    "sess.memory.extract_s": ({"value": None, "better": "lower", "void": "no valid samples"}
                                              if mode == "none" else {"value": extract, "better": "lower", "n": 3}),
                    **loop},
        "conformance": {"mem.no_leak": {"level": "MUST", "verdict": no_leak},
                        "sess.turns_complete": {"level": "MUST", "verdict": turns},
                        calls_case: {"level": "SHOULD", "verdict": calls}},
        "results": [{"case": "mem.recall", "tier": "mem", "level": "perf", "verdict": None,
                     "detail": {"llm_mode": mode, "config": config}},
                    {"case": "sess.turn", "tier": "sess", "level": "perf", "verdict": None,
                     "detail": {"llm_mode": mode, "config": config}},
                    {"case": "sess.memory", "tier": "sess", "level": "perf", "verdict": None,
                     "detail": {"by_purpose": by_purpose}},
                    {"case": "sess.turns_complete", "tier": "sess", "level": "MUST", "verdict": turns,
                     "detail": turns_complete},
                    {"case": calls_case, "tier": "sess", "level": "SHOULD", "verdict": calls,
                     "detail": {"reason": calls_reason, "llm_mode": mode}}],
        "system": {"during": during, "contention": list(contention)},
    }


def baseline(**kw):
    return [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38, **kw),
            leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, **kw)]


def verdict(candidates, base=None):
    return memory_verdict(candidates, base or baseline(), feature=FEATURE, omp_module_sha=MODULE_SHA, rev="abc1234")


FEATURE = "mnemopi-extraction"
MODULE_SHA = "s" * 64
# What the profile routes memory to after the study's winner is applied: the candidate, through omp's ollama provider.
APPLIED_CFG = {"modelRoles": {"smol": f"ollama/{CANDIDATE}"}, "memory.backend": "mnemopi", "mnemopi.llmMode": "smol"}
# The profile after memory:none: smol is still qwen3.8, but mnemopi makes no memory-model call (route off at `none`).
NONE_CFG = {"modelRoles": {"smol": f"ollama/{QWEN}"}, "memory.backend": "mnemopi", "mnemopi.llmMode": "none"}


def graded(receipt: dict, *, applied: bool = True, none_route: bool = False) -> tuple[str, str]:
    """features.grade of `receipt` for registries/features.tsv's mnemopi-extraction row on profile `lab`, routed to
    the candidate (APPLIED_CFG) or, with `none_route`, off at mnemopi.llmMode none (NONE_CFG). With `applied`, the
    presets applied record and rollback manifest say a local memory preset (memory:qwen36 or memory:none, in a temp
    presets registry) replaced smol ollama/qwen3.8:27b-mlx with llmMode smol, so the incumbent comes from
    features.incumbent_settings -> features.incumbent as `localbench features` computes it; without, the incumbent is
    the current route. The routed model, or the routed setting for a route that targets none, comes from
    features.route / features.incumbent of the profile's settings."""
    row = next(r for r in features.load() if r["feature"] == FEATURE)
    rid = "20261001T0500Z-mem1"
    cfg = NONE_CFG if none_route else APPLIED_CFG
    name = "memory:none" if none_route else "memory:qwen36"
    ops = ([{"op": "set", "key": "mnemopi.llmMode", "value": "none"}] if none_route else
           [{"op": "role", "role": "smol", "selector": f"ollama/{CANDIDATE}"},
            {"op": "set", "key": "mnemopi.llmMode", "value": "smol"}])
    steps = ([] if none_route else [{"profile": "lab", "kind": "set", "key": "modelRoles",
                                     "before": {"smol": f"ollama/{QWEN}"}, "after": cfg["modelRoles"]}]) + [
        {"profile": "lab", "kind": "set", "key": "mnemopi.llmMode", "before": "smol", "after": cfg["mnemopi.llmMode"]}]
    with tempfile.TemporaryDirectory() as tmp:
        home, registry = Path(tmp) / ".localbench", Path(tmp) / "presets.json"
        registry.write_text(json.dumps({"presets": [{"name": name, "local": True, "ops": ops}]}))
        if applied:
            state = home / "presets" / "applied.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"lab": {"memory": {"preset": name, "id": rid, "at": rid, "expect": []}}}))
            manifest = home / "rollback" / f"preset-{rid}" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"id": rid, "preset": name, "status": "applied", "plan": {"steps": steps}}))
        real_find = presets.find
        with mock.patch.object(presets, "_home", return_value=home), \
                mock.patch.object(presets, "find", side_effect=lambda n: real_find(n, registry)):
            settings = features.incumbent_settings("lab", row["preset"], cfg, {})
    before, provs, source = settings
    inc = features.incumbent(row, before, provs, source)
    routed = features.route(row, cfg, {})
    if routed["disabled"]:
        return features.grade(receipt, MODULE_SHA, None, None, inc,
                              features.incumbent(row, cfg, {}, "routed")["setting"])
    model = features.route_model(routed["target"].split(" > ")[0])
    return features.grade(receipt, MODULE_SHA, model, "sha256:" + CANDIDATE_DIGEST + "0" * 52, inc)


# A profile already routing memory extraction to local qwen3.8 with no memory preset apply recorded.
QWEN_CFG = {"modelRoles": {"smol": f"ollama/{QWEN}"}, "memory.backend": "mnemopi", "mnemopi.llmMode": "smol"}


def graded_already_local(receipt: dict) -> tuple[tuple[str, str], list[dict]]:
    """features.grade of `receipt` for mnemopi-extraction on profile `lab` routed to local qwen3.8 (QWEN_CFG) with an
    empty presets apply record: the incumbent is the current route (features.incumbent_settings) and the alternatives
    are features.alternatives over the real registries/presets.json, as features.report computes them."""
    row = next(r for r in features.load() if r["feature"] == FEATURE)
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(presets, "_home", return_value=Path(tmp)):
        before = features.incumbent_settings("lab", row["preset"], QWEN_CFG, {})
    assert before[2] == features.CURRENT_ROUTE, before
    inc, alts = features.incumbent(row, *before), features.alternatives(row, QWEN_CFG, {}, "lab")
    model = features.route_model(features.route(row, QWEN_CFG, {})["target"])
    return features.grade(receipt, MODULE_SHA, model, QWEN_DIGEST + "c" * 52, inc, alternatives=alts), alts


def qwen_legs(pre_main=1.0, **kw):
    return [leg("q1", smol=QWEN, digest=QWEN_DIGEST, pre_main=pre_main, **kw),
            leg("q2", smol=QWEN, digest=QWEN_DIGEST, pre_main=pre_main, **kw)]


def none_legs(pre_main=2.0, **kw):
    return [leg("n1", smol=None, digest=None, mode="none", pre_main=pre_main, **kw),
            leg("n2", smol=None, digest=None, mode="none", pre_main=pre_main, **kw)]


class DeclaredAlternatives(unittest.TestCase):
    """An already-local route without an apply record is proven BETTER than a declared non-local alternative of its
    preset family (features.alternatives): qwen3.8 extraction vs memory:none; embeddings vs embeddings:fts."""

    def test_an_already_local_route_is_proven_against_a_declared_alternative(self):
        receipt = verdict(qwen_legs(), none_legs())
        self.assertEqual(receipt["verdict"], {"compare": "BETTER", "baseline": {"kind": "fixed", "id": "none"}})
        self.assertEqual(receipt["problems"], [])
        (status, reason), alts = graded_already_local(receipt)
        self.assertEqual((status, reason), ("PROVEN", ""))
        # memory:qwen38 routes a local model; embeddings:* are another family: only memory:none is declared.
        self.assertEqual([(a["source"], a["setting"]) for a in alts], [("alternative memory:none", "none")])

    def test_another_local_model_route_or_the_route_itself_is_not_an_alternative(self):
        receipt = verdict(qwen_legs(), none_legs())
        for baseline, why in (({"kind": "route", "id": "ollama/qwen3.6:35b"}, "nor a declared alternative"),
                              ({"kind": "route", "id": f"ollama/{QWEN}"}, "is the routed model itself")):
            with self.subTest(baseline=baseline):
                (status, reason), _ = graded_already_local({**receipt, "verdict": {"compare": "BETTER",
                                                                                  "baseline": baseline}})
                self.assertEqual(status, "UNPROVEN")
                self.assertIn(why, reason)
        # Without the alternatives (an apply record exists), memory:none is not this profile's route.
        row = next(r for r in features.load() if r["feature"] == FEATURE)
        inc = features.incumbent(row, QWEN_CFG, {}, features.CURRENT_ROUTE)
        self.assertEqual(features.grade(receipt, MODULE_SHA, QWEN, QWEN_DIGEST, inc)[0], "UNPROVEN")
        # grade itself refuses a local model route handed to it as an alternative.
        local = features.incumbent(row, {**QWEN_CFG, "modelRoles": {"smol": "ollama/qwen3.6:35b"}}, {}, "alternative x")
        moved = {**receipt, "verdict": {"compare": "BETTER", "baseline": {"kind": "route", "id": "ollama/qwen3.6:35b"}}}
        self.assertEqual(features.grade(moved, MODULE_SHA, QWEN, QWEN_DIGEST, inc, alternatives=[local])[0],
                         "UNPROVEN")

    def test_the_same_legs_score_both_directions(self):
        q, n = qwen_legs(), none_legs()
        forward, backward = verdict(q, n), verdict(n, q)
        self.assertEqual((forward["verdict"]["baseline"], forward["verdict"]["compare"]),
                         ({"kind": "fixed", "id": "none"}, "BETTER"))
        self.assertEqual((backward["verdict"]["baseline"], backward["verdict"]["compare"]),
                         ({"kind": "route", "id": f"ollama/{QWEN}"}, "NOT_BETTER"))
        fd = forward["run"]["memory"]["compare"]["deltas"]["mem.recall.pre_main_s"]
        bd = backward["run"]["memory"]["compare"]["deltas"]["mem.recall.pre_main_s"]
        self.assertEqual((fd["candidate"], fd["baseline"]), (bd["baseline"], bd["candidate"]))
        self.assertEqual(forward["run"]["provenance"]["pins"]["model_digest"], QWEN_DIGEST)
        self.assertEqual(backward["run"]["provenance"]["pins"]["model_calls"], 0)
        self.assertEqual(forward["run"]["conformance"]["memory.baseline_route"]["verdict"], "PASS")

    def test_an_alternative_arm_must_differ_only_in_its_own_setting(self):
        # memory:none legs that also turned embeddings off change two things at once.
        receipt = verdict(qwen_legs(), none_legs(config=FTS_CONFIG))
        self.assertTrue(any(p.startswith("the arms differ in embeddings") for p in receipt["problems"]),
                        receipt["problems"])
        self.assertEqual(receipt["run"]["conformance"]["memory.baseline_route"]["verdict"], "FAIL")
        # A candidate that ran the alternative's own configuration is not compared against it.
        receipt = verdict(none_legs(), none_legs())
        self.assertTrue(any("ran the baseline alternative's own configuration (llm_mode none)" in p
                            for p in receipt["problems"]), receipt["problems"])

    def test_embeddings_pin_and_the_fts_alternative(self):
        on, off = qwen_legs(), qwen_legs(pre_main=2.0, config=FTS_CONFIG)
        receipt = verdict(on, off)
        self.assertEqual(receipt["verdict"], {"compare": "BETTER", "baseline": {"kind": "fixed", "id": "true"}})
        self.assertEqual((receipt["problems"], receipt["run"]["provenance"]["pins"]["embeddings"]), ([], "on"))
        self.assertEqual(verdict(off, on)["run"]["provenance"]["pins"]["embeddings"], "off")
        # embeddings:fts's id is the value features.incumbent names recall-embeddings' route off at.
        row = next(r for r in features.load() if r["feature"] == "recall-embeddings")
        self.assertEqual(features.incumbent(row, {**QWEN_CFG, "mnemopi.noEmbeddings": True}, {}, "x")["setting"],
                         "true")
        # The fts arm must run the candidate's memory model: another digest is a second difference.
        other = [leg(f"o{i}", smol=QWEN, digest="dddddddddddd", pre_main=2.0, config=FTS_CONFIG) for i in (1, 2)]
        self.assertTrue(any(p.startswith("the arms differ in digest") for p in verdict(on, other)["problems"]))

    def test_an_unverifiable_config_records_no_embeddings(self):
        tampered = qwen_legs()
        tampered[1]["provenance"]["pins"]["omp_mem_config"] = "0" * 16  # the file is not what the leg ran
        receipt = verdict(tampered, none_legs())
        self.assertIsNone(receipt["run"]["provenance"]["pins"]["embeddings"])
        self.assertTrue(any("a leg does not record it" in p for p in receipt["problems"]), receipt["problems"])
        # Against the incumbent route (qwen3.8 smol baseline legs) the arms must agree on embeddings too.
        cand = [leg("b1", post=2.06), leg("b2", post=2.10)]
        cand[1]["provenance"]["pins"]["omp_mem_config"] = "0" * 16
        receipt = verdict(cand)
        self.assertTrue(any(p.startswith("the arms differ in embeddings, or a leg does not record it")
                            and p.endswith("not the memory route alone") for p in receipt["problems"]),
                        receipt["problems"])
        mixed = verdict([leg("b1", post=2.06), leg("b2", post=2.10)],
                        [leg(f"a{i}", smol=QWEN, digest=QWEN_DIGEST, post=8.4, config=FTS_CONFIG) for i in (1, 2)])
        self.assertEqual(mixed["verdict"]["baseline"], {"kind": "fixed", "id": "true"})


FASTEMBED = "fast-bge-base-en-v1.5"


def fake_fastembed(root: Path) -> Path:
    """A fastembed cache dir holding FASTEMBED's files (two of them), as omp's ~/.omp/cache/fastembed does."""
    model = root / "fastembed" / FASTEMBED
    model.mkdir(parents=True)
    (model / "model_optimized.onnx").write_bytes(b"onnx-weights")
    (model / "config.json").write_text('{"_name_or_path": "BAAI/bge-base-en-v1.5"}')
    return root / "fastembed"


class EmbeddingsProof(unittest.TestCase):
    """recall-embeddings: the run pins the embedding model by name and on-disk digest; an embeddings-on vs
    embeddings:fts receipt proves it with that digest; its preset family is `embeddings`, apart from `memory`."""

    def test_embedding_pins_name_the_model_and_hash_its_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = fake_fastembed(Path(tmp))
            files = sorted((cache / FASTEMBED).iterdir())
            h = hashlib.sha256()
            for f in files:
                h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
            want = h.hexdigest()[:12]
            self.assertEqual(embedding_pins(MEM_CONFIG, cache), {"embedder": EMBEDDER, "embedder_digest": want})
            self.assertEqual(embedding_pins(NONE_CONFIG, cache)["embedder_digest"], want)   # llmMode none still embeds
            self.assertEqual(embedding_pins(FTS_CONFIG, cache), {"embedder": None, "embedder_digest": None})
            (cache / FASTEMBED / "model.onnx_data.corrupt-1").write_bytes(b"quarantined")
            self.assertEqual(fastembed_digest(FASTEMBED, cache), want)
            (cache / FASTEMBED / "model_optimized.onnx").write_bytes(b"other-weights!")
            self.assertNotEqual(fastembed_digest(FASTEMBED, cache), want)
            self.assertIsNone(fastembed_digest("fast-bge-small-en-v1.5", cache))
            # The run pins carry them (omp_pins is part of every run's pins).
            with mock.patch.object(cli, "omp_bin", return_value="omp"), \
                    mock.patch.object(cli, "_first_line", return_value="omp/18.4.6"), \
                    mock.patch.object(cli, "sha16", return_value="s"), \
                    mock.patch("localbench.workloads.FASTEMBED_CACHE", cache):
                self.assertEqual(cli.omp_pins(MEM_CONFIG).get("embedder"), EMBEDDER)
                self.assertIsNotNone(cli.omp_pins(MEM_CONFIG)["embedder_digest"])
                self.assertIsNone(cli.omp_pins(FTS_CONFIG)["embedder_digest"])

    def receipt(self, digest: str | None) -> dict:
        on = [leg(f"e{i}", smol=QWEN, digest=QWEN_DIGEST, pre_main=1.0, embedder_digest=digest) for i in (1, 2)]
        off = [leg(f"f{i}", smol=QWEN, digest=QWEN_DIGEST, pre_main=2.0, config=FTS_CONFIG) for i in (1, 2)]
        return memory_verdict(on, off, feature="recall-embeddings", omp_module_sha=MODULE_SHA, rev="abc1234")

    def test_an_embeddings_receipt_proves_the_installed_embedding_model(self):
        row = next(r for r in features.load() if r["feature"] == "recall-embeddings")
        with tempfile.TemporaryDirectory() as tmp:
            cache = fake_fastembed(Path(tmp))
            receipt = self.receipt(fastembed_digest(FASTEMBED, cache))
            pins = receipt["run"]["provenance"]["pins"]
            self.assertEqual((receipt["verdict"]["baseline"], receipt["problems"]),
                             ({"kind": "fixed", "id": "true"}, []))
            self.assertEqual((pins["model"], pins["model_digest"], pins["memory_model"]),
                             (EMBEDDER, fastembed_digest(FASTEMBED, cache), QWEN))
            self.assertNotIn("route", pins)
            with mock.patch.object(presets, "_home", return_value=Path(tmp) / "home"), \
                    mock.patch("localbench.workloads.FASTEMBED_CACHE", cache):
                on_cfg = {**QWEN_CFG, "mnemopi.noEmbeddings": False}   # omp config list reports the default
                before = features.incumbent_settings("lab", row["preset"], on_cfg, {})
                inc, alts = features.incumbent(row, *before), features.alternatives(row, on_cfg, {}, "lab")
                model = features.route_model(features.route(row, on_cfg, {})["target"])
                digest = features.installed_digest(row, model, {})
                self.assertEqual((model, digest), (FASTEMBED, fastembed_digest(FASTEMBED, cache)))
                self.assertEqual([(a["source"], a["setting"]) for a in alts], [("alternative embeddings:fts", "true")])
                self.assertEqual(features.grade(receipt, MODULE_SHA, model, digest, inc, alternatives=alts),
                                 ("PROVEN", ""))
                (cache / FASTEMBED / "model_optimized.onnx").write_bytes(b"re-downloaded")   # another build on disk
                status, reason = features.grade(receipt, MODULE_SHA, model,
                                                features.installed_digest(row, model, {}), inc, alternatives=alts)
        self.assertEqual(status, "UNPROVEN")
        self.assertIn("proves model digest", reason)

    def test_an_embeddings_receipt_beside_llm_mode_none_pins_the_embedder_not_the_no_model_route(self):
        on = [leg(f"e{i}", smol=None, digest=None, mode="none", pre_main=1.0) for i in (1, 2)]
        off = [leg(f"f{i}", smol=None, digest=None, mode="none", pre_main=2.0, config=FTS_CONFIG) for i in (1, 2)]
        receipt = memory_verdict(on, off, feature="recall-embeddings", omp_module_sha=MODULE_SHA, rev="abc1234")
        pins = receipt["run"]["provenance"]["pins"]
        self.assertEqual((receipt["problems"], pins["model"], pins["model_digest"]), ([], EMBEDDER, EMBEDDER_DIGEST))
        self.assertNotIn("route", pins)

    def test_candidate_legs_without_an_embedder_digest_cannot_prove_embeddings(self):
        receipt = self.receipt(None)
        self.assertTrue(any(p.startswith("candidate legs pin no single embedding model with a digest")
                            for p in receipt["problems"]), receipt["problems"])
        self.assertEqual(receipt["run"]["provenance"]["pins"]["model"], QWEN)

    def test_memory_and_embeddings_presets_gate_only_their_own_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = fake_fastembed(root)
            for rel in ("src/core/extraction.ts", "src/core/embeddings.ts"):
                (root / "pi-mnemopi" / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / "pi-mnemopi" / rel).write_text(f"// {rel}\n")
            (root / "pi-coding-agent").mkdir()
            sha = features.module_sha(root / "pi-mnemopi" / "src/core/embeddings.ts")
            receipt = memory_verdict(
                [leg(f"e{i}", smol=QWEN, digest=QWEN_DIGEST, pre_main=1.0,
                     embedder_digest=fastembed_digest(FASTEMBED, cache)) for i in (1, 2)],
                [leg(f"f{i}", smol=QWEN, digest=QWEN_DIGEST, pre_main=2.0, config=FTS_CONFIG) for i in (1, 2)],
                feature="recall-embeddings", omp_module_sha=sha, rev="abc1234")
            (root / "receipts").mkdir()
            (root / "receipts" / "embeddings.json").write_text(json.dumps(receipt))
            fts_now = {**QWEN_CFG, "mnemopi.noEmbeddings": True}   # the profile runs FTS-only recall today
            with mock.patch.object(features, "omp_settings", return_value=fts_now), \
                    mock.patch.object(features, "providers", return_value={}), \
                    mock.patch("localbench.workloads.FASTEMBED_CACHE", cache):
                def statuses(fam, settings):
                    return presets.proof_statuses(fam, {"lab": (fts_now["modelRoles"], {})},
                                                  receipts_dir=root / "receipts", package=root / "pi-coding-agent",
                                                  digests={}, settings={"lab": settings})
                memory = statuses("memory", {"mnemopi.llmMode": "none"})
                embeddings = statuses("embeddings", {"mnemopi.noEmbeddings": False})
        self.assertEqual(set(memory), {FEATURE})            # memory:* applies never need an embeddings proof
        self.assertEqual(set(embeddings), {"recall-embeddings"})
        on = embeddings["recall-embeddings"]["lab"]
        self.assertEqual((on["proof"], on["model"], on["digest"]),
                         ("PROVEN", FASTEMBED, receipt["run"]["provenance"]["pins"]["model_digest"]))


# A contention episode as sysstats.Sampler records one: proj-b's nimble resident and running beside the measured models.
NIMBLE_EPISODE = {"t": 1790900000.0, "foreign": {"ollama": ["nimble:latest"]},
                  "foreign_gpu": [{"name": "ollama", "pid": 4242, "model": "nimble:latest", "pct": 38.0}],
                  "resident": {"ollama": ["qwen3.6:35b-mlx", QWEN, "nimble:latest"]}, "gpu_device_pct": 97.0}
SIDE_STAMP = "20261002T030000Z"   # after workloads.SIDE_REGIME_FROM


def side_legs(names, *, created=SIDE_STAMP, regime="side", **kw):
    """Legs of an ab --side-regime run: real qwen3.8 as smol (no park alias), regime side in the pins."""
    return [leg(n, smol=QWEN, digest=QWEN_DIGEST, created=created, regime=regime, **kw) for n in names]


class SideRegime(unittest.TestCase):
    """Memory-route proofs under the side-model regime (bead kit-memory-study-vce DECISION, 2026-10-02T01:31:49Z):
    co-resident load is recorded, never a void, for side legs created on or after workloads.SIDE_REGIME_FROM."""

    def test_a_contended_side_leg_is_not_voided_and_reports_its_co_residents(self):
        flawed = dict(contended=True, contention=[NIMBLE_EPISODE], resident_unknown=2)
        cands = side_legs(["n1"], mode="none", pre_main=2.0) + side_legs(["n2"], mode="none", pre_main=2.0, **flawed)
        cands = [{**x, "provenance": {**x["provenance"], "pins": {**x["provenance"]["pins"], "smol_model": None,
                                                                   "smol_digest": None}}} for x in cands]
        base = side_legs(["q1"], pre_main=1.0) + side_legs(["q2"], pre_main=1.0, **flawed)
        receipt = verdict(base, cands)   # candidate: the real qwen3.8 route; baseline: memory:none
        self.assertEqual((receipt["verdict"]["compare"], receipt["problems"]), ("BETTER", []))
        entry = receipt["run"]["conformance"]["memory.co_resident"]
        self.assertEqual((entry["level"], entry["verdict"]), ("SHOULD", "FAIL"))
        self.assertEqual(entry["legs"]["candidate"]["q2"],
                         {"models": ["nimble:latest", "ollama/nimble:latest"], "episodes": 1,
                          "foreign_gpu_mean_pct": 38.0, "resident_unknown_samples": 2})
        self.assertEqual(entry["legs"]["candidate"]["q1"]["models"], [])
        self.assertEqual(entry["legs"]["baseline"]["n2"]["episodes"], 1)
        # The real qwen3.8 name (no park alias) is the baseline route by digest, and its pins name the regime.
        self.assertEqual(receipt["run"]["provenance"]["pins"]["regime"], "side")
        self.assertEqual(receipt["run"]["conformance"]["memory.baseline_route"]["verdict"], "PASS")

    def test_the_incumbent_route_under_its_real_name_is_the_baseline(self):
        receipt = verdict(side_legs(["b1", "b2"], post=2.06), side_legs(["a1", "a2"], post=8.4))
        self.assertEqual(receipt["verdict"]["baseline"], {"kind": "route", "id": f"ollama/{QWEN}"})
        self.assertFalse(any("did not run" in p or "served by" in p for p in receipt["problems"]),
                         receipt["problems"])

    def test_an_old_or_unflagged_leg_is_still_voided_by_contention(self):
        flawed = dict(contended=True, contention=[NIMBLE_EPISODE])
        for kw in ({"created": "20261001T220000Z"}, {"regime": None}, {"created": None}):
            with self.subTest(kw=kw):
                cands = side_legs(["b1"], post=2.06, **kw) + side_legs(["b2"], post=2.10, **{**kw, **flawed})
                base = side_legs(["a1", "a2"], post=8.4, **kw)
                receipt = verdict(cands, base)
                self.assertEqual(receipt["verdict"]["compare"], "NONE")
                self.assertTrue(any(p.startswith("comparison void: candidate leg(s) b2 CONTENDED")
                                    for p in receipt["problems"]), receipt["problems"])
                self.assertEqual(receipt["run"]["conformance"]["memory.co_resident"]["legs"]["candidate"], {})

    def test_legs_that_mix_regimes_are_a_problem(self):
        receipt = verdict(side_legs(["b1", "b2"], post=2.06), side_legs(["a1", "a2"], post=8.4, regime=None))
        self.assertTrue(any(p.startswith("legs mix measurement regimes") for p in receipt["problems"]),
                        receipt["problems"])
        self.assertEqual(graded(receipt)[0], "BAD")

    def test_quality_musts_still_fail_under_the_side_regime(self):
        receipt = verdict(side_legs(["b1"], post=2.06) + side_legs(["b2"], post=2.10, no_leak="FAIL"),
                          side_legs(["a1", "a2"], post=8.4))
        self.assertIn("MUST FAIL: b2:mem.no_leak", receipt["problems"])

    def test_unsound_and_banking_skip_co_residency_only_for_side_legs(self):
        side = side_legs(["s1"], contended=True, contention=[NIMBLE_EPISODE], resident_unknown=None)[0]
        side["verdicts"]["preflight_problems"] = []
        self.assertEqual(cli.unsound(side), [])
        self.assertFalse(cli._unknown_residency(side))
        old = side_legs(["o1"], created="20261001T220000Z", contended=True, resident_unknown=None)[0]
        old["verdicts"]["preflight_problems"] = []
        self.assertTrue(any(r.startswith("CONTENDED") for r in cli.unsound(old)))
        self.assertTrue(cli._unknown_residency(old))


class SideRegimeAb(unittest.TestCase):
    """`localbench ab --side-regime`: nothing is unloaded and co-residents do not refuse the start (fakes only)."""

    class Backend:
        name = "ollama"
        base_url = "http://127.0.0.1:9/v1"

        def __init__(self):
            self.isolated = []

        def isolate(self, model):
            self.isolated.append(model)
            return ["nimble:latest"]

        def fingerprint(self, model):
            return {"loaded_context": 4096}

        def pins(self, model):
            return {"model": model, "model_digest": QWEN_DIGEST}

    def run_ab(self, *extra: str):
        backend, warmed, busy_sides = self.Backend(), [], []
        sampler = mock.MagicMock()
        sampler.__enter__.return_value = sampler
        sampler.contention, sampler.series = [], []
        sampler.summary.return_value = {"resident_unknown_samples": 0}

        @contextlib.contextmanager
        def open_backend(spec, server_args):
            yield backend, spec.partition(":")[2]

        def busy_check(side=False):
            busy_sides.append(side)
            return {}, 5.0, []

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cli, "ROOT", Path(tmp)), \
                mock.patch.object(cli, "_check_root"), \
                mock.patch.object(cli, "RUNS", Path(tmp) / "runs"), \
                mock.patch.object(cli, "RECEIPTS", Path(tmp) / "rc"), \
                mock.patch.object(cli, "open_backend", open_backend), \
                mock.patch.object(cli, "_busy_check", side_effect=busy_check), \
                mock.patch.object(cli, "unload_ollama", side_effect=AssertionError("unload_ollama called")), \
                mock.patch.object(cli.backends, "_warm", side_effect=lambda url, m: warmed.append(m)), \
                mock.patch.object(cli, "run_pins", side_effect=lambda b, m, h, c=None: {"model": m}), \
                mock.patch.object(cli, "smol_pins", return_value={}), \
                mock.patch.object(cli, "_rev", return_value="abc1234"), \
                mock.patch.object(cli.sysstats, "snapshot", return_value={"host": {}, "live": {}}), \
                mock.patch.object(cli.sysstats, "Sampler", return_value=sampler), \
                mock.patch.object(cli.sysstats, "PowerSampler", return_value=sampler), \
                mock.patch.object(cli.sysstats, "CpuSampler", return_value=sampler), \
                mock.patch.dict(cli.TIERS, {"mem": lambda ctx: [], "sess": lambda ctx: [], "conf": lambda ctx: []}), \
                mock.patch.object(cli.golden, "listed_discrepancies", return_value=set()), \
                mock.patch.object(cli.render, "show", return_value=""), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = cli.main(["ab", "ollama:qwen3.6:35b-mlx", "ollama:qwen3.6:35b-mlx", "--tiers", "mem,sess",
                           "--repeats", "1", "--bank", "side-test", *extra])
            receipt = json.loads((Path(tmp) / "rc" / "side-test.json").read_text()) if rc in (0, 1) else None
        return rc, receipt, backend.isolated, warmed, busy_sides

    def test_side_regime_never_unloads_and_marks_every_leg(self):
        rc, receipt, isolated, warmed, busy_sides = self.run_ab("--side-regime")
        self.assertEqual(isolated, [])
        self.assertEqual(warmed, ["qwen3.6:35b-mlx"] * 3)
        self.assertEqual(busy_sides, [True] * 3)
        self.assertEqual([(leg["provenance"].get("regime"), leg["provenance"]["pins"].get("regime"))
                          for leg in receipt["legs"]], [("side", "side")] * 3)

    def test_without_the_flag_every_leg_isolates_under_the_one_model_preflight(self):
        rc, receipt, isolated, warmed, busy_sides = self.run_ab()
        self.assertEqual((isolated, warmed, busy_sides), (["qwen3.6:35b-mlx"] * 3, [], [False] * 3))
        self.assertTrue(all("regime" not in leg["provenance"]["pins"] for leg in receipt["legs"]))

    def test_side_regime_is_refused_outside_mem_and_sess(self):
        with mock.patch.object(cli, "execute", side_effect=AssertionError("leg measured")), \
                mock.patch.object(cli, "open_backend", side_effect=AssertionError("backend opened")), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.cmd_ab(argparse.Namespace(a="ollama:a", b="ollama:b", pairs=1, tiers="mem,conf", smol_model=None,
                                          b_smol_model=None, side_regime=True, b_omp=None, b_mlx_serve=None,
                                          b_mlxfast=None, server_arg=None, b_server_arg=None, mem_config=MEM_CONFIG,
                                          b_mem_config=None))
        self.assertEqual(caught.exception.code, 2)

    def test_the_side_preflight_refuses_only_on_unreadable_signals_or_swap(self):
        sampler = mock.MagicMock()
        sampler.__enter__.return_value = sampler
        nimble = {"name": "ollama", "pid": 4242, "model": "nimble:latest", "pct": 38.0, "cmd": "ollama runner"}
        sampler.summary.return_value = {"gpu_device_pct": {"mean": 90}, "gpu_by_process": [nimble],
                                        "resident_unknown_samples": 1, "swap_used_mb": {"max": 0}}
        with mock.patch.object(cli.sysstats, "Sampler", return_value=sampler), \
                mock.patch.object(cli.sysstats, "cpu_busy_pct", return_value=12.0), \
                mock.patch.object(cli.park, "reachable_smol", return_value=[QWEN]), \
                mock.patch.object(cli.smol, "load_state", return_value=None), \
                mock.patch.object(cli.park, "stuck_sessions", return_value=[]), \
                mock.patch.object(cli.sysstats, "omp_processes", return_value=[]):
            _, _, one_model = cli._busy_check()
            _, _, side = cli._busy_check(side=True)
            sampler.summary.return_value = {**sampler.summary.return_value, "swap_used_mb": {"max": 4096}}
            _, _, side_swapping = cli._busy_check(side=True)
        self.assertEqual(side, [])
        self.assertEqual(side_swapping, ["swapping"])
        self.assertTrue(any(p.startswith("another model is running") for p in one_model), one_model)
        self.assertTrue(any("resident model state unknown" in p for p in one_model))
        self.assertTrue(any("are loadable" in p for p in one_model))


class MemoryVerdict(unittest.TestCase):
    def test_equal_quality_and_faster_is_better_and_grades_proven(self):
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)])
        self.assertEqual(receipt["verdict"], {"compare": "BETTER",
                                              "baseline": {"kind": "route", "id": "ollama/qwen3.8:27b-mlx"}})
        self.assertEqual(receipt["problems"], [])
        self.assertEqual(receipt["run"]["label"], "memory")
        pins = receipt["run"]["provenance"]["pins"]
        self.assertEqual((pins["model"], pins["model_digest"], pins["main_model"]),
                         (CANDIDATE, CANDIDATE_DIGEST, "main:1"))
        self.assertIn("post_retain_latency", receipt["run"]["memory"]["compare"]["wins"])
        self.assertEqual(graded(receipt), ("PROVEN", ""))

    def test_a_correct_receipt_does_not_prove_against_the_candidate_route_itself(self):
        # No apply record: the incumbent is the current route, the candidate; a qwen3.8 baseline does not name it.
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)])
        status, reason = graded(receipt, applied=False)
        self.assertEqual(status, "UNPROVEN")
        self.assertIn("is not this profile's route", reason)

    def test_recall_drop_beyond_noise_is_worse_with_a_problem(self):
        # Four legs per arm: enough matched pairs for the cluster-robust CMH call.
        receipt = verdict([leg(f"b{i}", post=2.06, hit=0.6667) for i in (1, 2, 3, 4)],
                          [leg(f"a{i}", smol=QWEN, digest=QWEN_DIGEST, post=8.38) for i in (1, 2, 3, 4)])
        self.assertEqual(receipt["verdict"]["compare"], "WORSE")
        self.assertEqual(receipt["run"]["memory"]["compare"]["quality_losses"], ["mem.recall.hit_rate"])
        self.assertTrue(any("mem.recall.hit_rate" in p for p in receipt["problems"]))
        self.assertEqual(graded(receipt)[0], "BAD")

    def test_two_leg_pilot_cannot_call_a_clustered_recall_loss(self):
        # F1: CMH on 2 matched pairs called WORSE on what is effectively 2
        # observations; the design-effect fallback holds it to within noise.
        receipt = verdict([leg("b1", post=2.06, hit=0.6667), leg("b2", post=2.10, hit=0.6667)])
        row = receipt["run"]["memory"]["compare"]["deltas"]["mem.recall.hit_rate"]
        self.assertEqual(row["judgement"], "within_noise")
        self.assertEqual(row["test"]["method"], "deff-fisher")
        self.assertTrue(row["test"]["few_clusters"])

    def test_recall_drop_inside_the_baselines_own_spread_is_not_a_loss(self):
        base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38, hit=1.0),
                leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, hit=0.8889)]
        receipt = verdict([leg("b1", post=2.06, hit=0.8889), leg("b2", post=2.10, hit=0.8889)], base)
        self.assertEqual(receipt["run"]["memory"]["compare"]["deltas"]["mem.recall.hit_rate"]["judgement"],
                         "within_noise")
        self.assertEqual(receipt["verdict"]["compare"], "BETTER")

    def test_a_tie_is_not_better(self):
        receipt = verdict([leg("b1", post=8.38), leg("b2", post=8.44)])
        self.assertEqual(receipt["verdict"]["compare"], "NOT_BETTER")
        self.assertIn("not better than route ollama/qwen3.8:27b-mlx: NOT_BETTER", receipt["problems"])

    def test_drift_paired_gain_fires_on_a_baseline_first_chain(self):
        base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, pre_main=8.0, created="20261001T150001Z"),
                leg("a2", smol=QWEN, digest=QWEN_DIGEST, pre_main=8.0, created="20261001T150003Z"),
                leg("a3", smol=QWEN, digest=QWEN_DIGEST, pre_main=8.0, created="20261001T150005Z")]
        cand = [leg("b1", pre_main=2.0, created="20261001T150002Z"),
                leg("b2", pre_main=2.0, created="20261001T150004Z")]
        receipt = verdict(cand, base)
        row = receipt["run"]["memory"]["compare"]["deltas"]["mem.recall.pre_main_s"]
        self.assertEqual(row["test"]["method"], "drift-paired-t")
        self.assertEqual(row["judgement"], "gain")
        self.assertIn("recall_latency", receipt["run"]["memory"]["compare"]["wins"])
        self.assertEqual(receipt["verdict"]["compare"], "BETTER")

    def test_clustered_recall_falls_back_to_cmh(self):
        cand = [leg(f"b{i}", hit=h) for i, h in enumerate([1.0, 1.0, 0.0, 0.0])]
        base = [leg(f"a{i}", smol=QWEN, digest=QWEN_DIGEST, hit=1.0) for i in range(4)]
        receipt = verdict(cand, base)
        row = receipt["run"]["memory"]["compare"]["deltas"]["mem.recall.hit_rate"]
        self.assertEqual(row["test"]["method"], "cmh")
        self.assertEqual(row["judgement"], "loss")
        self.assertEqual(receipt["verdict"]["compare"], "WORSE")

    def test_band_counts_both_arms_spreads_on_conformance_rates(self):
        base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, calls="PASS"),
                leg("a2", smol=QWEN, digest=QWEN_DIGEST, calls="FAIL")]
        receipt = verdict(qwen_legs(), base)
        self.assertEqual(receipt["run"]["memory"]["compare"]["deltas"]["sess.memory_calls_ok"]["judgement"],
                         "within_noise")

    def test_conformance_loss_beyond_noise_is_worse(self):
        bad = [leg("b1", no_leak="FAIL"), leg("b2", no_leak="FAIL")]
        receipt = verdict(bad)
        self.assertEqual(receipt["run"]["memory"]["compare"]["deltas"]["mem.no_leak"]["judgement"], "loss")
        self.assertEqual(receipt["verdict"]["compare"], "WORSE")

    def test_delta_equal_to_the_band_is_a_tie_not_a_win(self):
        receipt = verdict(qwen_legs(), baseline())
        self.assertEqual(receipt["run"]["memory"]["compare"]["deltas"]["sess.memory_calls_ok"]["judgement"],
                         "within_noise")

    def test_any_must_not_passing_is_a_problem_even_when_faster(self):
        for failed in ("FAIL", "VOID"):
            with self.subTest(failed=failed):
                receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10, turns=failed)])
                self.assertEqual(receipt["verdict"]["compare"], "BETTER")
                self.assertIn("MUST FAIL: b2:sess.turns_complete", receipt["problems"])
                self.assertEqual(graded(receipt)[0], "BAD")

    def test_unmeasured_recall_cannot_be_better(self):
        for broken in ({"hit": None}, {"hit": 0.0, "hit_n": 0}):
            with self.subTest(broken=broken):
                receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10, **broken)])
                self.assertEqual(receipt["run"]["memory"]["compare"]["deltas"]["mem.recall.hit_rate"]["judgement"],
                                 "unmeasured")
                self.assertEqual(receipt["verdict"]["compare"], "NOT_BETTER")
                self.assertTrue(any("unmeasured" in p for p in receipt["problems"]))

    def test_llm_mode_none_proves_a_no_model_route_pinned_by_zero_model_calls(self):
        none = [leg("b1", smol=None, digest=None, mode="none", post=2.06),
                leg("b2", smol=None, digest=None, mode="none", post=2.10)]
        receipt = verdict(none)
        compare = receipt["run"]["memory"]["compare"]
        self.assertEqual(compare["deltas"]["sess.memory_calls_ok"]["judgement"], "within_noise")
        self.assertEqual((receipt["verdict"]["compare"], receipt["problems"]), ("BETTER", []))
        pins = receipt["run"]["provenance"]["pins"]
        self.assertEqual((pins["model_digest"], pins["route"], pins["model_calls"]),
                         (None, {"kind": "fixed", "id": "none"}, 0))
        self.assertEqual(cli.validate_doc(receipt), [])
        failed = verdict([leg(f"b{i}", smol=None, digest=None, mode="none", post=2.06, calls="FAIL") for i in (1, 2)])
        self.assertIn("sess.memory_calls_ok", failed["run"]["memory"]["compare"]["quality_losses"])

    def test_a_none_receipt_grades_proven_only_on_a_none_route(self):
        receipt = verdict([leg("b1", smol=None, digest=None, mode="none", post=2.06),
                           leg("b2", smol=None, digest=None, mode="none", post=2.10)])
        self.assertEqual(graded(receipt, none_route=True), ("PROVEN", ""))
        status, reason = graded(receipt)  # the profile routes memory to the candidate model
        self.assertEqual(status, "BAD")
        self.assertIn("proves no-model route fixed none; this profile's route targets model", reason)
        # Only the preset's pre-apply route is the baseline: graded against the none route itself it is not.
        status, reason = graded(receipt, none_route=True, applied=False)
        self.assertEqual(status, "UNPROVEN")
        # A model receipt on the none route proves nothing about it.
        model_receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)])
        self.assertEqual(graded(model_receipt, none_route=True)[0], "UNPROVEN")
        # A route off at another value of the gating settings (memory.backend off) is not the none route.
        row = next(r for r in features.load() if r["feature"] == FEATURE)
        inc = features.incumbent(row, {**NONE_CFG, "mnemopi.llmMode": "smol"}, {}, "before")
        status, reason = features.grade(receipt, MODULE_SHA, None, None, inc, "off")
        self.assertEqual(status, "BAD")
        self.assertIn("this profile's route is off at off", reason)

    def test_a_none_receipt_with_any_memory_model_call_is_refused(self):
        called = {"memory-extract": {"calls": 1, "models": [QWEN]}}
        receipt = verdict([leg("b1", smol=None, digest=None, mode="none", post=2.06),
                           leg("b2", smol=None, digest=None, mode="none", post=2.10, by_purpose=called)])
        self.assertEqual(receipt["run"]["provenance"]["pins"]["model_calls"], 1)
        self.assertTrue(any(p.startswith("llmMode none candidate legs made 1 memory-model calls")
                            for p in receipt["problems"]), receipt["problems"])
        self.assertEqual(receipt["run"]["conformance"]["memory.llm_mode"]["verdict"], "FAIL")
        self.assertEqual(graded(receipt, none_route=True)[0], "BAD")
        # grade itself refuses the pin, problems or not.
        clean = verdict([leg("b1", smol=None, digest=None, mode="none", post=2.06),
                         leg("b2", smol=None, digest=None, mode="none", post=2.10)])
        clean["run"]["provenance"]["pins"]["model_calls"] = 1
        status, reason = graded(clean, none_route=True)
        self.assertEqual(status, "BAD")
        self.assertIn("pins model_calls 1, not 0", reason)

    def test_preset_memory_none_is_graded_as_the_no_model_route_it_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = root / "pi-mnemopi" / "src" / "core" / "extraction.ts"
            module.parent.mkdir(parents=True)
            module.write_text("export function extractFactCategories() {}\n")
            (root / "pi-coding-agent").mkdir()
            sha = features.module_sha(module)
            receipt = memory_verdict([leg("b1", smol=None, digest=None, mode="none", post=2.06),
                                      leg("b2", smol=None, digest=None, mode="none", post=2.10)], baseline(),
                                     feature=FEATURE, omp_module_sha=sha, rev="abc1234")
            (root / "receipts").mkdir()
            (root / "receipts" / "none.json").write_text(json.dumps(receipt))
            now = {"modelRoles": {"smol": f"ollama/{QWEN}"}, "memory.backend": "mnemopi", "mnemopi.llmMode": "smol"}
            with mock.patch.object(features, "omp_settings", return_value=now), \
                    mock.patch.object(features, "providers", return_value={}):
                def grade_preset(settings):
                    return presets.proof_statuses("memory", {"lab": (now["modelRoles"], {})},
                                                  receipts_dir=root / "receipts", package=root / "pi-coding-agent",
                                                  digests={QWEN: QWEN_DIGEST}, settings=settings)[FEATURE]["lab"]
                applied = grade_preset({"lab": {"mnemopi.llmMode": "none"}})
                unset = grade_preset(None)
        self.assertEqual((applied["proof"], applied["setting"], applied["model"]), ("PROVEN", "none", None))
        self.assertEqual((unset["proof"], unset["model"]), ("BAD", QWEN))

    def test_an_ack_only_slip_in_either_arm_is_reported_not_voided(self):
        for arm in ("candidate", "baseline"):
            with self.subTest(arm=arm):
                # Both legs of the slipping arm slip, so its memory-calls quality reads the ack slip in every leg.
                c_slips, b_slips = ((1, 2), (0, 0)) if arm == "candidate" else ((0, 0), (1, 1))
                cands = [leg("b1", post=2.06, ack_slips=c_slips[0]), leg("b2", post=2.10, ack_slips=c_slips[1])]
                base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38, ack_slips=b_slips[0]),
                        leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, ack_slips=b_slips[1])]
                receipt = verdict(cands, base)
                self.assertEqual((receipt["verdict"]["compare"], receipt["problems"]), ("BETTER", []))
                entry = receipt["run"]["conformance"]["memory.main_ack_slips"]
                self.assertEqual((entry["level"], entry["verdict"]), ("SHOULD", "FAIL"))
                self.assertEqual(entry["slips"], {"candidate": dict(zip(("b1", "b2"), c_slips)),
                                                  "baseline": dict(zip(("a1", "a2"), b_slips))})
                self.assertEqual(graded(receipt), ("PROVEN", ""))

    def test_the_ack_slip_exemption_applies_only_to_legs_created_after_the_decision(self):
        # 20261001T113148Z: a stamp of the contrast-2 run, before the DECISIONS comment (2026-10-01T14:30:14Z).
        for created, exempt in (("20261001T113148Z", False), ("20261001T150000Z", True), (None, False)):
            for arm in ("candidate", "baseline"):
                with self.subTest(created=created, arm=arm):
                    c_kw, b_kw = ({"ack_slips": 1, "created": created}, {}) if arm == "candidate" else \
                        ({}, {"ack_slips": 1, "created": created})
                    cands = [leg("b1", post=2.06), leg("b2", post=2.10, **c_kw)]
                    base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                            leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, **b_kw)]
                    receipt = verdict(cands, base)
                    flawed = "b2" if arm == "candidate" else "a2"
                    slips = receipt["run"]["conformance"]["memory.main_ack_slips"]["slips"][arm][flawed]
                    want = ("MUST FAIL: b2:sess.turns_complete" if arm == "candidate"
                            else "comparison void: baseline MUST FAIL: a2:sess.turns_complete")
                    if exempt:
                        self.assertEqual(slips, 1)
                        self.assertNotIn(want, receipt["problems"])
                        self.assertEqual(graded(receipt), ("PROVEN", ""))
                    else:
                        self.assertIsNone(slips)
                        self.assertIn(want, receipt["problems"])
                        self.assertEqual(graded(receipt)[0], "BAD")

    def test_a_turns_complete_failure_beyond_ack_slips_still_fails_or_voids(self):
        flaws = ({"turns": "FAIL"},  # a missing turn, with a session failure
                 {"ack_slips": 1, "turns_detail": {"turns": 23}},  # a turn did not complete
                 {"ack_slips": 1, "turns_detail": {"failures": ["session 0 turn 5: TimeoutError"]}},
                 {"ack_slips": 1, "sess_timeouts": 1},
                 {"ack_slips": 1, "turns_detail": {"reason": "incomplete turns or failed session"}})  # e.g. rc != 0
        for flaw in flaws:
            for arm in ("candidate", "baseline"):
                with self.subTest(flaw=flaw, arm=arm):
                    c_flaw, b_flaw = (flaw, {}) if arm == "candidate" else ({}, flaw)
                    cands = [leg("b1", post=2.06), leg("b2", post=2.10, **c_flaw)]
                    base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                            leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, **b_flaw)]
                    receipt = verdict(cands, base)
                    want = ("MUST FAIL: b2:sess.turns_complete" if arm == "candidate"
                            else "comparison void: baseline MUST FAIL: a2:sess.turns_complete")
                    self.assertIn(want, receipt["problems"])
                    slips = receipt["run"]["conformance"]["memory.main_ack_slips"]["slips"]
                    self.assertIsNone(slips[arm]["b2" if arm == "candidate" else "a2"])
                    self.assertEqual(graded(receipt)[0], "BAD")

    def test_baseline_legs_must_run_the_baseline_route(self):
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)],
                          [leg("a1", smol="qwen3.8-uncensored:latest", post=8.38),
                           leg("a2", smol="qwen3.8-uncensored:latest", post=8.44)])
        self.assertTrue(any("did not run ollama/qwen3.8:27b-mlx" in p for p in receipt["problems"]))

    def test_baseline_legs_under_the_park_alias_are_the_baseline_route(self):
        # r2 pins: smol_model localbench-parked:5642e97495e1, smol_digest 5642e97495e1; calls served by the alias.
        base = [leg("ab_a1", smol=PARKED, digest=QWEN_DIGEST, post=8.38),
                leg("ab_a2", smol=PARKED, digest=QWEN_DIGEST, post=8.44)]
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)], base)
        self.assertEqual((receipt["verdict"]["compare"], receipt["problems"]), ("BETTER", []))
        # Pinned by route name but served under the alias: the same build, accepted.
        alias_served = {"memory-extract": {"calls": 3, "models": [PARKED]}}
        base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38, by_purpose=alias_served),
                leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, by_purpose=alias_served)]
        self.assertEqual(verdict([leg("b1", post=2.06), leg("b2", post=2.10)], base)["problems"], [])

    def test_the_route_name_on_another_digest_is_not_the_baseline(self):
        other = "dddddddddddd"
        base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                leg("a2", smol=QWEN, digest=other, post=8.44, by_purpose={"memory-extract": {"calls": 3,
                                                                                           "models": [QWEN]}})]
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)], base)
        self.assertTrue(any(p.startswith("baseline leg(s) a2 did not run ollama/qwen3.8:27b-mlx (digest 5642e97495e1")
                            for p in receipt["problems"]), receipt["problems"])
        # An alias-named leg on another digest may not borrow the baseline's names either.
        alias = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                 leg("a2", smol="x:1", digest=other, post=8.44,
                     by_purpose={"memory-extract": {"calls": 3, "models": [PARKED]}})]
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)], alias)
        self.assertTrue(any(p.startswith("baseline leg a2 memory calls served by") for p in receipt["problems"]))

    def test_candidate_legs_on_different_memory_builds_are_not_one_arm(self):
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10, digest="dddddddddddd")])
        self.assertTrue(any("different memory configurations" in p for p in receipt["problems"]))

    def test_one_leg_per_arm_has_no_noise_estimate(self):
        receipt = verdict([leg("b1", post=2.06)], baseline()[:1])
        self.assertEqual(receipt["verdict"]["compare"], "NOT_BETTER")
        self.assertTrue(any("A/A noise needs >= 2 legs" in p for p in receipt["problems"]))

    def test_a_contended_or_residency_unknown_leg_in_either_arm_voids_the_comparison(self):
        for flaw in ({"contended": True}, {"resident_unknown": 2}, {"resident_unknown": None}):
            for arm in ("candidate", "baseline"):
                with self.subTest(flaw=flaw, arm=arm):
                    c_flaw, b_flaw = (flaw, {}) if arm == "candidate" else ({}, flaw)
                    cands = [leg("b1", post=2.06), leg("b2", post=2.10, **c_flaw)]
                    base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                            leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, **b_flaw)]
                    receipt = verdict(cands, base)
                    self.assertEqual(receipt["verdict"]["compare"], "NONE")
                    self.assertEqual(receipt["run"]["memory"]["compare"]["computed"], "BETTER")
                    flawed = "b2" if arm == "candidate" else "a2"
                    self.assertTrue(any(p.startswith(f"comparison void: {arm} leg(s) {flawed}")
                                        for p in receipt["problems"]), receipt["problems"])
                    self.assertEqual(graded(receipt)[0], "BAD")

    def test_a_baseline_must_failure_voids_the_comparison(self):
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)],
                          [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                           leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, no_leak="VOID")])
        self.assertEqual(receipt["verdict"]["compare"], "NONE")
        self.assertIn("comparison void: baseline MUST FAIL: a2:mem.no_leak", receipt["problems"])

    def test_a_remote_memory_llm_is_not_a_local_candidate(self):
        receipt = verdict([leg("b1", mode="remote", post=2.06), leg("b2", mode="remote", post=2.10)])
        self.assertTrue(any(p.startswith("llmMode remote is not a local route") for p in receipt["problems"]))

    def test_memory_calls_must_be_served_by_the_pinned_memory_model(self):
        main_served = {"memory-extract": {"calls": 3, "models": ["main:1"]}}
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10, by_purpose=main_served)])
        self.assertTrue(any(p.startswith("candidate leg b2 memory calls served by ['main:1']")
                            for p in receipt["problems"]), receipt["problems"])
        unrecorded = {"memory-extract": {"calls": 3}}
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10, by_purpose=unrecorded)])
        self.assertTrue(any(p.startswith("candidate leg b2 records no models") for p in receipt["problems"]))
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)],
                          [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                           leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, by_purpose=main_served)])
        self.assertTrue(any(p.startswith("baseline leg a2 memory calls served by") for p in receipt["problems"]))

    def test_banked_receipt_legs_prove_like_summaries(self):
        # __main__._receipt_view drops `results` and keeps their details: a banked leg must read the same.
        def banked(summary):
            return cli._receipt_view({**summary, "run_dir": "runs/x",
                                      "system": {**summary["system"], "before": {"live": {}, "host": {}}}})
        cands = [banked(leg("b1", post=2.06)), banked(leg("b2", post=2.10))]
        base = [banked(x) for x in baseline()]
        self.assertTrue(all("results" not in x for x in cands + base))
        receipt = verdict(cands, base)
        self.assertEqual((receipt["verdict"]["compare"], receipt["problems"]), ("BETTER", []))
        self.assertEqual(graded(receipt), ("PROVEN", ""))
        stripped = {**cands[1], "details": {k: v for k, v in cands[1]["details"].items() if k != "sess.memory"}}
        receipt = verdict([cands[0], stripped], base)
        self.assertTrue(any(p.startswith("candidate leg b2 records no models") and p.endswith("refused")
                            for p in receipt["problems"]), receipt["problems"])
        no_mode = {**cands[1], "details": {"sess.memory": cands[1]["details"]["sess.memory"]}}
        receipt = verdict([cands[0], no_mode], base)
        self.assertTrue(any("b2 record no single mnemopi llmMode" in p for p in receipt["problems"]))

    def test_a_tool_call_explosion_or_a_timeout_is_worse(self):
        # Baseline legs' largest mem.turn.max_tool_calls is 11: 33 is the bound, 34 explodes (2026-09-28: 65 vs 11).
        base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38, mem_tool_calls=9),
                leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, mem_tool_calls=11)]
        at_bound = verdict([leg("b1", post=2.06, mem_tool_calls=33), leg("b2", post=2.10)], base)
        self.assertEqual((at_bound["verdict"]["compare"], at_bound["problems"]), ("BETTER", []))
        for flaw, needle in (({"mem_tool_calls": 34}, "b2 mem.turn.max_tool_calls 34 > 3x"),
                             ({"sess_tool_calls": 4}, "b2 sess.turn.max_tool_calls 4 > 3x"),
                             ({"mem_timeouts": 1}, "b2 mem.turn.timeouts 1 > 0"),
                             ({"sess_timeouts": 1}, "b2 sess.turn.timeouts 1 > 0")):
            with self.subTest(flaw=flaw):
                receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10, **flaw)], base)
                self.assertEqual(receipt["verdict"]["compare"], "WORSE")
                self.assertTrue(any(p.startswith("tool-call loop gate failed") and needle in p
                                    for p in receipt["problems"]), receipt["problems"])

    def test_a_missing_loop_metric_is_a_problem_not_a_pass(self):
        for arm, flaw in (("candidate", {"sess_timeouts": None}), ("candidate", {"mem_tool_calls": None}),
                          ("baseline", {"sess_tool_calls": None})):
            with self.subTest(arm=arm, flaw=flaw):
                cands = [leg("b1", post=2.06), leg("b2", post=2.10, **(flaw if arm == "candidate" else {}))]
                base = [leg("a1", smol=QWEN, digest=QWEN_DIGEST, post=8.38),
                        leg("a2", smol=QWEN, digest=QWEN_DIGEST, post=8.44, **(flaw if arm == "baseline" else {}))]
                receipt = verdict(cands, base)
                self.assertTrue(any(p.startswith("tool-call loop gate unmeasured") for p in receipt["problems"]),
                                receipt["problems"])
                self.assertEqual(graded(receipt)[0], "BAD")

    def test_the_receipt_is_valid_and_shows_its_own_checks(self):
        receipt = verdict([leg("b1", post=2.06), leg("b2", post=2.10)])
        self.assertEqual(cli.validate_doc(receipt), [])
        run = receipt["run"]
        self.assertIn("run_dir", run)
        self.assertIsNone(run["run_dir"])
        self.assertRegex(run["provenance"]["created"], r"^\d{8}T\d{6}Z$")
        self.assertEqual(run["provenance"]["localbench_rev"], "abc1234")
        fingerprint = run["provenance"]["fingerprint"]
        self.assertEqual((fingerprint["backend"], fingerprint["model"], fingerprint["host_id"]),
                         ("ollama", "main:1", "host0"))
        self.assertEqual([(s["arm"], s["label"], s["run_dir"]) for s in fingerprint["legs"]],
                         [("candidate", "b1", "runs/b1"), ("candidate", "b2", "runs/b2"),
                          ("baseline", "a1", "runs/a1"), ("baseline", "a2", "runs/a2")])
        self.assertEqual({c: e["verdict"] for c, e in run["conformance"].items()},
                         dict.fromkeys(("memory.baseline_route", "memory.served_model", "memory.comparison_valid",
                                        "memory.loop_gate", "memory.llm_mode", "memory.main_ack_slips",
                                        "memory.co_resident"), "PASS"))
        self.assertEqual({c: e["level"] for c, e in run["conformance"].items() if e["level"] != "MUST"},
                         {"memory.main_ack_slips": "SHOULD", "memory.co_resident": "SHOULD"})
        # A pin the candidate legs disagree on is not the arm's: it stays out of the fingerprint.
        split = leg("b2", post=2.10)
        split["provenance"]["pins"]["host_id"] = "host1"
        fingerprint = verdict([leg("b1", post=2.06), split])["run"]["provenance"]["fingerprint"]
        self.assertNotIn("host_id", fingerprint)
        self.assertEqual(fingerprint["model"], "main:1")

    def test_each_failed_own_check_is_a_failing_must_case_with_its_values(self):
        other = [leg("a1", smol=QWEN, digest="dddddddddddd", post=8.38),
                 leg("a2", smol=QWEN, digest="dddddddddddd", post=8.44, contended=True)]
        cases = (
            ("memory.baseline_route", [leg("b1", post=2.06), leg("b2", post=2.10)], other, "off_route", ["a1", "a2"]),
            ("memory.comparison_valid", [leg("b1", post=2.06), leg("b2", post=2.10)], other, "void", None),
            ("memory.served_model", [leg("b1", post=2.06),
                                     leg("b2", post=2.10, by_purpose={"memory-extract": {"calls": 3,
                                                                                         "models": ["main:1"]}})],
             None, "problems", None),
            ("memory.loop_gate", [leg("b1", post=2.06), leg("b2", post=2.10, mem_timeouts=1)], None, "failures",
             ["b2 mem.turn.timeouts 1 > 0"]),
            ("memory.llm_mode", [leg("b1", mode="remote", post=2.06), leg("b2", mode="remote", post=2.10)], None,
             "candidate", "remote"),
        )
        for case, cands, base, key, want in cases:
            with self.subTest(case=case):
                receipt = verdict(cands, base)
                entry = receipt["run"]["conformance"][case]
                self.assertEqual((entry["level"], entry["verdict"]), ("MUST", "FAIL"))
                self.assertTrue(entry[key])
                if want is not None:
                    self.assertEqual(entry[key], want)
                self.assertEqual(cli.validate_doc(receipt), [])
                self.assertIn(f"FAIL {case} (MUST)", render.show(receipt, "r.json"))

    def bank(self, tmp: Path, *, break_receipt: bool) -> tuple[int, Path]:
        """cmd_memory_verdict --bank with the legs and module sha patched in; `break_receipt` drops what
        `localbench validate` needs (provenance.created)."""
        def produce(*args, **kwargs):
            receipt = memory_verdict(*args, **kwargs, rev="abc1234")
            if break_receipt:
                del receipt["run"]["provenance"]["created"]
            return receipt

        legs = {"cand": [leg("b1", post=2.06), leg("b2", post=2.10)], "base": baseline()}
        args = argparse.Namespace(candidate="cand", baseline="base", feature=FEATURE, bank="memverdict-test")
        with mock.patch.object(cli, "RECEIPTS", tmp), \
                mock.patch.object(cli, "_memory_legs", side_effect=lambda src: legs[src]), \
                mock.patch.object(cli, "_feature_module_sha", return_value=(MODULE_SHA, None)), \
                mock.patch.object(cli.workloads, "memory_verdict", side_effect=produce), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            rc = cli.cmd_memory_verdict(args)
        self.stderr = err.getvalue()
        return rc, tmp / "memverdict-test.json"

    def test_bank_refuses_an_invalid_receipt_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, path = self.bank(Path(tmp), break_receipt=True)
            self.assertEqual(rc, 1)
            self.assertFalse(path.exists())
            self.assertIn("INVALID", self.stderr)
            self.assertIn("provenance: no created", self.stderr)
            rc, path = self.bank(Path(tmp), break_receipt=False)
            self.assertEqual(rc, 0, self.stderr)
            self.assertEqual(cli.validate_doc(json.loads(path.read_text())), [])


if __name__ == "__main__":
    unittest.main()
