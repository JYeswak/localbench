"""Interval math behind the call splits (busy seconds, memory-LLM overlap with main calls) and the omp child flags."""

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench.client import Sample
from localbench.render import CLIP
from localbench.sysstats import omp_client_identity
from localbench.workloads import (
    MEM_ROUNDS,
    THINK_MAX_TOKENS,
    THINK_TASKS,
    Ctx,
    _busy_s,
    _leaked,
    _overlap_s,
    _rel,
    answer_call,
    child_flags,
    e2e,
    final_answer,
    mem,
    prompt_token_verdict,
    recorded_answers,
    replay,
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

    def test_tool_free_children_get_no_tools_at_all(self):
        flags = child_flags("m", tools=False)
        self.assertIn("--no-tools", flags)
        self.assertFalse([f for f in flags if f.startswith("--tools")])
        self.assertTrue([f for f in child_flags("m") if f.startswith("--tools=")])


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


class MemPreMain(unittest.TestCase):
    """mem() must time its recall turns from the proxy log it writes: every call is tool-less (--no-tools), and a
    turn can make two answer calls (3 of 144 turns in runs/20260924T032536Z__ab_a1__…); the wait is to the first."""

    def test_pre_main_is_measured_to_the_first_answer_call_of_a_no_tools_turn(self):
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
            log = Path(tmp) / "mem_calls.jsonl"

            def omp_turn(argv, **kwargs):
                # What the proxy logs for one `omp -p --no-tools` turn: classifier, then two answer calls. Each
                # turn of a fact waits differently (plant 1.0 s, recall 0.2, control 3.0, derail 2.0), so pre_main
                # taken from any turn but the recall turn misses the band. The classifier starts 1 s into the turn:
                # omp's own startup is part of the wait a user sees, so pre_main counts from the launch.
                now, prompt, control = time.time(), argv[2], "control-" in str(kwargs.get("cwd"))
                first = (1.0 if prompt.startswith("PLANT") else 3.0 if control else
                         0.2 if prompt.startswith("QUESTION") else 2.0)
                rows = [{"t": now, "t_start": now + dt, "tools": 0, "purpose": p, "prompt_tokens": 2000}
                        for dt, p in ((1.0, "auto-thinking"), (1.0 + first, "aux"), (1.3 + first, "aux"))]
                log.write_text("".join(json.dumps(r) + "\n" for r in rows))
                return subprocess.CompletedProcess([], 0, "", "")

            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=1024, mem_rounds=1)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=omp_turn), \
                    mock.patch("localbench.workloads._mem_facts",
                               return_value=[(n, f"PLANT {n}", f"QUESTION {n}", n) for n in "abc"]), \
                    mock.patch("localbench.memory.banks", return_value=[]), \
                    mock.patch("localbench.memory.remove_banks"):
                results = mem(ctx)
        pre = next(r for r in results if r.case == "mem.recall").metrics["pre_main_s"]
        self.assertEqual(pre.get("n"), 3, pre)
        self.assertTrue(1.15 < pre["value"] < 1.4, pre)


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
    """c677837's miss was the wiring, not the clip helper. Dropping answers.append or the detail still passes
    a test that only calls recorded_answers."""

    FAIL = "WRONG " + ("x" * 400)

    def test_a_pass_and_a_fail_reach_e2e_and_rel_details(self):
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

        ok_n = {"n": 0}

        def run(args, **kwargs):
            prompt = args[2]
            if "exactly: OK" in prompt:
                text = self.FAIL if ok_n["n"] % 2 == 0 else "OK"
                ok_n["n"] += 1
            else:
                text = "4817"
            return subprocess.CompletedProcess(args, 0, _omp_stdout(text), "")

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Ctx(backend=_Backend(), model="m", repeats=1, run_dir=Path(tmp), emit=lambda ev: None,
                      loaded_context=8192)
            with mock.patch("localbench.proxy.Proxy", _Proxy), \
                    mock.patch("localbench.workloads.ensure_localbench_model"), \
                    mock.patch("localbench.workloads.omp_bin", return_value="omp"), \
                    mock.patch("localbench.workloads.subprocess.run", side_effect=run):
                e2e_rows = e2e(ctx)
                rel_rows = _rel(ctx, cold=False)
        correct = next(r for r in e2e_rows if r.case == "e2e.ok.correct")
        self.assertEqual(correct.verdict, "FAIL")
        self.assertIn("OK", correct.detail["answers"])
        self.assertTrue(any(a.startswith("WRONG ") and "…[+" in a for a in correct.detail["answers"]))
        self.assertNotIn(self.FAIL, correct.detail["answers"])
        read = next(r for r in e2e_rows if r.case == "e2e.tool_read.correct")
        self.assertEqual(read.verdict, "PASS")
        self.assertEqual(read.detail["answers"], ["4817", "4817"])
        rel_ok = next(r for r in rel_rows if r.case == "rel.ok")
        self.assertIn("OK", rel_ok.detail["answers"])
        self.assertTrue(any("…[+" in a for a in rel_ok.detail["answers"]))
        self.assertEqual(len(rel_ok.detail["answers"]), 20)



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


class ReplayTokenWiring(unittest.TestCase):
    """replay() must pass the sidecar's backend into the verdict. Testing the helper alone missed that."""

    def case(self, backend: str, tokens: int):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            omp = root / "omp"
            omp.mkdir()
            (omp / "lean.json").write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
            (omp / "lean.meta.json").write_text(json.dumps(
                {"prompt_tokens": 10000, "backend_pins": {"backend": "ollama"}}))
            sample = Sample(backend=backend, model="m", label="x", prompt_tokens=tokens, ttft_s=0.1, text="OK")

            class Backend:
                name = backend

            ctx = Ctx(backend=Backend(), model="m", repeats=1, run_dir=root, emit=lambda ev: None,
                      loaded_context=200000)
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


class AnswerCall(unittest.TestCase):
    """mem runs --no-tools, so the answer call has tools=0. Filtering on tools voids pre_main_s."""

    def test_the_answer_is_the_aux_call_not_the_classifier(self):
        rows = (
            {"t_start": 10.0, "tools": 0, "purpose": "auto-thinking"},
            {"t_start": 12.5, "tools": 0, "purpose": "aux", "prompt_tokens": 2079},
        )
        got = answer_call(rows)
        self.assertEqual(got["purpose"], "aux")
        self.assertEqual(got["t_start"] - 9.0, 3.5)


    def test_a_tooled_main_call_is_still_the_answer(self):
        rows = [{"t_start": 1.0, "tools": 0, "purpose": "auto-thinking"},
                {"t_start": 2.0, "tools": 7, "purpose": "main"}]
        self.assertEqual(answer_call(rows)["purpose"], "main")

    def test_only_side_calls_have_no_answer(self):
        self.assertIsNone(answer_call([{"t_start": 1.0, "tools": 0, "purpose": "auto-thinking"}]))

if __name__ == "__main__":
    unittest.main()
