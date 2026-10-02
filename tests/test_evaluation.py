"""Behavioral validation campaigns keep exact-generation evidence and never hide missing cases."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def require_api(test: unittest.TestCase):
    try:
        from localbench.evaluation import CampaignError, EvaluationCampaign
        from localbench.workloads import score_e2e_case
    except ImportError as exc:
        test.fail(f"behavioral evaluation API is unavailable: {exc}")
    return CampaignError, EvaluationCampaign, score_e2e_case


def stdout_answer(text: str) -> str:
    return json.dumps({"type": "message_end", "message": {
        "role": "assistant", "content": [{"type": "text", "text": text}]}}) + "\n"


class CampaignIdentity(unittest.TestCase):
    def setUp(self):
        self.CampaignError, self.EvaluationCampaign, _ = require_api(self)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "runs" / "eval-case"
        self.identity = {"model_digest": "model-a", "omp_sha": "omp-a"}
        self.cases = {"ok": "input-a", "tool_read": "input-b"}

    def create(self):
        return self.EvaluationCampaign.create(self.path, root=self.root, identity=self.identity,
                                              cases=self.cases, profile="omp-e2e.v1")

    def run_trace(self):
        run = self.root / "runs" / "one-case"
        run.mkdir(parents=True)
        output = run / "e2e.ok.first.omp.jsonl"
        output.write_text(stdout_answer("OK"))
        return run, output

    def test_resume_refuses_a_changed_model_pin_before_reusing_a_pass(self):
        campaign = self.create()
        run, output = self.run_trace()
        campaign.record_case("ok", input_sha256="input-a", status="PASS", run_dir=run,
                             trace_files=[output])

        with self.assertRaisesRegex(self.CampaignError, "identity"):
            self.EvaluationCampaign.open(self.path, root=self.root,
                                         expected_identity={"model_digest": "model-b", "omp_sha": "omp-a"})

    def test_a_changed_trace_cannot_be_reused_as_a_completed_case(self):
        campaign = self.create()
        run, output = self.run_trace()
        campaign.record_case("ok", input_sha256="input-a", status="PASS", run_dir=run,
                             trace_files=[output])
        output.write_text(stdout_answer("WRONG"))

        with self.assertRaisesRegex(self.CampaignError, "trace"):
            campaign.completed_case("ok", input_sha256="input-a")

    def test_a_recorded_failure_cannot_be_overwritten_by_a_retry_pass(self):
        campaign = self.create()
        run, output = self.run_trace()
        campaign.record_case("ok", input_sha256="input-a", status="FAIL", run_dir=run,
                             trace_files=[output])

        with self.assertRaisesRegex(self.CampaignError, "already recorded"):
            campaign.record_case("ok", input_sha256="input-a", status="PASS", run_dir=run,
                                 trace_files=[output])
        self.assertEqual(campaign.completed_case("ok", input_sha256="input-a")["status"], "FAIL")

    def test_score_artifact_is_immutable_for_one_scorer_identity(self):
        campaign = self.create()
        scorer = "a" * 64
        base = {"identity_sha256": campaign.identity_sha256, "status": "PASS"}
        artifact = campaign.write_scores(scorer, base)
        self.assertEqual(artifact.name, f"{scorer}.json")

        with self.assertRaisesRegex(self.CampaignError, "differs"):
            campaign.write_scores(scorer, {**base, "status": "FAIL"})


class OfflineRescoring(unittest.TestCase):
    def test_rescore_uses_saved_outputs_and_exit_codes_without_opening_backend(self):
        _, EvaluationCampaign, _ = require_api(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runs = root / "runs"
            campaign_path = runs / "eval-offline"
            campaign = EvaluationCampaign.create(campaign_path, root=root,
                                                 identity={"model_digest": "model-a"},
                                                 cases={"ok": "input-a"}, profile="omp-e2e.v1")
            run_dir = runs / "one-case"
            run_dir.mkdir(parents=True)
            traces = []
            for attempt, returncode in (("first", 0), ("repeat", 7)):
                result = run_dir / f"e2e.ok.{attempt}.result.json"
                output = run_dir / f"e2e.ok.{attempt}.omp.jsonl"
                result.write_text(json.dumps({"task": "ok", "attempt": attempt, "returncode": returncode}))
                output.write_text(stdout_answer("OK"))
                traces.extend((result, output))
            campaign.record_case("ok", input_sha256="input-a", status="PASS", run_dir=run_dir,
                                 trace_files=traces)
            from localbench import __main__ as cli

            stdout = io.StringIO()
            with (mock.patch.object(cli, "ROOT", root), mock.patch.object(cli, "RUNS", runs),
                  mock.patch.object(cli, "open_backend", side_effect=AssertionError("offline command opened backend")),
                  contextlib.redirect_stdout(stdout)):
                rc = cli.main(["eval", "rescore", str(campaign_path)])

            result = json.loads(stdout.getvalue())
            self.assertEqual((rc, result["status"]), (1, "FAIL"))
            score = json.loads((root / result["score"]).read_text())
            self.assertEqual((score["cases"][0]["attempts"][1]["returncode"],
                              score["cases"][0]["attempts"][1]["ok"]), (7, False))

class E2EScoring(unittest.TestCase):
    def setUp(self):
        _, _, self.score_e2e_case = require_api(self)

    def test_both_attempts_must_pass_the_expected_answer_and_process_exit(self):
        good = stdout_answer("OK")
        wrong = stdout_answer("not OK")
        self.assertEqual(self.score_e2e_case("ok", [{"stdout": good, "returncode": 0},
                                                      {"stdout": good, "returncode": 0}])["status"], "PASS")
        bad_answer = self.score_e2e_case("ok", [{"stdout": good, "returncode": 0},
                                                  {"stdout": wrong, "returncode": 0}])
        bad_exit = self.score_e2e_case("ok", [{"stdout": good, "returncode": 0},
                                                {"stdout": good, "returncode": 1}])
        self.assertEqual((bad_answer["status"], bad_answer["passed"], bad_exit["status"], bad_exit["passed"]),
                         ("FAIL", 1, "FAIL", 1))

    def test_tool_case_requires_the_file_value_in_each_attempt(self):
        good = stdout_answer("4817")
        wrong = stdout_answer("4818")
        result = self.score_e2e_case("tool_read", [{"stdout": good, "returncode": 0},
                                                      {"stdout": wrong, "returncode": 0}])
        self.assertEqual((result["status"], result["attempts"][0]["answer"], result["attempts"][1]["ok"]),
                         ("FAIL", "4817", False))

    def test_tool_read_requires_the_file_to_be_read_successfully_not_just_named_in_an_answer(self):
        answer_only = stdout_answer("4817")
        started = json.dumps({"type": "tool_execution_start", "toolCallId": "call-1", "toolName": "read",
                              "args": {"path": "/tmp/localbench-e2e/answer.txt"}}) + "\n"
        finished = json.dumps({"type": "tool_execution_end", "toolCallId": "call-1", "toolName": "read",
                               "isError": False, "result": {"details": {"displayContent": {"text": "4817"}}}}) + "\n"
        good = started + finished + answer_only
        failed = started + finished.replace('"isError": false', '"isError": true') + answer_only
        wrong_file = started.replace("answer.txt", "other.txt") + finished + answer_only
        stale_value = started + finished.replace('"text": "4817"', '"text": "4818"') + answer_only
        wrong_call = started + finished.replace("call-1", "call-2") + answer_only

        for trace in (answer_only, failed, wrong_file, stale_value, wrong_call):
            with self.subTest(trace=trace[:90]):
                attempts = [{"stdout": trace, "returncode": 0}] * 2
                self.assertEqual(self.score_e2e_case("tool_read", attempts)["status"], "FAIL")
        self.assertEqual(self.score_e2e_case("tool_read", [{"stdout": good, "returncode": 0}] * 2)["status"],
                         "PASS")



def varied_trace(*, tool: str, path: str, answer: str, content: str | None = None,
                 failed: bool = False, call_id: str = "call-1", result_path: str | None = None) -> str:
    if tool == "edit":
        # omp 18.4.5 hashline shape (edit-700 trajectory): args carry only `input`, the
        # file is in the `[PATH#TAG]` header and the end event reports details.path.
        args = {"input": f"[{path}#B7F8]\nPUT 1.=1:\n+EXPECTED-synthetic"}
        result = {"content": [{"type": "text", "text": f"[{path}#9D0C]\n1:EXPECTED-synthetic"}],
                  "details": {"op": "update", "path": result_path or f"/isolated/trial/{path}",
                              "firstChangedLine": 1}}
    else:
        args = {"path": path}
        result = ({"details": {"displayContent": {"text": content}}} if tool == "read"
                  else {"content": [{"type": "text", "text": "File written"}]})
    start = {"type": "tool_execution_start", "toolCallId": call_id, "toolName": tool, "args": args}
    end = {"type": "tool_execution_end", "toolCallId": call_id, "toolName": tool,
           "isError": failed, "result": result}
    return "\n".join(map(json.dumps, (start, end))) + "\n" + stdout_answer(answer)


class VariedTrialScoring(unittest.TestCase):
    def setUp(self):
        from localbench.evaluation import make_varied_spec, score_varied_trial
        self.make_spec = make_varied_spec
        self.score = score_varied_trial
        self.cwd = Path("/isolated/trial")

    def grade(self, spec, stdout, *, target=None, decoy=None, returncode=0, timed_out=False):
        files = {spec["file"]: spec["initial"] if target is None else target,
                 spec["decoy_file"]: spec["decoy_initial"] if decoy is None else decoy}
        return self.score(spec, stdout=stdout, returncode=returncode, timed_out=timed_out,
                          final_files=files, cwd=self.cwd, wall_s=1.5)

    def test_specs_are_fresh_deterministic_and_do_not_leak_read_answer(self):
        read = self.make_spec(family="read", seed=91, phase="exploratory")
        edit = self.make_spec(family="edit", seed=91, phase="exploratory")
        heldout = self.make_spec(family="read", seed=91, phase="heldout")
        other = self.make_spec(family="read", seed=92, phase="exploratory")
        self.assertEqual(read, self.make_spec(family="read", seed=91, phase="exploratory"))
        self.assertEqual(len({read["initial"], edit["initial"], heldout["initial"], other["initial"]}), 4)
        self.assertNotIn(read["initial"].strip(), read["prompt"])
        self.assertNotIn("4817", read["initial"])
        self.assertNotEqual(edit["initial"], edit["expected"])
        self.assertNotEqual(edit["decoy_initial"], edit["initial"])
        self.assertNotEqual(edit["decoy_initial"], edit["expected"])
        self.assertEqual((read["file"], read["decoy_file"], read["grader_version"]),
                         ("target.txt", "decoy.txt", "varied.v2"))
        self.assertIn("read", read["allowed_tools"])
        self.assertTrue({"edit", "write"} & set(edit["allowed_tools"]))
        json.dumps([read, edit, heldout, other])

    def test_read_requires_correct_answer_and_successful_matching_read(self):
        spec = self.make_spec(family="read", seed=93, phase="exploratory")
        value = spec["initial"].strip()
        good = varied_trace(tool="read", path="target.txt", answer=value, content=value)
        result = self.grade(spec, good)
        self.assertEqual((result["status"], result["ok"], result["wall_s"]), ("PASS", True, 1.5))
        self.assertEqual(result["final_files"][spec["file"]], spec["initial"])
        for stdout in (stdout_answer(value),
                       varied_trace(tool="read", path="decoy.txt", answer=value, content=value),
                       varied_trace(tool="read", path="../target.txt", answer=value, content=value),
                       varied_trace(tool="read", path="target.txt", answer=value, content="stale"),
                       varied_trace(tool="read", path="target.txt", answer=value, content=value,
                                    failed=True),
                       varied_trace(tool="read", path="target.txt", answer="wrong", content=value)):
            with self.subTest(stdout=stdout[:100]):
                self.assertEqual(self.grade(spec, stdout)["status"], "FAIL")
        wrong_call = good.replace('"toolCallId": "call-1"', '"toolCallId": "call-2"', 1)
        self.assertEqual(self.grade(spec, wrong_call)["status"], "FAIL")
        self.assertEqual(self.grade(spec, good, target="modified")["status"], "FAIL")

    def test_edit_requires_transition_matching_tool_and_unchanged_decoy(self):
        spec = self.make_spec(family="edit", seed=94, phase="heldout")
        good = varied_trace(tool="write", path=str(self.cwd / "target.txt"), answer="done")
        self.assertEqual(self.grade(spec, good, target=spec["expected"])["status"], "PASS")
        self.assertEqual(self.grade(
            spec, varied_trace(tool="edit", path="target.txt", answer="done"),
            target=spec["expected"])["status"], "PASS")
        for stdout, target, decoy in (
            (stdout_answer("done"), spec["expected"], spec["decoy_initial"]),
            (varied_trace(tool="write", path="decoy.txt", answer="done"), spec["expected"],
             spec["decoy_initial"]),
            (good, spec["initial"], spec["decoy_initial"]),
            (good, spec["expected"], "modified decoy"),
            (varied_trace(tool="edit", path="target.txt", answer="done", failed=True),
             spec["expected"], spec["decoy_initial"]),
        ):
            with self.subTest(stdout=stdout[:90], target=target, decoy=decoy):
                result = self.grade(spec, stdout, target=target, decoy=decoy)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result["reasons"])
        already_correct = {**spec, "initial": spec["expected"]}
        self.assertEqual(self.grade(already_correct, good, target=spec["expected"])["status"], "FAIL")
        failed_extra = good + varied_trace(tool="read", path="decoy.txt", answer="done",
                                           failed=True, call_id="call-2")
        self.assertEqual(self.grade(spec, failed_extra, target=spec["expected"])["status"], "FAIL")

    def test_hashline_edit_is_graded_by_its_header_and_result_path(self):
        spec = self.make_spec(family="edit", seed=96, phase="heldout")
        good = varied_trace(tool="edit", path="target.txt", answer="done")
        result = self.grade(spec, good, target=spec["expected"])
        self.assertEqual((result["status"], result["reasons"]), ("PASS", []))
        reported = '"path": "/isolated/trial/target.txt", '
        self.assertEqual(good.count(reported), 1)
        missing_result_path = good.replace(reported, "")
        malformed_result_path = good.replace(reported, '"path": ["/isolated/trial/target.txt"], ')
        for stdout in (varied_trace(tool="edit", path="decoy.txt", answer="done"),
                       varied_trace(tool="edit", path="../trial/target.txt", answer="done",
                                    result_path="/isolated/trial/target.txt"),
                       varied_trace(tool="edit", path="target.txt", answer="done",
                                    result_path="/isolated/trial/decoy.txt"),
                       missing_result_path, malformed_result_path):
            with self.subTest(stdout=stdout[:120]):
                self.assertEqual(self.grade(spec, stdout, target=spec["expected"])["status"], "FAIL")
        v1_spec = {**spec, "grader_version": "varied.v1"}
        self.assertEqual(self.grade(v1_spec, good, target=spec["expected"])["status"], "ERROR")

    def test_failed_process_and_broken_infrastructure_never_pass(self):
        spec = self.make_spec(family="read", seed=95, phase="exploratory")
        value = spec["initial"].strip()
        good = varied_trace(tool="read", path="target.txt", answer=value, content=value)
        self.assertEqual(self.grade(spec, good, timed_out=True)["status"], "FAIL")
        self.assertEqual(self.grade(spec, good, returncode=3)["status"], "FAIL")
        self.assertEqual(self.grade(spec, good, returncode=None)["status"], "ERROR")
        broken = self.score(spec, stdout="{bad json\n", returncode=0, timed_out=False,
                            final_files={spec["file"]: spec["initial"],
                                         spec["decoy_file"]: spec["decoy_initial"]},
                            cwd=self.cwd, wall_s=1.5)
        self.assertEqual(broken["status"], "ERROR")
        unreadable_result = good.replace('"displayContent": {"text":', '"displayContent": {"missing":')
        self.assertEqual(self.grade(spec, unreadable_result)["status"], "ERROR")
        missing = self.score(spec, stdout=good, returncode=0, timed_out=False, final_files={},
                             cwd=self.cwd, wall_s=1.5)
        self.assertEqual(missing["status"], "ERROR")
        boolean_wall = self.score(spec, stdout=good, returncode=0, timed_out=False,
                                  final_files={spec["file"]: spec["initial"],
                                               spec["decoy_file"]: spec["decoy_initial"]},
                                  cwd=self.cwd, wall_s=True)
        self.assertEqual(boolean_wall["status"], "ERROR")

class OfflineVariedRescoring(unittest.TestCase):
    def test_saved_pass_labels_cannot_hide_wrong_read_answer_and_success_wall_is_conditional(self):
        from localbench import __main__ as cli
        from localbench.evaluation import EvaluationCampaign, canonical_sha256, make_varied_spec

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            runs = root / "runs"
            specs = {f"{family}-{seed}": make_varied_spec(family=family, seed=seed, phase="heldout")
                     for seed in (11, 12) for family in ("read", "edit")}
            campaign = EvaluationCampaign.create(
                runs / "eval-varied-offline", root=root,
                identity={"varied_specs": specs, "profile": cli.VARIED_PROFILE},
                cases={name: canonical_sha256(spec) for name, spec in specs.items()},
                profile=cli.VARIED_PROFILE)
            for case_id, spec in specs.items():
                trial = runs / f"trial-{case_id}"
                attempt = trial / "attempt"
                work = attempt / "workspace"
                work.mkdir(parents=True)
                target = spec["expected"] if spec["family"] == "edit" else spec["initial"]
                (work / "target.txt").write_text(target)
                (work / "decoy.txt").write_text(spec["decoy_initial"])
                result = attempt / "result.json"
                result.write_text(json.dumps({"returncode": 0, "timed_out": False, "wall_s": 1.5}))
                snapshot = attempt / "final_state.json"
                snapshot.write_text(json.dumps({"target.txt": target, "decoy.txt": spec["decoy_initial"]}))
                stdout = attempt / "trajectory.jsonl"
                if spec["family"] == "read":
                    answer = "wrong" if case_id == "read-12" else spec["expected"].strip()
                    trace = varied_trace(tool="read", path="target.txt", answer=answer,
                                         content=spec["initial"].strip())
                else:
                    trace = varied_trace(tool="write", path="target.txt", answer="done")
                stdout.write_text(trace)
                # A stale persisted PASS is not the verdict: re-score from preserved observations.
                campaign.record_case(case_id, input_sha256=canonical_sha256(spec), status="PASS", run_dir=trial,
                                     trace_files=[result, snapshot, stdout, work / "target.txt", work / "decoy.txt"])
            output = io.StringIO()
            with mock.patch.object(cli, "ROOT", root), mock.patch.object(cli, "RUNS", runs), \
                    mock.patch.object(cli, "open_backend", side_effect=AssertionError("rescore opened model")), \
                    contextlib.redirect_stdout(output):
                rc = cli.cmd_eval_rescore(type("Args", (), {"campaign": str(campaign.path)})())
            self.assertEqual(rc, 1)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "FAIL")
            score = json.loads((root / result["score"]).read_text())
            self.assertEqual((score["success"]["passed"], score["success"]["graded"]), (3, 4))
            by_id = {row["case_id"]: row for row in score["cases"]}
            self.assertEqual(by_id["read-12"]["status"], "FAIL")
            self.assertIn("answer does not match", " ".join(by_id["read-12"]["reasons"]))
            self.assertEqual(score["success"]["wall_s_on_success"]["median"], 1.5)


if __name__ == "__main__":
    unittest.main()
