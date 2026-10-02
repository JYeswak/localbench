"""The offline subprocess fixture exercises the same workspace and evidence capture as a real omp trial."""

import argparse
import contextlib
import io
import json
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as cli
from localbench import proxy, varied
from localbench.evaluation import CampaignError, EvaluationCampaign, canonical_sha256, make_varied_spec, score_varied_trial


class VariedTrialRunner(unittest.TestCase):
    def test_edit_preserves_distinct_prestate_real_poststate_and_matched_tool_trace(self):
        spec = make_varied_spec(family="edit", seed=221, phase="heldout")
        self.assertNotEqual(spec["initial"], spec["expected"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_omp = root / "omp-fixture"
            fake_omp.write_text("#!/usr/bin/env python3\n"
                                "import json\n"
                                "from pathlib import Path\n"
                                "target = Path('target.txt')\n"
                                f"target.write_text({spec['expected']!r})\n"
                                "print(json.dumps({'type': 'tool_execution_start', 'toolCallId': 'write-1', "
                                "'toolName': 'write', 'args': {'path': 'target.txt'}}))\n"
                                "print(json.dumps({'type': 'tool_execution_end', 'toolCallId': 'write-1', "
                                "'toolName': 'write', 'isError': False, 'result': {'ok': True}}))\n")
            fake_omp.chmod(0o700)
            run_dir = root / "runs" / "trials" / "edit-221" / "attempt"
            with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                    mock.patch.object(varied, "child_env", return_value=os.environ.copy()):
                attempt, traces = varied.run_trial(spec, "offline-fixture", run_dir)
            before = json.loads((run_dir / "initial_state.json").read_text())
            after = json.loads((run_dir / "final_state.json").read_text())
            self.assertEqual(before["target.txt"], spec["initial"])
            self.assertEqual(after["target.txt"], spec["expected"])
            self.assertEqual(after["decoy.txt"], spec["decoy_initial"])
            self.assertEqual((run_dir / "workspace" / "target.txt").read_text(), spec["expected"])
            self.assertEqual(attempt["returncode"], 0)
            self.assertTrue(all(trace.is_file() for trace in traces))
            self.assertIn("--tools=read,edit,write", attempt["command"])
            grade = score_varied_trial(spec, stdout=(run_dir / "trajectory.jsonl").read_text(),
                                       returncode=attempt["returncode"], timed_out=attempt["timed_out"],
                                       final_files=after, cwd=attempt["cwd"], wall_s=attempt["wall_s"])
            self.assertEqual((grade["status"], grade["reasons"]), ("PASS", []))
            campaign = EvaluationCampaign.create(
                root / "runs" / "campaign", root=root,
                identity={"varied_specs": {"edit-221": spec}, "profile": cli.VARIED_PROFILE},
                cases={"edit-221": canonical_sha256(spec)}, profile=cli.VARIED_PROFILE)
            campaign.record_case("edit-221", input_sha256=canonical_sha256(spec), status="PASS",
                                 run_dir=run_dir.parent, trace_files=traces)
            score, artifact = cli._evaluation_score_varied(campaign)
            self.assertEqual((score["status"], score["success"]["passed"]), ("PASS", 1))
            self.assertTrue(artifact.is_file())


    def test_timeout_stops_child_before_final_workspace_state_is_recorded(self):
        spec = make_varied_spec(family="edit", seed=620, phase="heldout")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_omp = root / "omp-fixture"
            # Timings scale together: the fake must start and write `spawned` well inside the timeout even on a
            # loaded machine (timeout_s=1 flaked twice under a live run, 2026-10-01), and the grandchild's write
            # lands after it unless the timeout killed the process group.
            late_write = ("import time; from pathlib import Path; time.sleep(6); "
                          "Path('target.txt').write_text('late mutation')")
            fake_omp.write_text(
                "#!/usr/bin/env python3\n"
                "import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {late_write!r}], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "open('spawned', 'w').write('yes')\n"
                "time.sleep(30)\n")
            fake_omp.chmod(0o700)
            with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                    mock.patch.object(varied, "child_env", return_value=os.environ.copy()):
                attempt, _ = varied.run_trial(spec, "offline-fixture", root / "attempt", timeout_s=4)
            workspace = attempt["cwd"]
            self.assertTrue((workspace / "spawned").is_file())
            self.assertTrue(attempt["timed_out"])
            self.assertEqual(attempt["final_files"]["target.txt"], spec["initial"])
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and (workspace / "target.txt").read_text() == spec["initial"]:
                time.sleep(0.02)
            self.assertEqual((workspace / "target.txt").read_text(), spec["initial"],
                             "timed-out omp left a child that changed the recorded workspace")


    def test_timeout_records_fail_when_detached_tool_keeps_output_pipes_open(self):
        spec = make_varied_spec(family="edit", seed=625, phase="heldout")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_omp = root / "omp-fixture"
            fake_omp.write_text(
                "#!/usr/bin/env python3\n"
                "import subprocess, sys, time\n"
                "from pathlib import Path\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
                "start_new_session=True)\n"
                "Path('child.pid').write_text(str(child.pid))\n"
                "time.sleep(30)\n")
            fake_omp.chmod(0o700)
            run_dir = root / "attempt"
            try:
                with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                        mock.patch.object(varied, "child_env", return_value=os.environ.copy()):
                    attempt, _ = varied.run_trial(spec, "offline-fixture", run_dir, timeout_s=1)
                self.assertTrue(attempt["timed_out"])
                self.assertEqual(score_varied_trial(
                    spec, stdout=attempt["stdout"], returncode=attempt["returncode"],
                    timed_out=attempt["timed_out"], final_files=attempt["final_files"],
                    cwd=attempt["cwd"], wall_s=attempt["wall_s"])["status"], "FAIL")
            finally:
                pid_file = run_dir / "workspace" / "child.pid"
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def test_rescore_refuses_a_hashed_snapshot_that_disagrees_with_the_actual_workspace(self):
        spec = make_varied_spec(family="edit", seed=621, phase="heldout")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_omp = root / "omp-fixture"
            fake_omp.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "from pathlib import Path\n"
                f"Path('target.txt').write_text({spec['expected']!r})\n"
                "print(json.dumps({'type': 'tool_execution_start', 'toolCallId': 'write-1', "
                "'toolName': 'write', 'args': {'path': 'target.txt'}}))\n"
                "print(json.dumps({'type': 'tool_execution_end', 'toolCallId': 'write-1', "
                "'toolName': 'write', 'isError': False, 'result': {'ok': True}}))\n")
            fake_omp.chmod(0o700)
            run_dir = root / "runs" / "trial"
            with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                    mock.patch.object(varied, "child_env", return_value=os.environ.copy()):
                attempt, traces = varied.run_trial(spec, "offline-fixture", run_dir / "attempt")
            self.assertEqual(attempt["final_files"]["target.txt"], spec["expected"])
            (attempt["cwd"] / "target.txt").write_text("late mutation", encoding="utf-8")
            campaign = EvaluationCampaign.create(
                root / "runs" / "campaign", root=root,
                identity={"varied_specs": {"edit-621": spec}, "profile": cli.VARIED_PROFILE},
                cases={"edit-621": canonical_sha256(spec)}, profile=cli.VARIED_PROFILE)
            campaign.record_case("edit-621", input_sha256=canonical_sha256(spec), status="PASS",
                                 run_dir=run_dir, trace_files=traces)
            with self.assertRaisesRegex(CampaignError, "snapshot differs from workspace"):
                cli._evaluation_score_varied(campaign)


    def test_detached_late_writer_cannot_turn_a_timeout_into_unscorable_evidence(self):
        spec = make_varied_spec(family="edit", seed=624, phase="heldout")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_omp = root / "omp-fixture"
            late_write = ("import time; from pathlib import Path; time.sleep(1.5); "
                          "Path('target.txt').write_text('late mutation')")
            fake_omp.write_text(
                "#!/usr/bin/env python3\n"
                "import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {late_write!r}], start_new_session=True, "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
                "time.sleep(30)\n")
            fake_omp.chmod(0o700)
            run_dir = root / "runs" / "trial"
            with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                    mock.patch.object(varied, "child_env", return_value=os.environ.copy()):
                attempt, traces = varied.run_trial(spec, "offline-fixture", run_dir / "attempt", timeout_s=1)
            self.assertTrue(attempt["timed_out"])
            campaign = EvaluationCampaign.create(
                root / "runs" / "campaign", root=root,
                identity={"varied_specs": {"edit-624": spec}, "profile": cli.VARIED_PROFILE},
                cases={"edit-624": canonical_sha256(spec)}, profile=cli.VARIED_PROFILE)
            campaign.record_case("edit-624", input_sha256=canonical_sha256(spec), status="FAIL",
                                 run_dir=run_dir, trace_files=traces)
            target = attempt["cwd"] / "target.txt"
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and target.read_text() == spec["initial"]:
                time.sleep(0.02)
            self.assertEqual(target.read_text(), "late mutation",
                             "detached tool did not exercise the late-write boundary")
            score, _ = cli._evaluation_score_varied(campaign)
            self.assertEqual((score["status"], score["failed"]), ("FAIL", 1))

    def test_snapshot_does_not_read_a_symlink_when_path_metadata_is_stale(self):
        spec = make_varied_spec(family="edit", seed=622, phase="heldout")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside.txt"
            outside.write_text("external sentinel", encoding="utf-8")
            fake_omp = root / "omp-fixture"
            fake_omp.write_text("#!/usr/bin/env python3\n"
                                "from pathlib import Path\n"
                                "Path('target.txt').unlink()\n"
                                f"Path('target.txt').symlink_to({str(outside)!r})\n")
            fake_omp.chmod(0o700)
            run_dir = root / "attempt"
            target = run_dir / "workspace" / "target.txt"
            real_is_symlink = Path.is_symlink

            def stale_check(path):
                return False if path == target else real_is_symlink(path)

            with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                    mock.patch.object(varied, "child_env", return_value=os.environ.copy()), \
                    mock.patch.object(Path, "is_symlink", stale_check):
                attempt, traces = varied.run_trial(spec, "offline-fixture", run_dir)
            self.assertEqual(attempt["returncode"], 0)
            self.assertNotIn("target.txt", attempt["final_files"])
            self.assertNotIn(target, traces)
            self.assertNotIn("external sentinel", (run_dir / "final_state.json").read_text())

    def test_rescore_refuses_an_internal_symlink_even_if_path_metadata_is_stale(self):
        spec = make_varied_spec(family="edit", seed=623, phase="heldout")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_omp = root / "omp-fixture"
            fake_omp.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "from pathlib import Path\n"
                f"Path('target.txt').write_text({spec['expected']!r})\n"
                "print(json.dumps({'type': 'tool_execution_start', 'toolCallId': 'write-1', "
                "'toolName': 'write', 'args': {'path': 'target.txt'}}))\n"
                "print(json.dumps({'type': 'tool_execution_end', 'toolCallId': 'write-1', "
                "'toolName': 'write', 'isError': False, 'result': {'ok': True}}))\n")
            fake_omp.chmod(0o700)
            run_dir = root / "runs" / "trial"
            with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                    mock.patch.object(varied, "child_env", return_value=os.environ.copy()):
                attempt, traces = varied.run_trial(spec, "offline-fixture", run_dir / "attempt")
            campaign = EvaluationCampaign.create(
                root / "runs" / "campaign", root=root,
                identity={"varied_specs": {"edit-623": spec}, "profile": cli.VARIED_PROFILE},
                cases={"edit-623": canonical_sha256(spec)}, profile=cli.VARIED_PROFILE)
            campaign.record_case("edit-623", input_sha256=canonical_sha256(spec), status="PASS",
                                 run_dir=run_dir, trace_files=traces)
            target = attempt["cwd"].resolve() / "target.txt"
            alternate = run_dir / "alternate.txt"
            alternate.write_text(spec["expected"], encoding="utf-8")
            target.unlink()
            target.symlink_to(alternate)
            real_is_symlink = Path.is_symlink

            def stale_check(path):
                return False if path == target else real_is_symlink(path)

            self.assertTrue(real_is_symlink(target))
            with mock.patch.object(Path, "is_symlink", stale_check):
                self.assertIsNotNone(campaign.completed_case("edit-623"))
                with self.assertRaisesRegex(CampaignError, "final snapshot differs from workspace: target.txt"):
                    cli._evaluation_score_varied(campaign)

    def test_seeded_read_trial_is_rescored_from_workspace_trace_and_campaign_record(self):
        cases = {
            "read-222": (make_varied_spec(family="read", seed=222, phase="heldout"), "target.txt", "PASS"),
            "read-wrong-file-223": (
                make_varied_spec(family="read", seed=223, phase="heldout"), "decoy.txt", "FAIL"),
        }
        case_inputs = {case_id: canonical_sha256(spec) for case_id, (spec, _, _) in cases.items()}
        identity = {"varied_specs": {case_id: spec for case_id, (spec, _, _) in cases.items()},
                    "profile": cli.VARIED_PROFILE}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = EvaluationCampaign.create(
                root / "runs" / "campaign", root=root, identity=identity, cases=case_inputs,
                profile=cli.VARIED_PROFILE)
            fake_omp = root / "omp-fixture"

            for case_id, (spec, read_path, expected_status) in cases.items():
                fake_omp.write_text(
                    "#!/usr/bin/env python3\n"
                    "import json\n"
                    "from pathlib import Path\n"
                    f"read_path = {read_path!r}\n"
                    "read_text = Path(read_path).read_text(encoding='utf-8').removesuffix('\\n')\n"
                    "answer = Path('target.txt').read_text(encoding='utf-8').removesuffix('\\n')\n"
                    "events = [\n"
                    "    {'type': 'tool_execution_start', 'toolCallId': 'read-1', 'toolName': 'read', "
                    "'args': {'path': read_path}},\n"
                    "    {'type': 'tool_execution_end', 'toolCallId': 'read-1', 'toolName': 'read', "
                    "'isError': False, 'result': {'details': {'displayContent': {'text': read_text}}}},\n"
                    "    {'type': 'message_end', 'message': {'role': 'assistant', "
                    "'content': [{'type': 'text', 'text': answer}]}},\n"
                    "]\n"
                    "for event in events:\n"
                    "    print(json.dumps(event))\n")
                fake_omp.chmod(0o700)
                trial_dir = root / "runs" / "trials" / case_id
                with mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                        mock.patch.object(varied, "child_env", return_value=os.environ.copy()):
                    attempt, traces = varied.run_trial(spec, "offline-fixture", trial_dir / "attempt")

                initial = json.loads((trial_dir / "attempt" / "initial_state.json").read_text())
                final = json.loads((trial_dir / "attempt" / "final_state.json").read_text())
                self.assertEqual(initial, {"target.txt": spec["initial"], "decoy.txt": spec["decoy_initial"]})
                self.assertEqual(final, initial)
                self.assertEqual((attempt["cwd"] / "target.txt").read_bytes(), spec["initial"].encode())
                self.assertEqual((attempt["cwd"] / "decoy.txt").read_bytes(), spec["decoy_initial"].encode())
                self.assertEqual(attempt["returncode"], 0)
                self.assertFalse(attempt["timed_out"])
                self.assertIn("--tools=read", attempt["command"])

                events = [json.loads(line) for line in
                          (trial_dir / "attempt" / "trajectory.jsonl").read_text().splitlines()]
                tool_start, tool_end, assistant_end = events
                self.assertEqual(tool_start["args"]["path"], read_path)
                self.assertEqual(tool_end["toolCallId"], tool_start["toolCallId"])
                self.assertFalse(tool_end["isError"])
                read_value = spec["initial"] if read_path == "target.txt" else spec["decoy_initial"]
                self.assertEqual(tool_end["result"]["details"]["displayContent"]["text"],
                                 read_value.removesuffix("\n"))
                self.assertEqual(assistant_end["message"]["content"][0]["text"],
                                 spec["expected"].removesuffix("\n"))

                grade = score_varied_trial(
                    spec, stdout=attempt["stdout"], returncode=attempt["returncode"],
                    timed_out=attempt["timed_out"], final_files=attempt["final_files"],
                    cwd=attempt["cwd"], wall_s=attempt["wall_s"])
                self.assertEqual(grade["status"], expected_status)
                if expected_status == "FAIL":
                    self.assertIn("no matching successful read of planted text", grade["reasons"])

                row = campaign.record_case(
                    case_id, input_sha256=case_inputs[case_id], status=grade["status"],
                    run_dir=trial_dir, trace_files=traces)
                self.assertEqual(row["input_sha256"], case_inputs[case_id])
                self.assertEqual(row["identity_sha256"], campaign.identity_sha256)
                self.assertEqual(row["run_dir"], trial_dir.relative_to(root).as_posix())
                self.assertTrue({"attempt/initial_state.json", "attempt/trajectory.jsonl",
                                 "attempt/final_state.json", "attempt/workspace/target.txt",
                                 "attempt/workspace/decoy.txt"} <= set(row["trace_sha256"]))
                self.assertEqual(campaign.completed_case(case_id, input_sha256=case_inputs[case_id]), row)

            score, artifact = cli._evaluation_score_varied(campaign)
            by_id = {case["case_id"]: case for case in score["cases"]}
            self.assertEqual((score["status"], score["expected"], score["completed"],
                              score["passed"], score["failed"]), ("FAIL", 2, 2, 1, 1))
            self.assertEqual(by_id["read-222"]["status"], "PASS")
            self.assertEqual(by_id["read-wrong-file-223"]["status"], "FAIL")
            self.assertEqual(score["identity_sha256"], campaign.identity_sha256)
            self.assertTrue(artifact.is_file())



class VariedCampaignFailures(unittest.TestCase):
    def test_no_model_call_and_post_attempt_error_keep_distinct_evidence(self):
        class Backend:
            name = "ollama"
            base_url = "http://127.0.0.1:11299/v1"

            def fingerprint(self, model):
                return {"loaded_context": 4096}

            def isolate(self, model):
                return []

        class Sampler:
            def __init__(self, *args, **kwargs):
                self.contention = ["synthetic foreign GPU"]
                self.series = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def summary(self):
                return {"resident_unknown_samples": 0}

        class SilentProxy:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fake_omp = root / "fake-omp"
            fake_omp.write_text("#!/usr/bin/env python3\nprint('synthetic process; no model request')\n")
            fake_omp.chmod(0o700)
            host = {"host": {"host_id": "synthetic", "macos_build": "synthetic"}}
            args = argparse.Namespace(trials=2, seed=221, phase="heldout", resume=None,
                                      backend="ollama:synthetic", mem_config=cli.CHILD_CONFIG,
                                      server_arg=None, wait_idle=0, json=False, dry_run=False, explain=False)
            output = io.StringIO()
            with mock.patch.object(cli, "ROOT", root), mock.patch.object(cli, "RUNS", root / "runs"), \
                    mock.patch.object(cli.park, "parked_now", return_value=[{"name": "synthetic"}]), \
                    mock.patch.object(cli, "preflight", return_value={"problems": []}), \
                    mock.patch.object(cli, "open_backend", return_value=contextlib.nullcontext((Backend(), "synthetic"))), \
                    mock.patch.object(cli.sysstats, "snapshot", return_value=host) as snapshot, \
                    mock.patch.object(cli.sysstats, "Sampler", Sampler), \
                    mock.patch.object(cli, "run_pins", return_value={"host_id": "synthetic", "model": "synthetic"}), \
                    mock.patch.object(cli.golden, "pin_diff", return_value=[]), \
                    mock.patch.object(cli, "ensure_localbench_model"), mock.patch.object(cli, "_rev", return_value="synthetic"), \
                    mock.patch.object(varied, "omp_bin", return_value=str(fake_omp)), \
                    mock.patch.object(varied, "child_env", return_value=os.environ.copy()), \
                    mock.patch.object(proxy, "Proxy", SilentProxy), contextlib.redirect_stdout(output):
                self.assertEqual(cli.cmd_eval_varied(args), 1)
                first = json.loads(output.getvalue())
                score = json.loads((root / first["score"]).read_text())
                self.assertEqual((score["error"], score["void"]), (4, 0))
                system = json.loads(next((root / first["campaign"]).rglob("system.json")).read_text())
                self.assertEqual((system["status"], system["no_model_call"], system["contention"]),
                                 ("ERROR", True, ["synthetic foreign GPU"]))

                calls = [0]

                def fail_after_attempt():
                    calls[0] += 1
                    if calls[0] % 2 == 0:
                        raise OSError("synthetic after-state failure")
                    return host

                snapshot.side_effect = fail_after_attempt
                args.seed += 2
                output.seek(0)
                output.truncate(0)
                self.assertEqual(cli.cmd_eval_varied(args), 1)
                second = json.loads(output.getvalue())
                case = json.loads(next((root / second["campaign"] / "cases").glob("*.json")).read_text())
                self.assertEqual(case["status"], "ERROR")
                self.assertTrue({"attempt/initial_state.json", "attempt/final_state.json",
                                 "attempt/trajectory.jsonl", "attempt/result.json",
                                 "attempt/workspace/target.txt", "attempt/workspace/decoy.txt",
                                 "infrastructure-error.json"} <= set(case["trace_sha256"]))


if __name__ == "__main__":
    unittest.main()
