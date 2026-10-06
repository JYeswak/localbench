import hashlib
import json
import os
import plistlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.request import Request, urlopen

from localbench import mockomp, ompupdate

FAKE_OMP = r'''#!/usr/bin/env python3
"""Deterministic stand-in for the omp binary: speaks the RPC frames run_capture
waits for and replays fixed request shapes to the mock URL from the isolated
agent dir's models.yml. Adapts to the mock's configured judge choice and rank
by reading them back from its own classification responses."""
import json
import os
import sys
import urllib.request


def post(base, path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def mock_base():
    with open(os.path.join(os.environ["PI_CODING_AGENT_DIR"], "models.yml")) as stream:
        for line in stream:
            if "baseUrl:" in line:
                return line.split("baseUrl:")[1].strip().removesuffix("/v1")
    raise SystemExit("fake omp found no baseUrl in the isolated models.yml")


def rpc(base):
    chat = "/v1/chat/completions"
    sys1 = "/v1/systemone"
    print(json.dumps({"type": "ready"}), flush=True)
    count = 0
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        if command.get("type") != "prompt":
            continue
        count += 1
        level = post(base, sys1, {"model": "judge", "questions": {
            "level": {"type": "choice", "instructions": "rank",
                      "criteria": {"low": "easy", "medium": "mid", "high": "hard"}}}})
        effort = level["answers"]["level"]["choice"]
        tools = [{"type": "function", "function": {"name": "find"}}]
        call = {"id": "call_fake_find", "type": "function",
                "function": {"name": "find", "arguments": "{}"}}
        if count == 1:
            post(base, chat, {"model": "main", "reasoning_effort": effort, "tools": tools,
                              "messages": [{"role": "user", "content": "one"}]})
            for question in ("e000", "p00", "p00"):
                post(base, sys1, {"model": "judge", "questions": {
                    question: {"type": "noul", "instructions": "rank", "criteria": []}}})
            scored = post(base, sys1, {"model": "judge", "questions": {
                "rank": {"type": "noul", "instructions": "rank", "criteria": []}}})
            rank = scored["answers"]["rank"]["noul"]
            post(base, chat, {"model": "main", "reasoning_effort": effort, "tools": tools, "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "content": f"1 hit(s) for fake ({rank:.2f}), strongest first",
                 "tool_call_id": "call_fake_find"}]})
        else:
            post(base, chat, {"model": "main", "reasoning_effort": effort, "tools": tools,
                              "messages": [{"role": "user", "content": "fake turn"}]})
        if count == 4:
            post(base, chat, {"model": "smol", "temperature": 0, "messages": [
                {"role": "system", "content": "fake extractor"},
                {"role": "user", "content": "fake transcript"}]})
        print(json.dumps({"id": command.get("id"), "type": "agent_end"}), flush=True)


def interactive(base):
    post(base, "/v1/chat/completions", {"model": "tiny", "messages": [
        {"role": "system", "content": "Write a ~5 word title inside the <title> tag."},
        {"role": "user", "content": "<user>\nfake first message\n</user>"}]})
    for line in sys.stdin:
        if "quit" in line:
            return


if "--version" in sys.argv:
    print("omp/0.0.0-fake")
elif "--mode" in sys.argv or any(arg.startswith("--mode=") for arg in sys.argv):
    rpc(mock_base())
else:
    interactive(mock_base())
'''


class OmpUpdate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ompupdate-"))
        self.cwd = self.tmp / "work"
        self.cwd.mkdir()
        self.agent = self.tmp / "isolated-agent"


    def test_interactive_quit_terminates_a_child_that_ignores_sigterm(self):
        import shutil
        import time
        shell = shutil.which("sh") or "/bin/sh"
        child = ompupdate._InteractiveTitle(
            [shell, "-c", "trap '' TERM; sleep 60"], self.cwd, self.tmp / "quit.stderr.log",
            {"PATH": "/usr/bin:/bin", "TERM": "xterm-256color"})
        started = time.monotonic()
        try:
            code = child.quit(grace=1)
        finally:
            wall = time.monotonic() - started
        self.assertIsNotNone(code)
        self.assertLess(wall, 15)

    def test_mock_server_records_raw_body_and_serves_compat_endpoints(self):
        with mockomp.MockOmpServer() as server:
            body = b' { "model": "test", "messages": [] } \n'
            reply = mockomp.post_json(server.url + "/v1/chat/completions", body)
            self.assertEqual(reply["choices"][0]["message"]["content"], mockomp.DEFAULT_REPLY)
            judge_body = (b'{"model":"test","questions":{"difficulty":{"type":"choice","instructions":"rank",'
                          b'"criteria":{"low":"easy","high":"hard"}}}}')
            judged = mockomp.post_json(server.url + "/v1/systemone", judge_body)
            mockomp.post_json(server.url + "/api/tags/show", b'{"name":"localbench/test"}')
            self.assertEqual([r.path for r in server.requests],
                             ["/v1/chat/completions", "/v1/systemone", "/api/tags/show"])
            self.assertEqual(server.requests[0].body, body)
            self.assertEqual(server.requests[1].body, judge_body)
            self.assertEqual(judged["answers"]["difficulty"]["choice"], "high")
    def test_mock_server_preserves_configured_systemone_rank(self):
        with mockomp.MockOmpServer(systemone_noul_score=0.87) as server:
            body = (b'{"model":"localbench/judge","questions":{"candidate":{"type":"noul",'
                    b'"instructions":"rank the candidate","criteria":[]}}}')
            response = mockomp.post_json(server.url + "/v1/systemone", body)
            self.assertEqual(response["answers"]["candidate"]["noul"], 0.87)
            self.assertEqual(server.requests[0].body, body)
    def test_mock_find_call_is_selected_when_other_tools_are_also_advertised(self):
        with mockomp.MockOmpServer() as server:
            body = json.dumps({
                "model": "localbench/main",
                "tools": [{"type": "function", "function": {"name": "read"}},
                          {"type": "function", "function": {"name": "find"}}],
                "messages": [{"role": "user", "content": "find relevant files"}],
            }, separators=(",", ":")).encode()
            response = mockomp.post_json(server.url + "/v1/chat/completions", body)
        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "find")


    def test_streaming_openai_response_uses_sse_and_records_exact_request(self):
        with mockomp.MockOmpServer(reply="graded") as server:
            body = b'{"model":"test","stream":true,"messages":[]}'
            request = Request(server.url + "/v1/chat/completions", data=body,
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=3) as response:
                stream = response.read()
            self.assertEqual(response.headers["Content-Type"], "text/event-stream")
            self.assertIn(b"data: [DONE]", stream)
            self.assertIn(b'"content":"graded"', stream)
            self.assertEqual(server.requests[0].body, body)

    def test_capture_rejects_a_non_loopback_mock_endpoint_before_omp_or_agent_access(self):
        fake = self.tmp / "fake-omp-loopback"
        fake.write_text(FAKE_OMP, encoding="utf-8")
        fake.chmod(0o755)
        previous = os.environ.get("LOCALBENCH_OMP")
        os.environ["LOCALBENCH_OMP"] = str(fake)
        try:
            with self.assertRaisesRegex(ValueError, "must use HTTP loopback"):
                ompupdate.run_capture("https://example.invalid/v1", self.agent, self.cwd)
        finally:
            if previous is None:
                del os.environ["LOCALBENCH_OMP"]
            else:
                os.environ["LOCALBENCH_OMP"] = previous
        self.assertFalse(self.agent.exists())

    def test_capture_refuses_an_unresolvable_omp_binary(self):
        previous = os.environ.get("LOCALBENCH_OMP")
        os.environ["LOCALBENCH_OMP"] = "/missing/omp"
        try:
            with mockomp.MockOmpServer() as server:
                with self.assertRaisesRegex(ompupdate.CaptureError, "missing or not executable"):
                    ompupdate.run_capture(server.url, self.agent, self.cwd)
        finally:
            if previous is None:
                del os.environ["LOCALBENCH_OMP"]
            else:
                os.environ["LOCALBENCH_OMP"] = previous
        self.assertFalse(self.agent.exists())

    def test_isolated_configuration_refuses_to_overwrite_a_nonempty_agent_directory(self):
        self.agent.mkdir()
        keep = self.agent / "keep"
        keep.write_text("untouched")
        with mockomp.MockOmpServer() as server:
            with self.assertRaisesRegex(ompupdate.CaptureError, "must be empty"):
                ompupdate._isolated_configuration(self.agent, server.url)
        self.assertEqual(keep.read_text(), "untouched")
    def test_isolated_environment_cannot_inherit_or_use_cloud_typesafe_credentials(self):
        parent_env = {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "cloud-openai",
            "TYPESAFE_API_KEY": "cloud-typesafe",
            "TYPESAFE_BASE_URL": "https://api.typesafe.ai",
            "PI_CODING_AGENT_DIR": "/Users/x/.omp/agent",
            "OMP_PROFILE": "user-profile",
        }
        env = ompupdate._isolated_environment(self.agent, "http://127.0.0.1:8123", parent_env)
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("OMP_PROFILE", env)
        self.assertEqual(env["PI_CODING_AGENT_DIR"], str(self.agent))
        self.assertEqual(env["TYPESAFE_BASE_URL"], "http://127.0.0.1:8123")
        self.assertEqual(env["TYPESAFE_API_KEY"], "localbench-no-network")

    def test_byte_and_module_identity_carry_only_matching_proofs_and_queue_stale_routes(self):
        baseline = {"main-lean": b'{"tools":[]}', "auto-thinking": b'{"level":"low"}'}
        current = {"main-lean": b'{"tools":[]}', "auto-thinking": b'{"level":"high"}'}
        proofs = {
            "main-lean": {"feature": "main-lean", "proof_suite": "replay"},
            "auto-thinking": {"feature": "auto-thinking", "proof_suite": "decision"},
        }
        old_shas = {"main-lean": "b" * 64, "auto-thinking": "a" * 64}
        new_shas = {"main-lean": "b" * 64, "auto-thinking": "a" * 64}
        result = ompupdate.compare_captures(baseline, current, proofs, old_shas, new_shas)
        self.assertEqual(result["main-lean"]["status"], "CARRIED")
        self.assertEqual(result["auto-thinking"]["status"], "STALE")
        self.assertEqual(result["auto-thinking"]["queue"], ["decision", "through-omp"])
        self.assertEqual(result["main-lean"]["label"], "carried forward (identical request)")

    def test_timing_counts_and_date_variance_still_carries(self):
        old = ('{"messages": [{"role": "system", "content": "Today: 2026-10-01"},'
               '{"role": "tool", "content": "1 hit(s) listed 1 · judged 1 · read 1 files (82B)'
               ' · 3 requests · 3 tokens · $0.0000 · 8ms wall / 5ms api"}],'
               '"usage": {"prompt_tokens": 100}}').encode("utf-8")
        new = ('{"messages": [{"role": "system", "content": "Today: 2026-10-02"},'
               '{"role": "tool", "content": "1 hit(s) listed 2 · judged 2 · read 2 files (91B)'
               ' · 4 requests · 5 tokens · $0.0000 · 14ms wall / 6ms api"}],'
               '"usage": {"prompt_tokens": 140}}').encode("utf-8")
        result = ompupdate.compare_captures(
            {"find-judgments": [old]}, {"find-judgments": [new]},
            {"find-judgments": {"feature": "find-judgments", "proof_suite": "decision"}},
            {"find-judgments": "a" * 64}, {"find-judgments": "a" * 64})
        self.assertEqual(result["find-judgments"]["status"], "CARRIED")

    def test_prompt_wording_change_stales_with_a_named_span(self):
        old = b'{"messages": [{"role": "user", "content": "the fixture marker is cobalt"}]}'
        new = b'{"messages": [{"role": "user", "content": "the fixture marker is crimson"}]}'
        result = ompupdate.compare_captures(
            {"memory-extraction": [old]}, {"memory-extraction": [new]},
            {"memory-extraction": {"feature": "mnemopi-extraction", "proof_suite": "mem"}},
            {"mnemopi-extraction": "a" * 64}, {"mnemopi-extraction": "a" * 64})
        self.assertEqual(result["memory-extraction"]["status"], "STALE")
        self.assertIn("offset", ompupdate._stale_reason(
            "memory-extraction", {"memory-extraction": [old]}, {"memory-extraction": [new]},
            {"memory-extraction": {"feature": "mnemopi-extraction", "proof_suite": "mem"}},
            {"mnemopi-extraction": "a" * 64}, {"mnemopi-extraction": "a" * 64}))

    def test_judge_instruction_wording_change_stales(self):
        old = (b'{"questions": {"level": {"type": "choice", "instructions": "Choose the reasoning effort",'
               b'"criteria": {"low": "easy", "high": "hard"}}}}')
        new = (b'{"questions": {"level": {"type": "choice", "instructions": "Pick the thinking budget",'
               b'"criteria": {"low": "easy", "high": "hard"}}}}')
        result = ompupdate.compare_captures(
            {"auto-thinking": [old]}, {"auto-thinking": [new]},
            {"auto-thinking": {"feature": "auto-thinking", "proof_suite": "decision"}},
            {"auto-thinking": "a" * 64}, {"auto-thinking": "a" * 64})
        self.assertEqual(result["auto-thinking"]["status"], "STALE")

    def test_baseline_stores_raw_and_normalized_mirror(self):
        target = self.tmp / "baseline.json"
        bodies = [b'{"note": "8ms wall / 3ms api"}']
        ompupdate.write_baseline(target, {"titles": bodies}, {"titles": "a" * 64}, "18.4.8")
        loaded = ompupdate.read_baseline(target)
        self.assertEqual(loaded["requests"], {"titles": bodies})
        self.assertEqual(loaded["normalized"], {"titles": ompupdate.normalize_requests(bodies)})
        self.assertEqual(loaded["normalizers"], [name for name, _, _ in ompupdate._NORMALIZERS])

    def test_matching_request_stales_when_the_feature_module_sha_changes(self):
        result = ompupdate.compare_captures(
            {"memory-extraction": b'{"messages":[]}'}, {"memory-extraction": b'{"messages":[]}'},
            {"memory-extraction": {"feature": "mnemopi-extraction", "proof_suite": "mem"}},
            {"mnemopi-extraction": "a" * 64}, {"mnemopi-extraction": "b" * 64})
        self.assertEqual(result["memory-extraction"]["status"], "STALE")
        self.assertEqual(result["memory-extraction"]["queue"], ["mem", "through-omp"])

    def test_missing_feature_module_sha_cannot_carry_a_proof(self):
        result = ompupdate.compare_captures(
            {"auto-thinking": b"same"}, {"auto-thinking": b"same"},
            {"auto-thinking": {"feature": "auto-thinking", "proof_suite": "decision"}},
            {"auto-thinking": None}, {"auto-thinking": None})
        self.assertEqual(result["auto-thinking"]["status"], "STALE")
        self.assertEqual(result["auto-thinking"]["queue"], ["decision", "through-omp"])

    def test_unregistered_feature_module_sha_cannot_carry_a_proof(self):
        result = ompupdate.compare_captures(
            {"main-lean": b"same"}, {"main-lean": b"same"},
            {"main-lean": {"feature": "main-lean", "proof_suite": "replay"}}, {}, {})
        self.assertEqual(result["main-lean"]["status"], "STALE")
        self.assertEqual(result["main-lean"]["queue"], ["replay", "through-omp"])


    def test_feature_module_shas_follow_localbench_features_registry_rows(self):
        package = self.tmp / "pi-coding-agent"
        module = package / "src" / "judge.ts"
        module.parent.mkdir(parents=True)
        module.write_bytes(b"module")
        rows = [{"feature": "auto-thinking", "omp_package": "pi-coding-agent", "omp_module": "src/judge.ts"}]
        self.assertEqual(ompupdate.feature_module_shas(rows, package),
                         {"auto-thinking": hashlib.sha256(b"module").hexdigest()})

    def test_baseline_round_trip_preserves_request_bytes(self):
        target = self.tmp / "baseline.json"
        captures = {"main-lean": [b'{ "tools": [] }', b'{"level": "low"}']}
        ompupdate.write_baseline(target, captures, {"auto-thinking": "a" * 64}, "18.4.5")
        loaded = ompupdate.read_baseline(target)
        self.assertEqual(loaded["requests"], captures)
        self.assertEqual(loaded["omp_version"], "18.4.5")

    def test_watch_plist_watches_only_installed_omp_package_json_and_runs_refresh(self):
        payload = plistlib.loads(ompupdate.render_watch_plist())
        self.assertEqual(payload["WatchPaths"], [ompupdate.OMP_PACKAGE_JSON])
        self.assertEqual(payload["ProgramArguments"], [ompupdate.DEFAULT_LOCALBENCH, "omp", "refresh"])
        self.assertEqual(payload["Label"], "dev.localbench.omp-update")

    def test_watch_plist_carries_launchd_environment_and_refresh_log(self):
        payload = plistlib.loads(ompupdate.render_watch_plist(omp_path="/x/omp"))
        environment = payload["EnvironmentVariables"]
        self.assertEqual(environment["LOCALBENCH_OMP"], "/x/omp")
        for directory in (".bun/bin", "/opt/homebrew/bin", "/usr/local/bin"):
            self.assertIn(directory, environment["PATH"])
        log = os.path.join(str(Path.home()), ".localbench", "omp-watch", "refresh.log")
        self.assertEqual(payload["StandardOutPath"], log)
        self.assertEqual(payload["StandardErrorPath"], log)

    def test_refresh_writes_baseline_then_carries_identical_fake_captures(self):
        fake = self.tmp / "fake-omp"
        fake.write_text(FAKE_OMP, encoding="utf-8")
        fake.chmod(0o755)
        baseline = self.tmp / "baseline.json"
        previous = os.environ.get("LOCALBENCH_OMP")
        os.environ["LOCALBENCH_OMP"] = str(fake)
        real_settle, ompupdate._settle_package = ompupdate._settle_package, lambda *a, **k: None
        real_shas, ompupdate.feature_module_shas = (ompupdate.feature_module_shas,
            lambda: {p.get("feature", n): "0" * 64 for n, p in ompupdate.CAPTURE_PROOFS.items()})
        try:
            first = ompupdate.refresh(baseline_path=baseline)
            second = ompupdate.refresh(baseline_path=baseline)
        finally:
            ompupdate._settle_package = real_settle
            ompupdate.feature_module_shas = real_shas
            if previous is None:
                del os.environ["LOCALBENCH_OMP"]
            else:
                os.environ["LOCALBENCH_OMP"] = previous
        self.assertTrue(baseline.is_file())
        self.assertEqual(first["omp_version"], "0.0.0-fake")
        self.assertEqual(set(first["outcomes"]), set(ompupdate.CAPTURE_PROOFS))
        recorded = set(ompupdate.read_baseline(baseline)["requests"])
        uncaptured = set(ompupdate.CAPTURE_PROOFS) - recorded
        self.assertEqual(uncaptured, {"recall-embeddings", "unexpected-stop"})
        for name, outcome in first["outcomes"].items():
            self.assertEqual(outcome["status"], "NEW", name)
            if name in uncaptured:
                suite = ompupdate.CAPTURE_PROOFS[name]["proof_suite"]
                self.assertEqual(outcome["queue"], ([suite] if suite != "-" else []) + ["through-omp"])
        for name, outcome in second["outcomes"].items():
            if name in uncaptured:
                self.assertEqual(outcome["status"], "NEW", name)
                suite = ompupdate.CAPTURE_PROOFS[name]["proof_suite"]
                self.assertEqual(outcome["queue"], ([suite] if suite != "-" else []) + ["through-omp"])
            else:
                self.assertEqual(outcome["status"], "CARRIED", name)
                self.assertEqual(outcome["reason"], "carried forward (identical request)", name)
                self.assertEqual(outcome["queue"], [])
        self.assertEqual(first["outcomes"]["titles"]["queue"], ["through-omp"])
        self.assertEqual(first["outcomes"]["memory-extraction"]["feature"], "mnemopi-extraction")

    def test_settle_returns_when_quiet_and_refuses_a_changing_package(self):
        package = self.tmp / "package.json"
        package.write_text("{}")
        ompupdate._settle_package(str(package), quiet_secs=0, deadline_secs=5)
        with self.assertRaisesRegex(ompupdate.CaptureError, "still changing"):
            ompupdate._settle_package(str(package), quiet_secs=60, deadline_secs=0)
        with self.assertRaisesRegex(ompupdate.CaptureError, "missing"):
            ompupdate._settle_package(str(self.tmp / "gone" / "package.json"))

    def test_refresh_retries_once_after_a_transient_capture_failure(self):
        calls = []
        canned = {name: [b'{"shape":"fixed"}'] for name in ompupdate.CAPTURE_PROOFS}

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("no matching event within 90 s")
            return canned

        previous, original = os.environ.get("LOCALBENCH_OMP"), ompupdate.run_capture
        os.environ["LOCALBENCH_OMP"] = sys.executable
        ompupdate.run_capture = flaky
        real_settle, ompupdate._settle_package = ompupdate._settle_package, lambda *a, **k: None
        real_shas, ompupdate.feature_module_shas = (ompupdate.feature_module_shas,
            lambda: {p.get("feature", n): "0" * 64 for n, p in ompupdate.CAPTURE_PROOFS.items()})
        try:
            report = ompupdate.refresh(baseline_path=self.tmp / "baseline.json")
        finally:
            ompupdate.run_capture = original
            ompupdate._settle_package = real_settle
            ompupdate.feature_module_shas = real_shas
            if previous is None:
                del os.environ["LOCALBENCH_OMP"]
            else:
                os.environ["LOCALBENCH_OMP"] = previous
        self.assertEqual(len(calls), 2)
        for name, outcome in report["outcomes"].items():
            self.assertEqual(outcome["status"], "NEW", name)

    def test_refresh_second_failure_names_both_attempts(self):
        def always_slow(*args, **kwargs):
            raise EOFError("omp exited rc=1")

        previous, original = os.environ.get("LOCALBENCH_OMP"), ompupdate.run_capture
        os.environ["LOCALBENCH_OMP"] = sys.executable
        ompupdate.run_capture = always_slow
        real_settle, ompupdate._settle_package = ompupdate._settle_package, lambda *a, **k: None
        try:
            with self.assertRaisesRegex(ompupdate.CaptureError, "attempt 1.*attempt 2"):
                ompupdate.refresh(baseline_path=self.tmp / "baseline.json")
        finally:
            ompupdate.run_capture = original
            ompupdate._settle_package = real_settle
            if previous is None:
                del os.environ["LOCALBENCH_OMP"]
            else:
                os.environ["LOCALBENCH_OMP"] = previous

    def test_refresh_marks_changed_embedding_module_stale_and_queues_mem(self):
        proofs = ompupdate.capture_proofs()
        capture_name = next((name for name, proof in proofs.items()
                             if proof.get("feature") == "recall-embeddings"), None)
        self.assertIsNotNone(capture_name, "recall-embeddings must be tracked by refresh")
        if capture_name is None:
            return
        old_captures = {name: [b'{"shape":"stable"}'] for name in proofs}
        old_shas = {proof["feature"]: "a" * 64 for proof in proofs.values()}
        new_shas = {**old_shas, "recall-embeddings": "b" * 64}
        baseline = self.tmp / "recall-baseline.json"
        ompupdate.write_baseline(baseline, old_captures, old_shas, "18.4.8")
        original_work = ompupdate.UPDATE_WORK_DIR
        ompupdate.UPDATE_WORK_DIR = self.tmp / "refresh-work"
        try:
            with (mock.patch.object(ompupdate, "_settle_package"),
                  mock.patch.object(ompupdate, "_omp_version", return_value="18.4.9"),
                  mock.patch("localbench.workloads.omp_bin", return_value="/bin/true"),
                  mock.patch("localbench.backends.sha16", return_value="c" * 16),
                  mock.patch.object(ompupdate, "run_capture", return_value=old_captures),
                  mock.patch.object(ompupdate, "feature_module_shas", return_value=new_shas)):
                result = ompupdate.refresh(baseline_path=baseline)
        finally:
            ompupdate.UPDATE_WORK_DIR = original_work
        outcome = result["outcomes"][capture_name]
        self.assertEqual(outcome["status"], "STALE")
        self.assertEqual(outcome["queue"], ["mem", "through-omp"])

    def test_capture_proofs_cover_every_feature_with_a_registered_proof_spec(self):
        rows = ompupdate.features.load()
        proof_dir = ompupdate.features.REGISTRY.parent / "proofs"
        required = {row["feature"] for row in rows if any(proof_dir.glob(f"{row['feature']}__*.json"))}
        mapped = {proof.get("feature", name) for name, proof in ompupdate.CAPTURE_PROOFS.items()}
        excluded = getattr(ompupdate, "CAPTURE_EXCLUSIONS", {})
        self.assertTrue(all(isinstance(reason, str) and reason.strip() for reason in excluded.values()))
        self.assertFalse(mapped & set(excluded), "a feature cannot be both captured and excluded")
        self.assertEqual(required - mapped - set(excluded), set(), "proof-bearing features must be captured or excluded")
        by_feature = {row["feature"]: row for row in rows}
        for name, proof in ompupdate.CAPTURE_PROOFS.items():
            self.assertEqual(proof["proof_suite"], by_feature[proof.get("feature", name)]["proof_suite"])


if __name__ == "__main__":
    unittest.main()
