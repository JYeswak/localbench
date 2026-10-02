"""Residency lease, route, profile rollback, and launchd boundary tests."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from localbench import gateway, omp_profiles, sysstats


class RoutePolicy(unittest.TestCase):
    def test_routes_only_registered_profiles_and_keeps_discovery_metadata_unleased(self):
        profiles = {"default", "claude"}
        self.assertEqual(
            (gateway.route_request("POST", "/omp-profile/claude/responses", profiles).profile,
             gateway.route_request("POST", "/omp-profile/claude/responses", profiles).upstream_path),
            ("claude", "/v1/responses"),
        )
        self.assertEqual(gateway.route_request("GET", "/api/tags", profiles).kind, "discovery")
        self.assertEqual(gateway.route_request("POST", "/api/show", profiles).kind, "discovery")
        for method, path in (("POST", "/omp-profile/not-installed/responses"),
                             ("GET", "/omp-profile/default/responses"),
                             ("POST", "/omp-profile/default/../api/generate"),
                             ("POST", "/api/generate")):
            with self.subTest(method=method, path=path), self.assertRaises(gateway.RouteError):
                gateway.route_request(method, path, profiles)

    def test_systemone_routes_forward_to_ollama_unchanged(self):
        bare = gateway.route_request("POST", "/v1/systemone", {"default", "claude"})
        self.assertEqual((bare.kind, bare.profile, bare.upstream_path),
                         ("inference", None, "/v1/systemone"))
        profiled = gateway.route_request("POST", "/omp-profile/claude/v1/systemone",
                                         {"default", "claude"})
        self.assertEqual((profiled.kind, profiled.profile, profiled.upstream_path),
                         ("inference", "claude", "/v1/systemone"))

    def test_unknown_inference_paths_still_404(self):
        for method, path in (("POST", "/v1/unknown"),
                             ("POST", "/omp-profile/claude/v1/unknown"),
                             ("POST", "/omp-profile/claude/systemone/extra"),
                             ("GET", "/v1/systemone")):
            with self.subTest(method=method, path=path), self.assertRaises(gateway.RouteError) as raised:
                gateway.route_request(method, path, {"default", "claude"})
            self.assertIn(str(raised.exception.status), ("404", "405"))

    def test_gateway_socket_traffic_has_its_own_server_identity(self):
        port = sysstats.OLLAMA_GATEWAY_PORT
        before = {(port, 51000): (100, 20)}
        after = {(port, 51000): (150, 75)}
        self.assertEqual(sysstats.INFERENCE_PORTS[port], "ollama-gateway")
        self.assertEqual(sysstats.traffic(before, after, [[port, 51000]]),
                         {"ollama-gateway": {"up": 50, "down": 55}})


class DecisionPurpose(unittest.TestCase):
    def test_question_names_join_the_decision_purpose(self):
        self.assertEqual(gateway.decision_purpose({"model": "m", "questions": {
            "thinking": {"type": "choice", "instructions": "pick", "criteria": ["a"]}}}),
                         "decision:thinking")
        self.assertEqual(gateway.decision_purpose({"model": "m", "questions": [{"name": "effort"},
                                                                              {"name": "find"}]}),
                         "decision:effort,find")
        self.assertEqual(gateway.decision_purpose({"model": "m", "questions": ["ttsr"]}),
                         "decision:ttsr")

    def test_unknown_decision_shape_still_counts_as_decision(self):
        for payload in ({"model": "m"}, {"model": "m", "questions": "effort"},
                        {"model": "m", "questions": [{"label": "effort"}, 7, None]}):
            with self.subTest(payload=payload):
                self.assertEqual(gateway.decision_purpose(payload), "decision")

    def test_chat_uses_the_bench_proxy_shape_classification(self):
        route = gateway.Route("inference", profile="claude", upstream_path="/v1/responses")
        self.assertEqual(gateway.request_purpose(route, {"messages": [], "tools": [{"type": "x"}]}),
                         "main")
        systemone = gateway.Route("inference", profile=None, upstream_path="/v1/systemone")
        self.assertEqual(gateway.request_purpose(systemone, {"model": "m"}), "decision")

    def test_long_question_names_are_bounded_with_a_stable_hash_suffix(self):
        payload = {"model": "m", "questions": [{"name": f"q{i:02d}-{('x' * 121)}"} for i in range(32)]}
        first = gateway.decision_purpose(payload)
        self.assertTrue(first.startswith("decision:"))
        self.assertLessEqual(len(first), 200)
        self.assertIn("#", first)
        self.assertEqual(first, gateway.decision_purpose(payload))
        short = gateway.decision_purpose({"model": "m", "questions": [{"name": "effort"}]})
        self.assertEqual(short, "decision:effort")


class OmpIdentity(unittest.TestCase):
    def test_version_and_sha_come_only_from_an_explicit_omp_token(self):
        self.assertEqual(gateway._omp_identity("omp/18.4.5"),
                         {"omp_version": "18.4.5", "omp_sha": None, "omp_note": None})
        self.assertEqual(gateway._omp_identity("omp/18.4.5+ca9b8832ea05299f"),
                         {"omp_version": "18.4.5", "omp_sha": "ca9b8832ea05299f", "omp_note": None})
        self.assertEqual(gateway._omp_identity("Python-urllib/3.14"),
                         {"omp_version": None, "omp_sha": None,
                          "omp_note": "User-Agent carries no omp version token"})
        self.assertEqual(gateway._omp_identity(None),
                         {"omp_version": None, "omp_sha": None,
                          "omp_note": "no User-Agent header to read an omp version from"})

class StoreAndExpiry(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = gateway.GatewayStore(Path(self.temp.name) / "leases.sqlite")
        self.store.set_profiles({"default": "/omp-profile/default", "claude": "/omp-profile/claude"})

    def test_concurrent_requests_hold_lease_until_last_completion_then_idle_window(self):
        first = self.store.start_request("default", "m", "instance-a", now=100.0)
        second = self.store.start_request("claude", "m", "instance-a", now=105.0)
        self.store.finish_request(first, completed=True, outcome="completed", now=110.0)
        self.assertEqual(self.store.claim_due(now=10_000.0), [])
        self.store.finish_request(second, completed=True, outcome="completed", now=120.0)
        self.assertEqual(self.store.claim_due(now=419.0), [])
        due = self.store.claim_due(now=420.0)
        self.assertEqual([row["model"] for row in due], ["m"])
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["profiles"], ["claude", "default"])

    def test_manual_finite_lease_extends_idle_deadline_and_never_accepts_forever(self):
        request = self.store.start_request("default", "m", "instance-a", now=100.0)
        self.store.finish_request(request, completed=True, outcome="completed", now=110.0)
        self.store.set_manual_lease("m", expires_at=800.0, now=120.0)
        self.assertEqual(self.store.claim_due(now=500.0), [])
        self.assertEqual([row["model"] for row in self.store.claim_due(now=800.0)], ["m"])

    def test_manual_lease_refuses_nonfinite_expiry_or_clock_before_any_row_is_written(self):
        for expires_at, now in ((float("nan"), 100.0), (float("inf"), 100.0),
                                (float("-inf"), 100.0), (200.0, float("nan")),
                                (200.0, float("inf"))):
            with self.subTest(expires_at=expires_at, now=now):
                with self.assertRaisesRegex(gateway.GatewayError, "finite"):
                    self.store.set_manual_lease("m", expires_at, now=now)
                self.assertIsNone(self.store.lease("m"))

    def test_keep_state_refuses_nonfinite_duration_and_overflowed_expiry(self):
        for duration, now in ((float("nan"), 100.0), (float("inf"), 100.0),
                              (1e308, 1e308)):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(gateway.GatewayError, "finite"):
                    gateway.keep_state("m", duration, home=Path(self.temp.name), now=now)

    def test_unload_claim_excludes_new_requests_and_manual_keep_until_outcome(self):
        request_id = self.store.start_request("default", "m", "instance-a", now=10.0)
        self.store.finish_request(request_id, completed=True, outcome="completed", now=20.0)
        self.assertEqual([row["model"] for row in self.store.claim_due(now=320.0)], ["m"])
        with self.assertRaisesRegex(gateway.GatewayError, "unload is in progress"):
            self.store.start_request("default", "m", "instance-a", now=321.0)
        with self.assertRaisesRegex(gateway.GatewayError, "finite keep was refused"):
            self.store.set_manual_lease("m", expires_at=900.0, now=321.0)
        self.assertEqual(self.store.active_requests("m"), 0)
        self.store.mark_result("m", "unload_failed", "fake failure", now=322.0)
        request_id = self.store.start_request("default", "m", "instance-a", now=323.0)
        self.assertEqual(self.store.active_requests("m"), 1)
        self.store.finish_request(request_id, completed=True, outcome="completed", now=324.0)

    def test_draining_rejects_new_requests_before_any_inference_lease_is_created(self):
        self.store.set_accepting(False)
        with self.assertRaisesRegex(gateway.GatewayError, "draining"):
            self.store.start_request("default", "m", "instance-a", now=100.0)
        self.assertEqual(self.store.active_requests(), 0)
        self.assertIsNone(self.store.lease("m"))
        self.store.set_accepting(True)
        request_id = self.store.start_request("default", "m", "instance-a", now=101.0)
        self.assertEqual(self.store.active_requests("m"), 1)
        self.store.finish_request(request_id, completed=True, outcome="completed", now=102.0)

    def test_old_inflight_rows_are_reclaimed_only_after_death_is_proven(self):
        self.store.start_request("default", "m", "old-instance", now=100.0)
        with self.assertRaises(gateway.GatewayError):
            self.store.recover_instance("old-instance", previous_dead=False, now=200.0)
        self.assertEqual(self.store.active_requests("m"), 1)
        self.store.recover_instance("old-instance", previous_dead=True, now=200.0)
        self.assertEqual(self.store.active_requests("m"), 0)
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["last_outcome"], "interrupted_on_restart")


    def test_resident_reloaded_after_confirmed_gateway_unload_is_unowned(self):
        request_id = self.store.start_request("default", "m", "instance-a", now=100.0)
        self.store.finish_request(request_id, completed=True, outcome="completed", now=120.0)
        self.store.mark_result("m", "unloaded_confirmed", now=420.0)
        with mock.patch.object(gateway, "database_path", return_value=self.store.path), \
                mock.patch.object(gateway, "service_status", return_value={"health": False}):
            state = gateway.status(resident_state=[("m", "later")])
        self.assertEqual(state["unowned_residents"], ["m"])

    def test_removed_gateway_no_longer_claims_expired_model_leases(self):
        request_id = self.store.start_request("default", "m", "instance-a", now=10.0)
        self.store.finish_request(request_id, completed=True, outcome="completed", now=20.0)
        self.store.mark_unmanaged()
        self.assertEqual(self.store.claim_due(now=10_000.0), [])
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["last_outcome"], "unmanaged_on_remove")

    def test_purpose_report_aggregates_counts_and_busy_seconds_by_window(self):
        first = self.store.start_request("default", "m", "instance-a", now=36_005.0,
                                         purpose="decision:effort")
        self.store.finish_request(first, completed=True, outcome="completed", now=36_012.0)
        second = self.store.start_request("claude", "m", "instance-a", now=36_020.0,
                                          purpose="main")
        self.store.finish_request(second, completed=True, outcome="completed", now=36_023.0)
        report = self.store.purpose_report(36_000.0, 39_600.0)
        self.assertEqual(report, [{"profile": "claude", "purpose": "main",
                                   "requests": 1, "busy_s": 3.0},
                                  {"profile": "default", "purpose": "decision:effort",
                                   "requests": 1, "busy_s": 7.0}])
        self.assertEqual(self.store.purpose_report(0.0, 36_000.0), [])
        self.assertEqual(self.store.purpose_report(39_600.0, 43_200.0), [])


class ExpirySafety(StoreAndExpiry):
    class Ollama:
        def __init__(self, residents=None, fail_unload=False, fail_readback=False, retain_model=False):
            self.residents = set(residents or ())
            self.unload_calls = []
            self.fail_unload = fail_unload
            self.fail_readback = fail_readback
            self.retain_model = retain_model
            self.reads = 0

        def resident_models(self):
            self.reads += 1
            if self.fail_readback and self.reads > 1:
                raise OSError("simulated /api/ps failure")
            return set(self.residents)

        def unload(self, model):
            self.unload_calls.append(model)
            if self.fail_unload:
                raise OSError("simulated unload failure")
            if not self.retain_model:
                self.residents.discard(model)

    def _expired(self, model="m"):
        request = self.store.start_request("default", model, "instance-a", now=100.0)
        self.store.finish_request(request, completed=True, outcome="completed", now=120.0)

    def test_active_or_unknown_external_client_defers_unload(self):
        for safety in ((False, "external client active"), (None, "probe unavailable")):
            with self.subTest(safety=safety):
                self.store.set_profiles({"default": "/omp-profile/default"})
                self._expired()
                ollama = self.Ollama({"m"})
                policy = gateway.GatewayPolicy(
                    self.store, ollama, external_check=lambda _model, value=safety: value,
                )
                policy.expire_due(now=420.0)
                self.assertEqual(ollama.unload_calls, [])
                lease = self.store.lease("m")
                assert lease is not None
                self.assertEqual(lease["last_outcome"], "deferred")
                self.assertIn(safety[1], lease["last_error"])

    def test_persistent_external_connection_defers_unload_past_five_minutes(self):
        self._expired()
        now = [100.0]
        lsof = "p123\ncclient\nn127.0.0.1:40001->127.0.0.1:11434\n"

        def run(*_args, **_kwargs):
            return mock.Mock(returncode=0, stdout=lsof, stderr="")

        with mock.patch.object(gateway.shutil, "which", return_value="/usr/sbin/lsof"), \
                mock.patch.object(sysstats, "gpu_share", return_value=[]):
            probe = gateway.ExternalActivityProbe(run=run, clock=lambda: now[0], gpu_reader=dict)
            self.assertFalse(probe.check("m", gateway_pid=999)[0])
            now[0] += gateway.IDLE_SECONDS + 1
            ollama = self.Ollama({"m"})
            policy = gateway.GatewayPolicy(
                self.store, ollama, external_check=lambda model: probe.check(model, gateway_pid=999),
            )
            policy.expire_due(now=420.0)

        self.assertEqual(ollama.unload_calls, [])
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["last_outcome"], "deferred")
        self.assertIn("established non-gateway Ollama client", lease["last_error"] or "")

    def test_unload_is_confirmed_by_absence_and_not_retried_after_success(self):
        self._expired()
        ollama = self.Ollama({"m"})
        policy = gateway.GatewayPolicy(self.store, ollama, external_check=lambda _model: (True, None))
        policy.expire_due(now=420.0)
        policy.expire_due(now=421.0)
        self.assertEqual(ollama.unload_calls, ["m"])
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["last_outcome"], "unloaded_confirmed")

    def test_unload_and_readback_failures_remain_visible_and_do_not_claim_success(self):
        for ollama in (self.Ollama({"m"}, fail_unload=True), self.Ollama({"m"}, fail_readback=True),
                       self.Ollama({"m"}, retain_model=True)):
            with self.subTest(ollama=ollama.fail_unload, readback=ollama.fail_readback,
                              retained=ollama.retain_model):
                self.store.set_profiles({"default": "/omp-profile/default"})
                self._expired()
                gateway.GatewayPolicy(self.store, ollama,
                                      external_check=lambda _model: (True, None)).expire_due(now=420.0)
                lease = self.store.lease("m")
                assert lease is not None
                self.assertNotEqual(lease["last_outcome"], "unloaded_confirmed")
                self.assertTrue(lease["last_error"])

    def test_unowned_resident_is_never_claimed_or_unloaded(self):
        self.assertEqual(self.store.claim_due(now=10_000.0), [])
        ollama = self.Ollama({"external-model"})
        gateway.GatewayPolicy(self.store, ollama, external_check=lambda _model: (True, None)).expire_due(now=10_000.0)
        self.assertEqual(ollama.unload_calls, [])


class ExternalActivitySafety(unittest.TestCase):
    def test_lsof_visibility_warning_makes_external_activity_unknown(self):
        def warning_run(*_args, **_kwargs):
            return mock.Mock(returncode=0, stdout="", stderr="permission visibility warning")

        with mock.patch.object(gateway.shutil, "which", return_value="/usr/sbin/lsof"):
            probe = gateway.ExternalActivityProbe(run=warning_run, gpu_reader=dict)
            safe, reason = probe.check("m", gateway_pid=999)
        self.assertIsNone(safe)
        assert reason is not None
        self.assertIn("telemetry is unavailable", reason)

class Duration(unittest.TestCase):
    def test_finite_duration_parser_bounds_values_and_rejects_unbounded_spellings(self):
        self.assertEqual(gateway.parse_finite_duration("5m"), 300)
        self.assertEqual(gateway.parse_finite_duration("1h30m"), 5400)
        self.assertEqual(gateway.parse_finite_duration("0"), 0)
        for value in ("forever", "-1m", "inf", "", "1fortnight"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gateway.parse_finite_duration(value)


class OmpProfiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dirs = {}
        for name in ("default", "claude"):
            agent = self.root / name / "agent"
            agent.mkdir(parents=True)
            (agent / "config.yml").write_text("defaultThinkingLevel: high\n")
            self.dirs[name] = agent

    def test_install_adds_managed_provider_without_rewriting_other_providers_and_revert_preserves_edits(self):
        path = self.dirs["default"] / "models.yml"
        path.write_text("providers:\n  localbench:\n    baseUrl: http://127.0.0.1:11400/v1\nmodels: []\n")
        manager = omp_profiles.ProfileManager(self.dirs, self.root / "state")
        manifest = manager.install(port=11300)
        installed = path.read_text()
        self.assertIn("localbench:", installed)
        self.assertIn("baseUrl: http://127.0.0.1:11300/omp-profile/default", installed)
        self.assertIn("discovery:\n      type: ollama", installed)
        path.write_text(installed + "# manually added later\n")
        manager.revert(manifest)
        reverted = path.read_text()
        self.assertIn("localbench:", reverted)
        self.assertIn("# manually added later", reverted)
        self.assertNotIn("localbench ollama residency", reverted)
        self.assertFalse((self.dirs["claude"] / "models.yml").exists())

    def test_explicit_custom_ollama_provider_aborts_all_profiles_before_writing(self):
        conflict = self.dirs["claude"] / "models.yml"
        conflict.write_text("providers:\n  ollama:\n    baseUrl: http://127.0.0.1:12345\n")
        manager = omp_profiles.ProfileManager(self.dirs, self.root / "state")
        with self.assertRaises(omp_profiles.ProfileConflict):
            manager.install(port=11300)
        self.assertFalse((self.dirs["default"] / "models.yml").exists())
        self.assertEqual(conflict.read_text(), "providers:\n  ollama:\n    baseUrl: http://127.0.0.1:12345\n")

    def test_install_preserves_profile_edit_that_races_the_write(self):
        path = self.dirs["default"] / "models.yml"
        original = "providers:\n  localbench:\n    baseUrl: http://127.0.0.1:11400/v1\n"
        path.write_text(original)
        manager = omp_profiles.ProfileManager(self.dirs, self.root / "state")
        atomic_write = omp_profiles._atomic_write
        raced = {"value": False}

        def write_with_race(target, data, mode):
            if target.parent.name.startswith("omp-profiles-") and not raced["value"]:
                raced["value"] = True
                path.write_text(original + "# concurrent edit\n")
            return atomic_write(target, data, mode)

        with mock.patch.object(omp_profiles, "_atomic_write", side_effect=write_with_race):
            with self.assertRaises(omp_profiles.ProfileConflict):
                manager.install(port=11300)
        self.assertTrue(raced["value"])
        self.assertEqual(path.read_text(), original + "# concurrent edit\n")
        self.assertFalse((self.dirs["claude"] / "models.yml").exists())
        self.assertFalse(manager.manifest_path.exists())

    def test_revert_refuses_managed_block_edits_without_touching_other_profiles(self):
        manager = omp_profiles.ProfileManager(self.dirs, self.root / "state")
        manifest = manager.install(port=11300)
        changed = self.dirs["claude"] / "models.yml"
        changed.write_text(changed.read_text().replace("11300", "11301"))
        default_before = (self.dirs["default"] / "models.yml").read_bytes()
        with self.assertRaises(omp_profiles.ProfileConflict):
            manager.revert(manifest)
        self.assertEqual((self.dirs["default"] / "models.yml").read_bytes(), default_before)


class GatewayCli(unittest.TestCase):
    def test_install_dry_run_reports_profile_plan_without_writing_profile_or_manifest(self):
        from localbench import __main__ as cli

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent = root / "omp" / "agent"
            agent.mkdir(parents=True)
            (agent / "config.yml").write_text("defaultThinkingLevel: high\n")
            state = root / "gateway-state"
            manager = omp_profiles.ProfileManager({"default": agent}, state)
            mutation = cli.Mutation("gateway install", [], dry_run=True, as_json=True, audited=False)
            args = cli.argparse.Namespace(cmd="gateway", action="install", host=gateway.HOST,
                                          port=gateway.PORT, json=True, dry_run=True, explain=False,
                                          mutation=mutation)
            service = {"plist_installed": False, "plist_managed": False, "plist_error": None,
                       "launchd_loaded": False, "health": False}
            with mock.patch.object(omp_profiles.ProfileManager, "current", return_value=manager), \
                    mock.patch.object(gateway, "state_dir", return_value=state), \
                    mock.patch.object(gateway, "service_status", return_value=service), \
                    mock.patch.object(gateway, "ensure_port_free") as port_check, \
                    mock.patch.object(cli, "_run_alive", return_value=False), \
                    mock.patch.object(gateway, "install") as install_call, \
                    mock.patch("builtins.print") as output:
                self.assertEqual(cli.cmd_gateway(args), 0)
            plan = json.loads(output.call_args.args[0])
            self.assertEqual(plan["verb"], "gateway install")
            self.assertIsNone(plan["would_refuse"])
            self.assertIn("1 OMP profiles", plan["actions"][0])
            port_check.assert_called_once_with(port=gateway.PORT)
            install_call.assert_not_called()
            self.assertFalse((agent / "models.yml").exists())
            self.assertFalse(manager.manifest_path.exists())

    def fence_cli(self, store_path: Path, action: str, *, models=(), fence_id=None, park_ids=frozenset()):
        from localbench import __main__ as cli

        mutation = cli.Mutation(f"gateway {action}", [], dry_run=False, as_json=False, audited=False)
        args = cli.argparse.Namespace(cmd="gateway", action=action, model=list(models), fence_id=fence_id,
                                      wait=0.0, json=False, dry_run=False, explain=False, mutation=mutation)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(gateway, "database_path", return_value=store_path), \
                mock.patch.object(cli, "_park_fence_ids", return_value=set(park_ids)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.cmd_gateway(args)
        return rc, out.getvalue() + err.getvalue()

    def test_fence_holds_until_its_own_unfence_and_never_releases_park(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "leases.sqlite"
            store = gateway.GatewayStore(path)
            rc, text = self.fence_cli(path, "fence", models=["nimble:latest", "nimble"])
            self.assertEqual(rc, 0, text)
            fence_id = re.search(r"manual-[0-9a-f]+", text).group(0)
            self.assertEqual(sorted(f["model"] for f in store.fences()), ["nimble", "nimble:latest"])
            rc, text = self.fence_cli(path, "fence", models=["nimble"])
            self.assertEqual(rc, 1)
            self.assertIn("already fenced", text)
            self.assertIsNone(store.acquire_park_fence(["qwen"], "park-held"))
            rc, text = self.fence_cli(path, "unfence", fence_id="park-held", park_ids={"park-held"})
            self.assertEqual(rc, 1)
            self.assertEqual(store.park_fence("qwen"), "park-held", "a park fence must survive unfence")
            rc, text = self.fence_cli(path, "unfence", fence_id=fence_id)
            self.assertEqual(rc, 0, text)
            self.assertEqual([f["model"] for f in store.fences()], ["qwen"])

    def test_fence_waits_out_an_in_flight_request_then_refuses_when_time_is_up(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "leases.sqlite"
            store = gateway.GatewayStore(path)
            store.set_profiles({"default": "/omp-profile/default"})
            store.start_request("default", "nimble", "test")
            rc, text = self.fence_cli(path, "fence", models=["nimble"])
            self.assertEqual(rc, 1)
            self.assertIn("in flight", text)
            self.assertEqual(store.fences(), [], "a refused fence leaves nothing behind")

    def test_status_text_uses_actual_residency_payload_without_guessing_models(self):
        from localbench import __main__ as cli

        lease = {"model": "m", "profiles": ["default"], "active_requests": 0,
                 "idle_expires_at": None, "manual_expires_at": None, "last_completed_at": None,
                 "last_outcome": None, "last_error": None}
        for state, residents in (("unknown", None), ("available", ["orphan"]), ("available", [])):
            with self.subTest(state=state, residents=residents):
                status = {"service": {"health": True, "launchd_state": "running", "bind": "127.0.0.1:11300"},
                          "accepting_requests": True, "profiles": {"default": "/omp-profile/default"},
                          "active_requests": 0, "leases": [lease], "ollama_state": state,
                          "unowned_residents": residents}
                text = io.StringIO()
                with mock.patch.object(gateway, "status", return_value=status), contextlib.redirect_stdout(text):
                    rc = cli.cmd_gateway(cli.argparse.Namespace(action="status", json=False))
                self.assertEqual(rc, 0)
                self.assertIn("lease m:", text.getvalue())
                self.assertIn(f"Ollama residency API: {state}", text.getvalue())
                label = "unknown" if residents is None else ", ".join(residents) or "none"
                self.assertIn(f"unowned residents: {label}", text.getvalue())


    def test_unreadable_ollama_api_is_unknown_from_real_status_producer_to_text_and_json(self):
        from localbench import __main__ as cli

        service = {"health": True, "launchd_state": "running", "bind": "127.0.0.1:11300"}
        text, encoded = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gateway, "database_path", return_value=Path(tmp) / "absent.sqlite"), \
                mock.patch.object(gateway, "service_status", return_value=service), \
                mock.patch.object(gateway.sysstats, "ollama_residents", return_value=None):
            with contextlib.redirect_stdout(text):
                self.assertEqual(cli.cmd_gateway(cli.argparse.Namespace(action="status", json=False)), 0)
            with contextlib.redirect_stdout(encoded):
                self.assertEqual(cli.cmd_gateway(cli.argparse.Namespace(action="status", json=True)), 0)
        self.assertIn("Ollama residency API: unknown", text.getvalue())
        self.assertIn("unowned residents: unknown", text.getvalue())
        payload = json.loads(encoded.getvalue())
        self.assertEqual(payload["ollama_state"], "unknown")
        self.assertIsNone(payload["unowned_residents"])


class LaunchAgent(unittest.TestCase):
    def test_launch_agent_is_user_scoped_loopback_only_and_does_not_use_shell(self):
        with tempfile.TemporaryDirectory() as td:
            plist = gateway.launchd_plist(home=Path(td), python="/opt/localbench/bin/python")
        self.assertEqual(plist["Label"], gateway.LABEL)
        self.assertEqual(plist["ProgramArguments"][:3], ["/opt/localbench/bin/python", "-m", "localbench"])
        self.assertIn("127.0.0.1", plist["ProgramArguments"])
        self.assertTrue(plist["RunAtLoad"])
        self.assertNotIn("sh", plist["ProgramArguments"])
        self.assertTrue(plist["StandardOutPath"].startswith(str(Path(td) / ".localbench")))


class StopStartRace(unittest.TestCase):
    """`gateway stop` returns only after launchd unloads AND /healthz goes silent, so an
    immediate `gateway start` cannot read a stale healthy and no-op while dying."""

    def _script(self, prints, bootstraps=None):
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            if argv[:2] == ["launchctl", "print"]:
                loaded = prints.pop(0) if prints else False
                return SimpleNamespace(returncode=0 if loaded else 1,
                                       stdout="state = running\n" if loaded else "No such process\n",
                                       stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        return run, calls

    def _managed(self):
        return mock.patch.object(gateway, "_read_managed_launch_agent",
                                 return_value={"Label": gateway.LABEL})

    def test_stop_waits_out_stale_loaded_and_lingering_health(self):
        run, calls = self._script([True, True, False])
        probes = [{"ok": True, "latency_s": 0.01, "reason": None},
                  {"ok": False, "latency_s": 0.01, "reason": "refused"}]
        with self._managed(), \
                mock.patch.object(gateway, "_probe_health", side_effect=probes) as probed:
            self.assertTrue(gateway.stop_launch_agent(home=Path("/tmp/no-home"), run=run))
        self.assertEqual([c for c in calls if c[:2] == ["launchctl", "bootout"]],
                         [["launchctl", "bootout", gateway.launch_target()]])
        self.assertEqual(probed.call_count, 2)

    def test_stop_timeout_names_which_side_is_stuck(self):
        run, _ = self._script([True] * 100)
        with self._managed(), \
                mock.patch.object(gateway, "_probe_health",
                                  return_value={"ok": False, "latency_s": 0.0, "reason": None}):
            with self.assertRaisesRegex(gateway.GatewayError, "still loaded"):
                gateway.stop_launch_agent(home=Path("/tmp/no-home"), run=run, unload_timeout=0.4)
        run, _ = self._script([True, False])
        with self._managed(), \
                mock.patch.object(gateway, "_probe_health",
                                  return_value={"ok": True, "latency_s": 0.01, "reason": None}):
            with self.assertRaisesRegex(gateway.GatewayError, "still answers /healthz"):
                gateway.stop_launch_agent(home=Path("/tmp/no-home"), run=run, unload_timeout=0.4)

    def test_stop_then_start_actually_starts(self):
        run, calls = self._script([True, True, False])
        probes = [{"ok": True, "latency_s": 0.01, "reason": None},
                  {"ok": False, "latency_s": 0.01, "reason": None}]
        with self._managed(), \
                mock.patch.object(gateway, "_probe_health", side_effect=probes):
            self.assertTrue(gateway.stop_launch_agent(home=Path("/tmp/no-home"), run=run))
            self.assertTrue(gateway.start_launch_agent(home=Path("/tmp/no-home"), run=run,
                                                       wait_health=lambda timeout: True))
        bootstraps = [c for c in calls if c[:2] == ["launchctl", "bootstrap"]]
        self.assertEqual(len(bootstraps), 1)

    def test_start_reprobes_before_claiming_already_healthy(self):
        from localbench import __main__ as cli

        healthy = {"plist_installed": True, "plist_managed": True, "plist_error": None,
                   "launchd_loaded": True, "launchd_state": "running", "health": True,
                   "health_latency_s": 0.01, "health_reason": None, "bind": "127.0.0.1:11300"}
        down = dict(healthy, launchd_loaded=False, launchd_state="not_loaded", health=False,
                    health_reason="refused")

        def args():
            return cli.argparse.Namespace(
                action="start", json=False,
                mutation=cli.Mutation("gateway start", [], dry_run=False, audited=False))

        with mock.patch.object(gateway, "service_status", side_effect=[healthy, down]), \
                mock.patch.object(gateway, "start", return_value=True) as started, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.cmd_gateway(args()), 0)
        started.assert_called_once_with()
        with mock.patch.object(gateway, "service_status", side_effect=[healthy, healthy]), \
                mock.patch.object(gateway, "start", return_value=True) as started, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.cmd_gateway(args()), 0)
        started.assert_not_called()


class _StreamingUpstream(BaseHTTPRequestHandler):
    calls = []
    first_chunk = threading.Event()
    release_stream = threading.Event()
    systemone_calls = []
    systemone_arrived = threading.Event()
    release_systemone = threading.Event()
    SYSTEMONE_RESPONSE = b'{"decisions":[{"name":"effort","answer":"low"}]}'

    def log_message(self, format, *args):
        return

    def _json(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": "m", "digest": "fake"}]})
        elif self.path == "/api/version":
            self._json({"version": "0.35.0-fake"})
        else:
            self.send_error(404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(raw)
        type(self).calls.append((self.path, request))
        if self.path == "/v1/systemone":
            type(self).systemone_calls.append((self.path, raw))
            type(self).systemone_arrived.set()
            type(self).release_systemone.wait(10)
            body = type(self).SYSTEMONE_RESPONSE
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/v1/responses":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"event: response.created\n\n")
        self.wfile.flush()
        type(self).first_chunk.set()
        type(self).release_stream.wait(10)
        if request.get("disconnect"):
            for _ in range(128):
                try:
                    self.wfile.write(b"x" * 65_536)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            return
        self.wfile.write(b'event: response.completed\ndata: {"type":"response.completed"}\n\n')
        self.wfile.flush()


class HttpGateway(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = gateway.GatewayStore(Path(self.temp.name) / "leases.sqlite")
        self.store.set_profiles({"claude": "/omp-profile/claude"})
        _StreamingUpstream.calls = []
        _StreamingUpstream.first_chunk = threading.Event()
        _StreamingUpstream.release_stream = threading.Event()
        _StreamingUpstream.systemone_calls = []
        _StreamingUpstream.systemone_arrived = threading.Event()
        _StreamingUpstream.release_systemone = threading.Event()
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingUpstream)
        self.gateway_server = gateway.create_server(
            ("127.0.0.1", 0), self.store,
            gateway.GatewayPolicy(self.store, object(), lambda _model: (True, None)),
            upstream_root=f"http://127.0.0.1:{self.upstream.server_port}", instance_id="http-test",
        )
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.gateway_thread = threading.Thread(target=self.gateway_server.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.gateway_thread.start()
        self.addCleanup(self._close_servers)

    def _close_servers(self):
        _StreamingUpstream.release_stream.set()
        _StreamingUpstream.release_systemone.set()
        self.gateway_server.shutdown()
        self.gateway_server.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()

    def test_partial_request_body_times_out_before_upstream_forwarding(self):
        address = ("127.0.0.1", self.gateway_server.server_port)
        client = socket.create_connection(address, timeout=2)
        client.settimeout(0.8)
        headers = (f"POST /api/show HTTP/1.0\r\n"
                   f"Host: {address[0]}:{address[1]}\r\nContent-Type: application/json\r\n"
                   "Content-Length: 64\r\nConnection: close\r\n\r\n").encode()
        response = bytearray()
        try:
            with mock.patch.object(gateway, "REQUEST_BODY_TIMEOUT_SECONDS", 0.15, create=True):
                client.sendall(headers + b'{"model":"m"}')
                try:
                    while True:
                        chunk = client.recv(4096)
                        if not chunk:
                            break
                        response.extend(chunk)
                except socket.timeout:
                    pass
        finally:
            client.close()

        self.assertIn(b"408 Request Timeout", response)
        self.assertEqual(_StreamingUpstream.calls, [])
        self.assertEqual(self.store.active_requests(), 0)

    def test_metadata_is_unleased_and_sse_is_attributed_and_streamed_before_completion(self):
        root = f"http://127.0.0.1:{self.gateway_server.server_port}"
        with urlopen(root + "/api/tags", timeout=2) as response:
            self.assertEqual(json.load(response)["models"][0]["name"], "m")
        self.assertEqual(self.store.leases(), [])

        private_text = "private-prompt-not-for-the-lease-store"
        body = json.dumps({"model": "m", "input": private_text}).encode()
        request = Request(root + "/omp-profile/claude/responses", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=3) as response:
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            first = response.readline()
            self.assertEqual(first, b"event: response.created\n")
            self.assertTrue(_StreamingUpstream.first_chunk.wait(1))
            _StreamingUpstream.release_stream.set()
            rest = response.read()
        self.assertIn(b"response.completed", rest)
        self.assertEqual(_StreamingUpstream.calls[0][0], "/v1/responses")
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["profiles"], ["claude"])
        self.assertEqual(lease["active_requests"], 0)
        self.assertIsNotNone(lease["last_completed_at"])
        self.assertNotIn(private_text, json.dumps(lease))

    def test_client_disconnect_releases_inflight_request(self):
        root = ("127.0.0.1", self.gateway_server.server_port)
        body = json.dumps({"model": "m", "disconnect": True}).encode()
        client = socket.create_connection(root, timeout=3)
        client.sendall((f"POST /omp-profile/claude/responses HTTP/1.0\r\n"
                        f"Host: {root[0]}:{root[1]}\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body)
        stream = client.makefile("rb")
        try:
            while stream.readline() != b"\r\n":
                pass
            self.assertEqual(stream.readline(), b"event: response.created\n")
            self.assertTrue(_StreamingUpstream.first_chunk.wait(1))
        finally:
            stream.close()
            client.close()
            _StreamingUpstream.release_stream.set()
        deadline = time.monotonic() + 3
        while self.store.active_requests("m"):
            self.assertLess(time.monotonic(), deadline, "disconnected OMP request remained leased")
            time.sleep(0.01)
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["active_requests"], 0)
        self.assertIsNotNone(lease["idle_expires_at"])

    def _post(self, path, payload):
        root = f"http://127.0.0.1:{self.gateway_server.server_port}"
        body = json.dumps(payload).encode()
        request = Request(root + path, data=body, headers={"Content-Type": "application/json"},
                          method="POST")
        with urlopen(request, timeout=5) as response:
            return response.status, response.read()

    def test_systemone_returns_upstream_bytes_and_counts_lease_inflight_and_purpose(self):
        payload = {"model": "m", "questions": [{"name": "effort"}, {"name": "find"}]}
        sent = json.dumps(payload).encode()
        for path, profile in (("/v1/systemone", ""), ("/omp-profile/claude/v1/systemone", "claude")):
            with self.subTest(path=path):
                _StreamingUpstream.release_systemone.set()
                status, body = self._post(path, payload)
                self.assertEqual(status, 200)
                self.assertEqual(body, _StreamingUpstream.SYSTEMONE_RESPONSE)
                self.assertIn(("/v1/systemone", sent), _StreamingUpstream.systemone_calls)
        deadline = time.monotonic() + 3
        while self.store.active_requests("m"):
            self.assertLess(time.monotonic(), deadline, "finished systemone request remained leased")
            time.sleep(0.01)
        lease = self.store.lease("m")
        assert lease is not None
        self.assertEqual(lease["active_requests"], 0)
        self.assertIsNotNone(lease["last_completed_at"])
        report = self.store.purpose_report(0, time.time() + 3600)
        by_profile = {(row["profile"], row["purpose"]): row for row in report}
        self.assertEqual(by_profile[("", "decision:effort,find")]["requests"], 1)
        self.assertEqual(by_profile[("claude", "decision:effort,find")]["requests"], 1)
        self.assertGreater(by_profile[("claude", "decision:effort,find")]["busy_s"], 0)

    def test_many_long_question_names_still_forward_with_status_200(self):
        payload = {"model": "m", "questions": [{"name": f"q{i:02d}-" + "y" * 124} for i in range(32)]}
        self.assertTrue(all(len(q["name"]) == 128 for q in payload["questions"]))
        _StreamingUpstream.release_systemone.set()
        status, body = self._post("/omp-profile/claude/v1/systemone", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body, _StreamingUpstream.SYSTEMONE_RESPONSE)
        deadline = time.monotonic() + 3
        while self.store.active_requests("m"):
            self.assertLess(time.monotonic(), deadline, "finished systemone request remained leased")
            time.sleep(0.01)
        report = self.store.purpose_report(0, time.time() + 3600)
        self.assertEqual(len(report), 1)
        self.assertTrue(report[0]["purpose"].startswith("decision:"))
        self.assertLessEqual(len(report[0]["purpose"]), 200)
        self.assertIn("#", report[0]["purpose"])

    def test_chat_purpose_uses_the_proxy_shape_classification(self):
        root = f"http://127.0.0.1:{self.gateway_server.server_port}"
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                           "tools": [{"type": "function", "function": {"name": "f"}}]}).encode()
        request = Request(root + "/omp-profile/claude/responses", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        _StreamingUpstream.release_stream.set()
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            self.assertEqual(response.readline(), b"event: response.created\n")
            self.assertTrue(_StreamingUpstream.first_chunk.wait(2))
            rest = response.read()
        self.assertIn(b"response.completed", rest)
        report = self.store.purpose_report(0, time.time() + 3600)
        self.assertEqual([(row["profile"], row["purpose"], row["requests"]) for row in report],
                         [("claude", "main", 1)])

    def test_park_fence_refuses_while_a_decision_request_is_in_flight(self):
        payload = json.dumps({"model": "m", "questions": [{"name": "effort"}]}).encode()
        root = f"http://127.0.0.1:{self.gateway_server.server_port}"
        request = Request(root + "/omp-profile/claude/v1/systemone", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        outcome = {}

        def call():
            with urlopen(request, timeout=15) as response:
                outcome["status"] = response.status
                outcome["body"] = response.read()

        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        try:
            # The decision request is in flight once the store records it; poll the store
            # itself (bounded) rather than trusting fake-upstream event ordering.
            deadline = time.monotonic() + 10
            while self.store.active_requests("m") != 1:
                if not worker.is_alive():
                    self.fail("decision worker died before its request reached in-flight")
                self.assertLess(time.monotonic(), deadline,
                                "decision request never reached in-flight")
                time.sleep(0.01)
            refusal = self.store.acquire_park_fence(["m"], "fence-while-deciding")
            self.assertIsNotNone(refusal)
            self.assertIn("in flight", refusal)
            self.assertEqual(self.store.active_requests("m"), 1)
        finally:
            _StreamingUpstream.release_systemone.set()
            worker.join(15)
        self.assertEqual(outcome.get("status"), 200)
        self.assertEqual(outcome.get("body"), _StreamingUpstream.SYSTEMONE_RESPONSE)
        # The client holds the full body before the handler thread records completion;
        # wait for the drain (bounded) before asserting fence admission.
        deadline = time.monotonic() + 3
        while self.store.active_requests("m"):
            self.assertLess(time.monotonic(), deadline, "finished decision request remained leased")
            time.sleep(0.01)
        self.assertIsNone(self.store.acquire_park_fence(["m"], "fence-blocks-decision"))
        try:
            with self.assertRaises(HTTPError) as refused:
                self._post("/omp-profile/claude/v1/systemone",
                           {"model": "m", "questions": [{"name": "effort"}]})
            self.assertEqual(refused.exception.code, 503)
            refused.exception.close()
        finally:
            self.store.release_park_fence("fence-blocks-decision")

    def test_a_fenced_decision_request_is_refused_at_once_and_says_why(self):
        # proj-b 2026-10-01: its nimble gate falls back to paid decision service on a fast gateway error during an agreed window.
        self.assertIsNone(self.store.acquire_park_fence(["m"], "manual-window"))
        try:
            started = time.monotonic()
            with self.assertRaises(HTTPError) as refused:
                self._post("/omp-profile/claude/v1/systemone", {"model": "m", "questions": [{"name": "effort"}]})
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertEqual(refused.exception.code, 503)
            with refused.exception as response:   # an HTTPError holds its socket until closed
                self.assertIn("fenced", json.loads(response.read())["error"])
            self.assertEqual(_StreamingUpstream.systemone_calls, [])
        finally:
            self.store.release_park_fence("manual-window")


class HealthzIndependence(unittest.TestCase):
    """GET /healthz answers without the lease store and while a sweep blocks.
    After an unpark the probe took 2.1 s / 0.9 s / 0.01 s and once timed out:
    the handler read the store before routing, and the accept loop ran the
    lease sweep inline. Both paths now bypass the health check."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = str(Path(self.temp.name) / "leases.sqlite")
        self.store = gateway.GatewayStore(self.db_path)
        self.store.set_profiles({"claude": "/omp-profile/claude"})
        self.gateway_server = gateway.create_server(
            ("127.0.0.1", 0), self.store,
            gateway.GatewayPolicy(self.store, object(), lambda _model: (True, None)),
            upstream_root="http://127.0.0.1:1", instance_id="healthz-test",
        )
        self.thread = threading.Thread(target=self.gateway_server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._close_server)

    def _close_server(self):
        self.gateway_server.shutdown()
        self.gateway_server.server_close()

    def _healthz(self, timeout=10):
        root = f"http://127.0.0.1:{self.gateway_server.server_port}"
        start = time.monotonic()
        with urlopen(root + "/healthz", timeout=timeout) as response:
            payload = json.load(response)
        return payload, time.monotonic() - start

    def test_healthz_never_touches_the_lease_store(self):
        reads = []
        real_profiles = self.store.profile_paths

        def slow_profiles():
            reads.append(1)
            time.sleep(5)
            return real_profiles()

        with mock.patch.object(self.store, "profile_paths", side_effect=slow_profiles):
            payload, elapsed = self._healthz()
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(reads, [])
        self.assertLess(elapsed, 2.0)

    def test_healthz_answers_while_a_write_transaction_holds_the_store(self):
        holder = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        self.addCleanup(holder.close)
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO leases(model) VALUES('held-model') "
                       "ON CONFLICT(model) DO NOTHING")
        try:
            payload, elapsed = self._healthz(timeout=5)
        finally:
            holder.execute("ROLLBACK")
        self.assertEqual(payload, {"ok": True})
        self.assertLess(elapsed, 2.0)

    def test_lease_sweep_runs_off_the_accept_thread(self):
        sweeps = []

        def slow_expire(now=None):
            sweeps.append(1)
            time.sleep(3)
            return []

        stop = threading.Event()
        loop = threading.Thread(
            target=gateway.maintenance_loop, args=(self.gateway_server, stop, 0.05), daemon=True)
        with mock.patch.object(self.gateway_server.policy, "expire_due", side_effect=slow_expire):
            loop.start()
            deadline = time.monotonic() + 10
            while not sweeps:
                self.assertLess(time.monotonic(), deadline, "lease sweep never started")
                time.sleep(0.01)
            payload, elapsed = self._healthz(timeout=5)
            stop.set()
            loop.join(10)
        self.assertEqual(payload, {"ok": True})
        self.assertLess(elapsed, 2.0)
        self.assertTrue(sweeps)
        self.assertFalse(loop.is_alive())

    def test_maintenance_loop_keeps_sweeping_until_stopped(self):
        stop = threading.Event()
        with mock.patch.object(self.gateway_server.policy, "expire_due",
                               return_value=[]) as sweep:
            loop = threading.Thread(
                target=gateway.maintenance_loop, args=(self.gateway_server, stop, 0.05),
                daemon=True)
            loop.start()
            deadline = time.monotonic() + 10
            while sweep.call_count < 2:
                self.assertLess(time.monotonic(), deadline, "maintenance loop stopped sweeping")
                time.sleep(0.01)
            stop.set()
            loop.join(10)
            count = sweep.call_count
        self.assertGreaterEqual(count, 2)
        self.assertFalse(loop.is_alive())


class HealthProbeStatus(unittest.TestCase):
    """The timed probe names its latency and reason; both status verbs share it,
    so a slow probe cannot read healthy on one verb and UNAVAILABLE on the other."""

    def test_probe_names_refusal_timeout_and_bad_payload(self):
        with mock.patch.object(gateway, "urlopen",
                               side_effect=URLError("connection refused")):
            refused = gateway._probe_health(timeout=0.2)
        self.assertFalse(refused["ok"])
        self.assertGreaterEqual(refused["latency_s"], 0.0)
        self.assertIn("refused", refused["reason"])
        with mock.patch.object(gateway, "urlopen", side_effect=TimeoutError("timed out")):
            slow = gateway._probe_health(timeout=0.2)
        self.assertFalse(slow["ok"])
        self.assertIn("timed out after 0.2 s", slow["reason"])
        with mock.patch.object(gateway, "urlopen",
                               return_value=io.BytesIO(b'{"ok": false}')):
            bad = gateway._probe_health(timeout=0.2)
        self.assertFalse(bad["ok"])
        self.assertIn("unexpected payload", bad["reason"])
        with mock.patch.object(gateway, "urlopen",
                               return_value=io.BytesIO(b'{"ok": true}')):
            healthy = gateway._probe_health(timeout=0.2)
        self.assertTrue(healthy["ok"])
        self.assertIsNone(healthy["reason"])
        self.assertGreaterEqual(healthy["latency_s"], 0.0)

    def test_both_status_verbs_share_one_timed_probe(self):
        for probe, verdict in (({"ok": True, "latency_s": 0.004, "reason": None}, True),
                               ({"ok": False, "latency_s": 1.0, "reason": "connection refused"},
                                False)):
            with self.subTest(verdict=verdict):
                with tempfile.TemporaryDirectory() as tmp:
                    db = Path(tmp) / "leases.sqlite"
                    gateway.GatewayStore(db)
                    with mock.patch.object(gateway, "_probe_health",
                                           return_value=probe) as probed, \
                            mock.patch.object(gateway, "database_path", return_value=db), \
                            mock.patch.object(gateway.sysstats, "ollama_residents",
                                              return_value=[]):
                        service = gateway.service_status(home=Path(tmp))
                        full = gateway.status(home=Path(tmp))
                self.assertEqual(service["health"], verdict)
                self.assertEqual(full["service"]["health"], verdict)
                self.assertEqual(full["service"]["health_latency_s"], probe["latency_s"])
                self.assertEqual(full["service"]["health_reason"], probe["reason"])
                self.assertEqual(probed.call_count, 2)


class CaptureOptIn(unittest.TestCase):
    """Opt-in traffic capture: off writes nothing; on writes matching purposes once, honours the
    cap and expiry, refuses repo paths, and never breaks forwarding."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "corpora"
        self.store = gateway.GatewayStore(Path(self.temp.name) / "leases.sqlite")
        self.store.set_profiles({"claude": "/omp-profile/claude"})
        _StreamingUpstream.calls = []
        _StreamingUpstream.systemone_calls = []
        _StreamingUpstream.systemone_arrived = threading.Event()
        _StreamingUpstream.release_systemone = threading.Event()
        _StreamingUpstream.release_systemone.set()
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingUpstream)
        self.patched = mock.patch.object(gateway.corpus, "CORPORA", self.root)
        self.patched.start()
        self.addCleanup(self.patched.stop)
        self.gateway_server = gateway.create_server(
            ("127.0.0.1", 0), self.store,
            gateway.GatewayPolicy(self.store, object(), lambda _model: (True, None)),
            upstream_root=f"http://127.0.0.1:{self.upstream.server_port}", instance_id="capture-test",
        )
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.gateway_thread = threading.Thread(target=self.gateway_server.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.gateway_thread.start()
        self.addCleanup(self._close_servers)

    def _close_servers(self):
        _StreamingUpstream.release_systemone.set()
        self.gateway_server.shutdown()
        self.gateway_server.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()

    def _post(self, path, payload):
        root = f"http://127.0.0.1:{self.gateway_server.server_port}"
        body = json.dumps(payload).encode()
        request = Request(root + path, data=body, headers={"Content-Type": "application/json"},
                          method="POST")
        with urlopen(request, timeout=5) as response:
            return response.status, response.read()

    def _drain(self, model="m"):
        deadline = time.monotonic() + 3
        while self.store.active_requests(model):
            self.assertLess(time.monotonic(), deadline, "finished request remained leased")
            time.sleep(0.01)

    def test_off_writes_nothing(self):
        status, _ = self._post("/omp-profile/claude/v1/systemone",
                               {"model": "m", "questions": [{"name": "effort"}]})
        self.assertEqual(status, 200)
        self._drain()
        self.assertFalse((self.root / "captured").exists())
        gateway.corpus.capture_on(["decision"], max_items=10, minutes=60,
                                   root=self.root, now=time.time())
        gateway.corpus.capture_off(root=self.root)
        status, _ = self._post("/omp-profile/claude/v1/systemone",
                               {"model": "m", "questions": [{"name": "effort"}]})
        self.assertEqual(status, 200)
        self._drain()
        self.assertFalse((self.root / "captured").exists())

    def test_matching_purpose_is_captured_once_with_metadata_and_response(self):
        gateway.corpus.capture_on(["decision"], max_items=10, minutes=60,
                                   root=self.root, now=time.time())
        payload = {"model": "m", "questions": [{"name": "effort"}]}
        sent = json.dumps(payload).encode()
        for _ in range(2):
            status, body = self._post("/omp-profile/claude/v1/systemone", payload)
            self.assertEqual(status, 200)
            self.assertEqual(body, _StreamingUpstream.SYSTEMONE_RESPONSE)
        self._drain()
        purpose_dir = self.root / "captured" / "decision:effort"
        files = sorted(purpose_dir.glob("*.json"))
        self.assertEqual(len(files), 2)
        by_kind = {}
        for path in files:
            doc = json.loads(path.read_text(encoding="utf-8"))
            by_kind["request" if "request" in doc else "response"] = doc
        self.assertEqual(set(by_kind), {"request", "response"})
        self.assertEqual(by_kind["request"]["request"], payload)
        self.assertEqual(by_kind["request"]["meta"]["profile"], "claude")
        self.assertEqual(by_kind["request"]["meta"]["purpose"], "decision:effort")
        self.assertEqual(by_kind["request"]["meta"]["model"], "m")
        self.assertEqual(by_kind["request"]["meta"]["ollama_version"], "0.35.0-fake")
        self.assertEqual(by_kind["request"]["meta"]["omp_version"], None)
        self.assertEqual(by_kind["request"]["meta"]["omp_sha"], None)
        self.assertIn("no omp version token", by_kind["request"]["meta"]["omp_note"])
        self.assertEqual(oct(purpose_dir.stat().st_mode & 0o777), "0o700")
        for path in files:
            self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")

    def test_non_matching_purpose_is_not_captured(self):
        gateway.corpus.capture_on(["main"], max_items=10, minutes=60,
                                   root=self.root, now=time.time())
        status, _ = self._post("/omp-profile/claude/v1/systemone",
                               {"model": "m", "questions": [{"name": "effort"}]})
        self.assertEqual(status, 200)
        self._drain()
        self.assertFalse((self.root / "captured").exists())

    def test_cap_is_respected(self):
        gateway.corpus.capture_on(["decision"], max_items=1, minutes=60,
                                   root=self.root, now=time.time())
        first = {"model": "m", "questions": [{"name": "a"}], "state": "s1"}
        second = {"model": "m", "questions": [{"name": "a"}], "state": "s2"}
        self._post("/omp-profile/claude/v1/systemone", first)
        self._post("/omp-profile/claude/v1/systemone", second)
        self._drain()
        files = sorted((self.root / "captured" / "decision:a").glob("*.json"))
        self.assertEqual(len(files), 2)
        first_sha = hashlib.sha256(json.dumps(first).encode()).hexdigest()
        response_sha = hashlib.sha256(_StreamingUpstream.SYSTEMONE_RESPONSE).hexdigest()
        self.assertEqual({path.name for path in files},
                         {f"{first_sha}.json", f"{response_sha}.response.json"})

    def test_cap_is_per_purpose(self):
        gateway.corpus.capture_on(["decision", "main"], max_items=1, minutes=60,
                                   root=self.root, now=time.time())
        self._post("/omp-profile/claude/v1/systemone",
                   {"model": "m", "questions": [{"name": "a"}], "state": "s1"})
        self._post("/omp-profile/claude/v1/systemone",
                   {"model": "m", "questions": [{"name": "a"}], "state": "s2"})
        root = f"http://127.0.0.1:{self.gateway_server.server_port}"
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                           "tools": [{"type": "function", "function": {"name": "f"}}]}).encode()
        request = Request(root + "/omp-profile/claude/responses", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        _StreamingUpstream.release_stream.set()
        with urlopen(request, timeout=5) as response:
            response.read()
        self._drain()
        decision_files = sorted((self.root / "captured" / "decision:a").glob("*.json"))
        main_files = sorted((self.root / "captured" / "main").glob("*.json"))
        self.assertEqual(len(decision_files), 2)
        self.assertEqual(len(main_files), 1)

    def test_expiry_disables_capture(self):
        gateway.corpus.capture_on(["decision"], max_items=10, minutes=60,
                                   root=self.root, now=time.time() - 7200)
        status, _ = self._post("/omp-profile/claude/v1/systemone",
                               {"model": "m", "questions": [{"name": "effort"}]})
        self.assertEqual(status, 200)
        self._drain()
        self.assertFalse((self.root / "captured").exists())

    def test_capture_failure_still_forwards_with_200(self):
        gateway.corpus.capture_on(["decision"], max_items=10, minutes=60,
                                   root=self.root, now=time.time())
        captured = self.root / "captured"
        captured.parent.mkdir(parents=True, exist_ok=True)
        captured.write_text("not a directory", encoding="utf-8")
        try:
            status, body = self._post("/omp-profile/claude/v1/systemone",
                                      {"model": "m", "questions": [{"name": "effort"}]})
        finally:
            captured.unlink()
        self.assertEqual(status, 200)
        self.assertEqual(body, _StreamingUpstream.SYSTEMONE_RESPONSE)
        self._drain()

    def test_capture_on_rejects_purposes_the_gateway_cannot_capture(self):
        for bad in ("a/b", "a\\b", ".", ".."):
            with self.subTest(purpose=bad):
                with self.assertRaises(gateway.corpus.CorpusError):
                    gateway.corpus.capture_on([bad], max_items=1, minutes=60,
                                              root=self.root, now=time.time())
        self.assertFalse((self.root / "capture.json").exists())
        spec = gateway.corpus.capture_on(["decision"], max_items=1, minutes=60,
                                         root=self.root, now=time.time())
        self.assertEqual(spec["purposes"], ["decision"])

    def test_gateway_capture_uses_the_shared_purpose_validator(self):
        gateway.corpus.capture_on(["decision"], max_items=10, minutes=60,
                                   root=self.root, now=time.time())
        with mock.patch.object(gateway.corpus, "valid_capture_purpose",
                               return_value=False) as check:
            self.assertIsNone(gateway._capture_body(purpose="decision", meta={}, body=b"{}",
                                                    kind="request"))
        check.assert_called_once_with("decision")

    def test_repo_capture_root_is_refused(self):
        probe = Path(gateway.corpus.REPO_ROOT) / "runs" / "capture-probe"
        with mock.patch.object(gateway.corpus, "CORPORA", probe):
            gateway._capture_cache.pop(str(gateway.corpus.capture_spec_path()), None)
            spec = gateway.corpus.capture_spec_path()
            spec.parent.mkdir(parents=True, exist_ok=True)
            spec.write_text(json.dumps({"enabled": True, "purposes": ["decision"],
                                        "max_items": 10, "until": time.time() + 600}),
                            encoding="utf-8")
            try:
                self.assertIsNone(gateway.capture_body(purpose="decision", meta={}, body=b"{}",
                                                       kind="request"))
            finally:
                spec.unlink(missing_ok=True)
        self.assertFalse((probe / "captured").exists())


if __name__ == "__main__":
    unittest.main()
