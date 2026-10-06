"""CLI contracts that must hold before any backend is touched: overlay parsing, the golden-write refusals (the golden
for a spec is its default launch; variants are measured as ab's B leg), and what a banked receipt keeps."""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from localbench import __main__ as main_mod
from localbench import smol
from localbench.__main__ import DETAILS_NOT_BANKED, _overlay, _receipt_view

# Status snapshots are host-scoped user state; suite status calls stay in scratch.
from localbench.workloads import FIXTURES, MEM_CONFIG, TIERS

# Status snapshots are host-scoped user state; suite status calls stay in scratch.
main_mod.STATUS_SNAPSHOT = Path(tempfile.mkdtemp(prefix="status-snapshot-")) / "status-snapshot.json"

ROOT = Path(__file__).resolve().parent.parent
FTS = FIXTURES / "omp" / "child-config-mem-fts.yml"


# A CLI subprocess does not see tests/__init__.py's redirects; under this HOME its audit rows and smol state are
# scratch, never the user's ~/.localbench (2026-09-25: the aa refusal tests wrote rows into the live ledger).
SCRATCH_HOME = tempfile.mkdtemp(prefix="localbench-test-home-")


def cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "localbench", *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=60, env={**os.environ, "HOME": SCRATCH_HOME, **(env or {})})


class Overlay(unittest.TestCase):
    def test_explicit_default_path_is_the_default(self):
        self.assertEqual(_overlay("fixtures/omp/child-config-mem.yml"), MEM_CONFIG)

    def test_missing_file_is_a_usage_error(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _overlay("/nonexistent/overlay.yml")


class GoldenWriteRefusals(unittest.TestCase):
    """Refusals exit before open_backend: a nonexistent model name proves nothing was contacted."""

    def test_variant_overlay_cannot_write_a_golden(self):
        p = cli("aa", "ollama:no-such-model", "--tiers", "mem", "--mem-config", str(FTS), "--write-golden",
                "--force-load")
        self.assertEqual(p.returncode, 1)
        self.assertIn("refusing --write-golden", p.stderr)

    def test_server_flag_cannot_write_a_golden(self):
        p = cli("aa", "mlx-serve:/nonexistent", "--server-arg=--mtp", "--write-golden", "--force-load")
        self.assertEqual(p.returncode, 1)
        self.assertIn("refusing --write-golden", p.stderr)

    def test_a_non_default_round_count_cannot_write_a_golden(self):
        p = cli("aa", "ollama:no-such-model", "--tiers", "mem", "--mem-rounds", "9", "--write-golden",
                "--force-load")
        self.assertEqual(p.returncode, 1)
        self.assertIn("refusing --write-golden", p.stderr)
        self.assertIn("--mem-rounds", p.stderr)

    def test_zero_rounds_is_a_usage_error_before_any_backend(self):
        p = cli("aa", "ollama:no-such-model", "--mem-rounds", "0", "--write-golden", "--force-load")
        self.assertEqual(p.returncode, 2)
        self.assertIn("at least 1", p.stderr)
        self.assertNotIn("refusing --write-golden", p.stderr)

    def test_a_separate_smol_model_cannot_write_a_golden(self):
        # A separate memory model is a study leg; the golden for a spec is its one-artifact default launch.
        # In-process with open_backend faked: a refusal that stopped working must not reach a real server.
        args = argparse.Namespace(backend="ollama:no-such-model", tiers="mem", repeats=1, allow_busy=False,
                                  purge=False, wait_idle=0, server_arg=None, mem_config=MEM_CONFIG,
                                  mem_rounds=main_mod.MEM_ROUNDS, smol_model="no-such-smol", write_golden=True)
        err = io.StringIO()
        with mock.patch.object(main_mod, "open_backend", side_effect=AssertionError("backend opened")), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = main_mod.cmd_aa(args)
        self.assertEqual(rc, 1)
        self.assertIn("refusing --write-golden", err.getvalue())
        self.assertIn("--smol-model", err.getvalue())


class DuplicateTiers(unittest.TestCase):
    def test_repeated_tier_is_rejected_before_measurement(self):
        # A repeated tier reuses case IDs: a later leg would overwrite earlier verdict diagnostics in a receipt.
        for command, specs in (("run", ["ollama:no-such-model"]),
                               ("aa", ["ollama:no-such-model"]),
                               ("ab", ["ollama:no-such-model", "ollama:no-such-model"])):
            with self.subTest(command=command), mock.patch.object(
                    main_mod, {"run": "cmd_run", "aa": "cmd_aa", "ab": "cmd_ab"}[command],
                    side_effect=AssertionError("measurement handler reached")):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                    main_mod.main([command, *specs, "--tiers", "replay,replay"])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("duplicate tier", stderr.getvalue())




class RecordTokenizerIdentity(unittest.TestCase):
    def test_record_sidecar_keeps_the_backend_reported_tokenizer_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = root / "fixtures"
            omp = fixtures / "omp"
            omp.mkdir(parents=True)
            child_config = omp / "child-config.yml"
            child_config.write_text("memory: off\n")
            agent_dir = root / "runs" / "omp-agent"
            agent_config = agent_dir / "config.yml"
            agent_config.parent.mkdir(parents=True)
            agent_config.write_text("agents: []\n")
            identity = "ollama-ggml-sha256:measured"

            class Backend:
                name = "ollama"
                base_url = "http://127.0.0.1:11434/v1"

                def isolate(self, model):
                    return []

                def fingerprint(self, model):
                    return {"loaded_context": 8192}

                def tokenizer_identity(self, model):
                    return identity

                def pins(self, model):
                    return {"backend": self.name, "model": model}

            class CaptureProxy:
                def __init__(self, _url, calls, *, save_dir, label):
                    self.calls, self.save_dir, self.label = calls, save_dir, label

                def __enter__(self):
                    body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
                    (self.save_dir / f"{self.label}-000.json").write_text(json.dumps(body))
                    self.calls.write_text(json.dumps({"prompt_tokens": 246, "tools": 1}) + "\n")
                    return self

                def __exit__(self, *exc):
                    return None

            args = argparse.Namespace(backend="ollama:test-model", label="tokenizer", omp_flags=[])
            with mock.patch.object(main_mod, "ROOT", root), \
                    mock.patch.object(main_mod, "FIXTURES", fixtures), \
                    mock.patch.object(main_mod, "CHILD_CONFIG", child_config), \
                    mock.patch.object(main_mod, "AGENT_DIR", agent_dir), \
                    mock.patch.object(main_mod, "AGENT_CONFIG", agent_config), \
                    mock.patch.object(main_mod, "open_backend", return_value=contextlib.nullcontext((Backend(), "test-model"))), \
                    mock.patch.object(main_mod, "_mut", return_value=mock.Mock(gate=lambda _steps: None)), \
                    mock.patch.object(main_mod, "omp_pins", return_value={"omp_version": "18.4.3"}), \
                    mock.patch.object(main_mod, "sha16", return_value="child-config-sha"), \
                    mock.patch.object(main_mod, "child_env", return_value={}), \
                    mock.patch.object(main_mod, "omp_bin", return_value="omp"), \
                    mock.patch.object(main_mod.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")), \
                    mock.patch("localbench.proxy.Proxy", CaptureProxy), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main_mod.cmd_record(args), 0)

            sidecar = json.loads((omp / "tokenizer.meta.json").read_text())
            self.assertEqual(sidecar["backend_pins"]["tokenizer_identity"], identity)


class ReceiptView(unittest.TestCase):
    def test_receipts_keep_the_details_of_every_tier_a_reader_audits(self):
        # New tiers default to banked details; perf micro and passing conf/replay stay compact.
        results = [{"case": f"{t}.x", "tier": t, "verdict": "PASS" if t in ("conf", "replay") else None,
                    "detail": {"why": t}} for t in TIERS]
        summary = {"provenance": {}, "verdicts": {}, "metrics": {}, "conformance": {}, "run_dir": "r",
                   "results": results, "system": {"before": {"live": {}, "host": {}}}}
        kept = set(_receipt_view(summary)["details"])
        self.assertEqual(kept, {f"{t}.x" for t in TIERS} - {"micro.x", "conf.x", "replay.x"})
        self.assertEqual(DETAILS_NOT_BANKED, {"micro"})

    def test_nonpassing_conformance_and_replay_details_survive_banking(self):
        summary = {"provenance": {}, "verdicts": {}, "metrics": {}, "conformance": {}, "run_dir": "r",
                   "system": {"before": {"live": {}, "host": {}}},
                   "results": [
                       {"case": "conf.no_truncation_64k", "tier": "conf", "verdict": "FAIL",
                        "detail": {"short_tokens": 8000, "long_tokens": 8192}},
                       {"case": "replay.lean.prompt_tokens", "tier": "replay", "verdict": "FAIL",
                        "detail": {"recorded": 11433, "replayed": 10853}},
                       {"case": "replay.other.prompt_tokens", "tier": "replay", "verdict": "VOID",
                        "detail": {"recorded": 11433, "replayed": 10853, "reason": "different tokenizer"}},
                       {"case": "replay.full.prompt_tokens", "tier": "replay", "verdict": "PASS",
                        "detail": {"recorded": 73779, "replayed": 73779}},
                       {"case": "replay.full", "tier": "replay", "verdict": None,
                        "detail": {"prompt_tokens": 73779, "fixture_prompt_tokens": 73779}},
                   ]}
        receipt = _receipt_view(summary)
        self.assertEqual(receipt["details"], {
            "conf.no_truncation_64k": {"short_tokens": 8000, "long_tokens": 8192},
            "replay.lean.prompt_tokens": {"recorded": 11433, "replayed": 10853},
            "replay.other.prompt_tokens": {"recorded": 11433, "replayed": 10853,
                                            "reason": "different tokenizer"},
            "replay.full": {"prompt_tokens": 73779, "fixture_prompt_tokens": 73779},
        })
        all_pass = {**summary, "results": [{**r, "verdict": "PASS"} if r["verdict"] is not None else r
                                            for r in summary["results"]]}
        self.assertEqual(_receipt_view(all_pass)["details"], {
            "replay.full": {"prompt_tokens": 73779, "fixture_prompt_tokens": 73779},
        })



class NoGoldenConformance(unittest.TestCase):
    """A missing golden cannot conceal an unestablished required case."""

    def summary(self, conformance: dict) -> dict:
        return {
            "provenance": {"pins": {"host_id": "h", "backend": "ollama", "model": "m"},
                           "tiers": ["conf", "micro"], "label": "run"},
            "run_dir": "runs/r",
            "verdicts": {"contended": False, "must_fail": [], "preflight_problems": []},
            "metrics": {"micro.decode.tps": {"value": None, "void": "no samples", "spread": []}},
            "conformance": conformance,
            "system": {"before": {"host": {"chip": "test", "gpu_cores": 0, "mem_gb": 0, "macos": "test"}},
                       "during": {"resident_unknown_samples": 0}},
        }

    def judge_without_golden(self, conformance: dict) -> tuple[int, dict, str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs" / "r").mkdir(parents=True)
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod.golden, "GOLDENS",
                                                                             root / "goldens"), \
                    contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                rc = main_mod.judge(self.summary(conformance), as_json=True)
            report = (root / "runs" / "r" / "report.md").read_text()
            return rc, json.loads(out.getvalue()), report

    def test_must_void_is_unsound_without_a_golden_even_when_must_fail_is_empty(self):
        rc, result, report = self.judge_without_golden({
            "conf.required": {"level": "MUST", "verdict": "VOID"},
            "conf.second_required": {"level": "MUST", "verdict": "VOID"},
            "conf.optional": {"level": "SHOULD", "verdict": "VOID"},
        })
        self.assertEqual((rc, result["golden"], result["rows"]), (1, None, []))
        self.assertEqual(len(result["unsound"]), 2)
        for case in ("conf.required", "conf.second_required"):
            self.assertTrue(any(case in reason and "MUST VOID" in reason for reason in result["unsound"]),
                            result["unsound"])
        self.assertIn("UNSOUND", report)

    def test_should_and_performance_void_alone_remain_nonblocking_without_a_golden(self):
        rc, result, report = self.judge_without_golden({
            "conf.optional": {"level": "SHOULD", "verdict": "VOID"},
        })
        self.assertEqual((rc, result["golden"], result["unsound"]), (0, None, []))
        self.assertIn("# localbench · ollama / m (run) · SOUND", report)


class E2ECaseStatus(unittest.TestCase):
    def summary(self, *, unknown: int = 0, contended: bool = False, app_gpu_pct: float = 0) -> dict:
        return {
            "verdicts": {"contended": contended, "must_fail": [], "preflight_problems": [], "pins_changed": []},
            "system": {"during": {"resident_unknown_samples": unknown,
                                  "load": {"app_gpu_mean_pct": app_gpu_pct}}},
            "conformance": {"e2e.case.correct": {"level": "MUST", "verdict": "PASS"}},
        }

    def test_unknown_residency_voids_a_would_be_pass(self):
        self.assertEqual(main_mod._e2e_case_status(self.summary(unknown=1), "case"), "VOID")

    def test_known_isolation_and_app_only_load_preserve_a_pass(self):
        self.assertEqual(main_mod._e2e_case_status(self.summary(), "case"), "PASS")
        self.assertEqual(main_mod._e2e_case_status(self.summary(app_gpu_pct=80), "case"), "PASS")

    def test_known_second_model_remains_contended_not_unknown(self):
        summary = self.summary(contended=True)
        self.assertEqual(main_mod._e2e_case_status(summary, "case"), "VOID")
        reasons = main_mod.unsound(summary)
        self.assertTrue(any(reason.startswith("CONTENDED:") for reason in reasons), reasons)
        self.assertFalse(any("resident model state unknown" in reason for reason in reasons), reasons)

    def test_a_missing_residency_count_voids_and_is_unsound(self):
        # An incomplete sampler block is no proof of isolation: it must not read as zero unknown samples.
        summary = self.summary()
        del summary["system"]["during"]["resident_unknown_samples"]
        self.assertEqual(main_mod._e2e_case_status(summary, "case"), "VOID")
        self.assertTrue(any("residency sampling incomplete" in r for r in main_mod.unsound(summary)))
        del summary["system"]
        self.assertTrue(any("residency sampling incomplete" in r for r in main_mod.unsound(summary)))


class UnknownResidencyBank(unittest.TestCase):
    def bank(self, unknown: int | None) -> tuple[int, bool]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "r"
            run.mkdir(parents=True)
            summary = NoGoldenConformance().summary({})
            summary["system"]["before"]["live"] = {}
            if unknown is None:
                del summary["system"]["during"]["resident_unknown_samples"]
            else:
                summary["system"]["during"]["resident_unknown_samples"] = unknown
            (run / "summary.json").write_text(json.dumps(summary))
            receipts = root / "receipts"
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RECEIPTS", receipts), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = main_mod.cmd_bank(argparse.Namespace(run_dir=str(run), name="isolation", dry_run=False))
            return rc, (receipts / "isolation.json").exists()

    def test_unknown_residency_cannot_create_a_banked_receipt(self):
        self.assertEqual(self.bank(1), (1, False))

    def test_known_isolation_can_still_bank(self):
        self.assertEqual(self.bank(0), (0, True))

    def test_a_missing_residency_count_cannot_create_a_banked_receipt(self):
        self.assertEqual(self.bank(None), (1, False))


class UnknownResidencyComparisonBank(unittest.TestCase):
    def summary(self, label: str, unknown: int | None = 0, contended: bool = False) -> dict:
        return {
            "provenance": {"pins": {"host_id": "h", "backend": "ollama", "model": "m"},
                           "created": label, "label": label},
            "verdicts": {"contended": contended, "must_fail": [], "preflight_problems": [], "pins_changed": []},
            "metrics": {}, "conformance": {}, "run_dir": f"runs/{label}", "results": [],
            "system": {"before": {"live": {}, "host": {}},
                       "during": {} if unknown is None else {"resident_unknown_samples": unknown}},
        }

    def run_aa(self, unknowns: list[int], *, write_golden: bool, contended: tuple[bool, bool] = (False, False)
               ) -> tuple[int, dict | None, bool]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipts, goldens = root / "receipts", root / "goldens"
            summaries = [self.summary(f"aa{i}", unknown, contended[i]) for i, unknown in enumerate(unknowns)]
            backend = object()
            mutator = mock.Mock()
            mutator.gate.return_value = None
            mutator.detail = {}
            args = argparse.Namespace(
                backend="ollama:m", tiers="conf", repeats=1, allow_busy=False, purge=False, wait_idle=0,
                server_arg=[], mem_config=MEM_CONFIG, mem_rounds=main_mod.MEM_ROUNDS,
                write_golden=write_golden, dry_run=False, smol_model=None)
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RECEIPTS", receipts), \
                    mock.patch.object(main_mod.golden, "GOLDENS", goldens), \
                    mock.patch.object(main_mod, "_mut", return_value=mutator), \
                    mock.patch.object(main_mod, "open_backend", side_effect=lambda *_: contextlib.nullcontext(
                        (backend, "m"))), mock.patch.object(main_mod, "execute", side_effect=summaries), \
                    mock.patch.object(main_mod.golden, "pin_diff", return_value=[]), \
                    mock.patch.object(main_mod.golden, "from_aa", return_value=({}, [])), \
                    mock.patch.object(main_mod.golden, "golden_path", return_value=goldens / "g.json"), \
                    mock.patch.object(main_mod.golden, "load", return_value={}), \
                    mock.patch.object(main_mod.golden, "merge", return_value=({}, [])), \
                    mock.patch.object(main_mod.golden, "write") as write_golden_fn, \
                    mock.patch.object(main_mod.render, "show", return_value="receipt"), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = main_mod.cmd_aa(args)
                receipt = receipts / "aa__ollama__m__aa0.json"
                doc = json.loads(receipt.read_text()) if receipt.exists() else None
                result = rc, doc, write_golden_fn.called
        return result

    def run_ab(self, unknown_leg: int | None, contended_leg: int | None = None) -> tuple[int, dict | None]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipts = root / "receipts"
            summaries = [self.summary(f"ab{i}", int(i == unknown_leg), i == contended_leg) for i in range(3)]
            backend = object()
            mutator = mock.Mock()
            mutator.gate.return_value = None
            mutator.detail = {}
            args = argparse.Namespace(
                a="ollama:a", b="ollama:b", pairs=1, tiers="conf", repeats=1, allow_busy=False, purge=False,
                wait_idle=0, server_arg=[], b_server_arg=[], mem_config=MEM_CONFIG, b_mem_config=None,
                mem_rounds=main_mod.MEM_ROUNDS, b_omp=None, b_mlx_serve=None, b_mlxfast=None,
                bank="ab-isolation", dry_run=False, smol_model=None, b_smol_model=None, side_regime=False)
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RECEIPTS", receipts), \
                    mock.patch.object(main_mod, "_mut", return_value=mutator), \
                    mock.patch.object(main_mod, "_binaries_for_leg", return_value=contextlib.nullcontext()), \
                    mock.patch.object(main_mod, "open_backend", side_effect=lambda *_: contextlib.nullcontext(
                        (backend, "m"))), mock.patch.object(main_mod, "execute", side_effect=summaries), \
                    mock.patch.object(main_mod.golden, "arm_pin_drift", return_value=[]), \
                    mock.patch.object(main_mod.golden, "load_balance", return_value={"favours": None}), \
                    mock.patch.object(main_mod.golden, "ab_table", return_value=[]), \
                    mock.patch.object(main_mod.render, "show", return_value="receipt"), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = main_mod.cmd_ab(args)
                path = receipts / "ab-isolation.json"
                doc = json.loads(path.read_text()) if path.exists() else None
        return rc, doc

    def test_unknown_aa_pair_does_not_bank_or_write_a_golden(self):
        rc, doc, golden_written = self.run_aa([1, 0], write_golden=True)
        self.assertEqual(rc, 1)
        self.assertIsNone(doc)
        self.assertFalse(golden_written)

    def test_aa_leg_without_a_residency_count_does_not_bank_or_write_a_golden(self):
        rc, doc, golden_written = self.run_aa([0, None], write_golden=True)
        self.assertEqual((rc, doc, golden_written), (1, None, False))

    def test_known_aa_pair_still_banks(self):
        rc, doc, _ = self.run_aa([0, 0], write_golden=False)
        self.assertEqual(rc, 0)
        self.assertIsNotNone(doc)

    def test_unsound_aa_pair_banks_an_unsound_receipt_but_writes_no_golden(self):
        # README: an unsound pair banks its receipt but leaves the golden unwritten (kit-unsound-pair-banking-cku).
        rc, doc, golden_written = self.run_aa([0, 0], write_golden=True, contended=(False, True))
        self.assertEqual(rc, 1)
        self.assertEqual(doc["verdict"], "UNSOUND")
        self.assertTrue(any("CONTENDED" in p or "contend" in p.lower() for p in doc["problems"]), doc["problems"])
        self.assertFalse(golden_written)

    def test_sound_aa_pair_banks_and_writes_the_golden(self):
        rc, doc, golden_written = self.run_aa([0, 0], write_golden=True)
        self.assertEqual((rc, doc["verdict"], doc["problems"]), (0, "SOUND", []))
        self.assertTrue(golden_written)

    def test_unknown_ab_leg_does_not_bank_receipt(self):
        rc, doc = self.run_ab(1)
        self.assertEqual(rc, 1)
        self.assertIsNone(doc)

    def test_unsound_ab_leg_banks_a_diagnostic_receipt_and_exits_1(self):
        rc, doc = self.run_ab(None, contended_leg=1)
        self.assertEqual(rc, 1)
        self.assertEqual(doc["verdict"], "UNSOUND")
        self.assertTrue(doc["problems"] and all(p.startswith("ab1: ") for p in doc["problems"]), doc["problems"])

    def test_known_ab_legs_still_bank(self):
        rc, doc = self.run_ab(None)
        self.assertEqual((rc, doc["verdict"]), (0, "SOUND"))


class OllamaAutoUpdate(unittest.TestCase):
    """`ollama-app auto-update` on a fixture Ollama.app db.sqlite: backup before write, audit row, read-back."""

    def setUp(self):
        import sqlite3

        from localbench import audit, ollama_app
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.db = self.dir / "db.sqlite"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE settings (id INTEGER PRIMARY KEY, auto_update_enabled BOOLEAN, models TEXT)")
        con.execute("INSERT INTO settings VALUES (1, 1, '')")
        con.commit()
        con.close()
        self.rollback, self.updates = self.dir / "rollback", self.dir / "updates"
        for patch in (mock.patch.dict(os.environ, {ollama_app.DB_ENV: str(self.db)}),
                      mock.patch.object(ollama_app, "ROLLBACK", self.rollback),
                      mock.patch.object(ollama_app, "UPDATES", self.updates),
                      mock.patch.object(audit, "AUDIT_PATH", self.dir / "audit.jsonl")):
            patch.start()
            self.addCleanup(patch.stop)
        self.audit = audit

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main_mod.main(["ollama-app", "auto-update", *argv])
        return rc, out.getvalue(), err.getvalue()

    def flag(self) -> bool | None:
        from localbench import sysstats
        return sysstats.ollama_auto_update(self.db)

    def rows(self) -> list[dict]:
        return [r for r in self.audit.rows() if r["verb"] == "ollama-app auto-update"]

    def test_off_on_status_round_trip_backs_up_and_audits(self):
        rc, out, _ = self.run_cli("status")
        self.assertEqual((rc, out.splitlines()[0]), (1, "auto-update: ON"))
        self.assertEqual(self.run_cli("off")[0], 0)
        self.assertIs(self.flag(), False)
        backups = sorted(self.rollback.glob("ollama-db-*.sqlite"))
        self.assertEqual(len(backups), 1)
        # The backup is the db as it was before the write: still ON.
        from localbench import sysstats
        self.assertIs(sysstats.ollama_auto_update(backups[0]), True)
        rc, out, _ = self.run_cli("status")
        self.assertEqual((rc, out.splitlines()), (0, ["auto-update: OFF", "staged update: none"]))
        self.assertEqual(self.run_cli("on")[0], 0)
        self.assertIs(self.flag(), True)
        self.assertEqual(len(list(self.rollback.glob("ollama-db-*.sqlite"))), 2)
        rows = self.rows()
        self.assertEqual([(r["outcome"], r["detail"].get("auto_update")) for r in rows],
                         [("done", False), ("done", True)])
        self.assertTrue(all(Path(r["detail"]["backup"]).is_file() for r in rows))

    def test_dry_run_writes_nothing(self):
        rc, out, _ = self.run_cli("off", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("auto_update_enabled = 0", out)
        self.assertIs(self.flag(), True)
        self.assertFalse(self.rollback.exists())
        self.assertEqual(self.rows(), [])

    def test_requesting_the_current_state_is_a_noop(self):
        rc, out, _ = self.run_cli("on")
        self.assertEqual(rc, 0)
        self.assertIn("already ON", out)
        self.assertFalse(self.rollback.exists())
        self.assertEqual([(r["actions"], r["detail"].get("noop") is not None) for r in self.rows()], [([], True)])

    def test_status_reports_a_staged_update_even_when_off(self):
        self.run_cli("off")
        self.updates.mkdir()
        (self.updates / "Ollama-darwin.zip").write_bytes(b"x")
        rc, out, _ = self.run_cli("status")
        self.assertEqual(rc, 0)
        self.assertIn("staged update: Ollama-darwin.zip", out)
        self.assertIn("regardless of the setting", out)

    def doctor_row(self) -> dict:
        from localbench import doctor
        with mock.patch.object(doctor.backends.Ollama, "pins", return_value={"backend_version": "0.35.0"}):
            return doctor.check_ollama(False)

    def test_doctor_fails_when_auto_update_is_on(self):
        row = self.doctor_row()   # the fixture db starts ON
        self.assertEqual((row["status"], row["fix"]), ("FAIL", "localbench ollama-app auto-update off"))
        self.run_cli("off")
        self.assertEqual(self.doctor_row()["status"], "PASS")

    def test_doctor_warns_on_a_staged_update_even_when_off(self):
        self.run_cli("off")
        self.updates.mkdir()
        (self.updates / "Ollama-darwin.zip").write_bytes(b"x")
        row = self.doctor_row()
        self.assertEqual(row["status"], "WARN")
        self.assertIn("staged bundle", row["fix"])
        self.assertIn("Ollama-darwin.zip", row["detail"])

    def test_doctor_warns_when_the_switch_is_unknown(self):
        self.db.unlink()
        row = self.doctor_row()
        self.assertEqual((row["status"], row["fix"]), ("WARN", "localbench ollama-app auto-update status"))


class StatusSnapshot(unittest.TestCase):
    def test_suite_snapshot_target_is_not_user_home(self):
        self.assertNotEqual(main_mod.STATUS_SNAPSHOT, Path.home() / ".localbench" / "status-snapshot.json")

    def status_sandbox(self, sampled_at: float, host_id: str):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        snapshot = root / "status-snapshot.json"
        snapshot.write_text(json.dumps({"sampled_at": sampled_at, "data": {"host_id": host_id, "goldens": []}}))
        patches = [
            mock.patch.object(main_mod, "STATUS_SNAPSHOT", snapshot),
            mock.patch.object(main_mod.sysstats, "host", return_value={"host_id": "current-host"}),
            mock.patch.object(main_mod, "golden_states", return_value=[]),
            mock.patch.object(main_mod, "omp_pins", return_value={"omp_version": "v", "omp_sha": "sha", "omp_child_config": None}),
            mock.patch.object(main_mod.sysstats, "ollama_residents", return_value=[]),
            mock.patch.object(main_mod.smol, "load_state", return_value=None),
            mock.patch.object(main_mod.sysstats, "gpu_time_by_pid", return_value={}),
            mock.patch.object(main_mod.sysstats, "gpu_share", return_value={}),
            mock.patch.object(main_mod.sysstats, "omp_processes", return_value=[]),
            mock.patch.object(main_mod.park, "stuck_sessions", return_value=[]),
            mock.patch.object(main_mod.quiet, "paused", return_value=False),
            mock.patch.object(main_mod.gateway, "status", return_value={"test": True}),
            mock.patch.object(main_mod.time, "sleep"),
            mock.patch.object(main_mod.park, "STATE", root / "park-state.json"),
        ]
        return temp, snapshot, patches

    def test_fresh_same_host_snapshot_is_served_without_rewrite(self):
        now = 10_000.0
        temp, snapshot, patches = self.status_sandbox(now - 1, "current-host")
        try:
            before = snapshot.read_bytes()
            with mock.patch.object(main_mod.time, "time", return_value=now):
                with contextlib.ExitStack() as stack:
                    for patcher in patches:
                        stack.enter_context(patcher)
                    result = main_mod.status_report()
            self.assertEqual(result["status_snapshot"]["age_s"], 1.0)
            self.assertEqual(snapshot.read_bytes(), before)
        finally:
            temp.cleanup()

    def test_expired_snapshot_is_refreshed(self):
        now = 10_000.0
        temp, snapshot, patches = self.status_sandbox(now - main_mod.STATUS_SNAPSHOT_TTL - 1, "current-host")
        try:
            with mock.patch.object(main_mod.time, "time", return_value=now):
                with contextlib.ExitStack() as stack:
                    for patcher in patches:
                        stack.enter_context(patcher)
                    result = main_mod.status_report()
            self.assertEqual(result["host_id"], "current-host")
            self.assertEqual(result["status_snapshot"]["age_s"], 0.0)
            self.assertFalse(result["status_snapshot"]["stale"])
            saved = json.loads(snapshot.read_text())
            self.assertEqual(saved["data"]["host_id"], "current-host")
            self.assertEqual(saved["sampled_at"], now)
        finally:
            temp.cleanup()

    def test_foreign_host_snapshot_is_ignored_and_replaced(self):
        now = 10_000.0
        temp, snapshot, patches = self.status_sandbox(now, "foreign-host")
        try:
            with mock.patch.object(main_mod.time, "time", return_value=now):
                with contextlib.ExitStack() as stack:
                    for patcher in patches:
                        stack.enter_context(patcher)
                    result = main_mod.status_report()
            self.assertEqual(result["host_id"], "current-host")
            self.assertEqual(json.loads(snapshot.read_text())["data"]["host_id"], "current-host")
        finally:
            temp.cleanup()


class StaleGraderRescore(unittest.TestCase):
    def test_a_v1_spec_campaign_is_refused_and_its_scores_stay_unchanged(self):
        from localbench.evaluation import EvaluationCampaign, canonical_sha256, make_varied_spec
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            spec = {**make_varied_spec(family="read", seed=11, phase="heldout"), "grader_version": "varied.v1"}
            path = root / "runs" / "eval-varied-v1"
            EvaluationCampaign.create(path, root=root,
                                      identity={"varied_specs": {"read-11": spec}, "profile": main_mod.VARIED_PROFILE},
                                      cases={"read-11": canonical_sha256(spec)}, profile=main_mod.VARIED_PROFILE)
            scores = path / "scores"
            scores.mkdir(exist_ok=True)
            (scores / "old.json").write_text('{"status": "PASS"}')
            before = {p.name: p.read_bytes() for p in scores.iterdir()}
            err = io.StringIO()
            # _evaluation_path requires the campaign under RUNS: point both roots at this temp clone.
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RUNS", root / "runs"), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = main_mod.cmd_eval_rescore(argparse.Namespace(campaign=str(path)))
            self.assertEqual(rc, 1)
            self.assertEqual({p.name: p.read_bytes() for p in scores.iterdir()}, before)
            self.assertEqual(len(err.getvalue().strip().splitlines()), 1)
            self.assertIn("varied.v1", err.getvalue())
            self.assertIn("varied.v2", err.getvalue())
            self.assertIn("new campaign identity", err.getvalue())


class PureJsonStdout(unittest.TestCase):
    """`--json` read commands print one JSON document on stdout and nothing else (agents pipe them into jq)."""

    def test_memory_json(self):
        json.loads(cli("memory", "--json").stdout)

    def test_status_json_runs_with_closed_stdin(self):
        host = {"host_id": "test-host"}
        running = {"omp_version": "18.0", "omp_sha": "test-sha", "omp_child_config": "test-child"}
        stdin = io.StringIO()
        stdin.close()
        output = io.StringIO()
        park_state = mock.Mock()
        park_state.exists.return_value = False
        with mock.patch("sys.stdin", stdin), \
                mock.patch.object(main_mod, "STATUS_SNAPSHOT", Path(SCRATCH_HOME) / ".localbench" / "status-snapshot.json"), \
                mock.patch.object(main_mod.sysstats, "host", return_value=host), \
                mock.patch.object(main_mod, "golden_states", return_value=[]), \
                mock.patch.object(main_mod, "omp_pins", return_value=running), \
                mock.patch.object(main_mod.sysstats, "ollama_residents", return_value=[]), \
                mock.patch.object(main_mod.smol, "load_state", return_value=None), \
                mock.patch.object(main_mod.sysstats, "gpu_time_by_pid", return_value=[]), \
                mock.patch.object(main_mod.time, "sleep"), \
                mock.patch.object(main_mod.sysstats, "ollama_auto_update", return_value=False), \
                mock.patch.object(main_mod.quiet, "paused", return_value=[]), \
                mock.patch.object(main_mod.park, "STATE", park_state), \
                mock.patch.object(main_mod.park, "stuck_sessions", return_value=[]), \
                mock.patch.object(main_mod.sysstats, "omp_processes", return_value=[]), \
                mock.patch.object(main_mod.gateway, "status", return_value={}), \
                mock.patch.object(main_mod.heavyslot, "holder", return_value=None), \
                mock.patch.object(main_mod.sysstats, "gpu_share", return_value=[]), \
                contextlib.redirect_stdout(output):
            rc = main_mod.main(["status", "--json"])
        self.assertEqual(rc, 0)
        status = json.loads(output.getvalue())
        self.assertEqual(status["host_id"], "test-host")
        self.assertEqual(status["running_omp"], {"omp_version": "18.0", "omp_sha": "test-sha"})

    @unittest.skipUnless((ROOT / "runs" / "observe.db").exists(), "no runs/observe.db (localbench watch never ran)")
    def test_report_json(self):
        json.loads(cli("report", "--since", "1h", "--json").stdout)


class Version(unittest.TestCase):
    def test_version_needs_no_subcommand_and_names_the_installed_release(self):
        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        p = cli("--version")
        self.assertEqual((p.returncode, p.stdout.strip()), (0, f"localbench {version}"))


class DataRoot(unittest.TestCase):
    """A non-editable install resolves the data root to site-packages; every answer from there is silently empty."""

    def test_a_root_without_fixtures_is_refused_with_a_way_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = cli("status", env={"LOCALBENCH_HOME": tmp})
        self.assertEqual((p.returncode, p.stdout), (2, ""))
        self.assertIn("LOCALBENCH_HOME=<clone>", p.stderr)
        self.assertIn("uv tool install -e .", p.stderr)

    def test_localbench_home_is_where_state_is_read_and_an_empty_window_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "fixtures" / "omp").mkdir(parents=True)
            as_json = cli("report", "--since", "1h", "--json", env={"LOCALBENCH_HOME": tmp})
            text = cli("report", "--since", "1h", env={"LOCALBENCH_HOME": tmp})
            db = (Path(tmp) / "runs" / "observe.db").exists()
        self.assertTrue(db, "report read another root's runs/observe.db")
        r = json.loads(as_json.stdout)
        self.assertEqual((as_json.returncode, r["samples"], r["gpu"], r["traffic"]), (0, 0, [], []))
        self.assertEqual((text.returncode, text.stdout), (0, ""))
        self.assertIn("no samples", text.stderr)


class TrafficUnits(unittest.TestCase):
    def test_nonzero_sampled_traffic_is_bytes_not_tokens_or_request_model_attribution(self):
        report = {"samples": 1, "covered_s": 60, "first": 1_000, "last": 1_060,
                  "gpu": [], "resident": [], "clients": [],
                  "traffic": [{"down": 2_000, "up": 150, "who": "omp", "cwd": "/project",
                               "server": "ollama", "while_resident": "qwen"}]}
        output = io.StringIO()
        with mock.patch.object(main_mod.observe, "report", return_value=report), \
                contextlib.redirect_stdout(output):
            rc = main_mod.cmd_report(argparse.Namespace(since=3600, json=False, by_purpose=False, by_profile=False,
                                                       requests=False, req_purpose=None, req_profile=None, limit=200))
        text = output.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("2 KB down", text)
        self.assertIn("150 B up", text)
        self.assertIn("response bytes", text)
        self.assertIn("request bytes", text)
        self.assertIn("sample-time resident", text)
        self.assertNotIn("generated tokens", text)


class ModelIdentityText(unittest.TestCase):
    def test_hf_installed_artifact_and_upstream_commit_have_distinct_provenance(self):
        hf = {"server": "mlx-serve", "name": "publisher/model", "repo": "publisher/model", "gb": 1.0,
              "installed_artifact": "a" * 12, "installed_artifact_source": ".hf_commit (bytes not verified)",
              "upstream_sha": "b" * 12, "upstream_modified": "2026-09-28T12:34",
              "upstream_date_source": "lastModified", "freshness": "update available"}
        ollama = {"server": "ollama", "name": "model", "source": "model", "digest": "c" * 12,
                  "gb": 1.0, "freshness": "current", "upstream_modified": None,
                  "upstream_date_source": "unavailable"}
        text = io.StringIO()
        with mock.patch.object(main_mod.models, "ollama_models", return_value=[ollama]), \
                mock.patch.object(main_mod.models, "mlx_models", return_value=[hf]), \
                mock.patch.object(main_mod.models, "omp_cpu_models", return_value=[]), \
                mock.patch.object(main_mod.models, "profiles", return_value=["default"]), \
                mock.patch.object(main_mod.models, "routes_by_model", return_value={}), \
                mock.patch.object(main_mod.models, "releases", return_value=[]), \
                contextlib.redirect_stdout(text):
            rc = main_mod.cmd_models(argparse.Namespace(days=1, json=False))
            hf["upstream_modified"] = None
            hf["upstream_date_source"] = "unavailable"
            missing_date = io.StringIO()
            with contextlib.redirect_stdout(missing_date):
                main_mod.cmd_models(argparse.Namespace(days=1, json=False))
            del hf["upstream_sha"]
            unavailable = io.StringIO()
            with contextlib.redirect_stdout(unavailable):
                main_mod.cmd_models(argparse.Namespace(days=1, json=False))
        rows = text.getvalue().splitlines()
        self.assertEqual(rc, 0)
        self.assertIn("c" * 12, next(line for line in rows if line.startswith("ollama")))
        self.assertIn("upstream release date: unavailable", text.getvalue())
        hf_row = next(line for line in rows if line.startswith("mlx-serve"))
        self.assertIn("a" * 12, hf_row)
        self.assertNotIn("b" * 12, hf_row)
        self.assertIn(".hf_commit (bytes not verified)", text.getvalue())
        self.assertIn("upstream bbbbbbbbbbbb (lastModified 2026-09-28T12:34)", text.getvalue())
        self.assertIn("upstream bbbbbbbbbbbb (date unavailable)", missing_date.getvalue())
        self.assertIn("upstream unavailable (date unavailable)", unavailable.getvalue())


class TeachingErrors(unittest.TestCase):
    """Bad input is a usage error (exit 2) that says what to give instead, never a traceback, and never on stdout."""

    def assert_usage(self, p: subprocess.CompletedProcess, hint: str):
        self.assertEqual((p.returncode, p.stdout), (2, ""), p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn(hint, p.stderr)

    def test_a_refusal_is_one_stderr_line_and_exit_1_not_a_traceback(self):
        # A plain RuntimeError is how the code refuses with a reason (park's resolver behind a wrapper omp, 2026-09-27).
        err = io.StringIO()
        with mock.patch.object(main_mod, "cmd_stats", side_effect=RuntimeError("cannot ask omp: set LOCALBENCH_OMP")), \
                contextlib.redirect_stderr(err):
            rc = main_mod.main(["stats"])
        self.assertEqual(rc, 1)
        self.assertEqual(err.getvalue().strip(), "localbench stats: cannot ask omp: set LOCALBENCH_OMP")

    def test_a_bug_keeps_its_traceback(self):
        with mock.patch.object(main_mod, "cmd_stats", side_effect=RecursionError("bug")), \
                self.assertRaises(RecursionError):
            main_mod.main(["stats"])

    def test_a_window_that_is_not_a_duration(self):
        self.assert_usage(cli("report", "--since", "banana"), "90m, 24h, 7d")

    def test_a_run_dir_without_a_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_usage(cli("compare", tmp), "runs/")
            self.assert_usage(cli("bank", tmp, "x"), "runs/")

    def test_a_file_that_is_not_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md") as f:
            f.write("# a markdown receipt\n")
            f.flush()
            self.assert_usage(cli("show", f.name), "receipt .json")


class SmolActionsParse(unittest.TestCase):
    def test_every_action_reaches_its_handler(self):
        # A flag read by cmd_smol but defined on no parser crashed `smol revert` with AttributeError (2026-09-25).
        # With a run alive every changing action refuses before touching a server or a profile.
        for action in ("set", "start", "stop", "autostart", "revert", "status"):
            err = io.StringIO()
            with self.subTest(action), mock.patch.object(smol, "load_state", return_value=None), \
                    mock.patch.object(main_mod, "_run_alive", return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(err):
                rc = main_mod.main(["smol", action])
                if action == "status":
                    self.assertEqual((rc, out.getvalue().startswith("smol: not managed")), (0, True))
                else:
                    self.assertEqual((rc, out.getvalue()), (1, ""))
                    self.assertIn("run is alive", err.getvalue())


class EvaluationCommands(unittest.TestCase):
    def test_offline_rescore_is_a_real_command_and_refuses_a_missing_campaign_cleanly(self):
        result = cli("eval", "rescore", "runs/does-not-exist")
        self.assertEqual(result.returncode, 1)
        self.assertIn("campaign", result.stderr.lower())
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_varied_eval_plan_freezes_two_read_and_two_edit_trials_without_model_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fixtures" / "omp").mkdir(parents=True)
            env = {"LOCALBENCH_HOME": tmp, "HOME": tmp}

            def plan(seed: str, phase: str) -> dict:
                result = cli("eval", "varied", "ollama:no-such-model", "--phase", phase, "--seed", seed,
                             "--trials", "2", "--dry-run", "--json", env=env)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("park", result.stderr)
                return json.loads(result.stdout)

            heldout = plan("71", "heldout")
            self.assertEqual(heldout["verb"], "eval varied")
            self.assertIn("park", heldout["would_refuse"])
            self.assertEqual(len([a for a in heldout["actions"] if a.startswith("trial ")]), 4)
            self.assertNotEqual(heldout["actions"], plan("72", "heldout")["actions"])
            self.assertNotEqual(heldout["actions"], plan("71", "exploratory")["actions"])
            self.assertFalse((root / "runs").exists())
            self.assertFalse((root / ".localbench" / "audit.jsonl").exists())
            too_few = cli("eval", "varied", "ollama:no-such-model", "--phase", "heldout", "--seed", "71",
                          "--trials", "1", "--dry-run", env=env)
            self.assertEqual(too_few.returncode, 2)
            self.assertFalse((root / "runs").exists())


class DecisionRun(unittest.TestCase):
    """`decision run` with decision.run_suite, the Sampler and the machine checks faked: no model, no network."""

    def receipt(self, problems: list[str], verdict: str = "NONE") -> dict:
        return {"kind": "run", "problems": problems, "verdict": {"compare": verdict, "baseline": {"kind": "none"}},
                "run": {"provenance": {"created": "20260930T000000Z", "pins": {"model": "nimble:latest"}},
                        "verdicts": {"contended": False, "must_fail": [], "preflight_problems": [],
                                     "pins_changed": {}},
                        "metrics": {}, "conformance": {}, "run_dir": None, "details": {},
                        "system": {"during": {}, "contention": [{"model": "foreign:latest"}], "load_spikes": []}}}

    def run_decision(self, *, problems=(), verdict="NONE", hosted=False, alive=False, feature=None, module=True,
                     busy=("CONTENDED: foreign:latest resident",), env=None, context_error=None,
                     mutation=None, spec="ollama:nimble:latest") -> tuple[int, list[dict], mock.Mock, str]:
        """`fake` stands for decision.run_suite (ollama: specs) and decision.run_laya (laya: specs, whose venv is a
        temp dir holding bin/python3)."""
        from localbench import decision, features
        suite = argparse.Namespace(name="judge-auto", role="auto_thinking", items=({"id": "i1"},))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            pkg.mkdir()
            if module:
                (pkg / "thinking.ts").write_text("export const classify = 1;\n")
            rows = [{"feature": "auto-thinking", "omp_package": "pi-coding-agent", "omp_module": "thinking.ts"}]
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python3").write_text("")

            def run_suite(*_args, feature=None, omp_module_sha=None, **_kw):
                if context_error is not None:
                    raise decision.ContextError(context_error)
                out = self.receipt(list(problems), verdict)
                if feature is not None:
                    out.update(feature=feature, omp_module_sha=omp_module_sha)
                return out

            fake = mock.Mock(side_effect=run_suite)
            err = io.StringIO()
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RUNS", root / "runs"), \
                    mock.patch.object(main_mod, "RECEIPTS", root / "receipts"), \
                    mock.patch.object(decision, "resolve_suite", return_value=suite), \
                    mock.patch.object(decision, "run_suite", fake), mock.patch.object(decision, "run_laya", fake), \
                    mock.patch.object(decision, "laya_venv", return_value=venv), \
                    mock.patch.object(features, "load", return_value=rows), \
                    mock.patch.object(features, "package_root", return_value=pkg), \
                    mock.patch.object(main_mod.park, "omp_package", return_value=root), \
                    mock.patch.object(main_mod, "omp_bin", return_value="omp"), \
                    mock.patch.object(main_mod.sysstats, "Sampler"), \
                    mock.patch.object(main_mod, "_run_alive", return_value=alive), \
                    mock.patch.object(main_mod, "_busy_check", return_value=({}, 50.0, list(busy))), \
                    mock.patch.object(main_mod, "_rev", return_value="rev"), \
                    mock.patch.dict(os.environ, env or {}, clear=False), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                if env is None:
                    os.environ.pop("TYPESAFE_API_KEY", None)
                rc = main_mod.cmd_decision(argparse.Namespace(
                    spec=spec, suite="judge-auto", hosted_arm=hosted, repeats=1, wait_idle=0,
                    feature=feature, dry_run=False, explain=False, json=False, mutation=mutation))
                banked = [json.loads(p.read_text()) for p in sorted((root / "receipts").glob("*.json"))]
                summaries = list((root / "runs").glob("*__decision__*/summary.json"))
                if banked:
                    self.assertEqual(len(summaries), 1, "the banked run has its summary.json")
                self.module_sha = features.module_sha(pkg / "thinking.ts") if module else None
        return rc, banked, fake, err.getvalue()

    def test_a_context_refusal_exits_1_with_one_line_banks_nothing_and_is_audited_refused(self):
        reason = "nimble:latest needs num_ctx 32768 but is loaded at 8192 and in use by omp pid 4242; not reloaded"
        mutation = main_mod.Mutation("decision run", [], audited=False)
        rc, banked, run_suite, err = self.run_decision(context_error=reason, mutation=mutation)
        self.assertEqual((rc, banked), (1, []))
        self.assertEqual(err.strip().splitlines(), [reason])
        self.assertEqual((mutation.outcome, mutation.detail.get("reason")), ("refused", reason))
        self.assertIsNone(run_suite.call_args.kwargs["in_use"])
        self.assertEqual(run_suite.call_args.args[0], "http://127.0.0.1:11434")

    def test_a_not_better_comparison_banks_its_verdict_and_exits_1(self):
        problem = "not better than route ollama/qwen3.8:27b-mlx: WORSE"
        rc, banked, _, _ = self.run_decision(problems=[problem], verdict="WORSE")
        self.assertEqual(rc, 1)
        self.assertEqual((banked[0]["verdict"]["compare"], banked[0]["problems"]), ("WORSE", [problem]))

    def test_feature_stamps_the_installed_module_sha_on_the_receipt(self):
        rc, banked, run_suite, _ = self.run_decision(feature="auto-thinking")
        self.assertEqual(rc, 0)
        self.assertIsNotNone(self.module_sha)
        self.assertEqual(run_suite.call_args.kwargs["omp_module_sha"], self.module_sha)
        self.assertEqual((banked[0]["feature"], banked[0]["omp_module_sha"]), ("auto-thinking", self.module_sha))

    def test_an_unknown_feature_is_a_usage_error_before_any_request(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_decision(feature="no-such-feature")
        self.assertEqual(caught.exception.code, 2)

    def test_a_feature_whose_module_is_gone_is_refused(self):
        rc, banked, run_suite, err = self.run_decision(feature="auto-thinking", module=False)
        self.assertEqual((rc, banked, run_suite.called), (1, [], False))
        self.assertIn("not in the installed", err)

    def test_clean_run_under_foreign_load_banks_and_exits_0_never_contended(self):
        # Side-model law: a co-resident model is the measured condition, never a refusal or a CONTENDED void.
        rc, banked, run_suite, err = self.run_decision()
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(banked), 1)
        self.assertEqual(banked[0]["problems"], [])
        self.assertFalse(banked[0]["run"]["verdicts"]["contended"])
        self.assertEqual(run_suite.call_args.args[0], "http://127.0.0.1:11434")
        self.assertIsNone(run_suite.call_args.kwargs["hosted"])

    def test_problems_bank_a_receipt_carrying_them_and_exit_1(self):
        problem = "model digest of 'nimble:latest' unreadable from http://127.0.0.1:11434/api/tags: the run is not pinned"
        rc, banked, _, err = self.run_decision(problems=[problem])
        self.assertEqual(rc, 1)
        self.assertEqual(banked[0]["problems"], [problem])
        self.assertIn(problem, err)

    def test_hosted_arm_without_key_is_refused_before_any_request(self):
        rc, banked, run_suite, err = self.run_decision(hosted=True)
        self.assertEqual(rc, 1)
        self.assertFalse(run_suite.called)
        self.assertEqual(banked, [])
        self.assertIn("TYPESAFE_API_KEY", err)

    def test_hosted_arm_with_key_runs_both_arms(self):
        rc, _, run_suite, _ = self.run_decision(hosted=True, env={"TYPESAFE_API_KEY": "k"})
        self.assertEqual(rc, 0)
        self.assertIsNotNone(run_suite.call_args.kwargs["hosted"])

    def test_refused_while_a_localbench_run_is_alive(self):
        rc, banked, run_suite, err = self.run_decision(alive=True)
        self.assertEqual(rc, 1)
        self.assertFalse(run_suite.called)
        self.assertEqual(banked, [])
        self.assertIn("run is alive", err)


class DecisionRunLaya(unittest.TestCase):
    """`decision run laya:<repo>` keeps every guard of the ollama path: same harness, decision.run_laya faked."""

    receipt = DecisionRun.receipt
    _run = DecisionRun.run_decision

    def run_decision(self, **kw):
        return self._run(spec="laya:aac6fef/laya-mlx", **kw)

    def test_clean_run_under_foreign_load_goes_through_run_laya_banks_and_exits_0(self):
        from localbench import decision
        rc, banked, run_laya, err = self.run_decision()
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(banked), 1)
        self.assertFalse(banked[0]["run"]["verdicts"]["contended"])
        self.assertEqual(run_laya.call_args.args[0], decision.LayaSpec("aac6fef/laya-mlx"))
        self.assertIsNone(run_laya.call_args.kwargs["hosted"])

    def test_problems_bank_a_receipt_carrying_them_and_exit_1(self):
        problem = "the laya checkout of /x/laya_mlx/__init__.py has uncommitted changes"
        rc, banked, _, err = self.run_decision(problems=[problem])
        self.assertEqual(rc, 1)
        self.assertEqual(banked[0]["problems"], [problem])
        self.assertIn(problem, err)

    def test_refused_while_a_localbench_run_is_alive(self):
        rc, banked, run_laya, err = self.run_decision(alive=True)
        self.assertEqual((rc, banked, run_laya.called), (1, [], False))
        self.assertIn("run is alive", err)

    def test_hosted_arm_without_key_is_refused_before_any_request(self):
        rc, banked, run_laya, err = self.run_decision(hosted=True)
        self.assertEqual((rc, banked, run_laya.called), (1, [], False))
        self.assertIn("TYPESAFE_API_KEY", err)

    def test_a_feature_whose_module_is_gone_is_refused(self):
        rc, banked, run_laya, err = self.run_decision(feature="auto-thinking", module=False)
        self.assertEqual((rc, banked, run_laya.called), (1, [], False))
        self.assertIn("not in the installed", err)


class DecisionRunAllowEvict(unittest.TestCase):
    """`decision run --allow-evict` reaches run_suite through the real parser; without it the eviction guard holds."""

    def run_main(self, *flags: str) -> mock.Mock:
        from localbench import audit, decision
        suite = argparse.Namespace(name="judge-auto", role="decision.noul", items=({"id": "i1"},))
        receipt = DecisionRun.receipt(None, [])
        fake = mock.Mock(return_value=receipt)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RUNS", root / "runs"), \
                    mock.patch.object(main_mod, "RECEIPTS", root / "receipts"), \
                    mock.patch.object(audit, "AUDIT_PATH", root / "audit.jsonl"), \
                    mock.patch.object(main_mod, "_check_root"), \
                    mock.patch.object(decision, "resolve_suite", return_value=suite), \
                    mock.patch.object(decision, "run_suite", fake), \
                    mock.patch.object(main_mod.sysstats, "Sampler"), \
                    mock.patch.object(main_mod, "_run_alive", return_value=False), \
                    mock.patch.object(main_mod, "_rev", return_value="rev"), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = main_mod.main(["decision", "run", "ollama:tev1:latest", "--suite", "judge-auto", *flags])
        self.assertEqual(rc, 0)
        return fake

    def test_allow_evict_is_passed_to_run_suite_only_when_given(self):
        self.assertIs(self.run_main("--allow-evict").call_args.kwargs["allow_evict"], True)
        self.assertIs(self.run_main().call_args.kwargs["allow_evict"], False)


class DecisionDerive(unittest.TestCase):
    """`decision derive` with decision.plan_derive/derive faked (their HTTP is tested in tests/test_decision.py):
    what it gates, what it prints, what the audit ledger records."""

    PLAN = {"base": "tev1:latest", "name": "tev1-ctx21507", "num_ctx": 21507, "refuse": None, "noop": None,
            "base_weights": "a" * 64, "base_digest": "d" * 64,
            "request": {"model": "tev1-ctx21507", "from": "tev1:latest", "parameters": {"num_ctx": 21507},
                        "stream": False}}

    def run_derive(self, *argv: str, plan: dict | None = None, derive_error: str | None = None):
        from localbench import audit, decision
        derived = {"name": "tev1-ctx21507", "digest": "c" * 64, "num_ctx": 21507, "weights": "a" * 64,
                   "base": "tev1:latest", "base_digest": "d" * 64}
        derive = mock.Mock(return_value=derived,
                           side_effect=decision.DeriveError(derive_error) if derive_error else None)
        plan_derive = mock.Mock(return_value={**self.PLAN, **(plan or {})})
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(audit, "AUDIT_PATH", Path(tmp) / "a.jsonl"), \
                mock.patch.object(decision, "plan_derive", plan_derive), \
                mock.patch.object(decision, "derive", derive), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main_mod.main(["decision", "derive", "ollama:tev1:latest", "--num-ctx", "21507", *argv])
            rows = audit.rows()
        return rc, out.getvalue(), err.getvalue(), plan_derive, derive, rows

    def test_dry_run_prints_the_create_request_and_changes_nothing(self):
        rc, out, _, plan_derive, derive, rows = self.run_derive("--dry-run", "--name", "tev1-ctx21507")
        self.assertEqual(rc, 0)
        self.assertEqual(plan_derive.call_args.args, ("http://127.0.0.1:11434", "tev1:latest", 21507, "tev1-ctx21507"))
        self.assertIn('/api/create {"model": "tev1-ctx21507", "from": "tev1:latest", "parameters": {"num_ctx": 21507}',
                      out)
        self.assertEqual((derive.called, rows), (False, []))

    def test_derive_creates_and_records_the_derived_identity(self):
        rc, out, _, _, derive, rows = self.run_derive()
        self.assertEqual(rc, 0)
        self.assertEqual(derive.call_args.args[1]["request"]["parameters"], {"num_ctx": 21507})
        self.assertIn("derived tev1-ctx21507", out)
        self.assertEqual([(r["verb"], r["outcome"], r["detail"].get("num_ctx")) for r in rows],
                         [("decision derive", "done", 21507)])

    def test_a_refused_plan_creates_nothing_and_is_audited_refused(self):
        reason = "tev1-ctx21507 already exists and is not tev1:latest with num_ctx 21507; pick another --name"
        rc, _, err, _, derive, rows = self.run_derive(plan={"refuse": reason})
        self.assertEqual((rc, derive.called), (1, False))
        self.assertIn(reason, err)
        self.assertEqual([(r["verb"], r["outcome"]) for r in rows], [("decision derive", "refused")])

    def test_an_existing_identical_derivation_is_a_noop(self):
        rc, out, _, _, derive, rows = self.run_derive(plan={"noop": "tev1-ctx21507 already derives from tev1"})
        self.assertEqual((rc, derive.called), (0, False))
        self.assertIn("already derives", out)
        self.assertEqual([r["detail"].get("noop") for r in rows], ["tev1-ctx21507 already derives from tev1"])

    def test_a_readback_mismatch_exits_1_and_is_audited_failed(self):
        why = "tev1-ctx21507 was created but does not read back as asked: num_ctx reads back 2050, not 21507"
        rc, _, err, _, _, rows = self.run_derive(derive_error=why)
        self.assertEqual(rc, 1)
        self.assertIn(why, err)
        self.assertEqual([r["outcome"] for r in rows], ["failed"])


class RunAliveSeesDecisionRuns(unittest.TestCase):
    """aa/ab/run refuse while a decision run measures (it shares the GPU); a decision run does not refuse itself."""

    def alive(self, pids: list[int], cmdline: str) -> bool:
        import re

        def pgrep(argv, **_kw):
            hit = re.search(argv[-1], cmdline)
            return subprocess.CompletedProcess(argv, 0 if hit else 1, "".join(f"{p}\n" for p in pids) if hit else "", "")

        with mock.patch.object(main_mod.subprocess, "run", side_effect=pgrep):
            return main_mod._run_alive()

    def test_another_processs_decision_run_is_a_live_run(self):
        self.assertTrue(self.alive([os.getpid() + 100000], "localbench decision run ollama:nimble:latest --suite j"))

    def test_a_decision_run_does_not_see_itself_or_its_wrapper(self):
        self.assertFalse(self.alive([os.getpid(), os.getppid()],
                                    "uv run localbench decision run ollama:nimble:latest --suite j"))

    def test_other_verbs_named_decision_are_not_runs(self):
        self.assertFalse(self.alive([os.getpid() + 100000], "localbench features --json"))



class FeaturesCommand(unittest.TestCase):
    def row(self, proof: str) -> dict:
        # features.report() row shape: the feature's proof is its worst local profile's entry in `proofs`.
        receipt = None if proof == "UNPROVEN" else "r.json"
        reason = "no receipt names it" if proof == "UNPROVEN" else None
        return {"feature": "auto-thinking", "omp_package": "pkg", "omp_module": "m.ts", "omp_symbol": "f",
                "module_path": "/x/m.ts", "omp_module_sha": "a" * 64, "proof_suite": "judge-auto", "preset": "-",
                "routes": {"default": {"target": "ollama/tev1:latest", "local": True, "disabled": None}},
                "local_profiles": ["default"],
                "proofs": {"default": {"model": "tev1:latest", "digest": "cef45ef93cf6", "bypass": None,
                                       "proof": proof, "receipt": receipt, "receipt_sha": None, "reason": reason}},
                "proof": proof, "receipt": receipt, "receipt_sha": None, "reason": reason, "status": proof}

    def run_features(self, proof: str) -> int:
        from localbench import features
        with mock.patch.object(main_mod.models, "profiles", return_value=["default"]), \
                mock.patch.object(features, "report", return_value=[self.row(proof)]), \
                mock.patch.object(features, "ollama_digests", side_effect=AssertionError("read :11434")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main_mod.cmd_features(argparse.Namespace(json=False))

    def test_a_local_route_without_proof_fails(self):
        self.assertEqual(self.run_features("UNPROVEN"), 1)

    def test_a_proven_local_route_passes(self):
        self.assertEqual(self.run_features("PROVEN"), 0)


class DisabledAgentRoutes(unittest.TestCase):
    """scout is in task.disabledAgents: no scout turn reaches the local model, so gpu/models must not list it."""

    SCOUT = "scout subagents (task tool; whole agent turns)"

    def patches(self):
        return (mock.patch.object(main_mod.models.park, "local_routes",
                                  return_value={self.SCOUT: "ollama/qwen3.8", "session titles": "ollama/tev1:latest"}),
                mock.patch.object(main_mod.models, "disabled_agents", return_value={"scout"}))

    def test_gpu_prints_a_disabled_scout_as_disabled_not_a_route(self):
        client = {"pid": 1, "cmd": "omp", "cwd": "/", "servers": ["ollama"], "conns": [], "omp_profile": "default"}
        out = io.StringIO()
        a, b = self.patches()
        with a, b, mock.patch.object(main_mod.time, "sleep"), \
                mock.patch.multiple(main_mod.sysstats, gpu_time_by_pid=mock.DEFAULT, connection_bytes=mock.DEFAULT,
                                    gpu_share=mock.Mock(return_value=[]),
                                    gpu_utilization=mock.Mock(return_value={}),
                                    resident_models=mock.Mock(return_value={}),
                                    inference_clients=mock.Mock(return_value=[client]),
                                    traffic=mock.Mock(return_value={})), \
                mock.patch.object(main_mod.park, "stuck_sessions", return_value=[]), contextlib.redirect_stdout(out):
            main_mod.cmd_gpu(argparse.Namespace(seconds=0, json=False))
        text = out.getvalue()
        self.assertNotIn(f"{self.SCOUT}: ollama/qwen3.8", text)
        self.assertIn(f"{self.SCOUT}: {main_mod.models.DISABLED}", text)
        self.assertIn("session titles: ollama/tev1:latest", text)

    def test_models_lists_no_disabled_scout_under_the_model(self):
        qwen = {"server": "ollama", "name": "qwen3.8", "digest": "d" * 12, "gb": 1.0, "freshness": "current"}
        out = io.StringIO()
        a, b = self.patches()
        with a, b, mock.patch.object(main_mod.models, "ollama_models", return_value=[qwen]), \
                mock.patch.object(main_mod.models, "mlx_models", return_value=[]), \
                mock.patch.object(main_mod.models, "omp_cpu_models", return_value=[]), \
                mock.patch.object(main_mod.models, "profiles", return_value=["default"]), \
                mock.patch.object(main_mod.models, "releases", return_value=[]), contextlib.redirect_stdout(out):
            main_mod.cmd_models(argparse.Namespace(days=14, json=False))
        routed = [line for line in out.getvalue().splitlines() if line.startswith(" ") and self.SCOUT in line]
        self.assertEqual(routed, [])
        self.assertIn(f"disabled  {self.SCOUT}", out.getvalue())


class ReportByPurpose(unittest.TestCase):
    def report(self, *, as_json: bool = True, **flags):
        import sqlite3
        import time

        from localbench import gateway
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "gw" / "gateway.sqlite3"
            gateway.GatewayStore(db)
            bucket = int(time.time() // 3600 * 3600)
            con = sqlite3.connect(db)
            con.executemany("INSERT INTO purpose_stats(profile,purpose,bucket_start,requests,busy_s) VALUES(?,?,?,?,?)",
                            [("default", "judge", bucket, 3, 1.5), ("work", "judge", bucket, 2, 0.5),
                             ("work", "memory", bucket, 1, 4.0), ("old", "judge", bucket - 30 * 86400, 9, 9.0)])
            con.commit()
            con.close()
            out = io.StringIO()
            args = argparse.Namespace(since=86400.0, json=as_json, **{"by_purpose": False, "by_profile": False,
                                                                     "requests": False, "req_purpose": None,
                                                                     "req_profile": None, "limit": 200, **flags})
            with mock.patch.object(gateway, "database_path", return_value=db), contextlib.redirect_stdout(out):
                rc = main_mod.cmd_report(args)
            self.assertEqual(rc, 0)
            return json.loads(out.getvalue()) if as_json else out.getvalue()

    def test_text_table_shows_each_purpose_with_its_totals(self):
        rows = [[cell.strip() for cell in line.strip("|").split("|")]
                for line in self.report(as_json=False, by_purpose=True).splitlines()]
        self.assertIn(["memory", "1", "4.0"], rows)
        self.assertIn(["judge", "5", "2.0"], rows)
        self.assertFalse(any("9.0" in r or "old" in r for r in rows), rows)

    def test_by_purpose_sums_profiles_inside_the_window(self):
        doc = self.report(by_purpose=True)
        self.assertEqual(doc["by"], ["purpose"])
        self.assertEqual(doc["rows"], [{"purpose": "memory", "requests": 1, "busy_s": 4.0},
                                       {"purpose": "judge", "requests": 5, "busy_s": 2.0}])

    def test_by_profile_and_purpose_keeps_the_pairs(self):
        doc = self.report(by_purpose=True, by_profile=True)
        self.assertEqual({(r["profile"], r["purpose"]): r["requests"] for r in doc["rows"]},
                         {("default", "judge"): 3, ("work", "judge"): 2, ("work", "memory"): 1})


class SmolModelRuns(unittest.TestCase):
    """`--smol-model` on run/aa/ab: execute() with every machine and server touch faked."""

    class Backend:
        name = "ollama"
        base_url = "http://127.0.0.1:11434/v1"

        def isolate(self, _model):
            return []

        def fingerprint(self, _model):
            return {"loaded_context": 4096}

        def tokenizer_identity(self, _model):
            return None

        def pins(self, model):
            return {"model_digest": {"qwen3.8:27b": "5a1d0c0ffee0"}.get(model)}

    def execute(self, backend, smol_model):
        from types import SimpleNamespace
        seen: dict = {}

        def tier(ctx):
            seen["ctx"] = ctx
            return []

        def sampler(*_args, target, **_kw):
            seen["target"] = target
            return contextlib.nullcontext(SimpleNamespace(contention=[], series=[],
                                                          summary=lambda: {"resident_unknown_samples": 0}))

        quiet = mock.Mock(return_value=contextlib.nullcontext(SimpleNamespace(summary=lambda: {})))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RUNS", root / "runs"), \
                    mock.patch.object(main_mod.sysstats, "snapshot", return_value={"host": {}}), \
                    mock.patch.object(main_mod, "preflight", return_value={"problems": []}), \
                    mock.patch.object(main_mod, "run_pins", side_effect=lambda b, m, h, c: {"backend": b.name,
                                                                                              "model": m}), \
                    mock.patch.object(main_mod.backends, "_warm") as warm, \
                    mock.patch.object(main_mod.sysstats, "Sampler", side_effect=sampler), \
                    mock.patch.object(main_mod.sysstats, "PowerSampler", quiet), \
                    mock.patch.object(main_mod.sysstats, "CpuSampler", quiet), \
                    mock.patch.dict(main_mod.TIERS, {"conf": tier}), \
                    mock.patch.object(main_mod.golden, "listed_discrepancies", return_value=[]), \
                    mock.patch.object(main_mod, "_rev", return_value="rev"):
                summary = main_mod.execute(backend, "nimble:latest", tiers=["conf"], repeats=1, allow_busy=False,
                                           purge=False, smol_model=smol_model)
        return summary, seen, warm

    def test_smol_model_reaches_ctx_pins_sampler_and_is_warmed(self):
        summary, seen, warm = self.execute(self.Backend(), "qwen3.8:27b")
        self.assertEqual(seen["ctx"].smol_model, "qwen3.8:27b")
        pins = summary["provenance"]["pins"]
        self.assertEqual((pins["smol_model"], pins["smol_digest"]), ("qwen3.8:27b", "5a1d0c0ffee0"))
        self.assertEqual(seen["ctx"].pins, pins)
        self.assertEqual(seen["target"], ("ollama", "nimble:latest", "qwen3.8:27b"))
        warm.assert_called_once_with(self.Backend.base_url, "qwen3.8:27b")

    def test_without_smol_model_the_run_is_one_artifact(self):
        summary, seen, warm = self.execute(self.Backend(), None)
        self.assertIsNone(seen["ctx"].smol_model)
        self.assertNotIn("smol_model", summary["provenance"]["pins"])
        self.assertEqual(seen["target"], ("ollama", "nimble:latest"))
        warm.assert_not_called()

    def test_smol_model_on_a_one_model_server_is_a_usage_error(self):
        backend = self.Backend()
        backend.name = "mlx-serve"
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
            self.execute(backend, "qwen3.8:27b")
        self.assertEqual(caught.exception.code, 2)

    def test_ab_refuses_a_smol_model_on_a_non_ollama_arm_before_any_leg(self):
        with mock.patch.object(main_mod, "execute", side_effect=AssertionError("leg measured")), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main_mod.cmd_ab(argparse.Namespace(a="ollama:a", b="mlx-serve:/m", pairs=1, tiers="mem",
                                               smol_model=None, b_smol_model="qwen3.8:27b", side_regime=False))
        self.assertEqual(caught.exception.code, 2)



class WatchdogRun(unittest.TestCase):
    def test_violation_aborts_before_tier_and_emits_terminal_done_with_reason(self):
        from types import SimpleNamespace

        class Sampler:
            def __init__(self, *_args, on_sample=None, **_kwargs):
                self.on_sample = on_sample
                self.series = []
                self.contention = []

            def __enter__(self):
                sample = {"t": 123.0, "resident": {"ollama": []}, "gpu_procs": []}
                self.series.append(sample)
                self.on_sample(sample)
                return self

            def __exit__(self, *_exc):
                return None

            def summary(self):
                return {"resident_unknown_samples": 0}

        backend = SmolModelRuns.Backend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_pins = {"backend": "ollama", "model": "nimble:latest", "omp_sha": "pinned",
                        "omp_path": "/fixture/omp"}
            quiet = mock.Mock(return_value=contextlib.nullcontext(SimpleNamespace(summary=lambda: {})))
            tier = mock.Mock(return_value=[])
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RUNS", root / "runs"), \
                    mock.patch.object(main_mod.sysstats, "snapshot", return_value={"host": {}}), \
                    mock.patch.object(main_mod, "preflight", return_value={"problems": []}), \
                    mock.patch.object(main_mod, "run_pins", return_value=run_pins), \
                    mock.patch.object(main_mod, "sha16", return_value="pinned"), \
                    mock.patch.object(main_mod.golden, "pin_diff", return_value=[]), \
                    mock.patch.object(main_mod.backends, "_warm"), \
                    mock.patch.object(main_mod.sysstats, "Sampler", Sampler), \
                    mock.patch.object(main_mod.sysstats, "PowerSampler", quiet), \
                    mock.patch.object(main_mod.sysstats, "CpuSampler", quiet), \
                    mock.patch.dict(main_mod.TIERS, {"conf": tier}), \
                    mock.patch.object(main_mod.golden, "listed_discrepancies", return_value=[]), \
                    mock.patch.object(main_mod, "_rev", return_value="rev"), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = main_mod.execute(backend, "nimble:latest", tiers=["conf"], repeats=1, allow_busy=False,
                                           purge=False, watchdog_enabled=True)

            self.assertEqual(summary["verdicts"]["watchdog_abort"],
                             "expected_model_not_resident:nimble:latest")
            self.assertTrue(summary["watchdog"]["aborted"])
            tier.assert_not_called()
            progress = (root / summary["run_dir"] / "progress.jsonl").read_text()
            events = [json.loads(line) for line in progress.splitlines()]
            self.assertIn("watchdog_abort", [event["event"] for event in events])
            done = [event for event in events if event["event"] == "done"]
            self.assertEqual(len(done), 1)
            self.assertEqual(events[-1]["event"], "done")
            self.assertEqual(done[0]["watchdog_abort"], "expected_model_not_resident:nimble:latest")

    def test_resume_skips_completed_case_and_leaves_aborted_case_uncheckpointed(self):
        from types import SimpleNamespace

        from localbench.evaluation import EvaluationCampaign

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            runs = root / "runs"
            campaign_path = runs / "eval-resume"
            runs.mkdir()
            case_inputs = {"completed": "input-a", "pending": "input-b"}
            pins = {"model_digest": "model-a", "omp_sha": "omp-a"}
            fingerprint = {"context": 4096}
            source_sha256 = {"client.py": "source-a"}
            backend = SimpleNamespace(name="ollama", fingerprint=mock.Mock(return_value=fingerprint))
            identity = {"profile": main_mod.E2E_PROFILE, "backend": "ollama", "model": "model-a",
                        "pins": pins, "fingerprint": fingerprint, "server_args": [],
                        "localbench_rev": "revision-a", "source_sha256": source_sha256}
            campaign = EvaluationCampaign.create(campaign_path, root=root, identity=identity, cases=case_inputs,
                                                 profile=main_mod.E2E_PROFILE)
            completed_dir = runs / "completed-run"
            completed_dir.mkdir()
            trace = completed_dir / "summary.json"
            trace.write_text("{}")
            campaign.record_case("completed", input_sha256="input-a", status="PASS", run_dir=completed_dir,
                                 trace_files=[trace])
            args = argparse.Namespace(backend="ollama:model-a", server_arg=None, resume=str(campaign_path),
                                      wait_idle=0, watchdog=True)
            aborted = {"watchdog": {"aborted": True}}
            stderr = io.StringIO()
            with mock.patch.object(main_mod, "ROOT", root), mock.patch.object(main_mod, "RUNS", runs), \
                    mock.patch.object(main_mod.park, "parked_now", return_value=True), \
                    mock.patch.object(main_mod, "open_backend",
                                      return_value=contextlib.nullcontext((backend, "model-a"))), \
                    mock.patch.object(main_mod.sysstats, "snapshot", return_value={"host": {"host_id": "h"}}), \
                    mock.patch.object(main_mod, "run_pins", return_value=pins), \
                    mock.patch.object(main_mod, "_evaluation_case_inputs", return_value=case_inputs), \
                    mock.patch.object(main_mod, "_localbench_source_hashes", return_value=source_sha256), \
                    mock.patch.object(main_mod, "_rev", return_value="revision-a"), \
                    mock.patch.object(main_mod, "execute", return_value=aborted) as execute, \
                    contextlib.redirect_stderr(stderr):
                result = main_mod.cmd_eval_run(args)

            self.assertEqual(result, 1)
            execute.assert_called_once()
            self.assertEqual(execute.call_args.kwargs["e2e_case"], "pending")
            resumed = EvaluationCampaign.open(campaign_path, root=root, expected_identity=identity)
            self.assertIsNotNone(resumed.completed_case("completed", input_sha256="input-a"))
            self.assertIsNone(resumed.completed_case("pending", input_sha256="input-b"))
            self.assertIn("completed campaign cases remain checkpointed", stderr.getvalue())


class CorpusCommand(unittest.TestCase):
    def run_main(self, *argv: str):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            try:
                return main_mod.main(["corpus", *argv]), err.getvalue()
            except SystemExit as exc:
                return exc.code, err.getvalue()

    def test_import_without_a_gate_is_a_usage_error(self):
        from localbench import corpus
        with mock.patch.object(corpus, "import_profile", side_effect=AssertionError("imported")):
            rc, err = self.run_main("import", "--profile", "work", "--role", "decision.noul")
        self.assertEqual(rc, 2)
        self.assertIn("--gate", err)

    def test_a_malformed_gate_is_a_usage_error(self):
        from localbench import corpus
        for gate in ("decision.noul.accuracy=0.8", "decision.noul.accuracy=atleast:0.8", "=min:0.8",
                     "decision.noul.accuracy=min:nan"):
            with self.subTest(gate=gate), mock.patch.object(corpus, "import_profile",
                                                            side_effect=AssertionError("imported")):
                rc, _ = self.run_main("import", "--profile", "work", "--role", "decision.noul", "--gate", gate)
                self.assertEqual(rc, 2)

    def test_import_passes_the_parsed_gate(self):
        from localbench import corpus
        with mock.patch.object(corpus, "import_profile", return_value={}) as imp:
            rc, _ = self.run_main("import", "--profile", "work", "--role", "decision.noul",
                                  "--gate", "decision.noul.accuracy=min:0.8", "--gate", "decision.noul.ece=max:0.1")
        self.assertEqual(rc, 0)
        self.assertEqual(imp.call_args.kwargs["gate"],
                         {"decision.noul.accuracy": {"min": 0.8}, "decision.noul.ece": {"max": 0.1}})
        self.assertEqual(imp.call_args.kwargs["roles"], ("decision.noul",))

    def test_jev_build_requires_and_passes_the_gate(self):
        from localbench import jevsuites
        with mock.patch.object(jevsuites, "build", side_effect=AssertionError("built")):
            self.assertEqual(self.run_main("proj-b-build", "find")[0], 2)
        built = {"suite": {"name": "proj-b-find", "n_items": 3}, "directory": "/x", "excluded": {}, "notes": []}
        with mock.patch.object(jevsuites, "build", return_value=built) as build:
            rc, _ = self.run_main("proj-b-build", "find", "--gate", "decision.noul.accuracy=min:0.7")
        self.assertEqual(rc, 0)
        build.assert_called_once_with("find", gate={"decision.noul.accuracy": {"min": 0.7}})


class PresetCommand(unittest.TestCase):
    def plan(self, refused=None) -> dict:
        return {"preset": "judge:local-nimble", "target": "live", "profiles": ["default"], "force": False,
                "forced": None, "proof": {"auto-thinking": {"default": {"proof": "UNPROVEN", "model": "nimble:latest",
                                                                         "reason": "no receipt names it"}}},
                "refused": refused, "keeps_old": [], "notes": [],
                "steps": [{"profile": "default", "kind": "set", "key": "modelRoles", "file": "/x/config.yml",
                           "before": {"judge": "typesafe/proj-b-latest"}, "after": {"judge": "localbench-sys1/nimble"},
                           "changed": True, "expect": []}]}

    def run_main(self, *argv: str, plan: dict):
        from localbench import presets
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(presets, "plan", return_value=plan), \
                mock.patch.object(presets, "apply", return_value={"id": "20260930T000000Z-abc123", "forced": None}) \
                as apply, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main_mod.main(["preset", *argv])
        return rc, out.getvalue(), err.getvalue(), apply

    def rows(self) -> list[dict]:
        from localbench import audit
        return [r for r in audit.rows() if r["verb"] == "preset apply"]

    def test_live_apply_of_an_unproven_local_preset_is_refused_printed_and_audited(self):
        refusal = "live judge:local-nimble routes to a local model without proof: {'auto-thinking@default': 'UNPROVEN'}"
        before = len(self.rows())
        rc, _, err, apply = self.run_main("apply", "judge:local-nimble", "--profiles", "default", "--target",
                                          "live", plan=self.plan(refusal))
        self.assertEqual(rc, 1)
        self.assertFalse(apply.called)
        self.assertIn(refusal, err)
        new = self.rows()[before:]
        self.assertEqual([(r["outcome"], r["detail"].get("reason")) for r in new], [("refused", refusal)])

    def test_dry_run_prints_the_plan_lines_and_the_refusal_and_writes_nothing(self):
        refusal = "live judge:local-nimble routes to a local model without proof"
        before = len(self.rows())
        rc, out, err, apply = self.run_main("apply", "judge:local-nimble", "--profiles", "default", "--target",
                                            "live", "--dry-run", plan=self.plan(refusal))
        self.assertEqual(rc, 1)
        self.assertFalse(apply.called)
        self.assertIn("proof auto-thinking @default: UNPROVEN nimble:latest", out)
        self.assertIn('set  default: modelRoles {"judge": "typesafe/proj-b-latest"} -> {"judge": "localbench-sys1/nimble"}',
                      out)
        self.assertIn(refusal, err)
        self.assertEqual(len(self.rows()), before)

    def test_an_unrefused_apply_writes_and_records_the_rollback_id(self):
        rc, out, _, apply = self.run_main("apply", "judge:local-nimble", "--profiles", "default", "--target", "test",
                                          plan=self.plan())
        self.assertEqual(rc, 0)
        apply.assert_called_once_with("judge:local-nimble", ["default"], "test", force=False)
        self.assertIn("20260930T000000Z-abc123", out)
        self.assertEqual(self.rows()[-1]["detail"]["id"], "20260930T000000Z-abc123")

    def test_drift_exits_1_when_live_config_left_a_preset(self):
        from localbench import presets
        row = {"profile": "default", "preset": "judge:local-nimble", "id": "x", "key": "modelRoles.judge",
               "expected": {"selector": "localbench-sys1/nimble"}, "actual": "typesafe/proj-b-latest"}
        for rows, code in (([row], 1), ([], 0)):
            with self.subTest(drift=bool(rows)), mock.patch.object(presets, "drift", return_value=rows), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main_mod.main(["preset", "drift"]), code)


class WatchReleasesCommand(unittest.TestCase):
    def run_main(self, *argv: str):
        from localbench import releasewatch
        report = {"at": "t", "filed": [], "skipped": [], "deferred": [], "baselined": 0, "errors": [], "ok": True}
        with mock.patch.object(releasewatch, "run_once", return_value=report) as run_once, \
                mock.patch.object(releasewatch, "install_agent", side_effect=AssertionError("installed")), \
                mock.patch.object(releasewatch, "status", return_value={"dir": "/w", "queued": [], "seen": None}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = main_mod.main(["watch-releases", *argv])
        return rc, run_once

    def rows(self) -> list[dict]:
        from localbench import audit
        return [r for r in audit.rows() if r["verb"] == "watch-releases"]

    def test_once_runs_one_pass_and_is_audited(self):
        before = len(self.rows())
        rc, run_once = self.run_main("--once")
        self.assertEqual(rc, 0)
        run_once.assert_called_once_with()
        self.assertEqual([r["outcome"] for r in self.rows()[before:]], ["done"])

    def test_once_dry_run_makes_no_pass_and_no_row(self):
        before = len(self.rows())
        rc, run_once = self.run_main("--once", "--dry-run")
        self.assertEqual(rc, 0)
        run_once.assert_not_called()
        self.assertEqual(len(self.rows()), before)

    def test_bare_is_a_read_with_no_pass_and_no_row(self):
        before = len(self.rows())
        rc, run_once = self.run_main()
        self.assertEqual(rc, 0)
        run_once.assert_not_called()
        self.assertEqual(len(self.rows()), before)

    def test_an_unusable_watch_dir_is_refused_cleanly_before_any_pass(self):
        from localbench import releasewatch
        err = io.StringIO()
        before = len(self.rows())
        with mock.patch.object(releasewatch, "watch_dir", side_effect=releasewatch.WatchError("inside the repo")), \
                mock.patch.object(releasewatch, "run_once", side_effect=AssertionError("pass ran")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = main_mod.main(["watch-releases", "--once"])
        self.assertEqual(rc, 1)
        self.assertIn("inside the repo", err.getvalue())
        self.assertEqual([r["outcome"] for r in self.rows()[before:]], ["refused"])


class ProfileNames(unittest.TestCase):
    def test_paths_are_not_profile_names(self):
        for text in ("../other", "a/b", "..", "lab,..\\x"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                main_mod._profiles_arg(text)
        self.assertEqual(main_mod._profiles_arg("default, lab"), ["default", "lab"])


class CorpusCaptureAndMutations(unittest.TestCase):
    def run_main(self, *argv: str):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main_mod.main(["corpus", *argv])
            except SystemExit as exc:
                rc = exc.code
        return rc, out.getvalue(), err.getvalue()

    def rows(self, verb: str) -> list[dict]:
        from localbench import audit
        return [r for r in audit.rows() if r["verb"] == verb]

    def test_capture_on_needs_purpose_cap_and_expiry(self):
        from localbench import corpus
        with mock.patch.object(corpus, "capture_on", side_effect=AssertionError("enabled")):
            for argv in (["capture", "on"], ["capture", "on", "--purpose", "find", "--max-items", "5"],
                         ["capture", "on", "--max-items", "5", "--minutes", "30"]):
                with self.subTest(argv=argv):
                    self.assertEqual(self.run_main(*argv)[0], 2)

    def test_capture_on_is_an_audited_mutation_passing_its_bounds(self):
        from localbench import corpus
        before = len(self.rows("corpus capture on"))
        with mock.patch.object(corpus, "capture_on", return_value={"enabled": True}) as on:
            rc, _, _ = self.run_main("capture", "on", "--purpose", "find", "--purpose", "judge", "--max-items", "50",
                                     "--minutes", "30")
        self.assertEqual(rc, 0)
        on.assert_called_once_with(["find", "judge"], 50, 30.0)
        self.assertEqual([r["outcome"] for r in self.rows("corpus capture on")[before:]], ["done"])

    def test_capture_on_dry_run_writes_nothing(self):
        from localbench import corpus
        before = len(self.rows("corpus capture on"))
        with mock.patch.object(corpus, "capture_on", side_effect=AssertionError("enabled")):
            rc, out, _ = self.run_main("capture", "on", "--purpose", "find", "--max-items", "5", "--minutes", "10",
                                       "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("at most 5 item(s)", out)
        self.assertEqual(len(self.rows("corpus capture on")), before)

    def test_capture_status_is_a_read(self):
        from localbench import corpus
        before = len(self.rows("corpus capture status"))
        with mock.patch.object(corpus, "capture_status", return_value={"active": False}):
            self.assertEqual(self.run_main("capture", "status")[0], 0)
        self.assertEqual(self.run_main("capture", "status", "--dry-run")[0], 2)
        self.assertEqual(len(self.rows("corpus capture status")), before)

    def test_import_dry_run_reads_and_writes_nothing(self):
        from localbench import corpus
        with mock.patch.object(corpus, "import_profile", side_effect=AssertionError("imported")):
            rc, out, _ = self.run_main("import", "--profile", "work", "--role", "decision.noul",
                                       "--gate", "decision.noul.accuracy=min:0.8", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("judgment-cache.db", out)


class MemoryVerdictCommand(unittest.TestCase):
    """`memory-verdict --bank` over run dirs holding real mem+sess leg fixtures (tests.test_memory_study.leg); the
    receipt is the real workloads.memory_verdict's, wrapped in a Mock to see what the command passed it."""

    def leg(self, label: str, unknown=0, **kw) -> dict:
        from tests.test_memory_study import leg
        return leg(label, resident_unknown=unknown, **kw)

    def run_verdict(self, *, tie=False, unknown=0, feature="mnemopi-extraction"):
        from localbench.workloads import memory_verdict
        from tests.test_memory_study import QWEN, QWEN_DIGEST
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for side, n in (("c", 2), ("b", 2)):
                for i in range(n):
                    run = root / "runs" / f"{side}{i}"
                    run.mkdir(parents=True)
                    # Baseline post-retain walls 8.38/8.44 s; the candidate's are 2.06/2.10 s, or the same on a tie.
                    post = (8.38, 8.44)[i] if side == "b" or tie else (2.06, 2.10)[i]
                    leg = (self.leg(f"c{i}", unknown if i == 1 else 0, post=post) if side == "c" else
                           self.leg(f"b{i}", smol=QWEN, digest=QWEN_DIGEST, post=post))
                    (run / "summary.json").write_text(json.dumps(leg))
                    paths.setdefault(side, []).append(str(run))
            verdict = mock.Mock(side_effect=lambda *a, **kw: memory_verdict(*a, **kw, rev="abc1234"))
            err = io.StringIO()
            # ROOT stays the clone: main() refuses a data root without fixtures/omp.
            with mock.patch.object(main_mod, "RECEIPTS", root / "rc"), \
                    mock.patch.object(main_mod, "_feature_module_sha", return_value=("f" * 64, None)), \
                    mock.patch.object(main_mod.workloads, "memory_verdict", verdict), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = main_mod.main(["memory-verdict", "--candidate", *paths["c"], "--baseline", *paths["b"],
                                    "--feature", feature, "--bank", "mem-proof"])
            banked = root / "rc" / "mem-proof.json"
            doc = json.loads(banked.read_text()) if banked.exists() else None
        return rc, doc, verdict, err.getvalue()

    def test_a_clean_verdict_banks_with_the_installed_module_sha(self):
        rc, doc, verdict, _ = self.run_verdict()
        self.assertEqual(rc, 0)
        self.assertEqual(doc["verdict"]["compare"], "BETTER")
        self.assertEqual(verdict.call_args.kwargs, {"feature": "mnemopi-extraction", "omp_module_sha": "f" * 64})
        cand, base = verdict.call_args.args
        self.assertEqual(([leg["provenance"]["label"] for leg in cand], [leg["provenance"]["label"] for leg in base]),
                         (["c0", "c1"], ["b0", "b1"]))

    def test_problems_bank_as_evidence_and_exit_1(self):
        rc, doc, _, err = self.run_verdict(tie=True)
        self.assertEqual(rc, 1)
        self.assertEqual(doc["problems"], ["not better than route ollama/qwen3.8:27b-mlx: NOT_BETTER"])
        self.assertIn("UNSOUND", err)

    def test_a_leg_with_unknown_or_unrecorded_residency_is_refused_unbanked(self):
        for unknown in (2, None):
            with self.subTest(unknown=unknown):
                rc, doc, verdict, _ = self.run_verdict(unknown=unknown)
                self.assertEqual((rc, doc, verdict.called), (1, None, False))

    def test_an_unknown_feature_is_a_usage_error(self):
        from localbench import features
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(features, "load", return_value=[{"feature": "mnemopi-extraction"}]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            run = Path(tmp) / "r"
            run.mkdir()
            (run / "summary.json").write_text(json.dumps(self.leg("r")))
            main_mod.main(["memory-verdict", "--candidate", str(run), "--baseline", str(run), "--feature", "nope",
                           "--bank", "x"])
        self.assertEqual(caught.exception.code, 2)


class OmpCommand(unittest.TestCase):
    """`omp refresh` with ompupdate.refresh faked; `omp watch` under a scratch HOME with launchctl faked."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.launchctl: list[list[str]] = []

        def run(argv, **_kw):
            self.launchctl.append(list(argv))
            loaded = argv[:2] == ["launchctl", "print"] and self.plist().is_file()
            return subprocess.CompletedProcess(argv, 0 if argv[1] != "print" or loaded else 113, "", "")

        for patch in (mock.patch.dict(os.environ, {"HOME": str(self.home)}),
                      mock.patch.object(main_mod.subprocess, "run", side_effect=run)):
            patch.start()
            self.addCleanup(patch.stop)

    def plist(self) -> Path:
        from localbench import ompupdate
        return self.home / "Library" / "LaunchAgents" / f"{ompupdate.DEFAULT_LABEL}.plist"

    def run_main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main_mod.main(["omp", *argv])
        return rc, out.getvalue(), err.getvalue()

    def rows(self, verb: str) -> list[dict]:
        from localbench import audit
        return [r for r in audit.rows() if r["verb"] == verb]

    def refresh(self, outcomes: dict, *argv: str):
        from localbench import ompupdate
        report = {"omp_version": "18.4.6", "outcomes": outcomes}
        with mock.patch.object(ompupdate, "refresh", create=True, return_value=report) as refresh:
            rc, out, err = self.run_main("refresh", *argv)
        return rc, out, refresh

    def test_refresh_exits_1_when_any_feature_went_stale_and_names_its_reproof(self):
        before = len(self.rows("omp refresh"))
        rc, out, refresh = self.refresh({
            "auto-thinking": {"feature": "auto-thinking", "status": "CARRIED", "queue": []},
            "memory-extraction": {"feature": "mnemopi-extraction", "status": "STALE", "queue": ["mem", "through-omp"]}})
        self.assertEqual(rc, 1)
        refresh.assert_called_once_with()
        self.assertIn("1 carried, 1 STALE", out)
        self.assertIn("memory-extraction: STALE (feature mnemopi-extraction); re-prove: mem, through-omp", out)
        new = self.rows("omp refresh")[before:]
        self.assertEqual([(r["outcome"], r["detail"]["stale"]) for r in new], [("done", ["memory-extraction"])])

    def test_refresh_with_every_feature_carried_exits_0(self):
        rc, _, _ = self.refresh({"auto-thinking": {"feature": "auto-thinking", "status": "CARRIED", "queue": []}})
        self.assertEqual(rc, 0)

    def test_refresh_dry_run_captures_nothing(self):
        from localbench import ompupdate
        with mock.patch.object(ompupdate, "refresh", create=True, side_effect=AssertionError("captured")):
            rc, out, _ = self.run_main("refresh", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("mock server", out)

    def test_watch_install_writes_the_watchpaths_plist_bootstraps_and_audits(self):
        import plistlib

        from localbench import ompupdate
        rc, _, _ = self.run_main("watch", "install")
        self.assertEqual(rc, 0)
        doc = plistlib.loads(self.plist().read_bytes())
        self.assertEqual((doc["Label"], doc["WatchPaths"], doc["ProgramArguments"][-2:]),
                         (ompupdate.DEFAULT_LABEL, [ompupdate.OMP_PACKAGE_JSON], ["omp", "refresh"]))
        self.assertEqual([a[1] for a in self.launchctl], ["bootout", "bootstrap"])
        self.assertEqual(self.launchctl[1][-1], str(self.plist()))
        self.assertEqual([r["outcome"] for r in self.rows("omp watch install")][-1:], ["done"])
        rc, out, _ = self.run_main("watch", "status")
        self.assertEqual(rc, 0)
        self.assertIn("installed", out)
        self.assertIn(" loaded", out)

    def test_watch_install_refuses_a_foreign_plist_at_its_path(self):
        import plistlib
        self.plist().parent.mkdir(parents=True)
        self.plist().write_bytes(plistlib.dumps({"Label": "com.someone.else"}))
        rc, _, err = self.run_main("watch", "install")
        self.assertEqual(rc, 1)
        self.assertIn("not localbench's omp watch", err)
        self.assertEqual(plistlib.loads(self.plist().read_bytes())["Label"], "com.someone.else")
        self.assertEqual(self.launchctl, [])

    def test_watch_dry_run_and_remove(self):
        rc, out, _ = self.run_main("watch", "install", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("launchctl bootstrap", out)
        self.assertFalse(self.plist().exists())
        self.assertEqual(self.launchctl, [])
        self.assertEqual(self.run_main("watch", "remove")[0], 0)   # absent: a no-op
        self.assertEqual(self.launchctl, [])
        self.run_main("watch", "install")
        self.assertEqual(self.run_main("watch", "remove")[0], 0)
        self.assertFalse(self.plist().exists())
        self.assertEqual(self.launchctl[-1][1], "bootout")


if __name__ == "__main__":
    unittest.main()
