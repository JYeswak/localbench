import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import prove


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    if args and args[0] in ("commit", "add"):
        args = ("-c", "core.hooksPath=/dev/null", *args)
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True,
                          text=True, check=True)


def _init_export_repo(root: Path, version: str = "base") -> str:
    (root / "var" / "agent-tmp").mkdir(parents=True)
    package = root / "localbench"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "export = Path.cwd()\n"
        "root = Path(os.environ.get('LOCALBENCH_HOME') or export)\n"
        "head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()\n"
        "status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True, check=True).stdout\n"
        "owner = (export / '.owner').read_text(encoding='utf-8')\n"
        "spec = Path(sys.argv[2]).resolve()\n"
        "receipt = root / 'docs' / 'evidence' / 'receipts' / 'run-plan.json'\n"
        "receipt.parent.mkdir(parents=True, exist_ok=True)\n"
        "run_dir = root / 'runs' / 'plan-run-1'\n"
        "run_dir.mkdir(parents=True, exist_ok=True)\n"
        "(run_dir / 'summary.json').write_text('{}')\n"
        "receipt.write_text(json.dumps({'head': head, 'status': status, 'owner': owner, 'version': (export / 'version.txt').read_text(), 'spec': str(spec)}))\n"
    )
    (root / "spec.json").write_text("{}\n")
    (root / "version.txt").write_text(f"{version}\n")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "base")
    _git(root, "branch", "-M", "main")
    return _git(root, "rev-parse", "HEAD").stdout.strip()


class RunPlan(unittest.TestCase):
    def test_prove_spec_refuses_dirty_localbench_before_live_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_export_repo(root)
            stats = root / "localbench" / "stats.py"
            second = root / "localbench" / "zz_second.py"
            stats.write_text("before", encoding="utf-8")
            second.write_text("before", encoding="utf-8")
            _git(root, "add", "localbench/stats.py", "localbench/zz_second.py")
            _git(root, "commit", "--quiet", "-m", "track dirty-tree files")
            stats.write_text("dirty", encoding="utf-8")
            second.write_text("also dirty", encoding="utf-8")
            spec_path = root / "registries" / "proofs" / "auto-thinking__dirty.json"
            spec_path.parent.mkdir(parents=True)
            spec_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(prove, "load_spec", return_value={"feature": "auto-thinking"}):
                with self.assertRaises(prove.ProveError) as raised:
                    prove.prove_spec(spec_path, dry_run=False, repo=root)
            self.assertEqual(
                str(raised.exception),
                "proof requires a clean localbench tree; dirty files: "
                "localbench/stats.py, localbench/zz_second.py; use scripts/run_plan.py",
            )

    def test_main_removes_owned_export_after_spec_lookup_failure(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "var" / "agent-tmp"
            real = run_plan.subprocess.run

            def fake_run(argv, **kwargs):
                if argv[:2] == ["git", "archive"]:
                    return type("Result", (), {"stdout": b"", "returncode": 0})()
                if argv[:2] == ["tar", "-x"]:
                    return type("Result", (), {"stdout": b"", "returncode": 0})()
                if argv[0] == "git":
                    output = "a" * 40 if argv[1:3] == ["rev-parse", "HEAD"] else ".git/objects" if argv[1:3] == ["rev-parse", "--git-path"] else ""
                    return type("Result", (), {"stdout": output, "returncode": 0})()
                return real(argv, **kwargs)

            with mock.patch.object(run_plan, "ROOT", root), mock.patch.object(run_plan.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(run_plan, "__name__", "run_plan"), mock.patch("sys.argv", ["run_plan", "missing.json"]):
                with self.assertRaises(SystemExit):
                    run_plan.main()
            self.assertEqual(list(runs.iterdir()), [])

    def test_bare_invocation_starts_no_live_prove_process(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "var" / "agent-tmp"
            real = run_plan.subprocess.run
            prove_calls = []

            def fake_run(argv, **kwargs):
                if argv[:2] == ["git", "archive"]:
                    return type("Result", (), {"stdout": b"archive", "returncode": 0})()
                if argv[:2] == ["tar", "-x"]:
                    export = Path(argv[argv.index("-C") + 1])
                    (export / "spec.json").write_text("{}")
                    return type("Result", (), {"stdout": b"", "returncode": 0})()
                if argv[0] == "git":
                    output = "a" * 40 if argv[1:3] == ["rev-parse", "HEAD"] else ".git/objects" if argv[1:3] == ["rev-parse", "--git-path"] else ""
                    return type("Result", (), {"stdout": output, "returncode": 0})()
                if "-m" in argv and "localbench" in argv:
                    prove_calls.append(argv)
                    return type("Result", (), {"returncode": 0})()
                return real(argv, **kwargs)

            with mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.object(run_plan.subprocess, "run", side_effect=fake_run), \
                    mock.patch("sys.argv", ["run_plan", "spec.json"]):
                self.assertEqual(run_plan.main(), 0)
            self.assertEqual(len(prove_calls), 1)
            self.assertIn("--dry-run", prove_calls[0])
            self.assertNotIn("--execute", prove_calls[0])
            self.assertEqual(list(runs.iterdir()), [])

    def test_live_run_persists_receipts_and_runs_in_main_tree(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipts = root / "docs" / "evidence" / "receipts"
            receipts.mkdir(parents=True)
            (receipts / "old.json").write_text("unchanged")
            real = run_plan.subprocess.run
            seen_live = []

            def fake_run(argv, **kwargs):
                if argv[:2] == ["git", "archive"]:
                    return type("Result", (), {"stdout": b"archive", "returncode": 0})()
                if argv[:2] == ["tar", "-x"]:
                    export = Path(argv[argv.index("-C") + 1])
                    (export / "spec.json").write_text("{}")
                    return type("Result", (), {"stdout": b"", "returncode": 0})()
                if argv[0] == "git":
                    output = "a" * 40 if argv[1:3] == ["rev-parse", "HEAD"] else ".git/objects" if argv[1:3] == ["rev-parse", "--git-path"] else ""
                    return type("Result", (), {"stdout": output, "returncode": 0})()
                if argv[0] == "br":
                    return subprocess.CompletedProcess(argv, 0, json.dumps({"path": str(root / ".beads")}), "")
                if "-m" in argv and "localbench" in argv:
                    seen_live.append(argv)
                    export = Path(kwargs["cwd"])
                    self.assertEqual(Path(kwargs["env"]["LOCALBENCH_HOME"]), export.resolve())
                    self.assertEqual(Path(kwargs["env"]["LOCALBENCH_BEADS_ROOT"]), root.resolve())
                    live_receipts = export / "docs" / "evidence" / "receipts"
                    live_receipts.mkdir(parents=True)
                    (live_receipts / "new.json").write_text("new proof")
                    run_dir = export / "runs" / "plan-run-1"
                    run_dir.mkdir(parents=True)
                    (run_dir / "summary.json").write_text("{}")
                    return type("Result", (), {"returncode": 0})()
                return real(argv, **kwargs)

            with mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.object(run_plan.subprocess, "run", side_effect=fake_run), \
                    mock.patch("sys.argv", ["run_plan", "spec.json", "--execute"]):
                self.assertEqual(run_plan.main(), 0)
            self.assertEqual(len(seen_live), 1)
            self.assertNotIn("--dry-run", seen_live[0])
            self.assertIn("--omp-frozen", seen_live[0])
            self.assertEqual((receipts / "new.json").read_text(), "new proof")
            self.assertEqual((receipts / "old.json").read_text(), "unchanged")
            self.assertTrue((root / "runs" / "plan-run-1" / "summary.json").is_file())
            self.assertEqual(list((root / "var" / "agent-tmp").iterdir()), [])

    def test_runner_refuses_when_br_resolves_beads_outside_main_repo(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "main"
            wrong = Path(tmp) / "export"
            root.mkdir()
            result = subprocess.CompletedProcess(
                ["br", "where", "--json"], 0, json.dumps({"path": str(wrong / ".beads")}), ""
            )
            with mock.patch.object(run_plan.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(SystemExit, "resolves Beads"):
                    run_plan.verify_beads_root(root, {})


    def test_execute_refuses_misrouted_beads_before_starting_proof(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "main"
            wrong = Path(tmp) / "wrong"
            root.mkdir()
            _init_export_repo(root)
            real_run = run_plan.subprocess.run
            proof_calls = []

            def fake_run(argv, **kwargs):
                if argv[:2] == ["br", "where"]:
                    return subprocess.CompletedProcess(
                        argv, 0, json.dumps({"path": str(wrong / ".beads")}), ""
                    )
                if "-m" in argv and "localbench" in argv:
                    proof_calls.append(argv)
                    return subprocess.CompletedProcess(argv, 0, "", "")
                return real_run(argv, **kwargs)

            with mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.object(run_plan.subprocess, "run", side_effect=fake_run), \
                    mock.patch("sys.argv", ["run_plan", "spec.json", "--execute"]):
                with self.assertRaisesRegex(SystemExit, "resolves Beads"):
                    run_plan.main()
            self.assertEqual(proof_calls, [])
    def test_runner_accepts_beads_in_main_repo(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.CompletedProcess(
                ["br", "where", "--json"], 0, json.dumps({"path": str(root / ".beads")}), ""
            )
            with mock.patch.object(run_plan.subprocess, "run", return_value=result):
                run_plan.verify_beads_root(root, {})

    def test_proofqueue_bead_actions_persist_in_main_repo_for_export_runs(self):
        from localbench import proofqueue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beads = root / ".beads"
            beads.mkdir()
            bindir = root / "bin"
            bindir.mkdir()
            br = bindir / "br"
            br.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "if sys.argv[1:] == ['where', '--json']:\n"
                "    print(json.dumps({'path': str(Path.cwd() / '.beads')}))\n"
                "    raise SystemExit(0)\n"
                "with (Path.cwd() / '.beads' / 'actions.jsonl').open('a') as out:\n"
                "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            br.chmod(0o755)
            export = root / "export"
            export.mkdir()
            export_receipt = export / "docs" / "evidence" / "receipts" / "proof.json"
            with mock.patch.object(proofqueue, "REPO_ROOT", export), mock.patch.dict(os.environ, {
                "LOCALBENCH_BEADS_ROOT": str(root),
                "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", ""),
            }):
                self.assertEqual(proofqueue._br([
                    "close", "kit-sample", "--reason", f"receipt {export_receipt.resolve()}",
                ])[0], 0)
                self.assertEqual(proofqueue._br(["comments", "add", "kit-sample", "proof result"])[0], 0)
            actions = [json.loads(line) for line in (beads / "actions.jsonl").read_text().splitlines()]
            self.assertEqual(actions, [
                ["close", "kit-sample", "--reason",
                 f"receipt {(root / 'docs' / 'evidence' / 'receipts' / 'proof.json').resolve()}"],
                ["comments", "add", "kit-sample", "proof result"],
            ])

    def test_proofqueue_refuses_misrouted_beads(self):
        from localbench import proofqueue
        with tempfile.TemporaryDirectory() as tmp:
            root, wrong = Path(tmp) / "main", Path(tmp) / "wrong"
            root.mkdir()
            export = Path(tmp) / "export"
            result = subprocess.CompletedProcess(
                ["br", "where", "--json"], 0, json.dumps({"path": str(wrong / ".beads")}), ""
            )
            with mock.patch.object(proofqueue, "REPO_ROOT", export), \
                    mock.patch.dict(os.environ, {"LOCALBENCH_BEADS_ROOT": str(root)}), \
                    mock.patch.object(proofqueue.subprocess, "run", return_value=result) as run:
                rc, _out, err = proofqueue._br(["close", "kit-sample", "--reason", "proof"])
            self.assertEqual(rc, 1)
            self.assertIn("resolves Beads", err)
            run.assert_called_once()

    def test_archive_revision_stays_pinned_when_main_advances(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _init_export_repo(root)
            real_run = run_plan.subprocess.run
            archive_refs: list[str] = []
            advanced: list[str] = []

            def racing_run(argv, **kwargs):
                result = real_run(argv, **kwargs)
                if argv[:2] == ["git", "archive"] and not advanced:
                    archive_refs.append(argv[2])
                    (root / "version.txt").write_text("advanced\n")
                    _git(root, "add", "version.txt")
                    _git(root, "commit", "--quiet", "-m", "advance main")
                    advanced.append(_git(root, "rev-parse", "HEAD").stdout.strip())
                return result

            with mock.patch.object(run_plan, "verify_beads_root"), \
                    mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.object(run_plan.subprocess, "run", side_effect=racing_run), \
                    mock.patch("sys.argv", ["run_plan", "spec.json", "--execute"]):
                self.assertEqual(run_plan.main(), 0)

            receipt = json.loads((root / "docs" / "evidence" / "receipts" / "run-plan.json").read_text())
            self.assertEqual(archive_refs, [base])
            self.assertNotEqual(advanced, [base])
            self.assertEqual(receipt["head"], base)
            self.assertEqual(receipt["version"], "base\n")
            self.assertEqual(receipt["status"], "")
            owner = dict(part.split("=", 1) for part in receipt["owner"].split() if "=" in part)
            self.assertEqual(owner["pid"], str(os.getpid()))
            self.assertEqual(owner["label"], "run-plan")
            self.assertEqual(Path(owner["repo"]).resolve(), root.resolve())
            self.assertIn("created", owner)
            self.assertEqual(list((root / "var" / "agent-tmp").iterdir()), [])

    def test_spec_path_cannot_escape_export(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_export_repo(root)
            outside = root / "var" / "agent-tmp" / "outside.json"
            outside.write_text("{}\n")
            real_run = run_plan.subprocess.run
            proof_invocations = []

            def observe_proof(argv, **kwargs):
                if "-m" in argv and "localbench" in argv:
                    proof_invocations.append(argv)
                    return subprocess.CompletedProcess(argv, 0, "", "")
                return real_run(argv, **kwargs)

            with mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.object(run_plan.subprocess, "run", side_effect=observe_proof), \
                    mock.patch("sys.argv", ["run_plan", "../outside.json", "--execute"]):
                with self.assertRaisesRegex(SystemExit, "export"):
                    run_plan.main()
            self.assertEqual(proof_invocations, [])
            self.assertTrue(outside.is_file())

    def test_git_hook_environment_does_not_redirect_export(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            main_root, other_root = parent / "main", parent / "other"
            main_root.mkdir()
            other_root.mkdir()
            main_head = _init_export_repo(main_root, "main")
            other_head = _init_export_repo(other_root, "other")
            hook_env = {"GIT_DIR": str(other_root / ".git"),
                        "GIT_WORK_TREE": str(other_root),
                        "GIT_INDEX_FILE": str(other_root / ".git" / "index")}
            with mock.patch.object(run_plan, "verify_beads_root"), \
                    mock.patch.object(run_plan, "ROOT", main_root), \
                    mock.patch.dict(os.environ, hook_env), \
                    mock.patch("sys.argv", ["run_plan", "spec.json", "--execute"]):
                try:
                    self.assertEqual(run_plan.main(), 0)
                except (OSError, subprocess.CalledProcessError, SystemExit) as exc:
                    self.fail(f"run_plan inherited the caller's Git environment: {exc}")
            receipt = json.loads((main_root / "docs" / "evidence" / "receipts" / "run-plan.json").read_text())
            self.assertEqual(receipt["head"], main_head)
            self.assertEqual(receipt["version"], "main\n")
            self.assertEqual(_git(other_root, "rev-parse", "HEAD").stdout.strip(), other_head)



    def test_git_hook_environment_does_not_redirect_main_export(self):
        from scripts import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root, other = parent / "main", parent / "other"
            root.mkdir()
            other.mkdir()
            main_head = _init_export_repo(root, "main")
            other_head = _init_export_repo(other, "other")
            hook_env = {"GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other),
                        "GIT_INDEX_FILE": str(other / ".git" / "index")}
            with mock.patch.object(run_plan, "verify_beads_root"), \
                    mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.dict(os.environ, hook_env), \
                    mock.patch("sys.argv", ["run_plan", "spec.json", "--execute"]):
                try:
                    self.assertEqual(run_plan.main(), 0)
                except (OSError, subprocess.CalledProcessError, SystemExit) as exc:
                    self.fail(f"run_plan inherited outer GIT_* environment: {exc}")
            receipt = json.loads((root / "docs" / "evidence" / "receipts" / "run-plan.json").read_text())
            self.assertEqual(receipt["head"], main_head)
            self.assertEqual(receipt["version"], "main\n")
            self.assertEqual(_git(other, "rev-parse", "HEAD").stdout.strip(), other_head)

    def test_clean_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(prove, "_git", return_value=(0, "")):
                prove._refuse_dirty_code_tree(Path(tmp))


if __name__ == "__main__":
    unittest.main()
