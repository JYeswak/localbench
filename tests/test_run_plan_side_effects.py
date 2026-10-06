from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from localbench import proofqueue
from scripts import run_plan

REPO = Path(__file__).resolve().parent.parent


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    if env:
        clean_env.update(env)
    result = subprocess.run(["git", *args], cwd=root, env=clean_env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _scratch_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / ".beads").mkdir(parents=True)
    (repo / ".beads" / "issues.jsonl").write_text('{"id":"kit-sample","status":"open"}\n')
    spec = repo / "registries" / "proofs" / "sample.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}\n")
    (repo / "var" / "agent-tmp").mkdir(parents=True)

    _git(repo, "init", "--quiet")
    _git(repo, "add", ".")
    tree = _git(repo, "write-tree")
    commit = _git(repo, "commit-tree", tree, "-m", "scratch base",
                  env={"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                       "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"})
    _git(repo, "update-ref", "refs/heads/main", commit)
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")

    bindir = repo / "bin"
    bindir.mkdir()
    br = bindir / "br"
    br.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "log = Path.cwd() / '.beads' / 'actions.jsonl'\n"
        "if args == ['where', '--json']:\n"
        "    print(json.dumps({'path': os.environ.get('BR_WHERE_PATH', str(Path.cwd() / '.beads'))}))\n"
        "elif args == ['show', 'kit-sample', '--json']:\n"
        "    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []\n"
        "    close = next((row for row in rows if row[0] == 'close'), None)\n"
        "    comments = [row[row.index('-m') + 1] for row in rows if row[:2] == ['comments', 'add']]\n"
        "    reason = close[close.index('--reason') + 1] if close and '--reason' in close else None\n"
        "    print(json.dumps({'id': 'kit-sample', 'status': 'closed' if close else 'open',\n"
        "                      'close_reason': reason, 'comments': comments}))\n"
        "elif args and args[0] in ('close', 'comments'):\n"
        "    with log.open('a', encoding='utf-8') as out:\n"
        "        out.write(json.dumps(args) + '\\n')\n"
        "    print('{}')\n"
        "else:\n"
        "    print('{}')\n",
        encoding="utf-8",
    )
    br.chmod(0o755)
    return repo


def _owned_scratch() -> tempfile.TemporaryDirectory[str]:
    parent = REPO / "var" / "agent-tmp"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="mu71.17-", dir=parent)


class RunPlanSideEffects(unittest.TestCase):
    def test_execute_keeps_bead_updates_and_artifacts_in_main_repo(self):
        with _owned_scratch() as owned:
            scratch = Path(owned)
            (scratch / ".owner").write_text(f"pid={os.getpid()} label=mu71.17-test repo={REPO}\n")
            root = _scratch_repo(scratch)
            bindir = root / "bin"
            exports: list[Path] = []
            real_run = subprocess.run

            def fake_run(argv, **kwargs):
                if argv and argv[0] == "uv":
                    export = Path(kwargs["cwd"])
                    exports.append(export)
                    self.assertEqual(Path(kwargs["env"]["LOCALBENCH_HOME"]), export.resolve())
                    self.assertEqual(Path(kwargs["env"]["LOCALBENCH_BEADS_ROOT"]), root.resolve())
                    receipt = export / "docs" / "evidence" / "receipts" / "sample.json"
                    receipt.parent.mkdir(parents=True)
                    receipt.write_text('{"result":"known-good"}\n')
                    run_summary = export / "runs" / "sample-run" / "summary.json"
                    run_summary.parent.mkdir(parents=True)
                    run_summary.write_text('{"result":"known-good"}\n')
                    with mock.patch.object(proofqueue, "REPO_ROOT", export), \
                            mock.patch.dict(os.environ, {"LOCALBENCH_BEADS_ROOT": str(root)}):
                        path = str(receipt.resolve())
                        for args in (["close", "kit-sample", "--reason", f"receipt {path}"],
                                     ["comments", "add", "kit-sample", "-m", f"proof receipt {path}"]):
                            rc, _, err = proofqueue._br(args)
                            self.assertEqual(rc, 0, err)
                    return subprocess.CompletedProcess(argv, 0, "{}\n", "")
                return real_run(argv, **kwargs)

            with mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.dict(os.environ, {"PATH": str(bindir) + os.pathsep + os.environ.get("PATH", "")}), \
                    mock.patch.object(run_plan.subprocess, "run", side_effect=fake_run), \
                    mock.patch("sys.argv", ["run_plan", "registries/proofs/sample.json", "--execute"]):
                self.assertEqual(run_plan.main(), 0)
                shown = real_run(["br", "show", "kit-sample", "--json"], cwd=root, capture_output=True,
                                 text=True, check=True)

            self.assertEqual(len(exports), 1)
            self.assertFalse(exports[0].exists())
            receipt = root / "docs" / "evidence" / "receipts" / "sample.json"
            self.assertEqual(json.loads(receipt.read_text()), {"result": "known-good"})
            self.assertEqual(json.loads((root / "runs" / "sample-run" / "summary.json").read_text()),
                             {"result": "known-good"})
            issue = json.loads(shown.stdout)
            self.assertEqual(issue["status"], "closed")
            self.assertIn(str(receipt.resolve()), issue["close_reason"])
            self.assertIn(str(receipt.resolve()), issue["comments"][0])
            self.assertNotIn(str(exports[0]), issue["close_reason"] + issue["comments"][0])
            self.assertEqual(list((root / "var" / "agent-tmp").iterdir()), [])

    def test_execute_refuses_misrouted_beads_before_fake_prove(self):
        with _owned_scratch() as owned:
            scratch = Path(owned)
            (scratch / ".owner").write_text(f"pid={os.getpid()} label=mu71.17-test repo={REPO}\n")
            root = _scratch_repo(scratch)
            wrong = scratch / "wrong"
            (wrong / ".beads").mkdir(parents=True)
            prove_calls: list[list[str]] = []
            real_run = subprocess.run

            def fake_run(argv, **kwargs):
                if argv and argv[0] == "uv":
                    prove_calls.append(argv)
                    return subprocess.CompletedProcess(argv, 0, "{}\n", "")
                return real_run(argv, **kwargs)

            with mock.patch.object(run_plan, "ROOT", root), \
                    mock.patch.dict(os.environ, {"PATH": str(root / "bin") + os.pathsep + os.environ.get("PATH", ""),
                                                 "BR_WHERE_PATH": str(wrong / ".beads")}), \
                    mock.patch.object(run_plan.subprocess, "run", side_effect=fake_run), \
                    mock.patch("sys.argv", ["run_plan", "registries/proofs/sample.json", "--execute"]):
                with self.assertRaisesRegex(SystemExit, "br resolves Beads"):
                    run_plan.main()

            self.assertEqual(prove_calls, [])
            self.assertFalse((root / "docs" / "evidence" / "receipts" / "sample.json").exists())
            self.assertEqual(list((root / "var" / "agent-tmp").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
