from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
SITE_CUSTOMIZE = r'''
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("QKMI_FAKE_SLOT"):
    from localbench import heavyslot

    class FakeSlot:
        admission = "test-slot"

        def __init__(self):
            self.path = Path(os.environ["QKMI_SLOT_FILE"])

        def release(self):
            self.path.unlink(missing_ok=True)

        def __enter__(self):
            self.path.write_text("held\\n")
            return self

        def __exit__(self, *_exc):
            self.release()

    # Child interpreters do not inherit tests/__init__.py monkeypatches; keep the fake-slot child off live sensors.
    heavyslot.LOAD_FN = lambda: (1.0, 1.0, 1.0)
    heavyslot.CPU_FN = lambda: 1.0
    heavyslot.MEMORY_FN = lambda: {"pressure_level": "normal"}
    heavyslot.GPU_FN = lambda: {"device_pct": 1.0, "process_pct": 0.5, "coverage": None,
                               "unattributed_pct": 0.0, "status": "IDLE"}
    heavyslot.acquire = lambda *_args, **_kwargs: FakeSlot()

if os.environ.get("QKMI_MODE") == "laya":
    from localbench import decision, lifecycle
    import localbench.__main__ as cli

    class FakeMutation:
        def __init__(self, *_args, **_kwargs):
            self.dry_run = False
            self.detail = {}

        def gate(self, *_args, **_kwargs):
            return None

        def record(self, *_args, **_kwargs):
            pass

        def finish(self, *_args, **_kwargs):
            pass

    class FakeShim:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            child = lifecycle.spawn([sys.executable, "-c", "import time; time.sleep(60)"],
                                    start_new_session=True)
            Path(os.environ["QKMI_CHILD_INFO"]).write_text(json.dumps({
                "pid": child.pid,
                "session": os.getsid(child.pid),
            }))
            while True:
                time.sleep(1)

        def __exit__(self, *_exc):
            return False

    def fake_run_laya(_spec, _suite, *, shim, **_kwargs):
        with shim:
            time.sleep(60)

    cli.Mutation = FakeMutation
    cli.RUNS = Path(os.environ["QKMI_ROOT"]) / "runs"
    cli.ROOT = Path(os.environ["QKMI_ROOT"])
    cli._run_alive = lambda: False
    cli._decision_wait_idle = lambda _seconds: 0
    cli.sysstats.Sampler = lambda **_kwargs: object()
    decision.parse_laya_spec = lambda _spec: SimpleNamespace(name="fake-model")
    decision.resolve_suite = lambda _suite: SimpleNamespace(name="fake-suite", items=[object()], role="decision")
    venv = Path(os.environ["QKMI_VENV"])
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python3").write_text("fake executable marker\\n")
    decision.laya_venv = lambda: venv
    decision.LayaShim = FakeShim
    decision.run_laya = fake_run_laya
'''
FAKE_UV = r'''#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
try:
    python_args = args[args.index("python") + 1:]
except ValueError:
    raise SystemExit(2)
if "-c" in python_args or any(arg.endswith("scripts/heavy_run.py") for arg in python_args):
    os.execv(sys.executable, [sys.executable, *python_args])
if python_args[:2] != ["-m", "unittest"]:
    raise SystemExit(2)
count_path = Path(os.environ["QKMI_UV_COUNT"])
try:
    count = int(count_path.read_text())
except (OSError, ValueError):
    count = 0
count_path.write_text(str(count + 1))
if os.environ.get("QKMI_UV_MODE") == "mutate" and count == 0:
    raise SystemExit(0)
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(os.environ["QKMI_CHILD_INFO"]).write_text(json.dumps({
    "pid": os.getpid(),
    "grandchild": grandchild.pid,
    "session": os.getsid(0),
}))
grandchild.wait()
'''


class LifecycleCancellation(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "var" / "agent-tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(prefix="qkmi-lifecycle-", dir=scratch)
        self.tmp = Path(self._tmp.name)
        (self.tmp / ".owner").write_text(f"pid {os.getpid()} label qkmi-lifecycle repo={ROOT}\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _wait_for_info(self, path: Path, process: subprocess.Popen[str], timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                return json.loads(path.read_text())
            if process.poll() is not None:
                output = process.communicate()[0]
                self.fail(f"child harness exited {process.returncode} before readiness: {output}")
            time.sleep(0.01)
        self.fail(f"child harness did not become ready within {timeout:g}s")

    @staticmethod
    def _session_pids(session_id: int) -> list[int]:
        result = subprocess.run(["/bin/ps", "-A", "-o", "pid=", "-o", "sess="], capture_output=True, text=True,
                               timeout=3, check=True)
        pids = []
        for row in result.stdout.splitlines():
            fields = row.split()
            if len(fields) == 2 and fields[1] == str(session_id):
                pids.append(int(fields[0]))
        return pids

    def _assert_session_stopped(self, session_id: int, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._session_pids(session_id):
                return
            time.sleep(0.02)
        self.fail(f"process session {session_id} still has descendants: {self._session_pids(session_id)}")

    def _environment(self, mode: str, child_info: Path, slot_file: Path) -> dict[str, str]:
        site_dir = self.tmp / "site"
        site_dir.mkdir(exist_ok=True)
        (site_dir / "sitecustomize.py").write_text(SITE_CUSTOMIZE)
        return {
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(site_dir), str(ROOT), os.environ.get("PYTHONPATH", ""))),
            "QKMI_MODE": mode,
            "QKMI_FAKE_SLOT": "1",
            "QKMI_CHILD_INFO": str(child_info),
            "QKMI_SLOT_FILE": str(slot_file),
            "QKMI_ROOT": str(self.tmp / "laya-root"),
            "QKMI_VENV": str(self.tmp / "laya-venv"),
            "HOME": str(self.tmp / "home"),
        }

    def _runtime_environment(self, child_info: Path, slot_file: Path, uv_mode: str) -> dict[str, str]:
        env = self._environment("runner", child_info, slot_file)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        uv = bin_dir / "uv"
        uv.write_text(FAKE_UV)
        uv.chmod(0o755)
        env.update({
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "QKMI_UV_MODE": uv_mode,
            "QKMI_UV_COUNT": str(self.tmp / f"uv-count-{uv_mode}"),
            "LOCALBENCH_GATE_TREE": "qkmi-test-tree",
            "LOCALBENCH_GATE_LOG": str(self.tmp / "gates.jsonl"),
        })
        return env

    def _signal_child(self, command: list[str], env: dict[str, str], child_info: Path, signum: int,
                      slot_file: Path | None = None, cwd: Path = ROOT) -> dict:
        child_info.unlink(missing_ok=True)
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        session_id = None
        try:
            info = self._wait_for_info(child_info, process)
            session_id = info["session"]
            process.send_signal(signum)
            output = process.communicate(timeout=5)[0]
            self.assertNotEqual(process.returncode, 0, f"signal was ignored: {output}")
            self._assert_session_stopped(session_id)
            if slot_file is not None:
                self.assertFalse(slot_file.exists(), f"heavy slot still held after cancellation: {output}")
            return info
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            process.communicate()
            if session_id is not None and self._session_pids(session_id):
                try:
                    os.killpg(session_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_each_cancel_signal_terminates_the_registered_process_group(self) -> None:
        child_info = self.tmp / "child.json"
        grandchild_code = "import signal,time; signal.signal(signal.SIGTERM, lambda *_: None); time.sleep(60)"
        child_script = (
            "import json,os,signal,subprocess,sys,time; "
            "signal.signal(signal.SIGTERM, lambda *_: None); "
            f"grandchild=subprocess.Popen([sys.executable,'-c',{grandchild_code!r}]); "
            "open(os.environ['QKMI_CHILD_INFO'],'w').write(json.dumps({"
            "'grandchild':grandchild.pid,'session':os.getsid(0)})); time.sleep(60)"
        )
        driver = f"""
import os, subprocess, sys, time
from pathlib import Path
from localbench import lifecycle
lifecycle.install_cancel_handlers()
child_script = {child_script!r}
child = lifecycle.spawn([sys.executable, '-c', child_script], start_new_session=True)
while not Path(os.environ['QKMI_CHILD_INFO']).exists():
    time.sleep(0.01)
while True:
    time.sleep(1)
"""
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "QKMI_CHILD_INFO": str(child_info)}
        for signum in SIGNALS:
            with self.subTest(signal=signal.Signals(signum).name):
                info = self._signal_child([sys.executable, "-c", driver], env, child_info, signum)
                self.assertGreater(info["grandchild"], 0)

    def test_localbench_cancels_a_fake_laya_shim_for_each_signal(self) -> None:
        child_info = self.tmp / "laya-child.json"
        slot_file = self.tmp / "localbench.slot"
        env = self._environment("laya", child_info, slot_file)
        command = [sys.executable, "-c",
                   "from localbench import __main__; raise SystemExit(__main__.main(" + repr([
                       "decision", "run", "laya:fake/model", "--suite", "fake-suite"]
                   ) + "))"]
        for signum in SIGNALS:
            with self.subTest(signal=signal.Signals(signum).name):
                self._signal_child(command, env, child_info, signum, slot_file)

    def test_mutate_cancels_child_and_restores_the_plant_for_each_signal(self) -> None:
        target = ROOT / "localbench" / "lifecycle.py"
        journal = ROOT / "runs" / ".mutation-journal.json"
        lock = ROOT / "runs" / ".mutation.lock"
        for signum in SIGNALS:
            with self.subTest(signal=signal.Signals(signum).name):
                original = target.read_bytes()
                self.assertFalse(journal.exists(), "a prior mutation journal must be resolved before this test")
                self.assertFalse(lock.exists(), "the mutation lock must be free before this test")
                child_info = self.tmp / f"mutate-{signum}.json"
                slot_file = self.tmp / f"mutate-{signum}.slot"
                env = self._runtime_environment(child_info, slot_file, "mutate")
                count_path = Path(env["QKMI_UV_COUNT"])
                count_path.write_text("0")
                command = [sys.executable, "scripts/mutate.py", "tests/lifecycle-mutations.json"]
                process = subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                session_id = None
                try:
                    info = self._wait_for_info(child_info, process)
                    session_id = info["session"]
                    self.assertTrue(journal.is_file(), "the signal must arrive after the mutation journal is durable")
                    self.assertNotEqual(target.read_bytes(), original, "the runner must be in its planted-test phase")
                    process.send_signal(signum)
                    output = process.communicate(timeout=5)[0]
                    self.assertNotEqual(process.returncode, 0, f"signal was ignored: {output}")
                    self._assert_session_stopped(session_id)
                    self.assertEqual(target.read_bytes(), original, f"mutation plant was not restored: {output}")
                    self.assertFalse(journal.exists(), f"mutation journal remains after cancellation: {output}")
                    self.assertFalse(lock.exists(), f"mutation lock remains after cancellation: {output}")
                    self.assertFalse(slot_file.exists(), f"heavy slot remains held: {output}")
                    self.assertFalse(list(journal.parent.glob(f".{journal.name}.*.tmp")), "mutation journal temp remains")
                    self.assertFalse(list(target.parent.glob(f".{target.name}.*.tmp")), "source replacement temp remains")
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=2)
                    process.communicate()
                    if session_id is not None and self._session_pids(session_id):
                        try:
                            os.killpg(session_id, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if target.read_bytes() != original:
                        payload = json.loads(journal.read_text()) if journal.exists() else {}
                        current_sha = hashlib.sha256(target.read_bytes()).hexdigest()
                        if payload.get("original_sha") == hashlib.sha256(original).hexdigest() and \
                                payload.get("planted_sha") == current_sha:
                            target.write_bytes(original)
                            journal.unlink(missing_ok=True)

    def test_heavy_run_releases_slot_and_child_group_for_each_signal(self) -> None:
        for signum in SIGNALS:
            with self.subTest(signal=signal.Signals(signum).name):
                child_info = self.tmp / f"heavy-{signum}.json"
                slot_file = self.tmp / f"heavy-{signum}.slot"
                env = self._runtime_environment(child_info, slot_file, "always")
                self._signal_child([sys.executable, "scripts/heavy_run.py", "0"], env, child_info, signum,
                                   slot_file)

    def test_suite_verdict_trap_stops_heavy_run_tree_for_each_signal(self) -> None:
        repo = self.tmp / "suite-repo"
        (repo / "scripts").mkdir(parents=True)
        (repo / "localbench").mkdir()
        for relative in ("scripts/suite_verdict.sh", "scripts/heavy_run.py", "localbench/lifecycle.py"):
            shutil.copy2(ROOT / relative, repo / relative)
        (repo / "localbench/__init__.py").write_text("")
        (repo / "localbench/gates.py").write_text(
            "def working_tree_id(_root): return 'fixture'\n"
            "def parse_unittest_output(_text): return [], []\n"
            "def append_row(**_kwargs): return {}\n"
        )
        (repo / "localbench/heavyslot.py").write_text(
            "class SlotRefused(Exception): pass\n"
            "def acquire(*_args, **_kwargs): raise SlotRefused('slot injector was not loaded')\n"
        )
        subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
        tracked = [
            "scripts/suite_verdict.sh",
            "scripts/heavy_run.py",
            "localbench/__init__.py",
            "localbench/gates.py",
            "localbench/heavyslot.py",
            "localbench/lifecycle.py",
        ]
        subprocess.run(["git", "add", *tracked], cwd=repo, check=True)
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=Localbench tests",
                        "-c", "user.email=tests@example.invalid", "-c", "commit.gpgsign=false",
                        "commit", "--quiet", "-m", "suite fixture"],
                       cwd=repo, check=True)
        site_dir = self.tmp / "site"
        for signum in SIGNALS:
            with self.subTest(signal=signal.Signals(signum).name):
                child_info = self.tmp / f"suite-{signum}.json"
                slot_file = self.tmp / f"suite-{signum}.slot"
                env = self._runtime_environment(child_info, slot_file, "always")
                env["PYTHONPATH"] = os.pathsep.join((str(site_dir), str(repo)))
                env["SUITE_VERDICT_WAIT"] = "0"
                self._signal_child(["/bin/sh", str(repo / "scripts" / "suite_verdict.sh"), "HEAD"], env,
                                   child_info, signum, slot_file, cwd=repo)
