#!/usr/bin/env python3
"""Run the full pure-logic unittest discovery under macOS network isolation."""
from __future__ import annotations

import contextlib
import errno
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localbench.sysstats import INFERENCE_PORTS

SCRIPT = Path(__file__).resolve()
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
SCRATCH_ROOT = ROOT / "var" / "agent-tmp"
SYSTEM_PATHS = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
MODEL_SERVICE_PORTS = tuple(INFERENCE_PORTS)
TOOLS = ("omp", "ollama")
EXTERNAL_PROBE = ("1.1.1.1", 443)
CHILD_FLAG = "LOCALBENCH_HERMETIC_CHILD"
DENY_PORT_VAR = "LOCALBENCH_HERMETIC_DENY_PORT"
EGRESS_CONTROL_SOCKET_VAR = "LOCALBENCH_HERMETIC_EGRESS_CONTROL_SOCKET"
NETWORK_DENY = "(deny network-outbound)"
NETWORK_IP_DENY = NETWORK_DENY.replace(")", ' (remote ip "*:*"))')
NETWORK_UNIX_SOCKET_DENY = NETWORK_DENY.replace(")", " (remote unix-socket))")
LOCALHOST_ALLOW = '(allow network-outbound (remote ip "localhost:*"))'
REPORT_JSON_TEST = "tests.test_cli.PureJsonStdout.test_report_json"
REPORT_JSON_SKIP = "no runs/observe.db (localbench watch never ran)"


class HermeticError(RuntimeError):
    pass


def sanitized_path(python_executable: str | None = None) -> str:
    """Keep the selected Python and base macOS tools; omit user and package-manager bins."""
    python_executable = python_executable or sys.executable
    return os.pathsep.join((str(Path(python_executable).parent), *SYSTEM_PATHS))


def child_environment(python_executable: str | None = None,
                      parent_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if parent_env is None else parent_env)
    for key in tuple(env):
        if key.lower().endswith("_proxy") or key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
            env.pop(key, None)
    for key in ("PYTHONPATH", "PYTHONHOME", "LOCALBENCH_OMP", "OLLAMA_HOST", "OLLAMA_ORIGINS"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = sanitized_path(python_executable)
    env[CHILD_FLAG] = "1"
    return env


def suite_temp_root(environment: Mapping[str, str]) -> Path:
    if environment.get("GITHUB_ACTIONS", "").lower() != "true":
        return SCRATCH_ROOT
    runner_temp = environment.get("RUNNER_TEMP")
    if not runner_temp:
        raise HermeticError("GitHub Actions hermetic tests require RUNNER_TEMP")
    root = Path(runner_temp)
    if not root.is_absolute():
        raise HermeticError("GitHub Actions RUNNER_TEMP must be an absolute path")
    root = root.resolve()
    try:
        root.relative_to(ROOT)
    except ValueError:
        return root
    raise HermeticError("GitHub Actions RUNNER_TEMP must be outside the checkout")

def _owned_scratch(parent: Path, label: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f"{label}.{os.getpid()}.", dir=parent))
    (path / ".owner").write_text(
        f"pid={os.getpid()} label={label} repo={ROOT} "
        f"created={time.strftime('%FT%TZ', time.gmtime())}\n"
    )
    return path


def profile_source(denied_ports: tuple[int, ...]) -> str:
    ports = sorted(set((*MODEL_SERVICE_PORTS, *denied_ports)))
    rules = ["(version 1)", "(allow default)", NETWORK_IP_DENY, NETWORK_UNIX_SOCKET_DENY, LOCALHOST_ALLOW]
    rules.extend(f'(deny network-outbound (remote ip "localhost:{port}"))' for port in ports)
    return "\n".join(rules) + "\n"


@dataclass(frozen=True)
class Sandbox:
    profile: Path
    env: dict[str, str]
    denied_port: int
    egress_control_path: Path


@contextlib.contextmanager
def sandbox_context(parent_env: Mapping[str, str] | None = None,
                    python_executable: str | None = None) -> Iterator[Sandbox]:
    if sys.platform != "darwin":
        raise HermeticError("the hermetic CI gate requires macOS sandbox-exec")
    if not Path(SANDBOX_EXEC).is_file():
        raise HermeticError(f"missing macOS sandbox runner: {SANDBOX_EXEC}")
    if not (ROOT / "fixtures" / "omp").is_dir():
        raise HermeticError("declared local fixture tree is missing: fixtures/omp")
    if Path.cwd().resolve() != ROOT:
        raise HermeticError("run from the repository root for the local egress-control socket")

    environment = dict(os.environ if parent_env is None else parent_env)
    suite_root = suite_temp_root(environment)
    private = _owned_scratch(SCRATCH_ROOT, "hermetic-suite")
    profile = private / "network.sb"
    egress_control_path = private.relative_to(ROOT) / "e.sock"
    with contextlib.ExitStack() as sockets:
        denied_listener = sockets.enter_context(socket.socket())
        denied_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        denied_listener.bind(("127.0.0.1", 0))
        denied_listener.listen(1)
        denied_port = int(denied_listener.getsockname()[1])
        egress_control_listener = sockets.enter_context(
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
        egress_control_listener.bind(str(egress_control_path))
        egress_control_listener.listen(1)

        profile.write_text(profile_source((denied_port,)))
        suite_tmp = _owned_scratch(suite_root, "localbench-suite-tmp")
        env = child_environment(python_executable, environment)
        env["TMPDIR"] = str(suite_tmp)
        env[DENY_PORT_VAR] = str(denied_port)
        env[EGRESS_CONTROL_SOCKET_VAR] = str(egress_control_path)
        yield Sandbox(profile=profile, env=env, denied_port=denied_port,
                      egress_control_path=egress_control_path)


def _loopback_fixture_round_trip() -> None:
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(2)
        with socket.create_connection(server.getsockname(), timeout=2) as client:
            connection, _ = server.accept()
            with connection:
                client.sendall(b"fixture")
                if connection.recv(7) != b"fixture":
                    raise HermeticError("loopback fixture received different bytes")
                connection.sendall(b"ok")
                if client.recv(2) != b"ok":
                    raise HermeticError("loopback fixture reply was not received")


def _expect_blocked(address: tuple[str, int], *, permission_only: bool) -> None:
    try:
        with socket.create_connection(address, timeout=2):
            raise HermeticError(f"unexpectedly connected to {address[0]}:{address[1]}")
    except OSError as exc:
        if permission_only and exc.errno not in (errno.EPERM, errno.EACCES):
            raise HermeticError(
                f"sandbox did not refuse {address[0]}:{address[1]} with permission denial: {exc}"
            ) from exc


def _expect_unix_socket_blocked(path: str) -> None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(path)
            raise HermeticError(f"unexpectedly connected to local egress-control socket: {path}")
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.EACCES):
            raise HermeticError(
                f"sandbox did not refuse local egress-control socket with permission denial: {exc}"
            ) from exc



def probe_current_sandbox() -> str:
    if os.environ.get(CHILD_FLAG) != "1":
        raise HermeticError("network probe must run inside the isolated child")
    missing = [name for name in TOOLS if shutil.which(name) is None]
    if missing != list(TOOLS):
        found = ", ".join(name for name in TOOLS if name not in missing)
        raise HermeticError(f"unexpected executable(s) on hermetic PATH: {found}")
    for name in TOOLS:
        try:
            if name == "omp":
                subprocess.run(["omp", "--version"], cwd=ROOT, env=os.environ, timeout=2, check=False,
                               capture_output=True, text=True)
            elif name == "ollama":
                subprocess.run(["ollama", "--version"], cwd=ROOT, env=os.environ, timeout=2, check=False,
                               capture_output=True, text=True)
            else:
                raise HermeticError(f"no spawn probe declared for {name}")
        except FileNotFoundError:
            continue
        raise HermeticError(f"unexpectedly spawned {name} from the hermetic PATH")

    egress_control_path = os.environ.get(EGRESS_CONTROL_SOCKET_VAR)
    if not egress_control_path:
        raise HermeticError("missing local egress-control socket path")
    _expect_unix_socket_blocked(egress_control_path)
    try:
        with socket.create_connection(EXTERNAL_PROBE, timeout=2):
            raise HermeticError("external network probe unexpectedly connected")
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.EACCES):
            raise HermeticError(f"external network was not refused by the sandbox: {exc}") from exc

    _loopback_fixture_round_trip()
    denied_port = int(os.environ.get(DENY_PORT_VAR, "0"))
    if not denied_port:
        raise HermeticError("missing local-service denial probe port")
    _expect_blocked(("127.0.0.1", denied_port), permission_only=True)
    for port in MODEL_SERVICE_PORTS:
        _expect_blocked(("127.0.0.1", port), permission_only=True)
    return ("hermetic probe: omp=absent-from-PATH ollama=absent-from-PATH spawn=refused(FileNotFoundError) "
            "external-egress=blocked egress-control=blocked loopback-fixture=passed "
            "model-service-ports=denied")


def _run_child(sandbox: Sandbox, mode: str) -> subprocess.CompletedProcess[str]:
    command = [SANDBOX_EXEC, "-f", str(sandbox.profile), sys.executable, str(SCRIPT), mode]
    return subprocess.run(command, cwd=ROOT, env=sandbox.env, capture_output=True, text=True, check=False)


def run_isolated_probe(parent_env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    with sandbox_context(parent_env=parent_env) as sandbox:
        return _run_child(sandbox, "--probe-child")


def skip_is_allowed(test_id: str, reason: str) -> bool:
    return test_id == REPORT_JSON_TEST and reason == REPORT_JSON_SKIP


def suppression_violations(result: unittest.TestResult) -> list[str]:
    violations = [f"skipped {test.id()}: {reason}" for test, reason in result.skipped
                  if not skip_is_allowed(test.id(), reason)]
    violations.extend(f"expected failure {test.id()}" for test, _ in result.expectedFailures)
    violations.extend(f"unexpected success {test.id()}" for test in result.unexpectedSuccesses)
    return violations


def _run_unit_suite() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), pattern="test*.py", top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"hermetic suite: full discovery ran {result.testsRun} tests from tests/test*.py")
    for test, reason in result.skipped:
        if skip_is_allowed(test.id(), reason):
            print(f"hermetic suite: allowed local-data skip {test.id()}: {reason}")
    violations = suppression_violations(result)
    if violations:
        print("hermetic suite: refused suppressed tests:\n  " + "\n  ".join(violations), file=sys.stderr)
    return 0 if result.wasSuccessful() and not violations else 1


def _emit(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)


def _run_id_line() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID")
    repository = os.environ.get("GITHUB_REPOSITORY")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    if run_id and repository:
        return f"CI run URL: {server}/{repository}/actions/runs/{run_id}"
    return "CI run URL: unavailable (not running in GitHub Actions)"


def _run_parent(probe_only: bool) -> int:
    try:
        with sandbox_context() as sandbox:
            probe = _run_child(sandbox, "--probe-child")
            _emit(probe)
            if probe.returncode:
                return probe.returncode
            print("hermetic preflight: PATH excludes omp/ollama; external egress denied; loopback fixtures allowed")
            print(f"hermetic preflight: declared local fixtures=fixtures/omp; model-service ports={','.join(map(str, MODEL_SERVICE_PORTS))} denied")
            print(_run_id_line())
            if probe_only:
                return 0
            suite = _run_child(sandbox, "--suite-child")
            _emit(suite)
            return suite.returncode
    except (HermeticError, OSError) as exc:
        print(f"hermetic suite: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--probe-child"]:
        try:
            print(probe_current_sandbox())
            return 0
        except (HermeticError, OSError, ValueError) as exc:
            print(f"hermetic probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if argv == ["--suite-child"]:
        return _run_unit_suite()
    if argv == ["--probe-only"]:
        return _run_parent(probe_only=True)
    if argv:
        print("usage: hermetic_suite.py [--probe-only]", file=sys.stderr)
        return 2
    return _run_parent(probe_only=False)


if __name__ == "__main__":
    raise SystemExit(main())
