"""Execute one preregistered behavioral trial without feeding its reference state to omp."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .evaluation import workspace_text
from .workloads import CHILD_CONFIG, child_env, child_flags, omp_bin


def _stop_child_group(proc: subprocess.Popen[str]) -> None:
    """Stop omp and the tool processes it launched in the same session."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_trial(spec: dict, model: str, run_dir: Path, *, config: Path = CHILD_CONFIG,
              timeout_s: int = 1800) -> tuple[dict, list[Path]]:
    """Keep the pre-state, complete omp trajectory, exit, and post-state even when the model fails."""
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    workspace.mkdir()
    for filename, content in ((spec["file"], spec["initial"]),
                              (spec["decoy_file"], spec["decoy_initial"])):
        if Path(filename).name != filename or not isinstance(content, str):
            raise ValueError("trial fixture has an unsafe filename or non-text content")
        (workspace / filename).write_text(content, encoding="utf-8")
    initial = run_dir / "initial_state.json"
    initial.write_text(json.dumps({spec["file"]: spec["initial"], spec["decoy_file"]: spec["decoy_initial"]},
                                  sort_keys=True) + "\n", encoding="utf-8")

    flags = [arg for arg in child_flags(model, config=config) if not arg.startswith("--tools=")]
    flags.append("--tools=" + ",".join(spec["allowed_tools"]))
    command = [omp_bin(), "-p", spec["prompt"], *flags]
    started = time.perf_counter()
    with subprocess.Popen(command, cwd=workspace, env=child_env(), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL,
                          start_new_session=True) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            _stop_child_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired as exc:
                # A detached tool can retain the pipes even after omp exits.
                stdout, stderr = exc.stdout or b"", exc.stderr or b""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                proc.stdout.close()
                proc.stderr.close()
                proc.wait(timeout=5)
            timed_out = True
        except BaseException:
            _stop_child_group(proc)
            proc.communicate(timeout=5)
            raise
        returncode = proc.returncode
    wall_s = round(time.perf_counter() - started, 3)
    trajectory = run_dir / "trajectory.jsonl"
    trajectory.write_text(stdout, encoding="utf-8")
    final_files: dict[str, str] = {}
    for path in workspace.iterdir():
        content = workspace_text(path)
        if content is not None:
            final_files[path.name] = content
    final_state = run_dir / "final_state.json"
    final_state.write_text(json.dumps(final_files, sort_keys=True) + "\n", encoding="utf-8")
    result = {"returncode": returncode, "timed_out": timed_out, "wall_s": wall_s,
              "stderr": stderr, "command": [command[0], "-p", "<prompt in campaign identity>", *flags]}
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    traces = [initial, trajectory, final_state, result_path]
    if not timed_out:
        traces.extend(workspace / filename for filename in final_files)
    return {**result, "stdout": stdout, "final_files": final_files, "cwd": workspace}, traces
