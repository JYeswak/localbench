"""Offline capture and generation decisions for installed omp updates."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import plistlib
import pty
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping
from pathlib import Path

from . import features, lifecycle

OMP_PACKAGE_JSON = str(Path.home() / ".bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/package.json")
DEFAULT_LABEL = "dev.localbench.omp-update"
DEFAULT_LOCALBENCH = str(Path.home() / ".local/bin/localbench")


CAPTURE_FEATURES = {
    "auto-thinking": "auto-thinking",
    "find-judgments": "find-judgments",
    "memory-extraction": "mnemopi-extraction",
    "titles": "titles",
    "recall-embeddings": "recall-embeddings",
    "unexpected-stop": "unexpected-stop",
}
# These routes are intentionally excluded from request capture until a stable local fixture exists.
CAPTURE_EXCLUSIONS = {
    "skill-description-compression": "features.tsv has no proof suite; run_capture uses --no-skills",
}


def _capture_proofs_from_registry(rows: list[dict] | None = None) -> dict[str, dict]:
    """Take feature identity and proof suites from features.tsv, not a second registry."""
    by_feature = {row["feature"]: row for row in (features.load() if rows is None else rows)}
    missing = sorted(set(CAPTURE_FEATURES.values()) - set(by_feature))
    if missing:
        raise ValueError(f"omp-update capture map references features absent from features.tsv: {', '.join(missing)}")
    return {name: {"feature": feature, "proof_suite": by_feature[feature]["proof_suite"]}
            for name, feature in CAPTURE_FEATURES.items()}


CAPTURE_PROOFS = _capture_proofs_from_registry()
BASELINE_PATH = Path.home() / ".localbench" / "corpora" / "omp-update-baseline.json"


# Volatile fields, normalized before any shape comparison. Each entry is
# (name, pattern, replacement), applied in order to the decoded request text.
# Every pattern below was taken from a real 18.4.8 capture (runs/ and
# /tmp/omp-fields-probe): the find tool-result line embeds wall/api timings,
# per-call counts, byte sizes and a cost figure; usage blocks carry token
# counts; the `<system-reminder>` carries today's date and the cwd. Scores
# (the τ heat, the 1.00 file rank) are DELIBERATELY unnormalized: they are
# judgment signal, and a drift must stale the proof. Fixed prompt and system
# text normalizes stably, so normalization can only hide a wording change
# when the change lands inside a span below; the over-breadth test pins that
# a wording change outside them still goes STALE.
_NORMALIZERS: tuple[tuple[str, str, str], ...] = (
    ("find-report-line",
     r"listed \d+ · judged \d+ · read \d+ files \(\d+B\) · \d+ requests · \d+ tokens · \$\d+\.\d+ · \d+ms wall / \d+ms api",
     "listed N · judged N · read N files (NB) · N requests · N tokens · $N · Nms wall / Nms api"),
    ("find-wall-api", r"\d+ms wall / \d+ms api", "Nms wall / Nms api"),
    ("token-counts", r"\"(prompt_tokens|completion_tokens|total_tokens|cached_tokens|input_tokens|output_tokens|max_tokens|max_completion_tokens)\":\s*\d+",
     r'"\1":0'),
    ("iso-datetime", r"20\d\d-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d+)?)?(?:Z|[+-]\d\d:?\d\d)?",
     "<timestamp>"),
    ("iso-date", r"\b20\d\d-\d\d-\d\d\b", "<date>"),
    ("uuid", r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
     "<uuid>"),
    ("long-hex", r"\b[0-9a-fA-F]{16,}\b", "<hex>"),
    ("scratch-tmp", r"/(private/)?tmp/localbench-ompupdate-\d+/", "<scratch>/"),
    ("update-workdir", r"\.localbench/omp-update-work", "<update-work>"),
)
_NORMALIZERS_COMPILED: tuple[tuple[str, "re.Pattern[str]", str], ...] = tuple(
    (name, re.compile(pattern), replacement) for name, pattern, replacement in _NORMALIZERS)


def normalize_body(body: bytes) -> bytes:
    """Replace every volatile span with its placeholder. Pure and order-fixed."""
    text = body.decode("utf-8")
    for _, regex, replacement in _NORMALIZERS_COMPILED:
        text = regex.sub(replacement, text)
    return text.encode("utf-8")


def normalize_requests(bodies: list[bytes]) -> list[bytes]:
    """Normalize every body of one capture with the volatile-field patterns."""
    return [normalize_body(body) for body in bodies]

def _normalized_value(value: bytes | list[bytes]) -> bytes | list[bytes]:
    """Normalize one stored capture, preserving its bytes/list shape."""
    if isinstance(value, bytes):
        return normalize_body(value)
    return [normalize_body(body) for body in value]


def _first_span_diff(first: bytes, second: bytes, width: int = 40) -> str:
    """First differing offset with decoded context on both sides, one line."""
    offset = next((index for index, (left, right) in enumerate(zip(first, second)) if left != right),
                  min(len(first), len(second)))
    start = max(0, offset - width)
    left = first[start:offset + width].decode("utf-8", "replace").replace("\n", "\\n")
    right = second[start:offset + width].decode("utf-8", "replace").replace("\n", "\\n")
    return f"normalized shape differs at body offset {offset}: {left!r} vs {right!r}"[:240]

def capture_proofs() -> dict[str, dict]:
    """Copy the feature-to-proof mapping for this bounded capture set."""
    return {name: dict(proof) for name, proof in CAPTURE_PROOFS.items()}

class CaptureError(RuntimeError):
    """A capture could not safely complete against the local mock server."""


def _loopback_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("mock endpoint must use HTTP loopback")
    if not parsed.port:
        raise ValueError("mock endpoint must include its loopback port")
    return value.rstrip("/")


def _safe_write(path: Path, content: str) -> None:
    if path.is_symlink():
        raise CaptureError(f"refusing symlink in isolated omp agent dir: {path}")
    path.write_text(content, encoding="utf-8")


def _isolated_configuration(agent_dir: Path, mock_url: str) -> None:
    base = _loopback_url(mock_url)
    if agent_dir.exists() and (agent_dir.is_symlink() or not agent_dir.is_dir()):
        raise CaptureError(f"isolated omp agent path is not a real directory: {agent_dir}")
    if agent_dir.exists() and any(agent_dir.iterdir()):
        raise CaptureError(f"isolated omp agent path must be empty: {agent_dir}")
    agent_dir.mkdir(parents=True, exist_ok=True)
    roles = """providers:
  localbench:
    baseUrl: {base}/v1
    api: openai-completions
    apiKey: localbench-no-network
    models:
      - id: main
        name: Main mock
        contextWindow: 32768
        reasoning: true
        thinking:
          mode: effort
          efforts: [low, medium, high]
          defaultLevel: low
        compat:
          supportsReasoningEffort: true
      - id: tiny
        name: Tiny mock
        contextWindow: 32768
      - id: smol
        name: Smol mock
        contextWindow: 32768
  localbench-sys1:
    baseUrl: {base}
    api: typesafe
    apiKey: localbench-no-network
    models:
      - id: judge
        name: System One judge mock
        contextWindow: 32768
""".format(base=base)
    # An isolated HOME is a first-run state, so interactive omp opens the login
    # wizard (pi-tui ALL_SCENES, providers first) and waits for keys instead of
    # submitting the prompt. Both suppressions below are documented
    # non-interactive gates (wizard.ts selectSetupScenes): the setting and
    # OMP_SKIP_SETUP. Neither shapes the title request.
    config = """modelRoles:
  main: localbench/main
  judge: localbench-sys1/judge
  tiny: localbench/tiny
  smol: localbench/smol
  memory: localbench/smol
defaultThinkingLevel: auto
startup:
  setupWizard: false
  # checkUpdate runs getLatestRelease (5 s network timeout) on startup
  # (main.ts checkForNewVersion). It shapes no request; off here.
  checkUpdate: false
marketplace:
  # autoUpdate pulls the plugin manager at session build. Network on the
  # capture path, shapes no request; off here.
  autoUpdate: "off"
find:
  enabled: auto
memory:
  backend: mnemopi
mnemopi:
  llmMode: smol
  retainEveryNTurns: 4
  autoRetain: true
  # noEmbeddings: recall by FTS only. Otherwise each child spawns an
  # embedding worker that on-demand-installs fastembed over the network and
  # hangs the turn when the install stalls (observed: launchd run stuck with
  # 0 CPU in the install). Recall makes no LLM call either way, so no
  # captured shape changes; same key as the mem-fts tier variant.
  noEmbeddings: true
"""
    _safe_write(agent_dir / "models.yml", roles)
    _safe_write(agent_dir / "config.yml", config)


def _isolated_environment(agent_dir: Path, mock_url: str, parent_env: Mapping[str, str]) -> dict[str, str]:
    """Build an allowlisted environment for the isolated child without mutating process state."""
    home = agent_dir / ".home"
    env = {key: parent_env[key] for key in ("PATH", "TMPDIR", "LANG", "LC_ALL", "TERM") if key in parent_env}
    env.update({"HOME": str(home), "PI_CODING_AGENT_DIR": str(agent_dir), "OMPUPDATE_MOCK_URL": mock_url,
                "TYPESAFE_BASE_URL": mock_url, "TYPESAFE_API_KEY": "localbench-no-network",
                "OMP_SKIP_SETUP": "1"})
    return env


class _IsolatedRpc:
    """Drive an RPC child with an explicit argv, cwd, and isolated environment."""

    def __init__(self, argv: list[str], cwd: Path, stderr_path: Path, env: Mapping[str, str]):
        self._stderr = stderr_path.open("w", encoding="utf-8")
        self.proc = lifecycle.spawn(argv, cwd=cwd, env=dict(env), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._stderr, text=True, bufsize=1)
        self._lines: queue.Queue[str | None] = queue.Queue()
        try:
            threading.Thread(target=self._pump, daemon=True).start()
        except BaseException:
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
                self.proc.wait()
            finally:
                self.proc.stdin.close()
                self.proc.stdout.close()
                self._stderr.close()
            raise

    def _pump(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        finally:
            self.proc.stdout.close()  # EOF follows the child's exit or kill; unclosed, the GC warned into stderr
            self._lines.put(None)

    def send(self, command: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(command) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise EOFError(f"omp closed its stdin (rc={self.proc.poll()})") from None

    def until(self, accept: Callable[[dict], bool], timeout: float) -> list[str]:
        deadline, lines = time.monotonic() + timeout, []
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"no matching event within {timeout:.0f} s") from None
            if line is None:
                raise EOFError(f"omp exited rc={self.proc.wait()}")
            lines.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and accept(event):
                return lines

    def close(self, timeout: float) -> tuple[float, int]:
        started = time.perf_counter()
        try:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                return_code = self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                return_code = self.proc.wait()
        except BaseException:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            raise
        finally:
            self._stderr.close()
        return round(time.perf_counter() - started, 3), return_code


_TITLE_PROMPT = "Summarize the localbench feature route in one short sentence and answer OK."
_TITLE_MODELS = ("tiny", "commit", "smol")
# Timeouts sized for a cold start on a loaded machine (measured 2026-10-01
# under the launchd plist env: `omp --version` 16.0 s cold, then 4.5/3.6/2.8 s
# warm; backends._first_line allows only 20 s, so refresh reads --version here).
# A cold omp start alone exceeds 20 s; it is not TMPDIR.
_VERSION_TIMEOUT = 120.0
_RPC_READY_TIMEOUT = 300.0
_TITLE_TIMEOUT = 180.0
_TITLE_QUIT_GRACE = 10.0
_SETTLE_QUIET_SECS = 60.0
_SETTLE_DEADLINE_SECS = 600.0


def _is_title_request(request: object) -> bool:
    """A title-generation chat call: a tiny-role model asked for `<title>` markers."""
    if not isinstance(request, dict) or str(request.get("model", "")) not in _TITLE_MODELS:
        return False
    envelope = json.dumps(request.get("messages", []), ensure_ascii=False)
    return "<title>" in envelope and "<user>" in envelope


def _title_request_bodies(records: list[tuple[str, bytes]]) -> list[bytes]:
    """Every title-generation request body in a record slice, byte-exact."""
    bodies = []
    for path, body in records:
        if path not in ("/v1/chat/completions", "/chat/completions"):
            continue
        try:
            request = json.loads(body)
        except (UnicodeDecodeError, ValueError):
            continue
        if _is_title_request(request):
            bodies.append(body)
    return bodies


class _InteractiveTitle:
    """One interactive omp child on a pty: the only mode that generates titles.

    `--mode rpc` forces PI_NO_TITLE=1 and `-p` never calls
    maybeStartTitleGeneration, so the title shape is capturable only here.
    Output is drained and discarded; quitting is a bounded ladder
    (/quit, EOF, SIGTERM, SIGKILL) that always reaps the child."""

    def __init__(self, argv: list[str], cwd: Path, stderr_path: Path, env: Mapping[str, str]):
        self._stderr = stderr_path.open("w", encoding="utf-8")
        self._master, slave = pty.openpty()
        self._master_open = True
        try:
            self.proc = lifecycle.spawn(argv, cwd=str(cwd), env=dict(env), stdin=slave, stdout=slave,
                                         stderr=self._stderr, close_fds=True, start_new_session=True)
        finally:
            os.close(slave)
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        try:
            while os.read(self._master, 65536):
                pass
        except OSError:
            return

    def _wait(self, timeout: float) -> int | None:
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def _close(self) -> None:
        if self._master_open:
            with contextlib.suppress(OSError):
                os.close(self._master)
            self._master_open = False
        with contextlib.suppress(OSError, ValueError):
            self._stderr.close()

    def quit(self, grace: float = _TITLE_QUIT_GRACE) -> int:
        """Quit the child on a bounded ladder; always reap, never hang.

        The pty master is closed only after the child is reaped: closing it
        while a TUI child still holds the slave can block (observed: every
        real-omp title leg hung in teardown until the watchdog). The final
        wait is bounded too; a child that outlives SIGKILL is reported, not
        waited on."""
        if self._master_open:
            with contextlib.suppress(OSError):
                os.write(self._master, b"/quit\n")
        code = self._wait(grace)
        if code is None:
            self.proc.terminate()
            code = self._wait(grace)
        if code is None:
            self.proc.kill()
            code = self._wait(grace)
        self._close()
        if code is None:
            raise CaptureError("interactive omp child outlived SIGKILL; left running")
        return code


def _checked_omp(omp: str) -> Path:
    """Resolve the omp under test to an executable file, or refuse with a typed failure."""
    binary = Path(omp).expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise CaptureError(f"omp executable is missing or not executable: {binary}")
    return binary


def _capture_titles(agent: Path, work: Path, base: str, first_request: int,
                    parent_env: Mapping[str, str], timeout: float = _TITLE_TIMEOUT) -> list[bytes]:
    """Run one interactive omp turn on a pty and capture its background title request."""
    from .workloads import omp_bin

    omp = omp_bin()
    _checked_omp(omp)
    env = _isolated_environment(agent, base, parent_env)
    env["PWD"] = str(work)
    env.setdefault("TERM", "xterm-256color")
    argv = [omp, _TITLE_PROMPT, "--model", "localbench/main", "--smol", "localbench/smol",
            "--thinking=auto", "--config", str(agent / "config.yml"), "--no-extensions", "--no-skills",
            "--no-rules", "--no-lsp", "--no-tools"]
    child = _InteractiveTitle(argv, work, agent / "omp-title.stderr.log", env)
    try:
        deadline = time.monotonic() + timeout
        while True:
            if _title_request_bodies(_current_server_records(base)[first_request:]):
                break
            if child.proc.poll() is not None:
                break
            if time.monotonic() >= deadline:
                raise CaptureError(f"interactive title leg saw no title request within {timeout:.0f} s")
            time.sleep(0.2)
    finally:
        child.quit()
    titles = _title_request_bodies(_current_server_records(base)[first_request:])
    if not titles:
        raise CaptureError("interactive omp turn produced no title request")
    return titles


def run_capture(mock_url: str, agent_dir: Path, cwd: Path, *,
                timeout: float = _RPC_READY_TIMEOUT, extraction_timeout: float = 30,
                title_timeout: float = _TITLE_TIMEOUT,
                parent_env: Mapping[str, str] | None = None) -> dict[str, list[bytes]]:
    """Capture feature routes: a four-turn RPC session plus one interactive title turn.

    The RPC legs capture auto-thinking, find-judgments and memory-extraction.
    Titles need the interactive path (`--mode rpc` forces PI_NO_TITLE=1 and `-p`
    never calls maybeStartTitleGeneration), so a second leg runs one prompt on
    a pty and captures the background title request to the tiny role.
    """
    from . import mockomp
    from .workloads import omp_bin

    source_env = os.environ if parent_env is None else parent_env

    base = _loopback_url(mock_url)
    server = mockomp.server_for_url(base)
    if server is None:
        raise CaptureError("mock URL is not owned by a live localbench MockOmpServer")
    omp = omp_bin()
    _checked_omp(omp)
    work = cwd.expanduser().resolve()
    if not work.is_dir():
        raise CaptureError(f"omp capture cwd is not a directory: {work}")
    agent = agent_dir.expanduser().absolute()
    _isolated_configuration(agent, base)
    env = _isolated_environment(agent, base, source_env)
    env["PWD"] = str(work)
    argv = [omp, "--mode=rpc", "--model", "localbench/main", "--smol", "localbench/smol",
            "--thinking=auto", "--config", str(agent / "config.yml"), "--no-extensions", "--no-skills",
            "--no-rules", "--no-lsp", "--tools=find"]
    prompts = (
        "Use find to locate the feature route for the localbench mock provider. Think carefully, then answer OK.",
        "Remember this durable fact: the fixture marker is cobalt. Answer OK.",
        "Inspect the localbench feature route and answer OK.",
        "Remember this durable fact: the retention interval is four turns. Answer OK.",
    )
    first_request = len(server.requests)
    rpc = None
    try:
        rpc = _IsolatedRpc(argv, work, agent / "omp-rpc.stderr.log", env)
        rpc.until(lambda event: event.get("type") == "ready", timeout)
        for turn, prompt in enumerate(prompts, 1):
            rpc.send({"id": str(turn), "type": "prompt", "message": prompt})
            rpc.until(lambda event: event.get("type") == "agent_end", timeout)
            if turn < 4 and _smol_request_count(base, first_request):
                raise CaptureError("mnemopi extraction fired before the configured four-turn interval")
        deadline = time.monotonic() + extraction_timeout
        while not _smol_request_count(base, first_request) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not _smol_request_count(base, first_request):
            raise CaptureError("four-turn RPC session produced no smol memory-extraction request")
    except CaptureError:
        raise
    except (EOFError, OSError, TimeoutError) as exc:
        raise CaptureError(f"isolated OMP RPC capture failed: {exc}") from exc
    finally:
        if rpc is not None:
            _, return_code = rpc.close(15)
            if return_code:
                raise CaptureError(f"OMP RPC capture exited {return_code}")

    records = _current_server_records(base)[first_request:]
    parsed = [(path, body, json.loads(body)) for path, body in records]
    auto_index = next((index for index, (path, _, body) in enumerate(parsed)
                       if path == "/v1/systemone" and "level" in body.get("questions", {})), None)
    if auto_index is None:
        raise CaptureError("mock judge received no auto-thinking classification request")
    expected_effort = server.systemone_choice
    auto_main = next((body for path, body, request in parsed[auto_index + 1:]
                      if path in ("/v1/chat/completions", "/chat/completions")
                      and request.get("model") == "main"
                      and (request.get("reasoning_effort") == expected_effort
                           or (isinstance(request.get("reasoning"), dict)
                               and request["reasoning"].get("effort") == expected_effort))), None)
    if auto_main is None:
        raise CaptureError(f"mock judge choice {expected_effort!r} was not applied to a subsequent main request")
    auto_judge = parsed[auto_index][1]
    find_rows: list[bytes] = []
    found_rank = False
    expected_rank = f"{server.systemone_noul_score:.2f}"
    for path, body, request in parsed:
        if path == "/v1/systemone" and "level" not in request.get("questions", {}):
            find_rows.append(body)
        elif path in ("/v1/chat/completions", "/chat/completions") and request.get("model") == "main":
            tools = request.get("tools") or []
            has_find = any((tool.get("function") or {}).get("name") == "find" for tool in tools)
            tool_messages = [message for message in request.get("messages", [])
                             if message.get("role") == "tool"]
            if has_find or tool_messages:
                find_rows.append(body)
            for message in tool_messages:
                if expected_rank in json.dumps(message, ensure_ascii=False) and "hit(s)" in json.dumps(
                        message, ensure_ascii=False):
                    found_rank = True
    if not find_rows or not found_rank:
        raise CaptureError("find call did not return the configured mock ranking in its tool result")
    memory = [body for path, body, request in parsed
              if path in ("/v1/chat/completions", "/chat/completions")
              and str(request.get("model", "")).endswith("smol")]
    if not memory:
        raise CaptureError("four-turn RPC session did not capture its mnemopi extraction request")
    titles = _capture_titles(agent, work, base, len(server.requests), source_env,
                             title_timeout)
    return {
        "auto-thinking": [auto_judge, auto_main],
        "find-judgments": find_rows,
        "memory-extraction": memory,
        "titles": titles,
    }


def _smol_request_count(mock_url: str, first_request: int) -> int:
    return sum(
        1 for path, body in _current_server_records(mock_url)[first_request:]
        if path in ("/v1/chat/completions", "/chat/completions")
        and str(json.loads(body).get("model", "")).endswith("smol")
    )


def _current_server_records(mock_url: str) -> list[tuple[str, bytes]]:
    from . import mockomp

    server = mockomp.server_for_url(mock_url)
    if server is None:
        raise CaptureError("mock URL is not owned by a live localbench MockOmpServer")
    return [(request.path, request.body) for request in server.requests]


def feature_module_shas(rows: list[dict] | None = None, coding_agent: Path | None = None) -> dict[str, str | None]:
    """Read registered feature module hashes using localbench.features package resolution."""
    rows = features.load() if rows is None else rows
    if coding_agent is None:
        from . import park
        from .workloads import omp_bin
        coding_agent = park.omp_package(omp_bin())
    coding_agent = Path(coding_agent)
    return {row["feature"]: features.module_sha(
        features.package_root(row["omp_package"], coding_agent) / row["omp_module"])
        for row in rows}


def _request_bytes(value: bytes | list[bytes]) -> bytes:
    if isinstance(value, bytes):
        return value
    return b"".join(len(body).to_bytes(8, "big") + body for body in value)


def compare_captures(baseline: Mapping[str, bytes | list[bytes]], current: Mapping[str, bytes | list[bytes]],
                     proofs: Mapping[str, dict], old_module_shas: Mapping[str, str | None],
                     new_module_shas: Mapping[str, str | None]) -> dict[str, dict]:
    """Carry only normalized-identical request shapes with unchanged module hashes; queue the rest."""
    result = {}
    for capture_name, proof in proofs.items():
        feature = proof.get("feature", capture_name)
        suite = proof.get("proof_suite", "-")
        same_shape = (capture_name in baseline and capture_name in current
                      and _normalized_value(baseline[capture_name]) == _normalized_value(current[capture_name]))
        old_sha, new_sha = old_module_shas.get(feature), new_module_shas.get(feature)
        same_module = (feature in old_module_shas and feature in new_module_shas and old_sha is not None
                       and old_sha == new_sha)
        carried = same_shape and same_module
        digest = hashlib.sha256(_request_bytes(current[capture_name])).hexdigest() if capture_name in current else None
        queue = [] if carried else ([suite] if suite != "-" else []) + ["through-omp"]
        result[capture_name] = {
            "feature": feature,
            "status": "CARRIED" if carried else "STALE",
            "label": "carried forward (identical request)" if carried else "STALE",
            "request_sha256": digest,
            "old_module_sha": old_sha,
            "new_module_sha": new_sha,
            "queue": queue,
        }
    return result


def write_baseline(path: Path, captures: dict[str, list[bytes]], module_shas: dict[str, str | None],
                   omp_version: str) -> None:
    """Atomically persist exact request bytes, their normalized mirror, and module SHAs.

    The diff compares normalized shapes; the raw bytes stay for audit and for
    re-deriving the mirror after the pattern list changes."""
    def encode(bodies: list[bytes]) -> list[str]:
        return [base64.b64encode(body).decode("ascii") for body in bodies]
    document = {"format": 1, "omp_version": omp_version, "module_shas": module_shas,
                "requests": {name: encode(bodies) for name, bodies in captures.items()},
                "normalized": {name: encode(normalize_requests(bodies)) for name, bodies in captures.items()},
                "normalizers": [name for name, _, _ in _NORMALIZERS]}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"refusing symlink baseline path: {target}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_baseline(path: Path) -> dict:
    """Load and validate a stored baseline, decoding raw and normalized bytes."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != 1 or not isinstance(document.get("requests"), dict):
        raise ValueError("unsupported omp update baseline")
    decoded = {}
    for name, bodies in document["requests"].items():
        if not isinstance(bodies, list) or not all(isinstance(body, str) for body in bodies):
            raise ValueError(f"invalid request list for {name}")
        decoded[name] = [base64.b64decode(body, validate=True) for body in bodies]
    mirror = {}
    if isinstance(document.get("normalized"), dict):
        for name, bodies in document["normalized"].items():
            if not isinstance(bodies, list) or not all(isinstance(body, str) for body in bodies):
                raise ValueError(f"invalid normalized list for {name}")
            mirror[name] = [base64.b64decode(body, validate=True) for body in bodies]
    return {**document, "requests": decoded, "normalized": mirror}


def render_watch_plist(package_json: str = OMP_PACKAGE_JSON, localbench_bin: str = DEFAULT_LOCALBENCH,
                       label: str = DEFAULT_LABEL, omp_path: str | None = None) -> bytes:
    """Render the separate launchd WatchPaths job without installing or mutating launchd state.

    launchd runs jobs with a minimal PATH, so the job carries its own
    environment: LOCALBENCH_OMP is the absolute omp path resolved at render
    time, PATH covers the install locations, and both outputs append to one
    refresh log for post-update diagnosis."""
    home = str(Path.home())
    binary = (omp_path or os.environ.get("LOCALBENCH_OMP") or shutil.which("omp")
              or os.path.join(home, ".bun", "bin", "omp"))
    log = os.path.join(home, ".localbench", "omp-watch", "refresh.log")
    document = {"Label": label, "ProgramArguments": [localbench_bin, "omp", "refresh"],
                "RunAtLoad": False, "WatchPaths": [package_json], "ProcessType": "Background",
                "EnvironmentVariables": {
                    "LOCALBENCH_OMP": binary,
                    "PATH": ":".join([os.path.join(home, ".bun", "bin"), os.path.join(home, ".local", "bin"),
                                      "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"])},
                "StandardOutPath": log, "StandardErrorPath": log}
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=True)

UPDATE_WORK_DIR = Path.home() / ".localbench" / "omp-update-work"
UPDATE_FIXTURE_NAME = "notes.txt"
UPDATE_FIXTURE_TEXT = "localbench feature route behavior notes\n"
_REFRESH_RUNS = 0


def _stale_reason(name: str, baseline: dict[str, bytes | list[bytes]], current: dict[str, list[bytes]],
                  proofs: dict[str, dict], old_shas: dict[str, str | None],
                  new_shas: dict[str, str | None]) -> str:
    """Why a captured feature did not carry forward: normalized shape, module sha, or both."""
    if name not in current:
        return "not captured this run"
    reasons = []
    if name in baseline:
        old_norm, new_norm = _normalized_value(baseline[name]), _normalized_value(current[name])
        old_list = [old_norm] if isinstance(old_norm, bytes) else old_norm
        new_list = [new_norm] if isinstance(new_norm, bytes) else new_norm
        for index, (left, right) in enumerate(zip(old_list, new_list)):
            if left != right:
                reasons.append(f"body {index}: {_first_span_diff(left, right)}")
                break
        if len(old_list) != len(new_list):
            reasons.append(f"body count {len(old_list)} -> {len(new_list)}")
    else:
        reasons.append("no baseline entry")
    feature = proofs[name].get("feature", name)
    if old_shas.get(feature) != new_shas.get(feature):
        reasons.append(f"module sha {old_shas.get(feature)} -> {new_shas.get(feature)}")
    return "; ".join(reasons) if reasons else "differs"

def _reproof_queue(proof: Mapping[str, object]) -> list[str]:
    """The suite plus through-omp leg a non-carried proof must re-prove with."""
    suite = proof.get("proof_suite", "-")
    return ([suite] if isinstance(suite, str) and suite != "-" else []) + ["through-omp"]


def _is_transient(exc: BaseException) -> bool:
    """A capture failure worth one retry from scratch: a transport TimeoutError,
    EOFError or OSError, including one wrapped as the cause of a CaptureError
    (run_capture converts rpc.until's TimeoutError/EOFError). Assertion-style
    failures (no smol request, shape mismatch) fail fast instead."""
    if isinstance(exc, (TimeoutError, EOFError)):
        return True
    return isinstance(exc, CaptureError) and isinstance(exc.__cause__, (TimeoutError, EOFError, OSError))


def _settle_package(package_json: str = OMP_PACKAGE_JSON, quiet_secs: float = _SETTLE_QUIET_SECS,
                    deadline_secs: float = _SETTLE_DEADLINE_SECS) -> None:
    """Wait until the installed omp package dir and package.json stop changing.

    WatchPaths fires mid-install (uca rewrote package.json at 08:43:53 while
    the old binary was still being replaced); capturing against a half-written
    install voids the run. Bounded: CaptureError past the deadline."""
    package = Path(package_json)
    start = time.monotonic()
    while True:
        try:
            mtimes = [target.stat().st_mtime for target in (package, package.parent) if target.exists()]
        except OSError as exc:
            raise CaptureError(f"cannot stat installed omp package: {exc}") from exc
        if not mtimes:
            raise CaptureError(f"installed omp package is missing: {package}")
        quiet_for = time.time() - max(mtimes)
        if quiet_for >= quiet_secs:
            return
        if time.monotonic() - start >= deadline_secs:
            raise CaptureError(
                f"installed omp package still changing after {deadline_secs:.0f} s; refusing capture")
        time.sleep(min(5.0, max(0.5, quiet_secs - quiet_for)))


def _omp_version(timeout: float = _VERSION_TIMEOUT) -> str | None:
    """Read `omp --version` of workloads.omp_bin() with a cold-start-sized timeout (backends._first_line allows 20 s).
    argv[0] is bound to omp_bin() so the port map resolves the launch (tests/test_port_map.py EXEC_HELPERS)."""
    from .workloads import omp_bin

    omp = omp_bin()
    try:
        proc = subprocess.run([omp, "--version"], capture_output=True, text=True, timeout=timeout,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureError(f"omp --version failed after {timeout:.0f} s: {exc}") from exc
    return proc.stdout.removeprefix("omp/").strip().splitlines()[0].strip() if proc.stdout.strip() else None

def refresh(*, baseline_path: Path | str = BASELINE_PATH) -> dict:
    """Run the full offline refresh: capture against a self-owned mock, diff, carry or stale.

    Starts and stops the mockomp server itself, runs run_capture (RPC legs plus
    the interactive title leg) in an isolated agent dir under one per-process
    parent, and compares against the stored baseline, writing it on the first
    run. Proofs carry forward only on identical normalized request shape plus
    an identical module sha: volatile spans (find timings/counts, token
    counts, ISO dates, uuids, tmp paths) normalize via _NORMALIZERS first.
    The work cwd is fixed (UPDATE_WORK_DIR) and the baseline keeps raw bytes
    for audit next to the normalized mirror.
    """
    global _REFRESH_RUNS
    from . import mockomp
    from .backends import sha16
    from .workloads import omp_bin

    _settle_package()
    binary = omp_bin()
    version = _omp_version()
    sha = sha16(binary)
    parent = Path(tempfile.gettempdir()) / f"localbench-ompupdate-{os.getpid()}"
    work = UPDATE_WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    fixture = work / UPDATE_FIXTURE_NAME
    if fixture.is_symlink():
        raise CaptureError(f"refusing symlink fixture in refresh work dir: {fixture}")
    fixture.write_text(UPDATE_FIXTURE_TEXT, encoding="utf-8")
    failures: list[str] = []
    with mockomp.MockOmpServer() as server:
        for attempt in (1, 2):
            _REFRESH_RUNS += 1
            agent = parent / f"agent-{_REFRESH_RUNS}"
            try:
                captures = run_capture(server.url, agent, work)
                new_shas = feature_module_shas()
                break
            except Exception as exc:
                if not _is_transient(exc):
                    raise
                failures.append(f"attempt {attempt}: {exc}")
                if attempt == 2:
                    raise CaptureError(f"capture failed twice ({'; '.join(failures)})") from exc
    target = Path(baseline_path)
    proofs = capture_proofs()
    if not target.is_file():
        write_baseline(target, captures, new_shas, version or "unknown")
        return {"omp_version": version, "omp_sha": sha, "outcomes": {
            name: {"status": "NEW", "reason": "baseline written",
                   "feature": proofs[name].get("feature", name),
                   "queue": _reproof_queue(proofs[name])} for name in CAPTURE_PROOFS}}
    stored = read_baseline(target)
    old_requests, old_shas = stored["requests"], stored.get("module_shas", {})
    compared = compare_captures(old_requests, captures, proofs, old_shas, new_shas)
    outcomes = {}
    for name in CAPTURE_PROOFS:
        feature = proofs[name].get("feature", name)
        if name not in old_requests:
            outcomes[name] = {"status": "NEW", "reason": "no baseline entry",
                              "feature": feature, "queue": _reproof_queue(proofs[name])}
        elif name not in captures:
            outcomes[name] = {"status": "STALE", "reason": "not captured this run",
                              "feature": feature, "queue": _reproof_queue(proofs[name])}
        elif compared[name]["status"] == "CARRIED":
            outcomes[name] = {"status": "CARRIED", "reason": "carried forward (identical request)",
                              "feature": feature, "queue": []}
        else:
            outcomes[name] = {"status": "STALE",
                              "reason": _stale_reason(name, old_requests, captures, proofs, old_shas, new_shas),
                              "feature": feature, "queue": compared[name]["queue"]}
    return {"omp_version": version, "omp_sha": sha, "outcomes": outcomes}
