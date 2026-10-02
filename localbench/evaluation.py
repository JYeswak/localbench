"""Identity-bound, append-only evidence for deterministic behavioral evaluation campaigns."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path

SCHEMA_VERSION = "localbench.eval-campaign.v1"
CASE_STATUSES = frozenset({"PASS", "FAIL", "VOID", "ERROR"})
_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class CampaignError(ValueError):
    """A campaign cannot safely be created, resumed, or scored."""


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value independent of dict insertion order."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_text(path: Path) -> str | None:
    """Read only a regular workspace file; never follow its filename or directory symlink."""
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            return None
        raise
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                return None
            raise
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return None
            try:
                return stream.read()
            except UnicodeDecodeError:
                return None
    finally:
        os.close(parent_fd)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _case_file(path: Path, case_id: str) -> Path:
    if not _CASE_ID.fullmatch(case_id):
        raise CampaignError(f"invalid case id {case_id!r}")
    return path / "cases" / f"{case_id}.json"


class EvaluationCampaign:
    """One immutable input/pin identity with durable, independently verifiable case traces."""

    def __init__(self, path: Path, root: Path, manifest: dict):
        self.path = path.resolve()
        self.root = root.resolve()
        self.manifest = manifest
        self.identity_sha256 = manifest["identity_sha256"]
        self._case_inputs = {row["case_id"]: row["input_sha256"] for row in manifest["cases"]}

    @classmethod
    def create(cls, path: Path, *, root: Path, identity: dict, cases: dict[str, str], profile: str
               ) -> EvaluationCampaign:
        if not cases:
            raise CampaignError("a campaign needs at least one case")
        if any(not _CASE_ID.fullmatch(case_id) for case_id in cases):
            raise CampaignError("campaign case ids must be simple file-safe names")
        if path.exists():
            raise CampaignError(f"campaign already exists: {path}")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "profile": profile,
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "cases": [{"case_id": case_id, "input_sha256": digest} for case_id, digest in cases.items()],
        }
        path.mkdir(parents=True)
        try:
            _atomic_json(path / "manifest.json", manifest)
            (path / "cases").mkdir()
            (path / "scores").mkdir()
            return cls(path, root, manifest)
        except BaseException:
            # The manifest is the evidence root; leave an interrupted creation visible rather than erasing it.
            raise

    @classmethod
    def open(cls, path: Path, *, root: Path, expected_identity: dict | None = None) -> EvaluationCampaign:
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"cannot read campaign manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
            raise CampaignError(f"unsupported or invalid campaign manifest: {manifest_path}")
        identity = manifest.get("identity")
        if not isinstance(identity, dict) or canonical_sha256(identity) != manifest.get("identity_sha256"):
            raise CampaignError("campaign identity hash is invalid")
        if expected_identity is not None and identity != expected_identity:
            raise CampaignError("campaign identity mismatch; model/runtime/test pins changed, refusing reuse")
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise CampaignError("campaign manifest has no cases")
        ids = [row.get("case_id") for row in cases if isinstance(row, dict)]
        if len(ids) != len(cases) or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not _CASE_ID.fullmatch(i)
                                                                         for i in ids):
            raise CampaignError("campaign manifest has invalid or duplicate case ids")
        if not (path / "cases").is_dir() or not (path / "scores").is_dir():
            raise CampaignError("campaign artifact directories are missing")
        return cls(path, root, manifest)

    @property
    def case_ids(self) -> list[str]:
        return list(self._case_inputs)

    @property
    def case_inputs(self) -> dict[str, str]:
        """Return the immutable case-to-input identity map."""
        return self._case_inputs.copy()

    def record_case(self, case_id: str, *, input_sha256: str, status: str, run_dir: Path,
                    trace_files: list[Path]) -> dict:
        if case_id not in self._case_inputs:
            raise CampaignError(f"case {case_id!r} is not in this campaign")
        if self._case_inputs[case_id] != input_sha256:
            raise CampaignError(f"case {case_id!r} input identity mismatch")
        if status not in CASE_STATUSES:
            raise CampaignError(f"invalid completed case status {status!r}")
        run_dir = run_dir.resolve()
        try:
            run_rel = run_dir.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise CampaignError("case run directory must be inside the localbench data root") from exc
        files = {}
        for source in trace_files:
            source = source.resolve()
            try:
                relative = source.relative_to(run_dir).as_posix()
            except ValueError as exc:
                raise CampaignError("case trace file must be inside its run directory") from exc
            if not source.is_file():
                raise CampaignError(f"case trace is missing: {source}")
            files[relative] = file_sha256(source)
        if not files:
            raise CampaignError("a completed case must retain at least one trace file")
        row = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "input_sha256": input_sha256,
            "identity_sha256": self.identity_sha256,
            "status": status,
            "run_dir": run_rel,
            "trace_sha256": files,
        }
        path = _case_file(self.path, case_id)
        if path.exists():
            raise CampaignError(f"case {case_id!r} already recorded; completed outcomes are immutable")
        _atomic_json(path, row)
        return row

    def completed_case(self, case_id: str, *, input_sha256: str | None = None) -> dict | None:
        if case_id not in self._case_inputs:
            raise CampaignError(f"case {case_id!r} is not in this campaign")
        path = _case_file(self.path, case_id)
        if not path.exists():
            return None
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"invalid case record {path}: {exc}") from exc
        expected_input = input_sha256 or self._case_inputs[case_id]
        if (not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION
                or row.get("case_id") != case_id or row.get("input_sha256") != expected_input
                or row.get("identity_sha256") != self.identity_sha256):
            raise CampaignError(f"case {case_id!r} identity mismatch; refusing reuse")
        run_dir = (self.root / row.get("run_dir", "")).resolve()
        try:
            run_dir.relative_to(self.root)
        except ValueError as exc:
            raise CampaignError(f"case {case_id!r} run directory escapes the data root") from exc
        files = row.get("trace_sha256")
        if not isinstance(files, dict) or not files:
            raise CampaignError(f"case {case_id!r} has no trace hashes")
        for relative, digest in files.items():
            target = (run_dir / relative).resolve()
            try:
                target.relative_to(run_dir)
            except ValueError as exc:
                raise CampaignError(f"case {case_id!r} trace path escapes its run directory") from exc
            if not target.is_file() or file_sha256(target) != digest:
                raise CampaignError(f"case {case_id!r} trace is missing or changed: {relative}")
        if row.get("status") not in CASE_STATUSES:
            raise CampaignError(f"case {case_id!r} has invalid status")
        return row

    def write_scores(self, scorer_sha256: str, document: dict) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", scorer_sha256):
            raise CampaignError("scorer identity must be a full SHA-256 digest")
        if document.get("identity_sha256") != self.identity_sha256:
            raise CampaignError("score document belongs to a different campaign identity")
        payload = {"schema_version": SCHEMA_VERSION, "scorer_sha256": scorer_sha256, **document}
        path = self.path / "scores" / f"{scorer_sha256}.json"
        if path.exists():
            if path.read_text(encoding="utf-8") != json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                                 indent=2, default=str) + "\n":
                raise CampaignError("existing score artifact differs for the same scorer identity")
            return path
        _atomic_json(path, payload)
        return path


# v2: hashline edits (omp 18.4.5 default) carry their file in `[PATH#TAG]` headers, not args.path.
# A grader change is a new campaign identity; v1 specs are refused rather than rescored in place.
GRADER_VERSION = "varied.v2"
HASHLINE_HEADER = re.compile(r"^\[([^\]\n]+)#[0-9A-Za-z]+\]\s*$", re.MULTILINE)


def _tool_path(cwd: Path, value: str) -> Path | None:
    """Resolve a tool path lexically; traversal is rejected even if it would land on the target."""
    path = Path(value)
    if ".." in path.parts:
        return None
    return Path(os.path.normpath(cwd / path))


def make_varied_spec(*, family: str, seed: int, phase: str) -> dict:
    """Derive independent planted text for one phase/seed/family without ambient state."""
    if family not in ("read", "edit") or phase not in ("exploratory", "heldout"):
        raise ValueError("varied trial needs a read/edit family and exploratory/heldout phase")
    if type(seed) is not int or seed < 0:
        raise ValueError("varied trial seed must be a nonnegative integer")

    def planted(label: str) -> str:
        digest = canonical_sha256(["varied.v1", family, phase, seed, label])
        return f"{label.upper()}-{digest[:24]}\n"

    initial = planted("initial")
    expected = initial if family == "read" else planted("expected")
    prompt = ("Read target.txt in the current directory and reply with only its value."
              if family == "read" else
              f"Replace the entire contents of target.txt with exactly {expected.strip()!r} "
              "(followed by a newline). Do not change decoy.txt.")
    return {"family": family, "seed": seed, "phase": phase, "file": "target.txt",
            "initial": initial, "expected": expected, "decoy_file": "decoy.txt",
            "decoy_initial": planted("decoy"), "prompt": prompt,
            "allowed_tools": ["read"] if family == "read" else ["read", "edit", "write"],
            "grader_version": GRADER_VERSION}


def score_varied_trial(spec: dict, *, stdout: str, returncode: int, timed_out: bool,
                       final_files: dict[str, str], cwd: Path, wall_s: float) -> dict:
    """Score OMP JSONL and externally captured files; never infer a file change from an answer."""
    reasons: list[str] = []
    errors: list[str] = []
    visible = final_files.copy() if isinstance(final_files, dict) else {}

    if not isinstance(spec, dict) or spec.get("grader_version") != GRADER_VERSION:
        errors.append("invalid varied trial spec")
    elif (spec.get("family") not in ("read", "edit")
          or spec.get("phase") not in ("exploratory", "heldout")
          or type(spec.get("seed")) is not int or spec["seed"] < 0
          or spec.get("file") != "target.txt" or spec.get("decoy_file") != "decoy.txt"
          or not isinstance(spec.get("allowed_tools"), list)
          or any(not isinstance(tool, str) for tool in spec["allowed_tools"])
          or any(not isinstance(spec.get(key), str) for key in
                 ("initial", "expected", "decoy_initial", "prompt"))):
        errors.append("malformed or unsafe varied trial spec")
    if not isinstance(final_files, dict) or any(
            not isinstance(visible.get(key), str) for key in ("target.txt", "decoy.txt")):
        errors.append("missing or unreadable final file snapshot")
    if not isinstance(cwd, Path) or not cwd.is_absolute():
        errors.append("invalid trial working directory")
    if not isinstance(stdout, str):
        errors.append("unreadable OMP JSONL")
    if type(wall_s) not in (int, float) or not math.isfinite(wall_s) or wall_s < 0:
        errors.append("invalid trial wall time")
    if type(returncode) is not int:
        errors.append("invalid process exit code")
    if type(timed_out) is not bool:
        errors.append("invalid process timeout flag")
    if errors:
        return {"status": "ERROR", "ok": False, "reasons": errors,
                "wall_s": wall_s, "final_files": visible}

    target = Path(os.path.normpath(cwd / spec["file"]))
    starts: dict[str, tuple[str, Path | None, bool]] = {}
    seen_ids: set[str] = set()
    completed: list[tuple[str, Path | None, dict]] = []
    answer = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append("malformed OMP JSONL event")
            break
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            errors.append("malformed OMP event")
            break
        kind = event["type"]
        if kind == "message_end":
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if not isinstance(content, list):
                    errors.append("malformed assistant message")
                    break
                texts = [part.get("text") for part in content if isinstance(part, dict)
                         and part.get("type") == "text"]
                if any(not isinstance(text, str) for text in texts):
                    errors.append("malformed assistant text")
                    break
                if texts:
                    answer = "".join(texts)
        elif kind == "tool_execution_start":
            call_id, name, args = event.get("toolCallId"), event.get("toolName"), event.get("args")
            if (not isinstance(call_id, str) or not call_id or call_id in seen_ids
                    or not isinstance(name, str) or not isinstance(args, dict)):
                errors.append("malformed or duplicate tool start")
                break
            seen_ids.add(call_id)
            path = args.get("path")
            hashline = name == "edit" and "path" not in args and isinstance(args.get("input"), str)
            if hashline:
                # omp 18.4.5 hashline edits carry the file only in `[PATH#TAG]` section headers.
                headers = set(HASHLINE_HEADER.findall(args["input"]))
                if len(headers) > 1:
                    reasons.append("edit tool call names more than one file")
                path = headers.pop() if len(headers) == 1 else None
            if not isinstance(path, str) or not path:
                reasons.append(f"{name} tool call has no file path")
                file_path = None
            else:
                file_path = _tool_path(cwd, path)
                if file_path is None:
                    reasons.append("tool path traverses outside its named file")
            if name not in spec.get("allowed_tools", ()):
                reasons.append(f"unallowed {name} tool call")
            starts[call_id] = (name, file_path, hashline)
        elif kind == "tool_execution_end":
            call_id = event.get("toolCallId")
            if not isinstance(call_id, str) or not call_id:
                errors.append("malformed tool end")
                break
            started = starts.pop(call_id, None)
            if started is None:
                reasons.append("unmatched tool end")
                continue
            name, file_path, hashline = started
            if (event.get("toolName") != name or type(event.get("isError")) is not bool
                    or "result" not in event):
                errors.append("malformed tool result")
                break
            if event["isError"]:
                reasons.append(f"failed {name} tool call")
            if hashline and file_path is not None:
                # The header alone is model-authored; the edit counts only when omp reports the same file.
                details = event["result"].get("details") if isinstance(event["result"], dict) else None
                reported = details.get("path") if isinstance(details, dict) else None
                if not isinstance(reported, str) or not reported or _tool_path(cwd, reported) != file_path:
                    reasons.append("edit result path is missing or disagrees with its call")
                    file_path = None
            completed.append((name, file_path, event))
    if starts:
        reasons.append("unfinished tool call")
    if timed_out:
        reasons.append("process timed out")
    if returncode != 0:
        reasons.append("process did not exit successfully")
    if visible.get("decoy.txt") != spec["decoy_initial"]:
        reasons.append("decoy changed")

    if spec["family"] == "read":
        if visible.get("target.txt") != spec["initial"]:
            reasons.append("read target differs from planted text")
        if answer != spec["expected"].removesuffix("\n"):
            reasons.append("answer does not match planted value")
        read_ok = False
        for name, path, end in completed:
            if name != "read" or path != target or end["isError"]:
                continue
            result = end["result"]
            details = result.get("details") if isinstance(result, dict) else None
            display = details.get("displayContent") if isinstance(details, dict) else None
            if not isinstance(display, dict) or not isinstance(display.get("text"), str):
                errors.append("unreadable read tool result")
                continue
            text = display.get("text") if isinstance(display, dict) else None
            if text == spec["initial"].removesuffix("\n"):
                read_ok = True
        if not read_ok:
            reasons.append("no matching successful read of planted text")
    else:
        if spec["initial"] == spec["expected"]:
            reasons.append("edit prestate already matches target")
        if visible.get("target.txt") != spec["expected"]:
            reasons.append("target was not edited to expected text")
        if not any(name in ("edit", "write") and path == target and not end["isError"]
                   for name, path, end in completed):
            reasons.append("no matching successful edit/write tool call")
    status = "ERROR" if errors else "FAIL" if reasons else "PASS"
    return {"status": status, "ok": status == "PASS", "reasons": errors + reasons,
            "wall_s": wall_s, "final_files": visible}
