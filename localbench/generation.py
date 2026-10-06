"""Generation-proof corpora: real captured chat requests per feature, classified by the
system-prompt signature omp's call site sends.

Unlike decision suites (labelled questions with hosted answers), a generation item is a
request body captured byte-exact from live traffic plus the feature it belongs to. Replay
sends the same messages with the model field set per arm (local qwen3.8 vs the cloud
incumbent); a blind pairwise judge scores the two answers. Items live under
~/.localbench/corpora (never git), content-addressed by the request sha256.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import corpus, sysstats
from .heavyslot import held as _held

GENERATION_DIR = "generation"
MIN_ITEMS = 100


def _blob(body: dict) -> str:
    return json.dumps(body, ensure_ascii=False)


def _is_title_request(body: dict) -> bool:
    """A title-generation call: the tiny-role envelope asks for `<title>` markers."""
    text = _blob(body.get("messages", body))
    return "<title>" in text and "<user>" in text


def _is_skill_description_request(body: dict) -> bool:
    """A skill-description compression call: the single user turn carries the routing-hint prompt."""
    return "Compress into one routing hint of at most 12 words" in _blob(body)


def _is_commit_request(body: dict) -> bool:
    """An `omp commit` conventional-pipeline call: one of its family system prompts."""
    text = _blob(body)
    return ("Senior engineer writing a conventional commit message" in text
            or "You are a commit message specialist" in text)


FEATURE_SIGNATURES: tuple[tuple[str, Callable[[dict], bool]], ...] = (
    ("titles", _is_title_request),
    ("skill-description-compression", _is_skill_description_request),
    ("commit-messages", _is_commit_request),
)


def classify(body: dict) -> str | None:
    """The feature a captured request body belongs to, or None when no signature matches.
    Bodies carry resolved model names (not role aliases), so signatures are marker-only."""
    if not isinstance(body, dict):
        return None
    for feature, matches in FEATURE_SIGNATURES:
        if matches(body):
            return feature
    return None


def collect(capture_root: Path | str | None = None) -> dict[str, list[dict]]:
    """Captured aux request documents grouped by feature: {feature: [{body, source, meta}]}.
    Response documents (`*.response.json`) and unparseable files are skipped; `other`
    holds captures no signature claims, so a new omp shape shows up instead of vanishing."""
    root = Path(capture_root).expanduser() if capture_root is not None else corpus.CORPORA / "captured" / "aux"
    out: dict[str, list[dict]] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".response.json"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError):
            continue
        meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
        body = doc.get("request", doc) if isinstance(doc, dict) else None
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                continue
        if not isinstance(body, dict):
            continue
        out.setdefault(classify(body) or "other", []).append(
            {"body": body, "source": str(path),
             "meta": meta if isinstance(meta, dict) else {}})
    return out


def gateway_url(profile: str, body: dict) -> str:
    """The managed loopback route replaying a captured body must take: the gateway
    profile path for the body's shape (responses vs chat completions). Replay goes
    through the gateway, never direct to Ollama, so leases record it."""
    if not isinstance(profile, str) or not profile or "/" in profile or profile in {".", ".."}:
        raise corpus.CorpusError(f"replay profile {profile!r} is not a plain profile name")
    endpoint = "responses" if isinstance(body.get("input"), list) else "chat/completions"
    return (f"http://127.0.0.1:{sysstats.OLLAMA_GATEWAY_PORT}"
            f"/omp-profile/{profile}/{endpoint}")


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        os.fchmod(fh.fileno(), 0o600)
        fh.write(text)


def assemble(feature: str, records: list[dict], directory: Path | str, *, seed: int,
             version: int = 1) -> dict:
    """Build a write-once generation corpus with a manifest pinning its contents."""
    if feature not in {name for name, _ in FEATURE_SIGNATURES}:
        raise corpus.CorpusError(f"generation feature {feature!r} has no request signature")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise corpus.CorpusError(f"corpus seed {seed!r} must be an int")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise corpus.CorpusError(f"corpus version {version!r} must be an int starting at 1")
    d = Path(directory).expanduser()
    corpus.refuse_repo_path(d)
    corpus.secure_dir(d.parent)
    try:
        d.mkdir(mode=0o700)
    except FileExistsError:
        raise corpus.CorpusError(
            f"generation corpus directory {d} already exists; choose a new corpus id") from None
    corpus.refuse_repo_path(d.resolve())
    items, skipped = [], 0
    for record in records:
        body = record["body"]
        if not answerable(feature, body):
            skipped += 1
            continue
        digest = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, separators=(",", ":"),
                       sort_keys=True).encode("utf-8")).hexdigest()
        items.append({"id": digest, "feature": feature, "body": body, "source": record["source"],
                      "profile": record.get("meta", {}).get("profile") if isinstance(
                          record.get("meta"), dict) else None})
    items_path = d / "items.jsonl"
    _write_private(items_path, "".join(
        json.dumps(it, ensure_ascii=False, separators=(",", ":")) + "\n" for it in items))
    manifest = {"name": d.name, "kind": "generation", "feature": feature,
                "items": items_path.name,
                "items_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
                "sources": sorted({it["source"] for it in items}),
                "gate": {"min_items": MIN_ITEMS}, "n_items": len(items),
                "seed": seed, "version": version}
    manifest_path = d / "manifest.json"
    _write_private(manifest_path, json.dumps(manifest, indent=1) + "\n")
    items_path.chmod(0o400)
    manifest_path.chmod(0o400)
    d.chmod(0o500)
    return {"feature": feature, "directory": str(d), "items": len(items),
            "items_sha256": manifest["items_sha256"], "gate": manifest["gate"],
            "seed": seed, "version": version, "skipped_unanswerable": skipped}


def corpus_progress(capture_root: Path | str | None = None,
                    generation_root: Path | str | None = None) -> list[dict]:
    """Weekly drip readout per generation feature: built corpus items vs the MIN_ITEMS
    gate, captured pool size, oldest/newest capture dates. Newest corpus wins per
    feature (later builds superset earlier ones). Dates are capture times (meta t,
    else the source file mtime), UTC YYYY-MM-DD; None when the pool is empty."""
    grouped = collect(capture_root)
    gen = Path(generation_root).expanduser() if generation_root is not None \
        else corpus.CORPORA / "generation"
    built: dict[str, int] = {}
    if gen.is_dir():
        for manifest_path in sorted(gen.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError, UnicodeDecodeError):
                continue
            if manifest.get("kind") != "generation" or not manifest.get("feature"):
                continue
            n = manifest.get("n_items")
            if not isinstance(n, int):
                try:
                    n = sum(1 for line in (manifest_path.parent / "items.jsonl").read_text(
                        encoding="utf-8").splitlines() if line.strip())
                except OSError:
                    continue
            built[manifest["feature"]] = n
    rows = []
    for feature, _ in FEATURE_SIGNATURES:
        stamps = []
        for record in grouped.get(feature, []):
            meta = record.get("meta") or {}
            stamp = meta.get("t") if isinstance(meta.get("t"), (int, float)) else None
            if stamp is None:
                try:
                    stamp = Path(record["source"]).stat().st_mtime
                except OSError:
                    continue
            stamps.append(stamp)
        dates = sorted(time.gmtime(s)[:3] for s in stamps)
        def fmt(d):
            return f"{d[0]:04d}-{d[1]:02d}-{d[2]:02d}"
        rows.append({"feature": feature, "corpus": built.get(feature, 0),
                     "gate": MIN_ITEMS, "ready": built.get(feature, 0) >= MIN_ITEMS,
                     "captured": len(stamps),
                     "oldest": fmt(dates[0]) if dates else None,
                     "newest": fmt(dates[-1]) if dates else None})
    return rows


def corpus_progress_lines(rows: list[dict] | None = None, **kwargs) -> list[str]:
    """Text form of corpus_progress, one line per generation feature, for the
    features command: built items vs the gate, captured pool, capture date span."""
    out = []
    for row in rows if rows is not None else corpus_progress(**kwargs):
        span = (f", {row['oldest']}..{row['newest']}" if row["oldest"] else "")
        out.append(f"corpus {row['feature']}: {row['corpus']}/{row['gate']} "
                   f"(captured {row['captured']}{span})"
                   f"{' READY' if row['ready'] else ''}")
    return out


# --- builtin arms: what omp produces with no model role set (deterministic, no model).
# Pure ports of the 18.4.10 call sites (verified against bun evaluation during
# development); the runner reproduces them byte-faithfully without a model call.

class BuiltinUnavailable(Exception):
    """The feature has no builtin fallback: without a model the call errors."""


_CONTROL_CHARS = re.compile("[" + "".join(map(chr, range(0x00, 0x20))) + "".join(map(chr, range(0x7F, 0xA0))) + "]")


def _sanitize_terminal_part(value: str | None) -> str | None:
    """title-generator.ts sanitizeTerminalTitlePart: strip control chars, trim."""
    if not value:
        return None
    return _CONTROL_CHARS.sub("", value).strip() or None


def builtin_title(session_name: str | None, cwd: str | None) -> str | None:
    """The no-role title fallback (title-generator.ts getFallbackTerminalTitle): the
    sanitized session name, else the resolved cwd's basename (None at root/empty:
    the session stays unnamed). Returns None only when omp itself skips."""
    named = _sanitize_terminal_part(session_name)
    if named is not None:
        return named
    if not cwd:
        return None
    base = os.path.basename(os.path.normpath(cwd)) or None
    return _sanitize_terminal_part(base)


def builtin_skill_description(description: str) -> str:
    """The no-compressor fallback (skill-descriptions.ts previewSkillDescription):
    whitespace-collapsed text as-is within 100 chars, else a sentence (>= 40 chars)
    or word boundary cut with an ellipsis."""
    text = " ".join(description.split())
    if len(text) <= 100:
        return text
    boundary = text[:99]
    sentence = re.match(r"^.*?[.!?](?=\s|$)", boundary)
    if sentence and len(sentence.group(0)) >= 40:
        return sentence.group(0)
    word = boundary[:boundary.rfind(" ")].rstrip()
    return f"{word or boundary}\u2026"


def builtin_commit() -> str:
    """Commit generation without a model role: model-selection.ts throws and the
    pipeline rethrows (only VcsError/no-staged-changes are handled). No message."""
    raise BuiltinUnavailable("No model available for commit generation")


# --- deterministic assertions: per-feature shape checks returning violation lists.
# Empty means pass. Thresholds are the pre-registered proof rules.

CONVENTIONAL_TYPES = ("feat", "fix", "chore", "docs", "refactor", "test", "style",
                      "perf", "build", "ci")

_STOPWORDS = frozenset(
    "a an the and or but of to in on for with as at by from is are was were be been "
    "it its this that these those i you he she we they me him her us them my your his "
    "our their mine yours hers ours theirs not no yes do does did will would can could "
    "should have has had having s t re ll ve m d ll".split())


def check_title(text: str) -> list[str]:
    """Exactly one <title> block as the whole output, 3-8 inner words on one line, no
    surrounding quotes and no title: prefix. Interior punctuation is prose, not a tell."""
    found = []
    if text.count("<title>") != 1 or text.count("</title>") != 1:
        return ["want exactly one <title>..</title> block"]
    inner = text.split("<title>", 1)[1].split("</title>", 1)[0]
    if text.split("<title>", 1)[0].strip() or text.split("</title>", 1)[1].strip():
        found.append("title block must be the whole output")
    if "\n" in inner or "\r" in inner:
        found.append("title must be a single line")
    words = inner.split()
    if not 3 <= len(words) <= 8:
        found.append(f"title is {len(words)} words, want 3-8")
    lowered = inner.strip().lower()
    if inner.strip()[:1] in "\"'`" or inner.strip()[-1:] in "\"'`":
        found.append("title must not be quoted")
    if lowered.startswith(("title:", "title ")):
        found.append("title must not carry a prefix")
    return found


_COMMIT_SUBJECT = re.compile(
    r"^(feat|fix|chore|docs|refactor|test|style|perf|build|ci)(\([^()\n]{1,40}\))?: ([^\n]{1,72})$")


def check_commit_message(text: str, diff_files: list[str]) -> list[str]:
    """Conventional type in list, subject <= 72 chars, no trailing period or fences,
    scope (if present) occurs in the diff file list, and non-empty body lines wrap
    at 72 chars: the judge sees the subject line only, so body prose is checked,
    never judged."""
    found = []
    if "```" in text:
        found.append("message must not contain fences")
    lines = text.split("\n")
    match = _COMMIT_SUBJECT.match(lines[0])
    if match is None:
        return found + ["first line is not type[(scope)]: subject within 72 chars"]
    if lines[0].rstrip().endswith("."):
        found.append("subject must not end with a period")
    scope = match.group(2)
    if scope is not None and scope[1:-1] not in "\n".join(diff_files):
        found.append(f"scope {scope[1:-1]!r} occurs in no diff file")
    for lineno, line in enumerate(lines[1:], start=2):
        if line.strip() and len(line) > 72:
            found.append(f"body line {lineno} is {len(line)} chars, wrap at 72")
    return found


def _prompt_texts(body: dict) -> list[str]:
    """Every user/system text span in a chat or Responses body, in order."""
    texts = []
    if not isinstance(body, dict):
        return texts
    for key in ("messages", "input"):
        blocks = body.get(key)
        if isinstance(blocks, str):
            texts.append(blocks)
        elif isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                content = block.get("content", block.get("text"))
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    texts.extend(part.get("text") for part in content
                                 if isinstance(part, dict) and isinstance(part.get("text"), str))
    return texts


def _skill_description(body: dict) -> str | None:
    """The Description: tail of a skill-compression prompt, or None when absent."""
    for text in _prompt_texts(body):
        _, sep, tail = text.rpartition("Description:")
        if sep and tail.strip():
            return tail.strip()
    return None


def answerable(feature: str, body: dict) -> bool:
    """Whether an item can ever pass its checks: a skill compression needs a source
    description with at least one distinctive token to retain; titles and commits
    fail closed at run time instead (their whole request is the input)."""
    if feature != "skill-description-compression":
        return True
    description = _skill_description(body)
    return description is not None and bool(_distinctive_tokens(description))


def _distinctive_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z0-9_]+", text.lower())
            if len(w) > 4 and w not in _STOPWORDS}


def check_skill_compression(text: str, description: str) -> list[str]:
    """validCompression shape (single line, <= 12 words, <= 160 chars) plus keyword
    retention (>= 0.4 of the source's distinctive tokens) and a length ratio <= 0.6."""
    found = []
    if "\n" in text or "\r" in text:
        found.append("compression must be a single line")
    words = text.split()
    if len(words) > 12:
        found.append(f"compression is {len(words)} words, want at most 12")
    if len(text) > 160:
        found.append(f"compression is {len(text)} chars, want at most 160")
    if not text.strip():
        return found + ["compression is empty"]
    source, out = _distinctive_tokens(description), _distinctive_tokens(text)
    if source and len(source & out) / len(source) < 0.4:
        found.append("compression keeps fewer than 0.4 of the source distinctive tokens")
    if len(text) > 0.6 * len(description):
        found.append("compression is longer than 0.6 of the source description")
    return found


def response_text(body) -> str | None:
    """The generated text of a reply body, by route shape, or None when no text
    is present (an error outcome, never an empty string standing in for one):
    - chat/completions: choices[0].message.content (string or content blocks)
    - responses: the first output item's first content block text
    - completions: choices[0].text
    - Ollama generate: response
    Anything else is not a generation reply."""
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            texts = [block.get("text") for block in content
                     if isinstance(block, dict) and isinstance(block.get("text"), str)]
            return "".join(texts) or None
        text = first.get("text")
        return text if isinstance(text, str) else None
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            for block in item.get("content", []):
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    return block["text"]
        return None
    response = body.get("response")
    return response if isinstance(response, str) else None


FEATURE_CHECKS: dict[str, Callable[[str, dict], list[str]]] = {
    "titles": lambda text, context: check_title(text),
    "commit-messages": lambda text, context: check_commit_message(
        text, (context or {}).get("diff_files", [])),
    "skill-description-compression": lambda text, context: check_skill_compression(
        text, (context or {}).get("description", "")),
}

# --- judge hook: order assignment and win bounds. The model call itself belongs to the
# runner (%pane prove.py); this module fixes the randomization contract both arms share.

PAIRWISE_JUDGE_MODEL = "gemma3:27b"
PAIRWISE_JUDGE_DIGEST = "a418f5838eaf"  # localbench pull; the runner refuses any other digest


def judge_digest_ok(judge: "PairwiseJudge", installed_digest: str | None) -> bool:
    """The supply-chain pin: True only when an installed digest is present and shares
    a >= 12-hex prefix with the judge's pinned digest (same rule as features.same_digest)."""
    if judge.digest is None or installed_digest is None:
        return False
    short, long = sorted((judge.digest.lower(), installed_digest.lower().removeprefix("sha256:")),
                         key=len)
    return len(short) >= 12 and long.startswith(short)


@dataclass(frozen=True)
class PairwiseJudge:
    """A blind pairwise judge assertion: model of a family in neither arm, order
    randomized per item from subseed seed+":order", swap_subset from seed+":swap".
    Isolation is enforced, not assumed: the runner MUST call judge_isolated() with
    the run-time family map and refuse on False, MUST present judge_view() spans
    (not raw outputs) to the model, and MUST refuse a judge whose installed digest
    does not match digest (judge_digest_ok). This dataclass declares; those
    functions pin."""
    model: str = PAIRWISE_JUDGE_MODEL
    digest: str | None = PAIRWISE_JUDGE_DIGEST


def assign_order(item_ids: list[str], seed: str) -> dict[str, str]:
    """Item id -> "AB" (local first) or "BA", deterministically from seed+":order"."""
    out = {}
    for item_id in item_ids:
        digest = hashlib.sha256(f"{seed}:order:{item_id}".encode("utf-8")).hexdigest()
        out[item_id] = "AB" if int(digest, 16) % 2 == 0 else "BA"
    return out


def swap_subset(item_ids: list[str], seed: str, fraction: float) -> set[str]:
    """Deterministic swap-consistency subset: lowest seed+":swap" hashes first."""
    ranked = sorted(item_ids,
                    key=lambda item_id: hashlib.sha256(
                        f"{seed}:swap:{item_id}".encode("utf-8")).hexdigest())
    return set(ranked[:round(fraction * len(ranked))])


def win_lower_bound(wins: int, ties: int, n: int) -> float:
    """Normal-approximation lower bound on (wins + ties/2) / n, the pre-registered
    no-loss rule: >= min_win_lb means no quality loss beyond noise."""
    if n <= 0:
        return 0.0
    point = (wins + 0.5 * ties) / n
    return point - 1.96 * math.sqrt(point * (1.0 - point) / n)


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def pairwise_calibration(pairs: list[dict]) -> dict:
    """Summarize blind pairwise labels without treating raw preference as calibrated.

    Length adjustment is the candidate win-score intercept from an OLS model with
    log(candidate token count / baseline token count) as its sole covariate; this is
    the predicted preference at equal verbosity. Scores are candidate win=1, tie=.5,
    baseline win=0. An out-of-range intercept is refused, never clipped into a rate.
    `swap_winner` is already normalized back to candidate/baseline semantics.
    """
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("judge calibration needs at least one pair")
    ids: set[str] = set()
    scores, gaps, candidate_lengths, baseline_lengths = [], [], [], []
    swaps = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError(f"judge pair is not an object: {pair!r}")
        item_id = pair.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise ValueError(f"judge pair item_id is missing or duplicated: {item_id!r}")
        ids.add(item_id)
        for arm in ("candidate", "baseline"):
            if not isinstance(pair.get(f"{arm}_text"), str):
                raise ValueError(f"judge pair {item_id}: missing {arm}_text")
        winner = pair.get("winner")
        if winner not in ("candidate", "baseline", "tie"):
            raise ValueError(f"judge pair {item_id}: invalid winner {winner!r}")
        candidate_n = len(pair["candidate_text"].split())
        baseline_n = len(pair["baseline_text"].split())
        candidate_lengths.append(candidate_n)
        baseline_lengths.append(baseline_n)
        scores.append(1.0 if winner == "candidate" else 0.5 if winner == "tie" else 0.0)
        gaps.append(math.log((candidate_n + 0.5) / (baseline_n + 0.5)))
        swap_winner = pair.get("swap_winner")
        if swap_winner is not None:
            if swap_winner not in ("candidate", "baseline", "tie"):
                raise ValueError(f"judge pair {item_id}: invalid swap winner {swap_winner!r}")
            swaps.append(winner == swap_winner)
    if not swaps:
        raise ValueError("judge calibration needs at least one swap outcome")

    mean_score = math.fsum(scores) / len(scores)
    mean_gap = math.fsum(gaps) / len(gaps)
    variance = math.fsum((gap - mean_gap) ** 2 for gap in gaps)
    slope = (math.fsum((gap - mean_gap) * (score - mean_score)
                       for gap, score in zip(gaps, scores)) / variance if variance else 0.0)
    adjusted = mean_score - slope * mean_gap
    if not 0.0 <= adjusted <= 1.0:
        raise ValueError(f"length-adjusted win rate is out of [0, 1]: {adjusted!r}")
    return {
        "n": len(scores),
        "raw_win_rate": mean_score,
        "length_adjusted_win_rate": adjusted,
        "length_adjustment": {
            "method": "OLS candidate win score ~ log((candidate_tokens+0.5)/(baseline_tokens+0.5)); "
                      "intercept at equal verbosity",
            "log_length_ratio_slope": slope,
        },
        "verbosity": {
            "candidate": {"mean_tokens": math.fsum(candidate_lengths) / len(candidate_lengths),
                          "median_tokens": _median(candidate_lengths), "total_tokens": sum(candidate_lengths)},
            "baseline": {"mean_tokens": math.fsum(baseline_lengths) / len(baseline_lengths),
                         "median_tokens": _median(baseline_lengths), "total_tokens": sum(baseline_lengths)},
        },
        "swap_consistency": {"n": len(swaps), "consistent": sum(swaps),
                             "rate": sum(swaps) / len(swaps)},
    }


def require_pairwise_calibration(receipt: dict) -> None:
    """Refuse receipts with a raw judge win rate but missing calibration evidence."""
    if not isinstance(receipt, dict):
        raise ValueError("judge calibration receipt must be an object")
    calibration = receipt.get("judge_calibration")
    has_raw = "raw_win_rate" in receipt or "win_rate" in receipt
    if calibration is None:
        if has_raw:
            raise ValueError("raw judge win rate is refused without judge calibration")
        return
    if not isinstance(calibration, dict):
        raise ValueError("judge calibration must be an object")
    rate_names = ("raw_win_rate", "length_adjusted_win_rate")
    if any(not isinstance(calibration.get(name), (int, float))
           or isinstance(calibration.get(name), bool)
           or not math.isfinite(calibration[name]) or not 0.0 <= calibration[name] <= 1.0
           for name in rate_names):
        raise ValueError("judge calibration needs finite rates in [0, 1]")
    n = calibration.get("n")
    verbosity = calibration.get("verbosity")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or not isinstance(verbosity, dict):
        raise ValueError("judge calibration needs a positive n and per-arm verbosity")
    for arm in ("candidate", "baseline"):
        values = verbosity.get(arm)
        if not isinstance(values, dict):
            raise ValueError(f"judge calibration needs {arm} verbosity")
        for field in ("mean_tokens", "median_tokens", "total_tokens"):
            value = values.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"judge calibration needs valid {arm} {field}")
    swap = calibration.get("swap_consistency")
    if not isinstance(swap, dict):
        raise ValueError("judge calibration needs swap consistency")
    swap_n, consistent, swap_rate = swap.get("n"), swap.get("consistent"), swap.get("rate")
    if (not isinstance(swap_n, int) or isinstance(swap_n, bool) or swap_n < 1
            or not isinstance(consistent, int) or isinstance(consistent, bool)
            or not 0 <= consistent <= swap_n
            or not isinstance(swap_rate, (int, float)) or isinstance(swap_rate, bool)
            or not math.isfinite(swap_rate) or swap_rate != consistent / swap_n):
        raise ValueError("judge calibration needs a measured swap-consistency rate")


# --- replay: local model arms re-send captured messages through the loopback route.


def replay_body(body: dict, model: str) -> dict:
    """The captured messages with the model field set per arm; everything else identical."""
    return {**body, "model": model}


def judge_view(feature: str, text: str, context: dict | None = None) -> str:
    """The checked core text the judge sees: preamble, trailing explanations and body
    prose carry arm tells (style, length), so the judge gets exactly the span the
    deterministic check validates. Total: defined even for failing outputs, but the
    runner scores check failures through the deterministic assertions, not the judge."""
    _ = context
    if feature == "titles":
        if text.count("<title>") == 1 and text.count("</title>") == 1:
            return text.split("<title>", 1)[1].split("</title>", 1)[0].strip()
        return re.sub(r"</?title>", "", text).strip()
    if feature == "commit-messages":
        return text.split("\n", 1)[0].strip()
    if feature == "skill-description-compression":
        return text.strip().split("\n", 1)[0]
    raise corpus.CorpusError(f"generation feature {feature!r} has no judge view")


def judge_isolated(judge: str, arms: list[str], families: dict[str, str]) -> bool:
    """The isolation pin the PairwiseJudge docstring promises: True only when the
    judge's model family differs from BOTH arms' families. The runner derives families
    from ollama metadata at run time, pins the map in the receipt, and refuses to judge
    on False. Unknown families fail closed; the judge never shares an arm's model."""
    if judge in arms:
        return False
    judge_family = families.get(judge)
    if judge_family is None:
        return False
    arm_families = {families.get(arm) for arm in arms}
    return None not in arm_families and judge_family not in arm_families


def post_loopback(url: str, body: dict, timeout: float = 120.0) -> dict:
    """POST a JSON body to a loopback URL (anything else is refused). Returns
    {status, body, latency_s} on HTTP reply or {error, latency_s} on failure."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
    if urllib.parse.urlsplit(url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise corpus.CorpusError(f"replay target {url!r} is not loopback; refusing")
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    started = time.monotonic()
    try:
        request = Request(url, data=payload, headers={"Content-Type": "application/json"},
                          method="POST")
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
        latency = time.monotonic() - started
        try:
            return {"status": response.status, "body": json.loads(raw), "latency_s": latency}
        except (ValueError, UnicodeDecodeError):
            return {"status": response.status, "body": raw.decode("utf-8", "replace"),
                    "latency_s": latency}
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "latency_s": latency}

# --- runner entry point: what prove.py calls per candidate (mirrors run_decision_candidate).
# A screen replays hundreds of items through a local model: one heavy job at a time.


@_held("generation-replay")
def run_candidate(spec: dict, candidate: dict, items: list[dict], *,
                  post=post_loopback, timeout: float = 120.0) -> dict:
    """Run one generation candidate over corpus items and return evidence (no banking).
    A {"route": "ollama:<model>"} candidate replays each body with that model through its
    gateway profile route; a {"builtin": ...} candidate reproduces omp's no-model fallback
    with no model call (titles: unnamed without session context; commit: typed error).
    Per-item outcomes carry text (judge_view span extraction is the runner's job),
    violations from FEATURE_CHECKS, and latencies."""
    feature = spec.get("feature")
    if feature not in {name for name, _ in FEATURE_SIGNATURES}:
        raise corpus.CorpusError(f"generation spec feature {feature!r} has no request signature")
    check = FEATURE_CHECKS[feature]
    server = None
    pins = None
    if isinstance(candidate, dict) and "route" in candidate:
        route = candidate["route"]
        if not isinstance(route, str):
            raise corpus.CorpusError(f"generation candidate route {route!r}: expected string")
        if route.startswith("ollama:") and route[len("ollama:"):]:
            model = route[len("ollama:"):]
            builtin = None
        elif route.startswith("mlx-serve:") and route[len("mlx-serve:"):]:
            from .backends import MlxServe
            server = MlxServe(route[len("mlx-serve:"):])
            model, builtin = None, None
        else:
            raise corpus.CorpusError(
                f"generation candidate route {route!r}: ollama:<model> or mlx-serve:<model dir> only")
    elif isinstance(candidate, dict) and "builtin" in candidate:
        model, builtin = None, candidate["builtin"]
    else:
        raise corpus.CorpusError(f"generation candidate needs route or builtin: {candidate!r}")
    outcomes = []
    try:
        if server is not None:
            server.start()
            model = server.model_id()
            pins = server.fingerprint(model)
        for item in items:
            body = item["body"]
            context = _check_context(feature, body)
            if builtin is not None:
                outcomes.append({**_builtin_outcome(feature, body, item["id"]), "context": context})
                continue
            profile = item.get("profile")
            if server is None and (not isinstance(profile, str) or not profile):
                outcomes.append({"id": item["id"], "text": None, "context": context,
                                 "violations": ["item carries no replay profile"],
                                 "latency_s": 0.0, "error": "no-profile"})
                continue
            try:
                if server is None:
                    url = gateway_url(profile, body)
                else:
                    url = server.base_url + "/chat/completions"
                reply = post(url, replay_body(body, model), timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - record, never stop the screen
                reply = {"error": f"{type(exc).__name__}: {exc}", "latency_s": -1.0}
            text = response_text(reply.get("body")) if "error" not in reply else None
            outcomes.append({
                "id": item["id"], "text": text, "context": context,
                "violations": (check(text, context) if text is not None
                               else [f"no extractable text ({reply.get('error', 'empty')[:100]})"]),
                "latency_s": reply.get("latency_s", -1.0),
                **({"error": reply["error"]} if "error" in reply else {})})
    finally:
        if server is not None:
            server.stop()
    return {"candidate": candidate, "model": model, "builtin": builtin,
            "pins": pins, "outcomes": outcomes, "n": len(outcomes)}


def _check_context(feature: str, body: dict) -> dict:
    """The context FEATURE_CHECKS needs beyond the answer text: the source description
    for skill retention; an empty diff list for commits (scope without a recorded diff
    fails closed)."""
    if feature == "skill-description-compression":
        return {"description": _skill_description(body) or ""}
    if feature == "commit-messages":
        return {"diff_files": []}
    return {}


def _builtin_outcome(feature: str, body: dict, item_id: str) -> dict:
    """One builtin outcome without a model call: preview text, label, or typed error."""
    if feature == "skill-description-compression":
        description = _skill_description(body)
        if description is None:
            return {"id": item_id, "text": None,
                    "violations": ["no description in prompt"], "latency_s": 0.0,
                    "error": "no-description"}
        text = builtin_skill_description(description)
        return {"id": item_id, "text": text,
                "violations": check_skill_compression(text, description), "latency_s": 0.0}
    if feature == "titles":
        return {"id": item_id, "text": None,
                "violations": ["no title produced (session unnamed)"], "latency_s": 0.0,
                "error": "unnamed"}
    try:
        builtin_commit()
    except BuiltinUnavailable as exc:
        return {"id": item_id, "text": None, "violations": [f"builtin unavailable ({exc})"],
                "latency_s": 0.0, "error": "builtin-unavailable"}
    raise AssertionError("unreachable")
