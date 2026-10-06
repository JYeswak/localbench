#!/usr/bin/env python3
"""Replay recorded machine-invalid run telemetry to estimate time lost before a VOID was detected."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from localbench import sysstats
from localbench.__main__ import GPU_BUSY_MAX_PCT

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = Path("docs/evidence/receipts")


def _walk(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("run_dir"), str) and isinstance(value.get("verdicts"), dict):
            yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _void_reasons(run: dict) -> list[str]:
    verdicts = run["verdicts"]
    provenance = run.get("provenance") or {}
    side = provenance.get("regime") == "side"
    during = ((run.get("system") or {}).get("during") or {})
    reasons = []
    if verdicts.get("contended") is True:
        reasons.append("contended")
    if verdicts.get("preflight_problems"):
        reasons.append("preflight")
    if verdicts.get("pins_changed"):
        reasons.append("pins_changed")
    unknown = during.get("resident_unknown_samples")
    if not side and unknown is None:
        reasons.append("residency_count_missing")
    elif not side and type(unknown) is int and unknown > 0:
        reasons.append(f"residency_unknown={unknown}")
    if any(entry.get("level") == "MUST" and entry.get("verdict") == "VOID"
           for entry in (run.get("conformance") or {}).values() if isinstance(entry, dict)):
        reasons.append("MUST_VOID")
    return reasons


def collect_void_runs(root: Path) -> tuple[list[dict], list[str], int]:
    receipt_dir = root / RECEIPTS
    by_run: dict[str, dict] = {}
    errors = []
    receipt_count = 0
    for receipt_path in sorted(receipt_dir.glob("*.json")):
        receipt_count += 1
        try:
            document = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{receipt_path.relative_to(root)}: {exc}")
            continue
        for run in _walk(document):
            reasons = _void_reasons(run)
            if not reasons:
                continue
            run_dir = run["run_dir"]
            row = by_run.setdefault(run_dir, {"run": run, "receipts": [], "reasons": reasons})
            row["receipts"].append(receipt_path.name)
    return list(by_run.values()), errors, receipt_count


def _is_timestamp(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))

def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    return rows, f"line {line_number}: invalid JSON ({exc})"
                if not isinstance(row, dict):
                    return rows, f"line {line_number}: expected an object"
                rows.append(row)
    except (OSError, UnicodeDecodeError) as exc:
        return rows, str(exc)
    return rows, None


def _inside_root(root: Path, relative: str) -> Path | None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _sample_violations(sample: dict, run: dict) -> list[str]:
    provenance = run.get("provenance") or {}
    pins = provenance.get("pins") or {}
    backend, model = pins.get("backend"), pins.get("model")
    if not isinstance(backend, str) or not isinstance(model, str):
        return ["data_gap:target_identity_missing"]
    targets: tuple[str, ...] = (backend, model)
    if isinstance(pins.get("smol_model"), str):
        targets += (pins["smol_model"],)
    resident = sample.get("resident")
    if not isinstance(resident, dict) or backend not in resident or any(value is None for value in resident.values()):
        return ["residency_unknown"]
    names = resident.get(backend)
    if not isinstance(names, list):
        return ["residency_unknown"]
    violations = []
    missing = [target for target in targets[1:] if target not in names]
    if missing:
        violations.append("expected_model_not_resident:" + ",".join(missing))
    gpu_procs = sample.get("gpu_procs")
    if not isinstance(gpu_procs, list):
        return [*violations, "data_gap:gpu_process_samples_missing"]
    try:
        foreign, foreign_gpu, _ = sysstats.classify(gpu_procs, resident, targets, GPU_BUSY_MAX_PCT)
    except (KeyError, TypeError, ValueError) as exc:
        return [*violations, f"data_gap:gpu_process_sample_invalid:{type(exc).__name__}"]
    if foreign:
        violations.append("foreign_resident")
    if foreign_gpu:
        violations.append("foreign_gpu_client")
    return violations


def replay_run(root: Path, candidate: dict) -> dict:
    run = candidate["run"]
    run_path = _inside_root(root, run["run_dir"])
    report = {"run_dir": run["run_dir"], "receipts": sorted(set(candidate["receipts"])),
              "void_reasons": candidate["reasons"], "first_violation": None,
              "elapsed_s": None, "duration_s": None, "saved_fraction": None,
              "status": "unreplayed"}
    if run_path is None:
        report["status"] = "unsafe_run_path"
        return report
    progress_path, sampler_path = run_path / "progress.jsonl", run_path / "sampler.jsonl"
    if not progress_path.is_file() or not sampler_path.is_file():
        missing = [name for name, path in (("progress.jsonl", progress_path), ("sampler.jsonl", sampler_path))
                   if not path.is_file()]
        report.update(status="missing_artifacts", missing=missing)
        return report
    progress, progress_error = _read_jsonl(progress_path)
    samples, sampler_error = _read_jsonl(sampler_path)
    errors = [f"progress: {progress_error}" if progress_error else None,
              f"sampler: {sampler_error}" if sampler_error else None]
    errors = [error for error in errors if error]
    if errors:
        report.update(status="malformed_artifacts", errors=errors)
        return report
    starts = [event["t"] for event in progress if event.get("event") == "start"
              and _is_timestamp(event.get("t"))]
    ends = [event["t"] for event in progress if event.get("event") == "done"
            and _is_timestamp(event.get("t"))]
    if not starts:
        report["status"] = "missing_start_event"
        return report
    start = starts[0]
    complete = bool(ends)
    observed_ends = ends or [row["t"] for row in samples if _is_timestamp(row.get("t"))]
    if not observed_ends:
        report["status"] = "missing_end_time"
        return report
    end = max(observed_ends)
    duration = end - start
    if duration <= 0:
        report["status"] = "invalid_duration"
        return report
    report["duration_s"] = round(duration, 3)
    report["complete"] = complete
    candidates = []
    data_gaps = []
    for sample in samples:
        stamp = sample.get("t")
        if not _is_timestamp(stamp):
            report["status"] = "malformed_sample_time"
            return report
        for reason in _sample_violations(sample, run):
            if reason.startswith("data_gap:"):
                data_gaps.append((stamp, reason.removeprefix("data_gap:")))
            else:
                candidates.append((stamp, reason))
    if data_gaps:
        first_gap = min(data_gaps)
        reasons = ", ".join(sorted({gap[1] for gap in data_gaps}))
        report["data_gaps"] = [f"first={first_gap[0]:.3f}; samples={len(data_gaps)}; reasons={reasons}"]
    # These events are the Sampler's own first-of-episode reports; retain them as an independent replay path.
    for event in progress:
        if event.get("event") != "contention":
            continue
        stamp = event.get("t")
        if not _is_timestamp(stamp):
            report["status"] = "malformed_contention_time"
            return report
        if event.get("foreign"):
            candidates.append((stamp, "foreign_resident"))
        if event.get("foreign_gpu"):
            candidates.append((stamp, "foreign_gpu_client"))
    # A completed run only discovers pin drift in the end-of-run comparison; the onset is not in old telemetry.
    pin_drift = bool((run.get("verdicts") or {}).get("pins_changed"))
    if pin_drift:
        report["pin_drift_onset"] = "unknown; only end-of-run comparison was banked"
    if not candidates:
        report["status"] = ("first_time_uncertain_pin_drift" if pin_drift else
                            "incomplete_evidence" if data_gaps else "no_timed_sampler_violation")
        return report
    stamp = min(row[0] for row in candidates)
    first_reasons = sorted({reason for time_value, reason in candidates if time_value == stamp})
    if stamp < start or stamp > end:
        report["status"] = "violation_outside_run"
        return report
    if data_gaps and min(gap[0] for gap in data_gaps) < stamp:
        report["status"] = "incomplete_evidence"
        return report
    elapsed = stamp - start
    report["first_violation"] = {"elapsed_s": round(elapsed, 3), "reasons": first_reasons,
                                 "time_utc": datetime.fromtimestamp(stamp, timezone.utc).isoformat()}
    report["elapsed_s"] = round(elapsed, 3)
    if pin_drift:
        report["status"] = "first_time_uncertain_pin_drift"
        return report
    if not complete:
        report["status"] = "incomplete_run"
        return report
    report["saved_fraction"] = round(max(0.0, min(1.0, (end - stamp) / duration)), 6)
    report["status"] = "timed"
    return report


def render_report(root: Path) -> str:
    candidates, errors, receipt_count = collect_void_runs(root)
    runs = [replay_run(root, candidate) for candidate in candidates]
    timed = [run for run in runs if run["status"] == "timed" and run["saved_fraction"] is not None]
    median = statistics.median(run["saved_fraction"] for run in timed) if timed else None
    filter_reasons = ", ".join(("contention", "unknown/missing residency", "expected model absent",
                                "preflight", "pin drift"))
    lines = ["# Banked VOID-run telemetry replay", "",
             f"Receipts scanned: {receipt_count}; unique candidate runs: {len(candidates)}; "
             f"timed complete runs: {len(timed)}; non-timed/unavailable: {len(runs) - len(timed)}.",
             f"Machine invalidity filter: {filter_reasons}; "
             "MUST-VOID workload records are included but are not treated as machine aborts.",
             "Violation time is the first timestamp recoverable from sampler samples/progress events. "
             "Saved fraction is remaining run duration at that observed timestamp; it is an upper-bound opportunity, "
             "not a measured runtime saving. Pin-drift onset is not logged per second and is excluded from the median.", "",
             "| Run | Receipt(s) | Invalidity | First observed violation | Elapsed / duration (s) | Remaining fraction | Status |",
             "|---|---|---|---|---:|---:|---|"]
    for run in runs:
        first = run.get("first_violation")
        when = (", ".join(first["reasons"]) if first else "—")
        if first:
            when += f" @ {first['time_utc']}"
        ratio = (f"{run['elapsed_s']:.3f} / {run['duration_s']:.3f}"
                 if run.get("elapsed_s") is not None and run.get("duration_s") is not None else "—")
        saved = f"{100 * run['saved_fraction']:.1f}%" if run.get("saved_fraction") is not None else "—"
        receipts = ", ".join(run["receipts"])
        lines.append(f"| `{run['run_dir']}` | {receipts} | {', '.join(run['void_reasons'])} | {when} | {ratio} | {saved} | {run['status']} |")
        if run.get("missing"):
            lines.append(f"<!-- missing: {', '.join(run['missing'])} -->")
        if run.get("errors"):
            lines.append(f"<!-- artifact errors: {'; '.join(run['errors'])} -->")
        if run.get("pin_drift_onset"):
            lines.append(f"<!-- pin drift: {run['pin_drift_onset']} -->")
        if run.get("data_gaps"):
            lines.append(f"<!-- data gaps: {'; '.join(run['data_gaps'])} -->")
    lines.extend(["", f"Median remaining fraction: {100 * median:.1f}% (n={len(timed)} timed runs)." if median is not None
                  else "Median remaining fraction: unavailable (no timed complete runs).",
                  f"Watchdog threshold (>30%): {'EXCEEDED' if median is not None and median > 0.30 else 'NOT EXCEEDED' if median is not None else 'UNDECIDABLE'}."])
    if errors:
        lines.extend(["", "Receipt read errors:", *[f"- {error}" for error in errors]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="localbench workspace/data root")
    parser.add_argument("--output", type=Path, help="write the rendered replay report to this path")
    args = parser.parse_args()
    root = args.root.resolve()
    report = render_report(root)
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        try:
            output.resolve().relative_to(root)
        except ValueError:
            parser.error("--output must stay inside the workspace root")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
