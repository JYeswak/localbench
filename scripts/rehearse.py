#!/usr/bin/env python3
"""Run committed proof specs through fake-backed prove paths with one fault per job."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from localbench import prove

ROOT = Path(__file__).resolve().parent.parent
FAULTS = ("stall", "5xx", "slow-first-byte", "residency-loss", "stall", "5xx", "slow-first-byte")


def _rehearsal_provenance(label, tiers):
    pins = {
        "backend": "rehearsal-only",
        "backend_version": "rehearsal-only",
        "backend_sha": "rehearsal-only",
        "model": "rehearsal-only",
        "model_digest": "rehearsal-only",
        "macos_build": "rehearsal-only",
        "host_id": "rehearsal-only",
    }
    return {
        "label": label,
        "pins": pins,
        "tiers": tiers,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "localbench_rev": "rehearsal-only",
        "fingerprint": {"backend": pins["backend"], "model": pins["model"]},
    }




class FakeFaultServer:
    """Loopback fake used by the rehearsal to exercise the real HTTP client path."""

    def __init__(self, fault: str):
        state = {"fault": fault}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                current_fault = state["fault"]
                if current_fault == "stall":
                    time.sleep(1)
                    return
                if current_fault == "5xx":
                    self.send_response(503)
                    self.end_headers()
                    return
                if current_fault == "slow-first-byte":
                    time.sleep(2)
                if current_fault == "residency-loss":
                    state["fault"] = "5xx"
                    self.send_response(503)
                    self.end_headers()
                    return
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_exc) -> None:
        shutdown = __import__("threading").Thread(target=self.server.shutdown, daemon=True)
        shutdown.start()
        self.server.server_close()
        shutdown.join(timeout=2)
        self.thread.join(timeout=2)


def _fake_br(argv):
    if argv and argv[0] == "list":
        return 0, json.dumps({"issues": []}), ""
    if argv and argv[0] == "create":
        return 0, "kit-rehearse\n", ""
    return 0, "", ""


def _run_suite(_spec, _candidate, _resolved, suite, fault):
    """Exercise decision.post and preserve every suite item on endpoint failure."""
    from localbench import decision

    with FakeFaultServer(fault) as endpoint:
        probe_error = None
        if fault:
            if not suite.items:
                raise ValueError("fault rehearsal requires at least one suite item")
            try:
                decision.post(
                    endpoint,
                    {"model": "fake", "state": suite.items[0]["id"], "questions": []},
                    timeout=0.5,
                )
            except decision.DecisionError as exc:
                probe_error = exc
            else:
                raise RuntimeError(f"injected {fault} fault did not affect the endpoint probe")

        outcomes = []
        for index, item in enumerate(suite.items):
            if probe_error is not None:
                outcomes.append({
                    "id": item["id"],
                    "repeat": 0,
                    "cold": index == 0,
                    "n_questions": len(item.get("questions", {})),
                    "ok": False,
                    "error": probe_error.kind,
                    "detail": f"endpoint probe failed: {probe_error.detail}"[:500],
                    "latency_s": None,
                })
                continue
            try:
                reply, latency = decision.post(
                    endpoint,
                    {"model": "fake", "state": item["id"], "questions": []},
                    timeout=0.5,
                )
            except decision.DecisionError as exc:
                outcomes.append({
                    "id": item["id"],
                    "repeat": 0,
                    "cold": index == 0,
                    "n_questions": len(item.get("questions", {})),
                    "ok": False,
                    "error": exc.kind,
                    "detail": exc.detail[:500],
                    "latency_s": None if exc.latency_s is None else round(exc.latency_s, 6),
                })
            else:
                outcomes.append({
                    "id": item["id"],
                    "repeat": 0,
                    "cold": index == 0,
                    "n_questions": len(item.get("questions", {})),
                    "ok": True,
                    "model": reply.get("model", "fake"),
                    "answers": reply.get("answers", {}),
                    "usage": reply.get("usage", {}),
                    "latency_s": round(latency, 6),
                })
        run = {
            "provenance": _rehearsal_provenance("decision", ["rehearsal"]),
            "metrics": {},
            "verdicts": {},
            "rehearsal": True,
            "decision": {"local": {"outcomes": outcomes, "metrics": {},
                        "suite": suite.pin(), "endpoint": endpoint}},
        }
        return {"kind": "run", "run": run,
                "problems": ([f"rehearsal fault: {fault}"] if fault else [])}


def _outputs(spec, items, fault):
    outputs = {}
    for candidate in spec["candidates"]:
        key = prove._generation_candidate_key(candidate)
        outputs[key] = [{"id": item["id"], "text": None, "violations": [f"rehearsal fault: {fault}"],
                         "latency_s": -1.0, "error": fault} for item in items]
    return outputs


def _legs(_spec, fault):
    return [
        {
            "label": label,
            "provenance": _rehearsal_provenance(label, ["rehearsal"]),
            "metrics": {},
            "verdicts": {},
            "rehearsal_fault": fault,
        }
        for label in ("ab_a1", "ab_b1")
    ]


def _ensure_rehearsal_provenance(path: str) -> None:
    """Attach deterministic fake pins so validate checks artifact shape, not real backend identity."""
    target = Path(path)
    doc = json.loads(target.read_text())
    run = doc.setdefault("run", {})
    run.setdefault("metrics", {})
    run.setdefault("verdicts", {})
    prov = run.setdefault("provenance", {})
    prov.setdefault("fingerprint", {"backend": "fake", "model": "rehearse"})
    prov.setdefault("pins", {"backend": "fake", "backend_version": "rehearse",
                              "backend_sha": "fake", "model": "rehearse",
                              "model_digest": "fake", "macos_build": "rehearse", "host_id": "rehearse"})
    prov.setdefault("tiers", ["rehearse"])
    prov.setdefault("created", "rehearse")
    prov.setdefault("localbench_rev", "rehearse")
    target.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _show_validate(path: str, root: Path = ROOT) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": str(root)}
    failures = []
    results = (("show", subprocess.run(["python3", "-m", "localbench", "show", path, "--json"], capture_output=True, text=True, cwd=root, env=env)),
               ("validate", subprocess.run(["python3", "-m", "localbench", "validate", path, "--json"], capture_output=True, text=True, cwd=root, env=env)))
    for name, result in results:
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            failures.append(f"{name} exited {result.returncode}: {detail}")
    return not failures, "; ".join(failures)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    active_specs = [p for p in sorted((ROOT / "registries" / "proofs").glob("*.json"))
                    if "blocked" not in p.name]
    if len(active_specs) != len(FAULTS):
        raise RuntimeError(f"expected {len(FAULTS)} active proof specs, found {len(active_specs)}")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    receipts = output.with_name(f"{output.stem}-receipts")
    receipts.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, fault in zip(active_specs, FAULTS):
        try:
            spec = prove.load_spec(path)
            kwargs = {"repo": ROOT, "receipts_dir": receipts, "br": _fake_br,
                      "grade_fn": lambda receipt, model: ("UNKNOWN", "rehearsal fault or fake backend")}
            if spec["kind"] == "decision":
                kwargs["run_suite_fn"] = lambda spec, candidate, resolved, suite, f=fault: _run_suite(
                    spec, candidate, resolved, suite, f)
            elif spec["kind"] == "generation":
                kwargs["outputs_fn"] = lambda spec, items, f=fault: _outputs(spec, items, f)
            else:
                kwargs["legs_fn"] = lambda spec, f=fault: _legs(spec, f)
            report = prove.prove_spec(path, **kwargs)
            candidates = report["candidates"]
            receipt_paths = [candidate["receipt"] for candidate in candidates]
            checked = [_show_validate(receipt) for receipt in receipt_paths]
            candidate_verdicts = [candidate["grade"] for candidate in candidates]
            verdict = (candidate_verdicts[0] if candidate_verdicts and
                       len(set(candidate_verdicts)) == 1 else "MIXED")
            rows.append({"job": path.name, "fault": fault, "verdict": verdict,
                         "candidate_verdicts": candidate_verdicts, "receipts": receipt_paths,
                         "show_validate": bool(checked) and all(ok for ok, _ in checked),
                         "validation_errors": [error for ok, error in checked if not ok]})
        except Exception as exc:  # rehearsal records wiring breaks instead of hiding them
            rows.append({"job": path.name, "fault": fault, "verdict": "UNKNOWN",
                         "candidate_verdicts": [], "receipts": [], "show_validate": False,
                         "wiring_error": f"{type(exc).__name__}: {exc}"})
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_count = sum(len(row["receipts"]) for row in rows)
    output.write_text(json.dumps({"jobs": rows, "count": len(rows), "receipt_count": receipt_count},
                                 indent=2, default=str) + "\n")
    all_non_keep = all(row["candidate_verdicts"] and
                       all(verdict in {"VOID", "UNKNOWN"} for verdict in row["candidate_verdicts"])
                       for row in rows)
    all_show_validate = all(row["show_validate"] for row in rows)
    print(json.dumps({"count": len(rows), "all_non_keep": all_non_keep,
                      "all_show_validate": all_show_validate, "receipt_count": receipt_count}, sort_keys=True))
    return 0 if len(rows) == len(FAULTS) and all_non_keep and all_show_validate else 1


if __name__ == "__main__":
    raise SystemExit(main())
