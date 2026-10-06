"""File and update one open bead per unproven local feature route: the proof trigger.

Unit A of the owner's "anything unproven should trigger". For every feature x
profile route that is local (report()["routes"][p]["local"]), plus every local
preset candidate of a feature family's row that has no live local route at
all, and whose proof grades UNPROVEN, BAD or STALE, exactly one open bead
titled "prove <feature> on <route>" must exist. A route that becomes PROVEN
closes its bead citing the receipt. Idempotency rides on a machine line in
the bead body (`proof-trigger status=<STATUS> | <reason>`): same marker means
no-op, a changed marker rewrites the body plus one comment, and everything
goes through argv lists (the releasewatch.bead_argv discipline) so the port
map sees every launch.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIGGER_STATUSES = ("UNPROVEN", "BAD", "STALE")
TRIGGER_TYPE = "task"
TRIGGER_PRIORITY = "3"
TRIGGER_LABELS = "side-model"
MARKER = "proof-trigger status="
BR_TIMEOUT = 120.0


def proof_kind(row: dict) -> str:
    """Which proof a feature needs: the decision tier, a memory study, the generation tier, or an agent eval."""
    if row.get("proof_suite") == "decision":
        return "decision tier"
    if row.get("proof_suite") == "mem":
        return "memory study"
    if row.get("route_kind") == "agent":
        return "agent eval"
    return "generation tier"


def _marker(status: str, reason: str) -> str:
    single = " ".join(str(reason).split())[:200]
    return f"{MARKER}{status} | {single}"


def parse_marker(description: str) -> tuple[str, str] | None:
    """(status, reason) from a trigger bead body, or None when it carries no marker."""
    for line in str(description or "").splitlines():
        if line.startswith(MARKER):
            status, _, reason = line[len(MARKER):].partition(" | ")
            return status.strip(), reason.strip()
    return None


def render(feature: str, route: str, kind: str, status: str, reason: str, detail: dict) -> tuple[str, str]:
    """(title, body) for one trigger bead. The last line is the idempotency marker."""
    title = f"prove {feature} on {route}"
    lines = [f"{title} (proof-trigger)", "", f"status: {status} — {reason}", ""]
    for key in ("route", "target", "model", "digest", "profiles", "preset", "module", "module_sha",
                "incumbent", "receipt"):
        if detail.get(key) not in (None, "", [], {}):
            lines.append(f"- {key}: {detail[key]}")
    lines += ["",
              "proof contract (features.grade): kind=run with run.label in (decision, memory); "
              "omp_module_sha == installed sha; problems empty; verdict.compare == BETTER against the "
              "incumbent baseline the receipt names; pins.model_digest == the route model's digest.",
              f"proof needed: {kind}.", "", _marker(status, reason)]
    return title, "\n".join(lines)


def units(rows: list[dict], local_presets: dict[str, list[str]]) -> list[dict]:
    """Every local-route unit with its grade: live units grouped by (feature, model),
    plus one candidate unit per local preset of a family whose features have no live
    local route at all. Passing units are included so closes can be detected."""
    out = []
    for row in rows:
        feature = row["feature"]
        live = [(profile, row["routes"][profile]) for profile in row.get("local_profiles", [])
                if row["routes"][profile].get("local")]
        if live:
            by_model: dict[str, list[str]] = {}
            for profile, _ in live:
                by_model.setdefault(row["proofs"][profile]["model"], []).append(profile)
            for model, profiles in sorted(by_model.items()):
                worst = max((row["proofs"][p] for p in profiles),
                            key=lambda q: _status_rank(q["proof"]))
                targets = sorted({row["routes"][p]["target"] for p in profiles})
                out.append({"feature": feature, "route": model, "kind": "live", "profiles": sorted(profiles),
                            "target": targets[0] if len(targets) == 1 else targets,
                            "model": model, "digest": worst.get("digest"),
                            "status": worst["proof"], "reason": worst["reason"],
                            "receipt": worst.get("receipt"), "row": row})
        else:
            for preset in sorted(local_presets.get(row.get("preset", "-"), [])):
                out.append({"feature": feature, "route": preset, "kind": "candidate", "profiles": [],
                            "target": f"preset {preset}", "model": None, "digest": None,
                            "status": row.get("proof", "UNPROVEN"), "reason": row.get("reason", ""),
                            "receipt": row.get("receipt"), "row": row, "preset": preset})
    return out


def _status_rank(status: str) -> int:
    order = ("PROVEN", "CARRIED", "BAD", "STALE", "UNPROVEN", "DISABLED")
    return order.index(status) if status in order else len(order)


def _br(argv: list[str]) -> tuple[int, str, str]:
    """Run br against the main Beads root; reject path drift before every operation."""
    beads_root = Path(os.environ.get("LOCALBENCH_BEADS_ROOT") or REPO_ROOT).expanduser().resolve()
    try:
        if os.environ.get("LOCALBENCH_BEADS_ROOT"):
            check = subprocess.run(["br", "where", "--json"], cwd=beads_root,
                                   capture_output=True, text=True, timeout=BR_TIMEOUT, check=False)
            if check.returncode:
                return check.returncode, check.stdout, check.stderr
            actual = Path(json.loads(check.stdout)["path"]).expanduser().resolve()
            expected = (beads_root / ".beads").resolve()
            if actual != expected:
                return 1, "", f"br resolves Beads to {actual}; expected {expected}"
        export_root = str(REPO_ROOT.resolve())
        command = [argument.replace(export_root, str(beads_root)) for argument in argv]
        proc = subprocess.run(["br", *command], cwd=beads_root, capture_output=True, text=True,
                              timeout=BR_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


class BeadError(RuntimeError):
    """A br call failed or answered unusably; the queue stops rather than half-applies."""


def _adopted(open_issues: list[dict], feature: str, route: str) -> list[dict]:
    """Open beads already covering this unit: titled `prove <feature> on ...` naming the route.

    Exact-title beads are owned, never adopted. Deterministic by bead id."""
    prefix = f"prove {feature} on"
    return sorted((bead for bead in open_issues
                   if bead.get("title", "") != f"{prefix} {route}"
                   and bead.get("title", "").startswith(prefix)
                   and route in bead.get("title", "")[len(prefix):]),
                  key=lambda bead: bead.get("id", ""))


def _bead_marker(br, bead: dict):
    """The bead's marker: body first, then the latest marker comment. None when absent or unreadable."""
    marker = parse_marker(bead.get("description", ""))
    if marker is not None:
        return marker
    rc, out, _ = br(["comments", bead["id"], "--json"])
    if rc != 0:
        return None
    try:
        comments = json.loads(out)
    except ValueError:
        return None
    if isinstance(comments, dict):
        comments = comments.get("comments", [])
    found = None
    for comment in comments if isinstance(comments, list) else []:
        marker = parse_marker((comment or {}).get("text", ""))
        if marker is not None:
            found = marker
    return found


def _close_action(bead_id: str, title: str, unit: dict) -> dict:
    receipt = unit.get("receipt") or "the banked receipt"
    return {"op": "close", "id": bead_id, "title": title, "unit": unit,
            "reason": f"route proven: receipt {receipt} grades PROVEN for "
                      f"{unit['feature']} on {unit['route']}"}

def plan(units: list[dict], open_issues: list[dict], br=_br) -> list[dict]:
    """Reconcile units with open beads: create, comment (+rewrite the marker), adopt, close, or noop.

    An open bead that already covers a unit (title starts with `prove <feature>
    on` and names the route) is adopted: the marker attaches by comment, and
    its body is never rewritten (it may hold a pre-registration)."""
    by_title = {issue.get("title", ""): issue for issue in open_issues}
    actions = []
    for unit in units:
        title, body = render(unit["feature"], unit["route"], proof_kind(unit["row"]), unit["status"],
                             unit["reason"], _detail(unit))
        current = (unit["status"], _short(unit["reason"]))
        bead = by_title.get(title)
        if bead is None:
            candidates = _adopted(open_issues, unit["feature"], unit["route"])
            if unit["status"] in TRIGGER_STATUSES:
                if not candidates:
                    actions.append({"op": "create", "title": title, "body": body, "unit": unit})
                else:
                    stale = [b for b in candidates if _bead_marker(br, b) != current]
                    if stale:
                        target = stale[0]
                        note = (f"{MARKER}{unit['status']} | {_short(unit['reason'])}\nadopted: "
                                f"this bead already covers prove {unit['feature']} on {unit['route']}; "
                                "body left untouched (pre-registration). Status updates arrive as comments.")
                        actions.append({"op": "adopt", "id": target["id"], "title": target["title"],
                                        "note": note, "unit": unit})
                    else:
                        actions.append({"op": "noop", "title": candidates[0]["title"], "unit": unit})
            elif unit["status"] == "PROVEN" and candidates:
                actions.append(_close_action(candidates[0]["id"], candidates[0]["title"], unit))
            continue
        if unit["status"] in TRIGGER_STATUSES:
            if parse_marker(bead.get("description", "")) != current:
                actions.append({"op": "comment", "id": bead["id"], "title": title, "body": body,
                                "note": f"status changed to {unit['status']}: {unit['reason']}", "unit": unit})
            else:
                actions.append({"op": "noop", "title": title, "unit": unit})
        elif bead is not None and unit["status"] == "PROVEN":
            actions.append(_close_action(bead["id"], title, unit))
    return actions


def _short(reason: str) -> str:
    return " ".join(str(reason).split())[:200]


def _detail(unit: dict) -> dict:
    row = unit["row"]
    proofs = row.get("proofs", {})
    first = next((proofs[p] for p in unit.get("profiles", []) if p in proofs), {})
    return {"route": f"live profiles {unit['profiles']}" if unit["kind"] == "live"
                      else f"candidate preset {unit.get('preset')}",
            "target": unit.get("target"), "model": unit.get("model"), "digest": unit.get("digest"),
            "profiles": unit.get("profiles"), "preset": unit.get("preset"),
            "module": f"{row.get('omp_package')}/{row.get('omp_module')}",
            "module_sha": row.get("omp_module_sha"),
            "incumbent": (first.get("incumbent") or {}).get("selector", None) or first.get("incumbent"),
            "receipt": unit.get("receipt")}


def open_issues(br=_br) -> list[dict]:
    """Every open bead (id, title, description). Raises BeadError when br fails."""
    rc, out, err = br(["list", "--status", "open", "--json"])
    if rc != 0:
        raise BeadError(f"br list --status=open failed rc={rc}: {err.strip()[-300:]}")
    try:
        return json.loads(out).get("issues", [])
    except ValueError as exc:
        raise BeadError(f"br list returned non-JSON: {exc}") from exc


def create_argv(title: str, body: str) -> list[str]:
    """`br create` arguments for one trigger bead (releasewatch.bead_argv discipline)."""
    return ["create", "--silent", "--type", TRIGGER_TYPE, "--priority", TRIGGER_PRIORITY,
            "--labels", TRIGGER_LABELS, "--title", title, "--description", body]


def apply(actions: list[dict], br=_br) -> dict:
    """Execute a plan's br argv lists. Returns {filed, updated, adopted, closed, noop, errors}."""
    summary = {"filed": [], "updated": [], "adopted": [], "closed": [], "noop": [], "errors": []}
    for action in actions:
        op = action["op"]
        try:
            if op == "create":
                rc, out, err = br(create_argv(action["title"], action["body"]))
                _check(rc, err, action)
                summary["filed"].append(action["title"])
            elif op == "comment":
                rc, out, err = br(["update", action["id"], "--description", action["body"]])
                _check(rc, err, action)
                rc, out, err = br(["comments", "add", action["id"], "-m", action["note"]])
                _check(rc, err, action)
                summary["updated"].append(action["title"])
            elif op == "adopt":
                rc, out, err = br(["comments", "add", action["id"], "-m", action["note"]])
                _check(rc, err, action)
                summary["adopted"].append(action["title"])
            elif op == "close":
                rc, out, err = br(["close", action["id"], "--reason", action["reason"]])
                _check(rc, err, action)
                summary["closed"].append(action["title"])
            else:
                summary["noop"].append(action["title"])
        except BeadError as exc:
            summary["errors"].append(f"{action.get('title')}: {exc}")
    if summary["errors"]:
        raise BeadError("; ".join(summary["errors"]))
    return summary

def _check(rc: int, err: str, action: dict) -> None:
    """Raise BeadError on a failed br call, naming the action and the tail of stderr."""
    if rc != 0:
        raise BeadError(f"br {action['op']} {action.get('id', action.get('title'))} "
                        f"rc={rc}: {err.strip()[-300:]}")


def describe(action: dict) -> tuple[str, str]:
    """(action line, why line) for --dry-run/--explain, mirroring __main__.Step."""
    op, title = action["op"], action["title"]
    if op == "create":
        return (f"file bead {title!r}", "the route is local and grades UNPROVEN/BAD/STALE with no open trigger bead")
    if op == "comment":
        return (f"update bead {title!r}: {action['unit']['status']}",
                "the graded status changed since the bead was filed; the body marker is rewritten")
    if op == "close":
        return (f"close bead {title!r}: route proven",
                f"graded PROVEN; {action['reason']}")
    if op == "adopt":
        return (f"adopt bead {title!r}: {action['unit']['status']}",
                "an existing bead already covers this unit; the marker attaches by comment, "
                "the body is never rewritten (pre-registration)")
    return (f"leave bead {title!r} (unchanged)", "the open bead already carries this status")


def collect(profiles: list[str] | None = None, br=_br) -> tuple[list[dict], list[dict], list[dict]]:
    """Read-only planning input: graded units, the open beads, and skipped presets (no mutation).

    A local preset with `"queue": false` is an owner decision, not a hypothesis:
    the queue skips it and reports it with its `queue_reason` instead of filing."""
    from . import features, presets

    rows = features.report(profiles)
    families = {row["preset"] for row in rows if row.get("preset") != "-"}
    local_presets: dict[str, list[str]] = {}
    skipped: list[dict] = []
    for preset in presets.load()["presets"]:
        if not preset.get("local"):
            continue
        family = presets.family(preset["name"])
        if family not in families:
            continue
        if preset.get("queue", True) is False:
            skipped.append({"preset": preset["name"],
                            "reason": preset.get("queue_reason") or "opted out of the proof queue"})
        else:
            local_presets.setdefault(family, []).append(preset["name"])
    return units(rows, local_presets), open_issues(br), skipped


def queue(profiles: list[str] | None = None, br=_br, dry_run: bool = False) -> dict:
    """Collect grades, plan against open beads, and apply unless dry_run. Returns the summary."""
    collected, open_beads, skipped = collect(profiles, br)
    planned = plan(collected, open_beads, br)
    summary: dict = {"actions": [(a["op"], a["title"]) for a in planned],
                     "filed": [], "updated": [], "adopted": [], "closed": [],
                     "noop": [a["title"] for a in planned if a["op"] == "noop"],
                     "skipped": skipped}
    if not dry_run:
        applied = apply(planned, br)
        summary.update({key: applied[key] for key in ("filed", "updated", "adopted", "closed", "noop")})
    return summary
