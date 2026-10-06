"""Bounded interleavings for gateway lease/residency safety."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from localbench import gateway

CLIENTS = ("client-0", "client-1")
MODELS = ("model-0", "model-1")
MODEL_STEPS = (("start", CLIENTS[0]), ("finish", CLIENTS[0]),
               ("start", CLIENTS[1]), ("finish", CLIENTS[1]))

# Residency phase and lease phase cross with the in-flight request count.
# Each transition is exercised against GatewayStore + GatewayPolicy below.
TRANSITIONS = {
    ("start", "absent"): ("loading", "active"),
    ("start", "resident"): ("resident", "active"),
    ("load_complete", "loading"): ("resident", "active"),
    ("finish", "resident, requests remain"): ("resident", "active"),
    ("finish", "resident, no requests"): ("resident", "idle"),
    ("claim_due", "resident, zero in flight"): ("unloading", "unload_pending"),
    ("claim_due", "resident, in flight"): ("resident", "active"),
    ("unload_confirmed", "unloading, zero in flight"): ("absent", "released"),
    ("unload_refused_or_failed", "unloading"): ("resident", "idle"),
    ("start", "unloading"): ("unloading", "unload_pending"),
}


@dataclass
class ModelState:
    phase: str = "absent"
    lease_state: str = "absent"
    expected_resident: bool = False
    requests: dict[tuple[str, str], str] = field(default_factory=dict)

    def transition(self, event: str, condition: str) -> None:
        self.phase, self.lease_state = TRANSITIONS[event, condition]


class FakeOllama:
    """Observable residency with a hard assertion at the unload boundary."""

    def __init__(self, store: gateway.GatewayStore):
        self.store = store
        self.actual_residents: set[str] = set()
        self.before_residency_read: Callable[[], None] | None = None

    def load(self, model: str) -> None:
        self.actual_residents.add(model)

    def resident_models(self) -> set[str]:
        if self.before_residency_read is not None:
            self.before_residency_read()
        return set(self.actual_residents)

    def unload(self, model: str) -> None:
        active = self.store.active_requests(model)
        if active:
            raise AssertionError(f"unloaded {model} with {active} in-flight request(s)")
        self.actual_residents.discard(model)


def interleavings() -> Iterator[tuple[tuple[str, str, str], ...]]:
    """Interleave two models' four ordered start/finish steps (2 clients per model)."""

    def visit(progress: tuple[int, int], trace: tuple[tuple[str, str, str], ...]):
        if progress == (len(MODEL_STEPS), len(MODEL_STEPS)):
            yield trace
            return
        for model_index, model in enumerate(MODELS):
            step = progress[model_index]
            if step == len(MODEL_STEPS):
                continue
            event, client = MODEL_STEPS[step]
            next_progress = (
                progress[0] + int(model_index == 0),
                progress[1] + int(model_index == 1),
            )
            yield from visit(next_progress, trace + ((event, client, model),))

    yield from visit((0, 0), ())


class LeaseTransitionModel(unittest.TestCase):
    def test_every_two_client_two_model_interleaving_preserves_lease_and_residency(self):
        expected_schedule_count = 70  # Eight events; each model contributes four ordered steps.
        current_time = [0.0]
        explored = 0
        race_attempts = 0
        expired_while_active_checks = 0

        with tempfile.TemporaryDirectory() as tmp:
            store = gateway.GatewayStore(f"{tmp}/leases.sqlite")
            store.set_profiles({client: f"/omp-profile/{client}" for client in CLIENTS})
            ollama = FakeOllama(store)
            states = {model: ModelState() for model in MODELS}

            def try_request_during_unload() -> None:
                nonlocal race_attempts
                for model in MODELS:
                    lease = store.lease(model)
                    state = states[model]
                    if (lease is None or lease["last_outcome"] != "unload_pending"
                            or state.phase == "unloading"):
                        continue
                    self.assertEqual(store.active_requests(model), 0,
                                     f"claim_due selected {model} with an active request")
                    state.transition("claim_due", "resident, zero in flight")
                    race_attempts += 1
                    try:
                        store.start_request(CLIENTS[0], model, "race-client", now=current_time[0])
                    except gateway.GatewayError as exc:
                        self.assertIn("unload is in progress", str(exc))
                    else:
                        state.transition("start", "unloading")
                        self.fail(f"{model} admitted a request after its unload claim")

            ollama.before_residency_read = try_request_during_unload
            policy = gateway.GatewayPolicy(
                store, ollama, external_check=lambda model: (model in MODELS, None),
            )

            for trace in interleavings():
                explored += 1
                states = {model: ModelState() for model in MODELS}
                ollama.actual_residents.clear()
                with store._connect() as db:
                    for table in ("active_requests", "leases", "lease_profiles", "park_fences",
                                  "purpose_stats", "requests"):
                        db.execute(f"DELETE FROM {table}")
                    db.commit()

                for offset, (event, client, model) in enumerate(trace):
                    now = 100.0 + offset * 100.0
                    state = states[model]
                    key = (client, model)
                    if event == "start":
                        request_id = store.start_request(client, model, "bounded-model", now=now)
                        state.requests[key] = request_id
                        if state.phase == "absent":
                            state.transition("start", "absent")
                            ollama.load(model)
                            state.expected_resident = True
                            state.transition("load_complete", "loading")
                        else:
                            state.transition("start", "resident")
                    else:
                        request_id = state.requests.pop(key)
                        store.finish_request(request_id, completed=True, outcome="completed", now=now)
                        condition = "resident, requests remain" if state.requests else "resident, no requests"
                        state.transition("finish", condition)

                    sweep_now = now
                    if event == "start" and client == CLIENTS[1]:
                        # The prior request's idle expiry is now stale; its new request must
                        # prevent claim_due from racing an external keep_alive=0.
                        sweep_now = now + gateway.IDLE_SECONDS + 1
                    elif event == "finish" and client == CLIENTS[1]:
                        sweep_now = now + gateway.IDLE_SECONDS + 1
                    current_time[0] = sweep_now
                    outcomes = policy.expire_due(now=sweep_now)
                    for outcome in outcomes:
                        state = states[outcome["model"]]
                        if outcome["outcome"] in {"unloaded_confirmed", "not_resident_confirmed"}:
                            state.transition("unload_confirmed", "unloading, zero in flight")
                            state.expected_resident = False
                        else:
                            state.transition("unload_refused_or_failed", "unloading")

                    observed = ollama.resident_models()
                    expected = {name for name, row in states.items() if row.expected_resident}
                    self.assertEqual(observed, expected, f"residency mismatch after {trace[:offset + 1]}")
                    for name, row in states.items():
                        lease = store.lease(name)
                        active = store.active_requests(name)
                        self.assertEqual(active, len(row.requests),
                                         f"in-flight mismatch for {name} after {trace[:offset + 1]}")
                        if (active and lease and lease["idle_expires_at"] is not None
                                and lease["idle_expires_at"] <= sweep_now):
                            expired_while_active_checks += 1
                            row.transition("claim_due", "resident, in flight")
                            self.assertNotEqual(lease["last_outcome"], "unload_pending",
                                                f"claim_due selected active {name}")
                        actual_lease = "absent" if lease is None else (
                            "unload_pending" if lease["last_outcome"] == "unload_pending" else
                            "active" if lease["active_requests"] else
                            "released" if lease["last_outcome"] in {
                                "unloaded_confirmed", "not_resident_confirmed"} else "idle"
                        )
                        self.assertEqual(actual_lease, row.lease_state,
                                         f"lease state mismatch for {name} after {trace[:offset + 1]}")
                        self.assertEqual(row.phase != "absent", name in observed,
                                         f"phase/residency mismatch for {name} after {trace[:offset + 1]}")

            self.assertEqual(explored, expected_schedule_count)
            self.assertGreater(race_attempts, 0, "no request raced a claimed unload")
            self.assertGreater(expired_while_active_checks, 0, "no expired lease overlapped an active request")
            print(f"bounded lease model: state count={explored}; events per state=8; "
                  f"unload-race probes={race_attempts}; active-expiry checks={expired_while_active_checks}")


if __name__ == "__main__":
    unittest.main()
