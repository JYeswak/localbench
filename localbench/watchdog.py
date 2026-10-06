"""Fail-closed run invariants evaluated from each sampler record."""

from __future__ import annotations

from collections.abc import Callable

from . import sysstats


class RunWatchdog:
    def __init__(self, target: tuple[str, ...], expected_omp_sha: str, current_omp_sha: Callable[[], str | None],
                 gpu_foreign_max_pct: float = 25.0):
        if len(target) < 2 or not expected_omp_sha:
            raise ValueError("watchdog requires a backend, model, and pinned omp sha")
        self.target = target
        self.expected_omp_sha = expected_omp_sha
        self.current_omp_sha = current_omp_sha
        self.gpu_foreign_max_pct = gpu_foreign_max_pct

    def violations(self, sample: dict) -> list[str]:
        """Return every violated or unverifiable invariant in this one sampler sample."""
        violations = []
        resident = sample.get("resident")
        if not isinstance(resident, dict):
            violations.append("residency_unknown")
        else:
            if any(value is None for value in resident.values()):
                violations.append("residency_unknown")
            backend, *models = self.target
            server_models = resident.get(backend)
            if not isinstance(server_models, list):
                violations.append(f"residency_unknown:{backend}")
            else:
                violations.extend(f"expected_model_not_resident:{model}" for model in models
                                  if model not in server_models)

        gpu_procs = sample.get("gpu_procs")
        if not isinstance(gpu_procs, list) or not isinstance(resident, dict):
            violations.append("gpu_client_state_unknown")
        else:
            try:
                foreign, foreign_gpu, _ = sysstats.classify(gpu_procs, resident, self.target,
                                                           self.gpu_foreign_max_pct)
            except (KeyError, TypeError, ValueError):
                violations.append("gpu_client_state_unknown")
            else:
                if foreign:
                    violations.append(f"foreign_resident:{foreign}")
                if foreign_gpu:
                    violations.append(f"foreign_gpu_client:{foreign_gpu}")

        try:
            current_sha = self.current_omp_sha()
        except (OSError, RuntimeError, ValueError) as exc:
            violations.append(f"omp_sha_unavailable:{type(exc).__name__}")
        else:
            if current_sha is None:
                violations.append("omp_sha_unavailable:missing_identity")
            elif current_sha != self.expected_omp_sha:
                violations.append(f"omp_sha_changed:{self.expected_omp_sha}->{current_sha}")
        return violations
