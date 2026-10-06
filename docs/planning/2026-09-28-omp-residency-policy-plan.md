# OMP Ollama Residency Policy — Implementation Plan

**Status:** User-approved 2026-09-28; all 13 OMP profiles route to the installed loopback gateway and repository gates pass. Review fixes make any established non-gateway Ollama socket defer unload and cap inbound POST-body reads at 30 seconds. Full suite: 390 passed; 12 mutations caught; readiness, claim discipline, and Ruff passed. The managed LaunchAgent was reloaded only after `active_requests: 0`; post-reload status/listener/profile routes are healthy. OMP PID 9819 was not restarted or killed; its pre-existing direct-socket routing remains unverified until that session reloads. The earlier outage smoke had no direct fallback but exceeded 60 seconds, so clean OMP exit latency remains unverified. No staging or commit.

## Constraints and existing ownership

- The approved spec is `docs/planning/omp-residency-policy.md`.
- The initial pane-2 busy warning was superseded by Agent Mail thread 42964's asynchronous assignment to the current pane (%pane). No second NTM prompt was sent; the existing shared-tree changes were inspected and preserved.
- The tree contains unrelated uncommitted evaluation work in README, localbench/__main__.py, localbench/proxy.py, localbench/workloads.py, and tests. Those hunks were preserved. The user prohibited staging/committing this shared tree; no commit evidence is claimed. Target paths are reserved through Agent Mail.
- Use localbench’s existing profile/LaunchAgent conventions in `localbench/smol.py` (`profile_dirs`, `omp_bin`, `omp_env`, `plistlib`, `launchctl`, backup/rollback markers). Use existing native Ollama operations in `localbench/backends.py` / `localbench.__main__` rather than adding a dependency. Keep gateway state under `~/.localbench`, outside the clone.
- The built-in Ollama override was verified against the installed OMP version 18.4.2. The isolated default/named-profile smoke passed through the profile path prefix, including a separate direct fallback trap; no live profiles were edited before that proof.

## Work packages

### 1. Validate provider routing without changing user profiles

1. Preserve the pre-existing evaluation changes and reserve one owner per target path; do not dispatch another agent.
2. The OMP 18.4.2 prerequisite passed in a disposable HOME with default and named profiles, each using the built-in ollama provider's profile-prefixed baseUrl.
3. The actual OMP models command resolved the fake Ollama catalog through the loopback gateway; OMP requests returned ROUTED_OK through /v1/responses for both profiles.
4. A second loopback-only Ollama fallback listener received zero requests. The smoke and response content were not persisted.
5. PASS before any user profile or LaunchAgent changes; no local model inference was used.

### 2. Implement gateway and finite residency leases

1. Inspect the final `localbench.proxy` implementation after pane 2 finishes; reuse its request/stream forwarding behavior where appropriate, but keep gateway policy and run-recording behavior separate if their lifecycle contracts differ.
2. Add a stdlib-only loopback gateway service for Ollama’s OpenAI-compatible API. Preserve status codes, headers required by OMP, JSON bodies, streaming chunks, disconnects, and bounded timeouts. Strip only the profile-routing prefix/header used for attribution; never log request/response bodies.
3. Add a host-local SQLite lease store under `~/.localbench`; schema records profile identity, model, active-request count/instance identity, last completed request, finite expiry, and unload outcome. Use a single-instance lock so two services cannot retire the same model concurrently. No prompt, response, auth key, or full request body in the store.
4. Track request completion (including streaming terminal events and client disconnects) and reset the 300-second idle deadline from completion. A scheduler retires only gateway-managed models with no in-flight request. Before retirement, fail closed if external-client activity cannot be ruled out; run native Ollama unload (`keep_alive: 0`) and verify `/api/ps` no longer lists the model. On expiry/unload failure or external-client uncertainty, retain the lease/resident status as unresolved and surface the reason.
5. Recover the last persisted lease state on restart. Clear an old in-flight count only after proving the owning gateway process is dead; never adopt an unknown resident as gateway-owned.
6. Add `localbench` gateway lifecycle commands for `serve`, `status`, `start`, `stop`, and reversible LaunchAgent `install`/`remove`. Use the existing user-level LaunchAgent approach; no sudo, Ollama restart, firewall change, or remote bind. `status` distinguishes service health, configured profiles, active calls, lease expiry, residents, and unknown/unowned residents. Port conflicts and unavailable Ollama are explicit errors; local OMP requests fail closed.
7. Change `localbench keep` to a finite-only contract: default to the five-minute policy, reject `forever`/negative/unbounded values, preserve `0`/`unload`, verify actual residency, and audit the finite expiry. Migrate every command, test, README/context instruction, and operational callsite that currently says `forever`; do not silently re-pin it under another spelling.

### 3. Reversible OMP profile migration

1. Enumerate every OMP agent directory with the existing `smol.profile_dirs()` helper, including the default profile where present. Determine relevance from actual profile provider/role resolution rather than assuming only profiles whose current modelRole names Ollama; users can select another installed model.
2. Add a marked, idempotent `ollama` provider override pointing to the gateway and carrying profile identity. Preserve each existing `models.yml` entry, `config.yml`, roles, auth settings, and unrelated provider blocks. Back up original files under the established `~/.localbench/rollback` tree before mutation.
3. Apply changes transactionally: preflight every profile, verify no conflict with existing explicit Ollama provider or custom local endpoint, write atomically, read every profile back through OMP, and roll back all written profiles if any profile fails resolution or points at `11434` directly.
4. The localbench-owned benchmark child was inspected and excluded: ensure_localbench_model() exposes its isolated localbench provider, not built-in Ollama; benchmark payloads and smol routing were not changed.
5. Provide a reversible profile revert that removes only the marked gateway block when unchanged; preserve manual edits made after installation and report profiles that need operator reconciliation.

### 4. Integrate status, docs, and tests

1. Extend `localbench status` and gateway status with distinct states: gateway unavailable; active; finite lease; expired and unload confirmed; expiry deferred due to external activity/uncertainty; unowned resident; Ollama query unknown. `localbench gpu` reports clients but socket presence is not proof of active work. Since there is no reliable request-idle signal, an established non-gateway socket defers unload until it closes, regardless of elapsed time, nettop bytes, or GPU silence.
2. Replace only affected retention docs in README and `AGENTS.md`; document the five-minute finite contract, gateway install/status, fail-closed behavior, rollback, and direct non-OMP bypass boundary. Preserve concurrent README edits from pane 2. No new public performance claim or golden regeneration.
3. Add consumer-visible tests in `tests/test_gateway.py` and update `tests/test_keep_pull.py` / existing CLI tests. A fake Ollama must observe forwarded requests and native unload. Cover five-minute gateway-owned expiry, in-flight overlap, streaming/disconnect, persistent external sockets beyond 300 seconds, external-client uncertainty, unload failure, restart recovery, finite CLI parsing, unowned residents, rollback, and bounded partial-body reads.
4. Add only mutation cases that plant plausible consumer-visible defects and prove the named test catches them; use `scripts/mutate.py` per `AGENTS.md`. Never weaken gates or regenerate measurement goldens to pass.

### 5. Live wiring smoke and repository gates

1. PASS: after the invoking CLI exited, LaunchAgent com.localbench.ollama-gateway remained running and healthy; the listener readback was 127.0.0.1:11300 only.
2. PASS: OMP 18.4.2 made actual default and named-profile requests through the gateway to a fake Ollama; both returned ROUTED_OK, profile leases were attributed, streaming completed, and the separate direct fallback received zero requests. An injected five-minute expiry sent fake keep_alive:0 and /api/ps confirmed absence. No local inference occurred.
3. PASS: all 13 user profiles now target the gateway; OMP models ollama --json catalog readback succeeded for every profile (five catalog entries each). gateway remove --dry-run validated the reversible markers without changing them.
4. PASS: 390 unit tests; readiness (12 sections); claim discipline (3 passed, 0 failed, 4 skipped); Ruff; all 12 gateway mutations caught and restored.
   The full suite emitted one uninvestigated Python 3.14 ResourceWarning for an unclosed SQLite connection at `argparse.py:1769`; all 32 gateway tests passed with ResourceWarning escalated to error.
5. No golden was regenerated. No files were staged or committed, per the user instruction; commit-only cross-grade is therefore not applicable.
6. OUTAGE LIMITATION: with the isolated gateway stopped, OMP made no direct-fallback request, but the smoke harness had to terminate it after 60 seconds. A clean OMP exit/error latency is unverified. Final `localbench gpu --seconds 1 --json` observed an existing OMP process (profile=claude, PID 9819) holding an Ollama :11434 socket with 0/0 bytes in that sample; it was not restarted or killed. `localbench gateway status --json` reported two unowned residents (`qwen3.6:35b-mlx`, `qwen3.8:27b-mlx`); neither was unloaded. Restart existing OMP sessions before treating their cached routing as migrated. Direct non-OMP clients remain outside the policy.
7. REVIEW FOLLOW-UP: every established non-gateway Ollama socket now defers unload regardless of age, bytes, or GPU silence; inbound POST bodies are capped at 128 MiB and limited to a 30-second total read deadline. The persistent-socket and partial-body tests failed before the fix and pass afterward. Item 6's PID/socket observation is from the prior-turn snapshot, not a post-reload process query; no command targeted PID 9819.
8. Managed LaunchAgent reload was gated on `gateway status` reporting `active_requests: 0`; post-reload status is healthy/accepting on loopback with 13 profile routes, zero leases/requests, and qwen3.6 reported unowned. The OMP session PID 9819 was not restarted or killed.
