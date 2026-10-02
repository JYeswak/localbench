# OMP Ollama Residency Policy

**Status:** User-approved 2026-09-28; gateway implementation and 13-profile OMP/LaunchAgent configuration readback were verified, but live routing of already-running OMP processes is not. An isolated OMP 18.4.2 default/named-profile test routed built-in Ollama traffic through a fake backend; it is not proof of the current sessions' routes. The gateway code probes established non-gateway Ollama sockets and treats a detected socket as busy; quiet byte counters, GPU silence, and elapsed time are not idle signals. The probe is not synchronized with direct-client admission, so a new direct client can arrive after the check and before unload. Inbound POST bodies have a 128 MiB cap and a 30-second total read deadline in the implementation. At the earlier validation point, 390 tests passed, 12 mutations were caught, and readiness, claim discipline, and Ruff passed; those results are not banked same-generation live safety receipts. A managed gateway reload was gated on one `active_requests: 0` readback and then read back healthy; later proj-a observations of nine then eight gateway requests in flight demonstrate that the zero snapshot was not a persistent quiescence guarantee. OMP PID 9819 was not restarted/killed, and its cached route remains unverified until reload. The isolated outage smoke had no direct fallback but exceeded 60 seconds; a clean stopped-gateway OMP failure/exit and live restart recovery have not been proved.

## Problem

Before this policy change, `localbench keep` accepted `forever` as its default and translated it to Ollama's negative `keep_alive` value. Ollama documents negative `keep_alive` as indefinite residency. That made a resident model’s lifetime depend on a past command rather than current demand. The local audit log records CLI invocations, but Ollama’s `/api/ps` reports model identity and expiry, not the caller or reason.

## Decision

The approved behavior is to route every configured OMP Ollama provider through a loopback-only localbench policy gateway, attribute admitted traffic to its OMP profile, track in-flight calls and last completion per model, and unload a gateway-managed model after five minutes with no active request **only when unload safety is established**. Use Ollama’s native unload API (`keep_alive: 0`), not an OpenAI-compatible request field. The five-minute window follows Ollama’s documented default retention period. This is a policy target, not evidence that all already-running OMP sessions use the gateway or that direct-client unload races have been excluded.

The gateway is configured as a user LaunchAgent with a loopback bind. Configured profile routes are intended to fail closed when the gateway is unavailable rather than fall back to `127.0.0.1:11434`; this is not a live stopped-gateway guarantee for already-running OMP processes. The implementation forwards request/response bytes and streaming behavior without persisting prompts or completions; live route and failure behavior still require a same-generation receipt.
Installation edits profile files but does not reload already-running OMP processes; restart existing sessions before treating their requests as gateway-routed.

## Scope

- Configured user OMP profiles that can issue a local Ollama request are intended to route through the gateway after the relevant process loads the new configuration. The isolated localbench-owned benchmark child is excluded because its private `models.yml` exposes only the localbench provider, not built-in Ollama.
- Gateway status is intended to distinguish configured profiles, requests in flight, last request completion, next expiry, Ollama residency, and expiry/unload failures. A status snapshot is not proof that subsequent requests have drained.
- On gateway restart, persisted lease state must be recovered; stale in-flight markers may be cleared only after the prior gateway process is proven dead. The live stopped-gateway/restart transition remains unverified.
- The `localbench keep` CLI must reject unbounded retention and default to a finite lease. Explicit finite retention must be represented in status/audit; affected callsites and docs using `forever` must be migrated.
- Before unloading, the gateway must defer when an established non-gateway Ollama client is observed or safety is uncertain. The current socket probe is a point-in-time check, not exclusive control over new direct clients; it cannot guarantee that no direct request starts between probe and unload.

## Non-goals and boundary

- No host firewall, Ollama bind-address change, or patch to Ollama/OMP.
- Direct non-OMP callers that connect to Ollama and request negative `keep_alive` remain outside this OMP policy. The implementation must state this limitation; it must not claim host-wide enforcement.
- No changes to MLX/oMLX residency policy, model weights, cloud fallback, or measured inference payloads.
- Do not silently unload a pre-existing resident during profile migration. Surface a resident without a gateway lease as unowned/unknown; explicit reconciliation is a separate operator action.

Park is a separate consumer of gateway admission fences. Its SQLite fence serializes gateway requests for named models, but `runs/PARKED.json` is a separate, atomically replaced journal, not a transaction or lock shared by concurrent park/unpark operations. A digest-derived parked alias may be shared by several original tags or pre-exist outside the journal; matching digest does not establish alias ownership. Do not describe an unsynchronized concurrent park/unpark or an already-existing alias as safely recoverable solely because the journal is durable. Direct Ollama clients are outside the fence and can race the socket probe.

## Protocol and configuration constraints

Ollama’s native `/api/chat` and `/api/generate` support per-request `keep_alive`; its OpenAI-compatible `/v1/chat/completions` path does not document per-request `keep_alive`. Therefore the gateway must enforce expiry by scheduling a native unload after its own idle deadline, not by injecting an OpenAI request field. `/api/ps` is residency evidence, not ownership evidence.

Inbound POST bodies are capped at 128 MiB and must finish within a 30-second total client-read deadline; a timed-out or incomplete body is not forwarded. This deadline covers only the inbound body, not upstream inference or response streaming.

The active OMP profile directories contain `models.yml`; the local OMP kit’s fixture creates a provider `baseUrl` there. The isolated OMP 18.4.2 default/named-profile test in the dated implementation plan exercised the **built-in** `ollama` override through a fake loopback Ollama, beyond the custom-provider fixture. Profile and LaunchAgent readback does not prove current already-running sessions use that route, nor a clean failure/exit while the gateway is stopped. Keep those live proof gaps open.

## Lifecycle and safety invariants (required acceptance, not live certification)

1. Only loopback clients can connect to the gateway.
2. Once a session has loaded its migrated profile, its OMP local Ollama requests must not bypass the gateway; when the gateway is stopped, failure must remain visible and fail closed. Existing sessions with cached direct routing are not covered by configuration readback.
3. A model name is not eligible for idle unload while a gateway request admitted under that name is in flight; this does not establish digest-wide protection for another alias.
4. The five-minute idle deadline starts at completion of the last request, not request start.
5. No unload is reported as complete until Ollama confirms the model is no longer resident.
6. A detected established non-gateway Ollama connection must defer unload until it closes. Quiet nettop counters, GPU silence, or elapsed time are not request-idle signals; long prefills can be silent. A point-in-time socket probe cannot exclude a new direct client arriving after the probe, so this required invariant is not yet a proven live safety guarantee.
7. No prompt, response, API credential, or model request body is written to the lease/audit store.
8. A gateway restart must not convert an expired or unowned resident into a claimed lease without evidence. The stopped-gateway/restart case requires a live receipt; code recovery and a healthy post-reload snapshot do not establish it.
9. Neither policy lease storage nor the CLI may represent an infinite lease. The finite CLI parser alone does not prove this for internal numeric lease calls; `GatewayStore.set_manual_lease` and `keep_state` must reject nonfinite numeric values before this invariant can be claimed.

## Validation acceptance

- Current OMP resolves a profile-specific Ollama provider endpoint through the gateway using a disposable profile and fake loopback Ollama; a direct bypass is rejected or absent.
- Gateway preserves successful JSON and streaming response behavior, propagates upstream errors, and handles disconnected clients without leaving a permanent in-flight lease.
- Concurrent requests prevent unload until all finish; each completion resets the five-minute deadline; expiry unloads once and reports confirmed absence.
- A non-gateway client or uncertain client state defers unload; injected unload failure remains visible and does not claim success.
- Exercise direct-client arrival between the last external-client probe and native unload; either prevent unload under that race or keep this safety invariant uncertified. Also verify that an existing direct socket causes live unload refusal without disrupting that client.
- Gateway restart recovers expired leases and safely clears only stale process-owned in-flight state.
- Exercise a stopped gateway and subsequent restart with an actual OMP request and persisted lease: verify no direct fallback, observable failure/exit, stale process-owned request recovery, and expiry/unowned handling after restart. Profile readback or a single zero-active-requests snapshot does not satisfy this acceptance.
- `localbench keep forever` and equivalent negative/unbounded values are rejected; default/explicit finite keeps expire. Existing CLI callers/tests/docs are migrated.
- Exercise nonfinite numeric durations through the internal lease API as well as `localbench keep`; a CLI-only rejection does not establish the infinite-lease invariant.
- `localbench status` distinguishes gateway configuration, owned leases, active requests, actual residents, and unknown/unowned residents.
- User-profile edits are applied only after the isolated OMP test passes; each changed profile is read back and verified to target the gateway.
- For park integration, exercise concurrent park/unpark journal writes and pre-existing/shared digest aliases; require a serialized recovery path and proven alias ownership before claiming safe restoration or deletion. The gateway admission fence alone does not cover either case.

## Evidence and references

- `localbench/__main__.py` now accepts only finite keep durations (default 5m); `localbench/gateway.py` owns profile attribution, leases, guarded expiry, and gateway status. The former indefinite default is retained above as the problem statement.
- `localbench/audit.py`: CLI audit records the invocation and outcome, not OMP request ownership.
- `localbench/observe.py` and `/api/ps`: observation reports residents, but `/api/ps` has no owner/reason field.
- `~/Developer/omp-kit/tests/live/lib.mjs`: isolated custom-provider fixture writes a provider `baseUrl` in `models.yml`; alone it does not prove the built-in Ollama override. The separate OMP 18.4.2 default/named-profile smoke recorded in `docs/planning/2026-09-28-omp-residency-policy-plan.md:18-21,50-56` exercised that override against fake Ollama, not current already-running session routing.
- Ollama FAQ: <https://docs.ollama.com/faq> — five-minute default; negative native `keep_alive` means indefinite; `0` unloads; native per-request values override `OLLAMA_KEEP_ALIVE`.
- Ollama `/api/ps`: <https://docs.ollama.com/api/ps> — resident model and expiry fields.
- Ollama OpenAI compatibility: <https://docs.ollama.com/api/openai-compatibility> — compatibility endpoint contract; it does not expose the native per-request `keep_alive` control.
- Ollama native chat: <https://docs.ollama.com/api/chat> — native endpoint supports per-request retention.
- Lease pattern reference: <https://kubernetes.io/docs/concepts/architecture/leases/> — holder and renewal/expiry model; no Kubernetes dependency is proposed.
