<!--
  Planning packet (working copy of templates/planning-packet.md).
  Machine-checked by scripts/check-readiness.sh. Keep the CHECK markers.

  Anti-ceremony (CHECKLIST.md A12):
  - Consumer: the agent executing localbench beads; the reviewer (§11); check-readiness.sh (A3).
  - Gate: A3 execution-readiness — missing field means NOT READY.
  - Defect class: plans with holes the agents find for you; optional fields nobody fills.
  - Delete when: superseded by a machine packet schema (none in this kit; keep this file).
-->

# Planning Packet — localbench

localbench answers one question for one machine: is omp, driven by a local model on this Mac Studio
(M3 Ultra, 80 GPU cores, 512 GB), fast enough to work in — and did a change make it slower? It measures
the path the user actually runs (omp → local server → model) with the machine's state recorded next to
every number, and it gates regressions against frozen, reviewed goldens for this host only.
"Working" means: a single command produces a receipt whose numbers a second reader can trust, and the
fastest local configuration is chosen from same-invocation A/B receipts, not from vendor charts.

## 1. Problem
<!-- CHECK: PROBLEM -->
The user runs omp against `ollama/qwen3.8:27b-mlx` and reports every local model "runs really slow".
The spike (docs/evidence/receipts/2026-09-22-spike.md, single runs, non-proof per §8) located the likely cost:
omp's default-profile request is 74,289 prompt tokens [Verified, High — recorded fixture], and the 27B dense
model prefilled at 371.7–422.0 tok/s here [Verified single run, Medium], so the prompt alone implies roughly
2.9–3.3 minutes before the first token on a cold cache [Inference, Medium — 74,289 ÷ 422.0 to ÷ 371.7 s].
The observed 267.7 s wall for a one-word answer was measured under the `claude` omp profile (70,715 tokens),
not the user's default profile. Decode (72.9 tok/s dense, 105.3 tok/s MoE) is not the suspected bottleneck.
The user is the only consumer; the machine is the only machine that matters (cross-machine numbers are
explicitly unwanted).
Success condition: a local configuration whose omp first-turn wall time on this host is ≤ 10 s for a
tool-using task, passing every MUST conformance case, proven by a same-invocation A/B receipt against the
pinned incumbent, and protected by a golden that fails loudly when it regresses.
The single most important outcome is the chosen configuration plus the regression gate that keeps it fast.

## 2. Non-goals — what this is NOT
<!-- CHECK: NON-GOALS -->
- Not a cross-machine benchmark: numbers never compare across hosts; goldens are keyed by host_id.
- Not a model-quality leaderboard: the SWE-bench slice measures agent-loop latency and resolves pass/fail on a
  handful of pinned instances; it will not publish accuracy rankings.
- Not a general LLM benchmark suite: every workload is shaped by what omp sends (recorded fixtures) or
  what omp does (real `omp -p` turns).
- Not an inference engine: localbench will not patch ollama, mlx-serve, or model weights.
- No remote or cloud endpoints: loopback only; a failed local call is a finding, never a fallback.
- Will not tune omp internals: it chooses flags/providers the user can set, not omp source changes.
- Does not validate vendor numbers (mlx-serve's M4 Max chart, PonyExl3's M5 Max table) except by
  measuring the same software here.

## 3. State-of-the-art survey
<!-- CHECK: SOTA -->
Incumbent pinned in docs/evidence/incumbents.md: ollama 0.32.15 (binary sha `eee609f0a6da58b9`),
`qwen3.8:27b-mlx` digest `5642e97495e1`, omp 18.2.10 (binary sha `acf06c76a4969558`), full default profile
request (fixtures/omp/full.json). Every "faster" claim is measured against this pinned incumbent live in the
same invocation (A2), never against a remembered number.

Sources, with verdicts:
- **ollama 0.32.15** — adopt as incumbent and as a backend. Native MLX (nvfp4) runner. Prefix reuse verified
  only for an identical rerun (spike: 3.6 s warm vs 102.3 s cold, claude profile); whether an intervening
  different request evicts it is unverified and becomes lb-03's A,B,A eviction probe.
- **mlx-serve 26.9.2** (tap commit `30f32ccc9f9f`) — adopt as candidate backend. [Maintainer claim] multi-entry
  prefix cache (2 GB default; SSD tier off unless `--prefix-cache-disk`), native Qwen MTP, OpenAI/Anthropic/
  Ollama APIs. Its M4 Max performance chart is non-proof here; measure it on this host.
- **Qwen3.6-35B-A3B** (ollama digest `e92a3e94bbca`; mlx-serve build HF commit `6122e2b2`) — adopt as the
  first candidate model: ~3B active parameters; spike prefill 2197.7 vs 371.7 tok/s at 28,393 tokens (5.9×,
  single run, non-proof).
- **PonyExl3 v0.3.0** (github.com/beamivalice/PonyExl3 HEAD `8e7fa6b`) — reject for this machine. Its value is
  quality per GB (EXL3 at 4 bpw); it ships no OpenAI-compatible server, so omp cannot reach it, and its own
  table reports 27B plain decode at 16.6 tok/s (M5 Max) where ollama measured 72.9 tok/s here (different
  hardware; not a same-invocation result). Retry condition: PonyExl3 ships an OpenAI-compatible server AND
  a local same-invocation run beats the MoE baseline prefill on this host.
- **llama.cpp** (standalone /opt/homebrew/bin/llama-server installed; mlx-serve also embeds llama.cpp b10472 for
  GGUF) — not evaluated; retry condition: mlx-serve and ollama both fail a MUST conformance case that
  llama-server's `--jinja` tool-call parser is documented to handle.
- **mlx-lm server** (mlx-lm 0.31.x, installed in ~/Developer/PonyExl3/.venv) — not evaluated; deferred because
  mlx-serve covers the same MLX kernels with a documented multi-entry prefix cache and omp integration. Retry
  condition: mlx-serve fails a MUST conformance case, or lb-07 shows mlx-serve slower than ollama on the same model.
- **SWE-bench harness** — adopt pinned: swebench 5.0.2 on PyPI, dataset `princeton-nlp/SWE-bench_Lite`
  commit `6ec7bb89b934`, run through OrbStack 2.2.3, for the agent-loop tier only.
- **llmprobe** (mlx-serve's benchmark driver) — reject as the harness: it measures the server, not the omp
  path; retry condition: it gains a replay-of-recorded-requests mode.

## 4. Work packets
<!-- CHECK: PACKETS -->
Packet IDs live as beads `lb-01`…`lb-10` in .beads/issues.jsonl. Legacy anchors are the spike files already
in `localbench/`. Every packet's acceptance gate states a positive observable, a planted negative, and a
no-claim line (A10). Cross-cutting rules every packet obeys: a run is CONTENDED (non-proof, exit non-zero)
when the sampler series shows a second resident model or a GPU consumer other than the backend under test;
the harness exits non-zero on any MUST FAIL whether or not a golden exists.

### lb-01 — machine-state receipts, preflight, contention detection
- **Goal:** every run records host fingerprint, power, thermal, memory pressure, swap, GPU utilization, top
  processes, resident models on every local server (ollama `/api/ps`, mlx-serve `/v1/models`), and powermetrics
  when the sudoers grant exists; runs refuse to start on a busy machine and are marked CONTENDED if contention
  appears mid-run.
- **Anchors / target files:** localbench/sysstats.py (Sampler gains a resident-model probe),
  localbench/__main__.py (`preflight`, contention verdict), scripts/install-sudoers.sh.
- **Oracle tests:** `localbench stats` JSON has host_id, power.source, live.gpu_device_pct, live.resident_models;
  samples.jsonl rows carry resident models.
- **Fixture manifest:** happy = idle desktop; edge = no sudoers grant (PowerSampler reports available=false);
  adversarial = GPU saturated by another model during preflight; ioreg keys unparseable; a second model
  loaded mid-run.
- **Risk:** ioreg utilization keys change across macOS builds; preflight must fail closed when absent.
- **Acceptance gate:** positive: summary.json carries system.before/after/during/preflight and a contention
  verdict; planted negatives: (1) with a second model generating in a loop, `localbench run` exits non-zero with
  "preflight refused"; (2) with the GPU signal unavailable, preflight refuses with "GPU signal unavailable";
  (3) a second model loaded mid-run marks the run CONTENDED and exits non-zero; no-claim: an idle sampler
  does not prove the absence of non-GPU contention (disk, network).

### lb-02 — conformance tier (MUST/SHOULD)
- **Goal:** prove the backend+model does what omp depends on: structured tool calls, usage reporting,
  no silent truncation at 64k, incremental streaming; greedy determinism as SHOULD.
- **Anchors / target files:** localbench/workloads.py `conformance`, localbench/__main__.py exit status,
  docs/evidence/DISCREPANCIES.md (every XFAIL listed; compare treats an unlisted XFAIL as FAIL).
- **Oracle tests:** each case returns PASS/FAIL with captured shape; tool-call shape is a structural golden.
- **Fixture manifest:** happy = single `read_file` tool; edge = 64k prompt vs 8k token ratio;
  adversarial = ollama loaded with `OLLAMA_CONTEXT_LENGTH=8192`.
- **Risk:** a model that answers from memory instead of calling the tool looks like a backend bug.
- **Acceptance gate:** positive: all MUST cases PASS for the chosen config; planted negatives: (1) with the
  context forced to 8192, `conf.no_truncation_64k` FAILs and the run exits non-zero with no golden present;
  (2) `UPDATE_GOLDENS=1` refuses to bank a run with a MUST FAIL that DISCREPANCIES.md does not list;
  no-claim: one tool schema passing does not prove every omp tool schema parses.

### lb-03 — micro tier
- **Goal:** engine speed at omp-relevant sizes: decode, cold prefill 1k/8k/32k, prefix-cache warm TTFT, and an
  A,B,A eviction probe (does an intervening different request evict the cached prefix?).
- **Anchors / target files:** localbench/workloads.py `micro`, localbench/client.py, golden schema (median AND
  spread per metric).
- **Oracle tests:** server-reported prompt_tokens and streamed timings; median and min/max of ≥3.
- **Fixture manifest:** happy = deterministic synthetic corpus (seeded); edge = 1k prompt (jitter-dominated);
  adversarial = nonce-prefixed prompts so no cache can serve them.
- **Risk:** reasoning models emit reasoning tokens first; TTFT counts the first token of any kind
  (reasoning, content, or tool call) — documented, applied identically to every backend.
- **Acceptance gate:** positive: medians with spread recorded; eviction probe verdict recorded per backend;
  planted negative: removing the nonce collapses cold TTFT (proves the nonce is what defeats the cache);
  no-claim: synthetic code text is not omp's prompt (lb-04 covers that).

### lb-04 — replay tier
- **Goal:** replay omp's exact recorded request bodies (fixtures/omp/lean.json, full.json) cold, warm, and
  as turn 2, against the model under test.
- **Anchors / target files:** localbench/workloads.py `replay`, localbench/proxy.py, `localbench record`, and a
  committed provenance sidecar per fixture: fixtures/omp/<label>.meta.json (omp version + binary sha, recorded
  prompt_tokens, date, profile, provider path, backend/model pins), written by `record`.
- **Oracle tests:** replay prompt_tokens within 2% of the sidecar's recorded count; turn-2 TTFT < cold TTFT.
- **Fixture manifest:** happy = lean; edge = full (74,289 tokens); adversarial = sidecar omp version differs from
  the running omp (replay refuses: GENERATION-MISMATCH).
- **Risk:** fixtures go stale when omp or skills change; the loaded context can be below the fixture size.
- **Acceptance gate:** positive: three TTFTs per fixture plus the sidecar comparison; planted negatives:
  (1) a warm replay with a changed first byte is as slow as cold; (2) a sample whose loaded context is below the
  sidecar prompt_tokens is marked VOID, never averaged; no-claim: replay excludes omp client overhead (lb-05).

### lb-05 — end-to-end omp tier
- **Goal:** real `omp -p` turns, default profile, lean flags, routed through the timing proxy so every LLM call
  is timed; answers are checked; "first" means cold KV for the system-prompt prefix.
- **Anchors / target files:** localbench/workloads.py `e2e`, `ensure_localbench_model`, localbench/proxy.py
  (TTFT stamped only on a non-empty content/reasoning delta or a tool-call delta — the client.py rule).
- **Cold mechanism:** the backend is re-isolated immediately before the e2e tier (ollama: unload + reload;
  mlx-serve: server restart without `--prefix-cache-disk`); oracle: the first proxied call of task 1 reports
  cached_tokens 0 or absent, otherwise the e2e.first metric is VOID.
- **Oracle tests:** answer checks (`OK`, file-read value 4817); omp_calls.jsonl rows per call.
- **Fixture manifest:** happy = "Reply OK"; edge = tool round trip (read then answer); adversarial = model
  that never calls the tool (answer check fails).
- **Risk:** the localbench provider fails `omp -p` resolution when its entry carries `apiKey:` (ledger UNKNOWN row,
  2026-09-22); the managed block in models.yml omits it and must stay correct; a proxy TTFT stamped on an empty
  delta skews per-call timing.
- **Acceptance gate:** positive: first/repeat wall times with correct answers and cold-verified first call;
  planted negatives: (1) a wrong expected answer marks e2e.*.correct FAIL; (2) a pre-warmed cache makes the
  first call report cached tokens and e2e.first is marked VOID; no-claim: two short tasks do not represent a
  long coding session.

### lb-06 — goldens, A/A-derived tolerance, generation binding, same-invocation A/B
- **Goal:** tolerance bands come from a measured A/A null (same config run twice), not guesses; compare refuses
  goldens from another generation (backend version/binary, model digest, omp version, macOS build); an
  `ab` command runs incumbent and candidate back-to-back in one invocation with an A/A null; golden provenance
  records preflight state and `--allow-busy`, and UPDATE_GOLDENS refuses non-idle or CONTENDED runs.
- **Anchors / target files:** localbench/golden.py, localbench/__main__.py.
- **Oracle tests:** a golden whose provenance pins differ produces GENERATION-MISMATCH, never PASS; every banked
  golden's tol values carry a pointer to the A/A receipt that derived them.
- **Fixture manifest:** happy = identical config rerun; edge = metric exactly at the band edge;
  adversarial = golden edited to a 30% better value (must REGRESS); golden whose tol lacks A/A provenance.
- **Risk:** A/A spread on a desktop with other apps is wide; bands become too loose to catch anything.
- **Acceptance gate:** positive: `localbench ab` receipt with A/A spread and A/B ratio per metric; planted
  negatives: (1) the doctored golden REGRESSes and the run exits 1; (2) a golden with a different backend version
  reports GENERATION-MISMATCH; (3) a golden whose tol lacks an A/A pointer fails compare; no-claim: an A/B on this
  host says nothing about other hosts.

### lb-07 — mlx-serve backend and omp provider
- **Goal:** serve the candidate through mlx-serve, reach it from omp as `mlx-serve/<id>`, and measure it with
  every tier against ollama in the same invocation, with cold state equalized (lb-05 cold mechanism).
- **Anchors / target files:** localbench/backends.py `MlxServe`, ~/.omp/agent/models.yml `mlx-serve` provider
  (static model entries), the server's real context length (no fallback constant).
- **Oracle tests:** `omp -p --model mlx-serve/<id>` exits 0 with the correct answer.
- **Fixture manifest:** happy = MoE 4-bit build; edge = server already running on :11234 (refuse);
  adversarial = a provider entry with `apiKey:` on the localbench provider (reproduces "Model not found").
- **Risk:** mlx-serve model ids differ from directory names; static entries drift; mlx-serve's in-process
  prefix cache survives between tiers.
- **Acceptance gate:** positive: mlx-serve golden + A/B receipt vs ollama MoE; planted negative: stopping mlx-serve
  makes `omp -p --model mlx-serve/<id>` fail loudly (no silent fallback to another provider); no-claim: one model
  on mlx-serve does not rank the engines for other models.

### lb-08 — SWE-bench Lite slice (agent-loop latency)
- **Goal:** five pinned SWE-bench Lite instances solved by `omp -p` with the chosen config; record wall time,
  LLM calls, tokens, and resolved/unresolved from the official harness on OrbStack.
- **Anchors / target files:** new localbench/swe.py; swebench 5.0.2; dataset commit `6ec7bb89b934`.
- **Oracle tests:** swebench evaluation report per instance.
- **Fixture manifest:** happy = instance with a small single-file fix; edge = instance with a long test file;
  adversarial = empty patch (must be unresolved).
- **Risk:** x86 images under Rosetta dominate wall time; evaluation time is separated from agent time.
- **Acceptance gate:** positive: per-instance agent wall + resolved flag; planted negative: the empty-patch
  prediction is reported unresolved; no-claim: five instances are a latency workload, not an accuracy score.

### lb-09 — monitors in ntm session omp-test
- **Goal:** watcher agents tail runs/LATEST/progress.jsonl and the sampler, flag contention and regressions, and
  never run a local model while a run is active.
- **Anchors / target files:** docs/MONITOR.md (watcher brief), scripts/check-monitor-pane.sh (inspects a pane's
  omp process args/env and exits non-zero when its model provider is local), ntm session `omp-test`.
- **Oracle tests:** a planted contention event appears in a monitor's report; the pane check's exit status.
- **Fixture manifest:** happy = idle run; edge = run ends mid-watch; adversarial = a local-model omp pane
  starts generating during a run.
- **Risk:** a monitor pane on a local model contaminates the measurement it watches.
- **Acceptance gate:** positive: monitor names the contention with timestamps; planted negative:
  scripts/check-monitor-pane.sh exits non-zero for the pane running `ollama/qwen3.8:27b-mlx`; no-claim: monitors
  observe, they do not grade.

### lb-10 — gate break-tests
- **Goal:** each gate (preflight, conformance exit status, golden compare, claim-discipline hook including the
  new spike-receipt exclusion) is broken on purpose once and the loud failure recorded (B14 at day-1 cost).
- **Anchors / target files:** docs/evidence/break-tests.md; scripts/check-claim-discipline.sh (rows with
  enforce=yes may not cite docs/evidence/receipts/*spike* — a gate strengthening, recorded with two-direction
  evidence per B7).
- **Oracle tests:** each break-test records command, planted fault, observed failure text.
- **Fixture manifest:** happy = n/a by design; edge = fault at threshold; adversarial = fault that should be
  caught by two gates.
- **Risk:** break-tests that only exercise the happy failure path.
- **Acceptance gate:** positive: one recorded loud failure per gate; planted negative: the break-test itself is the
  planted negative; no-claim: a gate that catches one planted fault may still miss others.

## 5. Claim inventory
<!-- CHECK: CLAIM-INVENTORY -->
Registered in registries/claims.tsv before the README makes them. All start `planned` unless a spike receipt
exists (then `documented`, which is below `validated` and cannot appear in the README as proven). No row points
its proof_path at the spike receipt; proof slots name the future banked harness receipts.
- `lean_prompt_tokens` — lean omp flags cut the default-profile prompt from 74,289 to 11,650 tokens.
  Status: documented. Proof slot: fixtures/omp/lean.meta.json + full.meta.json sidecars (lb-04).
- `moe_prefill_ratio` — Qwen3.6-35B-A3B prefills ≥5× faster than the qwen3.8 27B incumbent on this host.
  Status: planned. Proof slot: `localbench ab` receipt with A/A null (lb-06).
- `first_turn_under_10s` — lean omp + chosen local config answers a tool-using task in ≤10 s wall, first turn.
  Status: planned. Proof slot: e2e golden with cold-verified first call (lb-05, lb-06).
- `must_conformance` — the chosen config passes every MUST conformance case. Status: planned. Proof slot: golden
  conformance block (lb-02).
- `mlx_serve_vs_ollama` — outcome unknown; registered so either result is reported. Status: planned. Proof slot:
  lb-07 A/B receipt.
- `swe_slice_latency` — per-instance agent wall time on five pinned SWE-bench Lite instances. Status: planned.
  Proof slot: lb-08 receipt.

## 6. Evidence design
<!-- CHECK: EVIDENCE-DESIGN -->
Evidence classes: conformance verdicts with captured shapes; engine timings (median and spread of ≥3,
nonce-cold); replayed omp requests; real omp turns with checked answers; same-invocation A/B with A/A null;
SWE-bench harness reports.
Receipt format: `runs/<UTC stamp>__<backend>__<model>/summary.json` with provenance — localbench commit
(+`-dirty`), backend version and binary sha256, model digest or HF commit, omp version, macOS version+build,
host id (the "worker" is the backend+model instance named in the fingerprint), preflight result and
`--allow-busy` flag, contention verdict, sampler and powermetrics summaries, every raw sample in samples.jsonl.
A receipt cited by a claim is banked (copied) into docs/evidence/receipts/ and committed; runs/ itself is
gitignored scratch.
Generation binding (B3): a golden, fixture sidecar, or receipt whose pins differ from the current run's pins is
from another generation and is not evidence for it; compare reports GENERATION-MISMATCH instead of PASS (lb-06).

## 7. Honesty machinery
<!-- CHECK: HONESTY-MACHINERY -->
The negative-evidence ledger lives at docs/evidence/NEGATIVE_EVIDENCE.md with the kit's row schema
(hypothesis, A/B, A/A null, verdict, retry predicate, lesson); the pre-commit hook lints every staged row and
blocks weasel retry predicates. Demotion rules are docs/evidence/demotion-rules.md (D1–D7); D6 proof expiry is
tightened to "any backend, model, omp, or macOS pin change" because a generation change invalidates speed
proofs faster than 90 days. Ledger preflight: a REJECT without a same-invocation A/A null is recorded as
VOID-<class>, not REJECT. Known conformance divergences are listed in docs/evidence/DISCREPANCIES.md; an XFAIL
not listed there is a FAIL. Resurrection cadence: rejected rows are re-audited whenever a pinned component
changes version, and at least monthly.

## 8. Proof taxonomy [PROVISIONAL]
<!-- CHECK: PROOF-TAXONOMY -->
Proof categories: (P1) same-invocation A/B receipt with A/A null; (P2) golden compare receipt from an idle
preflight and an uncontended run, median of ≥3; (P3) conformance verdict with captured shape; (P4) real omp turn
with a checked answer and a cold-verified first call; (P5) SWE-bench harness report for a pinned instance.
Non-proof classifications: a single unrepeated run (including every number in the 2026-09-22 spike receipt);
any run with `--allow-busy`, a failed preflight, or a CONTENDED verdict (a second resident model or a GPU
consumer other than the backend under test anywhere in the sampler series); vendor/maintainer numbers from
other hardware (mlx-serve's M4 Max chart, PonyExl3's M5 Max/M1 Max tables); a model listed by `omp models`
(listing ≠ resolvable in `omp -p`); omp runs under a profile other than the one the claim names; any sample
whose loaded context is below its prompt size; any number from before a pin change.

## 9. Release gate
<!-- CHECK: RELEASE-GATE -->
"Publication" here means either (a) switching the user's omp config (default or smol role) to a local model,
or (b) a README sentence claiming a speed or capability.
Non-waivable clauses: every MUST conformance case PASS for the configuration being published; the claim's
receipt is banked, same-generation, uncontended, and enforced in registries/claims.tsv; the preflight of the
cited run was idle. Enforcement today: proof existence and content are machine-checked by the pre-commit hook;
the spike-receipt exclusion becomes machine-checked in lb-10; "same-generation" is review-enforced until lb-06
adds the pin comparison to receipts — until then a reviewer must diff the receipt's pins against
docs/evidence/incumbents.md before any row flips to enforce=yes.
Waivable only with a public, expiring, recorded waiver (owner, rationale, expiry, compensating controls) in
docs/evidence/waivers.md: a perf metric REGRESSED by less than twice its A/A-derived band; the SWE slice not yet run.
Incomplete producer evidence blocks publication.

## 10. Phase exit criteria
<!-- CHECK: EXIT-CRITERIA -->
Rule: no phase gate may claim a result whose transitive dependency closure contains an unresolved [OPEN].
- **Phase A (planning).** Entry: kit installed (`scripts/init.sh` output). Exit: `scripts/check-readiness.sh`
  prints READY, §11 records an independent review, §12 signed.
- **Phase B1 (harness core: lb-01…lb-06, lb-10).** Entry: Phase A exit. Exit: incumbent and MoE goldens banked
  with A/A-derived bands; `localbench ab` receipt incumbent-vs-MoE with A/A null; break-tests recorded for
  preflight, conformance exit status, golden compare, and the claim-discipline hook (lb-10 closed).
- **Phase B2 (lb-07 mlx-serve).** Entry: B1 exit + model downloaded at the pinned HF commit. Exit: mlx-serve golden,
  A/B receipt vs ollama MoE, `omp -p --model mlx-serve/<id>` answer-checked.
- **Phase B3 (lb-08 SWE slice).** Entry: B1 exit + OrbStack daemon up. Exit: five-instance receipt.
- **Phase B4 (lb-09 monitors).** Entry: B1 exit. Exit: a planted contention event reported by a monitor and the
  pane check's non-zero exit recorded.
- **Publication (§9).** Entry: B1 exit, plus B2 exit iff the published configuration involves mlx-serve. Exit:
  release-gate clauses satisfied for the published config.

## 11. Independent review
<!-- CHECK: REVIEW -->
Reviewer: `PacketReviewer`, a separate reviewer agent with a fresh context and no authorship stake,
2026-09-22. Method: read-only static review of this packet, CHECKLIST.md, every evidence file, claims.tsv,
AGENTS.md, the DoD, the lb-XX beads, and all spike source files, hunting for believable-only acceptance,
unsupported claims, code/packet contradictions, measurement confounds, and phase-gate holes; no model or
inference commands were run. Caveat: the reviewer runs under the same orchestrating session and likely the same
model family as the author — independent context, not independent lineage. Full findings banked at
docs/evidence/reviews/2026-09-22-packet-review.json (14 findings: 2 BLOCKER, 8 MAJOR, 4 MINOR; verdict "yes, conditionally, once BLOCKERs are fixed").

What changed as a result:
- BLOCKER 1 (e2e "first" not guaranteed cold, unfair to the mlx-serve A/B): lb-05 now defines "first" as cold,
  re-isolates immediately before the e2e tier, verifies the first proxied call reports no cached tokens, and VOIDs
  e2e.first otherwise; lb-07 equalizes cold state by restarting mlx-serve without `--prefix-cache-disk`.
- BLOCKER 2 (release gate had no machine form; a claim row pointed at the non-proof spike receipt):
  `lean_prompt_tokens` now points at the lb-04 sidecar; §9 states which clauses are machine-checked today and which
  are review-enforced; lb-10 adds a machine exclusion of spike receipts to check-claim-discipline.sh.
- MAJORs: §1 now tiers its numbers and names the profile confound; lb-04 gains committed fixture sidecars and a
  loaded-context VOID rule; lb-01 gains resident-model polling, a GPU-signal-absent refusal, and a mid-run
  CONTENDED verdict (also added to §8 non-proof); lb-02/lb-06 now exit non-zero on any MUST FAIL without a golden,
  refuse to bank non-idle/contended runs, and require A/A provenance on every tolerance; the proxy TTFT rule is
  aligned with the client's; B1 exit now includes the claim-discipline break-test.
- MINORs: lb-09's planted negative is now a script exit status, not a self-rejection; "single-slot" is relabeled
  unverified and becomes lb-03's eviction probe; Publication entry is scoped to B2 only when mlx-serve is published.

## 12. Execution sign-off
<!-- CHECK: SIGN-OFF -->
I sign off — Phase A of docs/CHECKLIST.md is green; this plan is execution-ready. Signed: omp agent
(anthropic/claude-opus-5-5) acting for the owner in the 2026-09-22 session, 2026-09-22. The sign-off covers the plan
as revised after the §11 review; the owner's countersignature is invited and any objection reopens Phase A.

Amendment 2026-09-22 (same session, no scope change): lb-05 risk, lb-07 fixture manifest and planted negative
corrected after the ledger's same-day resurrection and demotion — discovery-only providers do resolve in
`omp -p`; the observed failure is specific to the localbench provider with `apiKey:` (UNKNOWN row).
