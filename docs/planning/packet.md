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

localbench answers one question for one machine (Mac Studio M3 Ultra, 80 GPU cores, 512 GB): which omp and proj-b
side features should a local model serve, because it is measurably better there than today's route, and did a
change make it worse? It measures the path the user actually runs, records the machine's state next to every
number, and gates regressions against frozen, reviewed goldens for this host only. Decisions come from
same-item, same-invocation A/B receipts, not from vendor charts.

## 1. Problem
<!-- CHECK: PROBLEM -->
Re-scoped by the owner on 2026-09-30; epic bead `kit-side-model-mission-hgy` (it supersedes `kit-mission-gate-ad7`).
Local models are no longer evaluated as the MAIN coding agent: under this machine's contention none beat the
cloud main (bead record: qwen3.8:27b-mlx held 78% of 24 h GPU-seconds in `localbench report`, and the
default-profile auto-thinking classifier on qwen3.8 aborted 193/197 calls at omp's 4 s cap).
Local models serve only features proven better than today's route. "Better" means no answer-quality loss beyond
A/A noise on the same items, plus at least one measured win in p95 latency, availability (aborts/timeouts), GPU
cost or privacy.
Retained uses:
- proj-b-like decision work through Ollama 0.35's `POST /v1/systemone` with local `nimble:latest` (9B),
  `tev1:latest` (4B) and `tev1:0.8b`, in omp's judge role (auto-thinking classifier, find judgments) and in the
  proj-b CLI;
- memory: mnemopi extraction (today the smol role, `ollama/qwen3.8:27b-mlx`).
History: the 2026-09-22 spike (docs/evidence/receipts/2026-09-22-spike.md, non-proof per §8) traced the original
"every local model runs really slow" report to omp's 74,289-token default-profile prompt prefilling at
371.7–422.0 tok/s on the 27B dense model.
Main-agent evaluation is retired because of contention; every tier and all harness code stay as the regression suite.
Success condition: every feature that routes to a local model has a banked, current-generation receipt showing
"better" against today's route, a preset applied with omp readback and one-command rollback, a through-omp
acceptance leg, and `localbench doctor` GREEN; the omp-update refresh and the release watch have each run once
for real (§4).

## 2. Non-goals — what this is NOT
<!-- CHECK: NON-GOALS -->
- Not a cross-machine benchmark: numbers never compare across hosts; goldens are keyed by host_id.
- Not a model-quality leaderboard: decision and memory suites compare a local route with today's route on the
  same items; they will not publish rankings. The SWE-bench slice (lb-08) is retired.
- Not a main-agent evaluation: no local model is evaluated as the main coding agent (retired 2026-09-30).
- Not a general LLM benchmark suite: every workload is shaped by what omp or the proj-b CLI sends (recorded or
  captured traffic) or what omp does (real `omp -p` turns).
- Not an inference engine: localbench will not patch ollama, mlx-serve, or model weights.
- Loopback only for all local measurement and serving; a failed local call is a finding, never a fallback.
  Hosted decision service is allowed ONLY as an explicitly flagged comparison arm in an A/B, with its cost recorded, never
  as a fallback. No other remote or cloud endpoint.
- Will not tune omp internals: it chooses flags/providers the user can set, not omp source changes.
- Does not validate vendor numbers (mlx-serve's M4 Max chart, PonyExl3's M5 Max table) except by
  measuring the same software here.

## 3. State-of-the-art survey
<!-- CHECK: SOTA -->
Current pinned incumbent in docs/evidence/incumbents.md: ollama 0.34.4 (binary sha `bba8b79eac84ab09`),
`qwen3.8:27b-mlx` digest `5642e97495e1`, and omp 18.4.3 (binary sha `b72ee39b7feb2d59`,
read back by `localbench status --json` on 2026-09-29). The full default-profile replay fixture
`fixtures/omp/full.json` was recorded under omp 18.2.11 with the child overlay: historical request
shape, not the current profile. The 0.32.15 / 18.2.10 figures below are historical survey pins.
Every "faster" claim needs the receipt-generation incumbent live in the same invocation (A2), never
a remembered number.

Sources, with verdicts:
- **Historical survey — ollama 0.32.15** — adopted as incumbent and as a backend. Native MLX (nvfp4) runner. Prefix reuse verified
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
Work packet IDs live as beads `lb-01`…`lb-10` in `.beads/issues.jsonl`; the epic is
`kit-side-model-mission-hgy` (defined below; it superseded the main-agent mission gate `kit-mission-gate-ad7`
on 2026-09-30). The lb-XX packets stay as the harness and regression suite; their main-agent goals are retired
where noted. Legacy anchors are the spike files already in `localbench/`. Every packet's acceptance gate states
a positive observable, a planted negative, and a no-claim line (A10). Cross-cutting rules for chat-model goldens
and main-tier runs: a run is CONTENDED (non-proof, exit non-zero) when the sampler series shows a second
resident model or a GPU consumer other than the backend under test; the harness exits non-zero on any MUST
FAIL whether or not a golden exists. Side-model decision quality is deterministic and measured anytime; its
latency, availability and GPU cost are measured live with co-resident models under recorded load, which is
recorded, never used to void the run.

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
  prompt_tokens, date, profile, provider path, backend/model pins, and a reported tokenizer identity when available),
  written by `record`.
- **Oracle tests:** prompt_tokens within 2% when the recorded and run tokenizer identities match; a known different
  tokenizer on the same backend is VOID, not FAIL; absent identity preserves the existing backend comparison;
  turn-2 TTFT < cold TTFT.
- **Fixture manifest:** happy = lean; edge = full (74,289 tokens); adversarial = sidecar omp version differs from
  the running omp (replay refuses: GENERATION-MISMATCH).
- **Risk:** fixtures go stale when omp or skills change; the loaded context can be below the fixture size.
- **Acceptance gate:** positive: three TTFTs per fixture plus the sidecar comparison; planted negatives:
  (1) a warm replay with a changed first byte is as slow as cold; (2) a sample whose loaded context is below the
  sidecar prompt_tokens is marked VOID, never averaged; no-claim: replay excludes omp client overhead (lb-05) and
  a matching tokenizer identity does not establish an identical chat template.

### lb-05 — end-to-end omp tier
- **Goal:** real `omp -p` turns, default profile, lean flags, routed through the timing proxy so every LLM call
  is timed; answers are checked; "first" means cold KV for the system-prompt prefix.
  The main-agent ≤10 s first-turn target is retired (2026-09-30); the tier stays as a regression check.
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
- **Status: historical.** Closed on its 2026-09-23 mlx-serve evidence; mlx-serve as a main-agent backend is
  retired with main-agent evaluation (2026-09-30). The `MlxServe` backend stays in the regression suite.
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
- **Status: retired.** Closed won't-do 2026-10-01 (main-agent evaluation retired); kept below as history.
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

### Side-model mission (epic) — `kit-side-model-mission-hgy`

- **Goal:** every feature that routes to a local model on this Mac is proven better than today's route (§1),
  locked in through the CLI, and kept current across omp updates. Supersedes `kit-mission-gate-ad7` (closed
  superseded 2026-09-30); the main-agent paired study `kit-paired-agent-session-decision-4a2` closed won't-do
  with it, and its memory question moved to `kit-memory-study-vce`.
- **Child beads:** `kit-gateway-systemone-route-siv`, `kit-private-corpus-capture-c2q`, `kit-decision-tier-j4y`,
  `kit-omp-feature-map-b7s`, `kit-omp-update-refresh-o3e`, `kit-presets-switcher-r3l`, `kit-memory-study-vce`,
  `kit-auto-thinking-route-kmt`, `kit-proj-b-tasks-local-zmm`, `kit-release-watch-y5n`,
  `kit-ollama-autoupdate-verb-9ki`, `kit-unsound-pair-banking-cku`.
- **Kept infrastructure:** `kit-unknown-residency-fail-closed-qqj`, `kit-memory-final-state-oracle-g73`,
  `kit-mission-gate-ad7.1`, `kit-verify-live-omp-residency-policy-gs7`, `kit-unpark-omp-catalog-sfv`, `kit-ghr`,
  `lb-06`.
- **Required evidence:** per-feature receipts (banked, current-generation; quality within A/A noise on the same
  items plus at least one measured win) with a through-omp leg; live readback after every preset flip;
  `localbench doctor` GREEN; one real omp update absorbed by the refresh; one release-watch cycle.
- **Acceptance gate:** positive: `localbench features` shows every local route PROVEN with a current receipt;
  doctor exits 0; one real omp update was absorbed by the refresh (carried-forward and STALE routes both
  demonstrated); one release-watch cycle filed and screened a candidate. Planted negative: a local route with no
  receipt, a stale module hash, or a preset drifted by hand turns doctor RED. No-claim: proves only the listed
  features on this host with the pinned models/runtimes; not main-agent quality, not cross-host.
- **Failure/rollback:** a feature that fails its gate keeps today's route; write negative evidence. Never weaken
  a gate, use hosted decision service as a fallback, or bank a failed run as a golden.

## 5. Claim inventory
<!-- CHECK: CLAIM-INVENTORY -->
Registered in registries/claims.tsv before the README makes them. A historical fixture can support a dated,
documented observation but is not a banked, same-generation proof for `enforce=yes`. Unenforced rows may cite
historical sidecars for provenance; an enforced row requires a banked harness receipt for the claim's generation.
The 2026-09-30 re-scope retires the main-agent claims below (`moe_prefill_ratio`, `first_turn_under_10s`,
`mlx_serve_vs_ollama`, `swe_slice_latency`): they stay unenforced as history and the README must not make them.
Side-model claims are registered per feature, before the README makes them, under `kit-side-model-mission-hgy`.
- `lean_prompt_tokens` — the omp 18.2.11 default-profile fixture recorded 73,779 prompt tokens without lean flags
  and 11,433 with them. Status: documented, not a current-generation performance claim. Provenance:
  `fixtures/omp/full.meta.json` and `fixtures/omp/lean.meta.json` (lb-04).
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
Generation binding (B3) is **per exercised tier**: backend, model, host and macOS pins bind all rows;
child/agent overlays bind live-omp tiers; the recorded fixture binds replay. `golden.tier_keys` does
not stale a golden for an omp version/SHA change alone, although A/B arm drift checks do. Therefore
`localbench status` reporting CURRENT means exercised-pin agreement, **not** proof that a dated
first-turn wall time or behavior still holds on today's omp. Current-version speed/behavior claims
require current-version runs and an A/A null; a missing pin required by a tier is GENERATION-MISMATCH.

External evaluation-method cross-check (researched 2026-09-29; methodology only, not current-host results or new acceptance gates):
- Anthropic's agent-eval guide separates task, trial, grader, trajectory, and environment outcome; OpenAI's workflow guide starts with full traces, then repeatable datasets. This supports keeping structural conformance, answer/state-checked `e2e`, and multi-turn `mem`/`sess` evidence distinct. One passing tool schema is not general tool competence. ([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), [OpenAI](https://developers.openai.com/api/docs/guides/agent-evals))
- MLPerf Inference v6.1 Edge Agentic separates a deterministic BFCL v4 accuracy gate from recorded single-stream multi-turn replay, fixes served context at 32K, and reports TTFT, TPOT, and per-turn latency. Transfer the workload separation and context/latency reporting, not its scores, model, or edge hardware assumptions. ([MLCommons](https://mlcommons.org/2026/07/mlperf-inference-v61-edge-agentic/))
- Google DeepMind's FACTS suite separates parametric, search, multimodal, and grounded factuality, with public and held-out examples. Localbench's performance and checked-task receipts do not establish general factuality. ([FACTS](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/))
- Prior GPU work at [`gpu-optimization@f6f1b377`](https://github.com/JYeswak/gpu-optimization/tree/f6f1b377691ab73112cac7869a8e040cfd5e177e) includes direct/before-after GenAI-Perf results and rejected optimization trials (`benchmarks/20260221-post-optimization-COMPARISON.md`; `scripts/run-benchmarks.sh`). Reuse the explicit workload profiles and the willingness to reject regressions; treat all H200/RTX PRO 6000 measurements, CUDA/TP/NVLink/NCCL and kernel settings as hardware-specific, not M3 Ultra evidence.
- Prior agent work at [`local-agents@8169f373`](https://github.com/JYeswak/local-agents/tree/8169f3736277f424ccc972690c36a161518439d3) includes a create/edit/test workflow with an independent test run (`tests/validation/test_multi_step.py`). Its streaming-order test is `xfail(strict=False)` and skips errors (`tests/validation/test_streaming_ordering.py`), so neither XFAIL nor SKIP is positive conformance evidence.
- Transfer only methods that fit the pinned Mac path: controlled task/context, complete traces and end-state checks, repeatable same-generation comparisons, and explicit rejection/rollback records. Do not carry over absolute throughput/latency, discrete-VRAM budgets, CUDA/NVLink topology, or concurrent-server results as Apple Silicon claims.
- Do not import an earlier report's roll-up verdict as data. The 2026-02-22 GPU report says
  “No performance regressions” in its executive summary, but its per-model section records a
  12–36% qwen3-vl-30b throughput decrease and defers cause to a targeted rerun; it also rejects
  EAGLE3 and FP8 KV on those GPUs and notes a small `torch.compile` gain with a 50 s startup cost
  ([pinned report, lines 11–39 and 216–231](https://github.com/JYeswak/gpu-optimization/blob/f6f1b377691ab73112cac7869a8e040cfd5e177e/benchmarks/20260221-post-optimization-COMPARISON.md)).
  Those are *historical GPU* observations, not Mac estimates. Here, preserve per-leg single-stream
  results and total cold-start-to-verified-task wall; a decode win or a favorable roll-up does not
  outweigh a startup penalty, an unresolved regression, or a wrong final state.
- Before a new held-out coding task enters a selection campaign, require a known-good reference
  outcome to pass its grader and a feasible wrong-state specimen to fail it; otherwise an all-fail
  task may be a broken rubric, not a model result. Keep clean per-trial workspaces and code-based
  end-state grading, using traces to diagnose failures without grading arbitrary valid tool paths
  ([Anthropic's task/reference-solution guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
  The pinned `local-agents` multi-step test checks files and independently executes generated
  tests, but its missing-test path skips; if a localbench task requires tests, SKIP cannot mean PASS.
- Oracle audit on this host (2026-09-29, **offline**, not a new model measurement): `score_e2e_case("tool_read", ...)` formerly returned PASS for two assistant-only `4817` traces with **zero file reads**. A stored real `omp -p` trace (`runs/20260929T090453Z__ab_a1__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit/e2e.tool_read.{first,repeat}.omp.jsonl`) contains a successful `read` start/end and the matching file value. The corrected scorer now rejects the answer-only and mismatched/failed-tool controls while accepting both recorded real attempts; prior banked receipts retain their historical proof scope. Even this stronger two-task gate cannot establish general coding-agent reliability. The next live selection experiment needs independent trials with changing task data, verified tool outcomes, and latency conditional on success, not another isolated tok/s race; do not launch it while another OMP session is using the local GPU. This implements Anthropic's [outcome-versus-claim distinction](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) and OpenAI's [trace-first evaluation](https://developers.openai.com/api/docs/guides/agent-evals) without importing MLPerf's other-hardware scores.


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
"Publication" here means either (a) switching a user omp profile so a role or feature uses a local model, or
(b) a README sentence claiming a speed or capability.
Profile switches go through presets (`kit-presets-switcher-r3l`). Agents may switch a LIVE profile only to a
preset that just passed its gate, after a heads-up to the owner, with one-command rollback; the owner may switch
anything anytime. Test flips are always allowed.
Non-waivable clauses: the feature's gate passed (no answer-quality loss beyond A/A noise on the same items, plus
at least one measured win versus today's route); the claim's receipt is banked, same-generation and enforced in
registries/claims.tsv; hosted decision service, if present, is an explicitly flagged comparison arm only. Chat-model goldens
cited by a claim also need an uncontended run with an idle preflight; side-model latency legs record their
co-resident load instead. Enforcement today: scripts/check-claim-discipline.sh (pre-commit hook) machine-checks
only that each enforced row's proof file exists, is non-empty, is not a spike receipt, and contains the row's
expected substring when one is given. Proof semantics (verdict BETTER versus the baseline, model digest matching
the live route, module sha) are checked by `localbench features`/doctor (`kit-omp-feature-map-b7s`) once its
proof-contract fix lands. "Same-generation" is review-enforced until lb-06 adds the pin comparison to receipts —
until then a reviewer must diff the receipt's pins against docs/evidence/incumbents.md before any row flips to
enforce=yes.
Waivable only with a public, expiring, recorded waiver (owner, rationale, expiry, compensating controls) in
docs/evidence/waivers.md: a perf metric REGRESSED by less than twice its A/A-derived band.
Incomplete producer evidence blocks publication.

- **Post-cutover verification:** after a user-profile switch, read back the live profile and verify that an existing or explicitly restarted user OMP session uses the intended local route with no direct fallback. An isolated `child_env()` receipt does not prove the user session; failure leaves publication incomplete and requires the preset's one-command rollback.

## 10. Phase exit criteria
<!-- CHECK: EXIT-CRITERIA -->
Rule: no phase gate may claim a result whose transitive dependency closure contains an unresolved [OPEN].
Closed bead status alone is not phase evidence when its dependency closure still contains OPEN work; inspect the transitive blockers and required receipts before declaring a phase exit.
- **Phase A (planning).** Entry: kit installed (`scripts/init.sh` output). Exit: `scripts/check-readiness.sh`
  prints READY, §11 records an independent review, §12 signed.
- **Phase B1 (harness core: lb-01…lb-06, lb-10).** Entry: Phase A exit. Exit: incumbent and MoE goldens banked
  with A/A-derived bands; `localbench ab` receipt incumbent-vs-MoE with A/A null; all break-tests recorded for
  preflight, conformance exit status, golden compare, and claim-discipline hook. Re-run `lb-10` after `lb-01`…`lb-06`
  close against the resulting source and pinned generation; B1 exit requires no OPEN transitive dependency.
- **Phase B2 (lb-07 mlx-serve main) — retired 2026-09-30** with main-agent evaluation; `lb-07` is historical
  evidence. No entry or exit applies.
- **Phase B3 (lb-08 SWE slice) — retired;** `lb-08` closed won't-do 2026-10-01. No entry or exit applies.
- **Phase B4 (lb-09 monitors).** Entry: B1 exit (`lb-10`). Exit: a planted contention event reported by a monitor and the
  pane check's non-zero exit recorded.
- **Phase C1 (decision routes: `kit-auto-thinking-route-kmt`, `kit-proj-b-tasks-local-zmm`).** Entry: each route
  bead's dependencies closed (both need `kit-gateway-systemone-route-siv` and `kit-decision-tier-j4y`;
  auto-thinking also needs `kit-private-corpus-capture-c2q` and `kit-presets-switcher-r3l`). Exit, per route: a
  banked current-generation receipt with deterministic decision quality on the same items within A/A noise of
  today's route, at least one measured live win (p95 latency, availability, GPU cost or privacy) under recorded
  load, a through-omp leg for omp routes, and the preset applied with readback and one-command rollback.
- **Phase C2 (memory study: `kit-memory-study-vce`).** Entry: `kit-memory-final-state-oracle-g73` and
  `kit-presets-switcher-r3l` closed. Exit: a memory configuration that passes the G73 oracle with no quality loss
  beyond A/A noise versus today's qwen3.8 extraction and at least one measured win, with a through-omp leg and
  preset readback; if none qualifies, today's route stays and the result is a ledger row.
- **Publication (§9).** Entry: B1 exit (`lb-10`) and the published feature's C1 or C2 exit. Exit: release-gate
  clauses satisfied for the published feature.

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

## 13. 2026-09-29 reality check and bridge (planning amendment, not a release decision)

**2026-09-30 re-scope (the owner).** localbench no longer evaluates local models as the main coding agent; local
models serve only side features proven better than today's route (§1), tracked by `kit-side-model-mission-hgy`,
which supersedes `kit-mission-gate-ad7`. The main-agent bridge below (items 1–4, the paired study, the ≤10 s
memory-on target and the SWE lane) is retired; its harness code and tiers stay as the regression suite. The text
below is kept unchanged as the 2026-09-29 record.

**Answer:** not finished. A live CLI can attribute GPU activity and show model routes; the harness
measures real omp turns and A/A goldens. No configuration has yet met the *conjunction* of reliable
memory with the main model, ≤10 s cold tool use, sound same-invocation selection against the pinned
incumbent, an enforced current claim, and verified existing-session routing. This is a capability
gap, not a bead-count problem. Read-only `br`/`bv` at this snapshot: 55 issues, 34 closed, 4 open,
15 blocked, 2 in progress, `br ready --json` empty; a closed component bead is not a mission verdict.
No model was run for this review. `localbench status --json`, `localbench models --json`,
`localbench gpu --seconds 5 --json`, `localbench report --since 24h`, and saved receipts supplied
the runtime and measurement observations. Two cloud-model fresh-context agents independently read
source/receipts without inference; their observations are corroborating static review, not a live
cross-model grade.

| # | Testable promise (source) | Reality and evidence | Coverage / missing step |
|---|---|---|---|
| V1 | Explain GPU users, OMP feature routes, model freshness (AGENTS mission; README 5-6, 87-92) | **PARTIAL.** `localbench models --days 1` on 2026-09-29 listed 13 profiles and a current registry-checked Ollama Qwen3.8; mlx-serve, Splash, omp-tiny and fastembed showed **unknown** HF freshness, since their local artifacts have no repo-linked recorded commit. Their displayed installed identities (partial config digest or unknown) are separate from the upstream SHA and sourced `lastModified` date; neither local mtime nor the upstream SHA establishes installed freshness. `localbench report --since 24h` showed nonzero **response/request bytes** and labels each model as a sample-time resident, not per-request attribution. `gpu --seconds 5 --json` also found a foreign direct Ollama socket, not proof that it was generating. | `kit-chosen-model-freshness-proof-3gj` implements fail-closed provenance; selected-model freshness remains unknown where no recorded installed commit exists, not a current-release claim. `kit-ghr` covers one client-directory gap; `kit-verify-live-omp-residency-policy-gs7` owns live existing-session routing. |
| V2 | Finite gateway leases and no OMP direct fallback (README 137-153; residency policy) | **PARTIAL.** `localbench gateway status` on 2026-09-29 exited 0: healthy loopback gateway, 13 configured profiles, four in-flight requests, active finite qwen3.8 lease, `Ollama residency API: available`, and no unowned residents. The previous `KeyError: 'resident_models'` was reproduced red-first and fixed by rendering the producer's `ollama_state`; synthetic unknown/available/empty states and one causal mutation were checked. A later five-second `localbench gpu` sample no longer listed the foreign direct socket, but showed qwen3.8 at 61.9% GPU and active muse gateway traffic; a live existing-session route and no-direct-fallback claim remain unverified. | `kit-cli-status-report-truth-hcm` addresses only truthful CLI output; `kit-verify-live-omp-residency-policy-gs7` still requires live fail-closed routing. Do not restart another user's process to manufacture evidence. |
| V3 | Memory and main model both reliable in an interactive session (AGENTS mission; packet 245-263) | **PARTIAL.** `mem` plants randomized facts and `sess` drives 12 turns. A 2026-09-29 descriptive oMLX session completed 180/180 turns, yet its B legs had 5/9 and 4/9 aborted memory calls (`docs/evidence/receipts/ab-mlxserve-vs-omlx-qwen36-sess-descriptive-20260929.json`); completion is not recall. The prior oracle accepted unverified controls and zero extraction. The current synthetic good/bad exercise checks clean fresh-process recall for one-shot print mode (which emitted zero extractions in the 2026-09-23 receipt), all three retention boundaries per interactive session, and MUST VOID rejection. No post-change live memory run has validated the producer or banked a new generation. | `kit-memory-final-state-oracle-g73` remains open until live memory-on proof on the chosen pins; `kit-mission-gate-ad7` then requires successful extraction, zero failures and positive overlap in each interactive session. Keep the failed MLX smol trial a DROP, not a speed win. |
| V4 | ≤10 s cold tool-using first turn, fastest eligible local arm (packet 32-35, 255-259) | **UNPROVEN as a selection.** Current omp 18.4.3 Ollama Qwen3.6 A/A `docs/evidence/receipts/aa__ollama__qwen3.6_35b-mlx__20260929T082037Z.json` reports 5.0477/5.6795 s for the short tool-read task; the 2026-09-29 mlx-serve/oMLX A,B,A,B,A tool wall is WITHIN-NOISE (band 0.5293) and omits the pinned dense Ollama incumbent. The old golden is CURRENT for its exercised pins but its wall numbers are from older omp. The tool-read scorer was corrected offline; historical receipts retain their original oracle scope. | Existing `lb-02` → `lb-06` → `lb-10` → mission, plus chosen-model current conf/e2e/mem/sess and ≥2-pair pinned-incumbent A/B. No speed claim from the single A/A or from tok/s. |
| V5 | Regression gate and public claims tied to proof (README 3-6, 35-51; packet 332-345) | **PARTIAL.** A/A bands, sound receipts and per-tier pin comparison are implemented. README now limits its task-evaluation claim to two fixed OMP cases and says a golden is written only for a sound A/A pair; the two corresponding registry rows remain `enforce=no`. `sh scripts/check-claim-discipline.sh` still fails with zero `enforce=yes` rows. The consumer rejects MUST VOID without a golden and refuses to bank stable MUST VOID; this has synthetic good/bad coverage only. `golden.FAILING` still excludes VOID *metrics*, so a shell exit 0 alone does not certify a voided cold-first metric. `localbench status` CURRENT is not a current omp-version performance observation. | `kit-b12` remains in progress on the wider claim audit. Existing `kit-b6`, `kit-b3`, `lb-06`, `lb-10`, mission; publication waits for a current banked receipt, enforced row, known-good and known-bad comparison, and an explicitly non-VOID cold-first metric. |
| V6 | Agent-loop latency on five real coding issues (packet 208-217) | **NOT_STARTED for SWE.** `lb-08` remains open; `localbench/swe.py` is not shipped. The separate `eval varied` path preregisters two independently seeded read and edit trials, preserves subprocess file state/JSONL and re-scores trace-bound outcomes offline. Its dry-run listed four hashed cases and refused an unparked launch; a fake OMP subprocess changed an edit file and offline re-scoring rejected a wrong read despite a saved PASS label. After correcting a real nested-artifact rescore bug found in fresh-context review, a synthetic-only CLI campaign scored four of four cases PASS and offline rescore agreed; a planted flat-path regression failed both relevant tests. A synthetic contended no-model-call and a post-attempt snapshot failure preserved ERROR case evidence. None of these are a live OMP model campaign, a measured Mac success rate, or paired A/B. Active qwen3.8 gateway traffic still precludes a sound live case. | `kit-agent-outcome-varied-trials-iae` remains in progress until a parked live read/edit trial and paired-seed interleaving are proven. `kit-paired-agent-session-decision-4a2` uses that output; `lb-08` independently covers five pinned SWE-bench Lite issues. Neither is replaced by synthetic traces. |
| V7 | A measured run is isolated from another model (packet 90-111, 320-330) | **PARTIAL—offline fail-closed proof only.** A deterministic fake Ollama/MLX timeout below the inference GPU threshold reproduced unknown residency with an otherwise sound verdict; `_busy_check()` now refuses unknown preflight and `unsound()` makes any during-run `resident_unknown_samples` non-proof, distinct from CONTENDED. Known own-model, app-only load and server-down/no-process controls remain sound; a known competitor is CONTENDED. Two causal mutations were caught and restored. No live run or historical receipt has been rejudged under this code: the 2026-09-29 preflight found an active foreign OMP client and 70.3% qwen3.8 runner GPU. | `kit-unknown-residency-fail-closed-qqj` stays in_progress pending a scheduled idle, parked live break-test; `lb-10` still requires current sound B1 evidence. App/screen activity remains recorded, not vetoed. |

**Bridge, ordered by buyer outcome rather than easy fixes:**

1. **Prove or reject the mission, not a backend ranking.** Finish the existing B1 chain and live
   residency/unpark/client-route blockers. On one current pinned generation, run the chosen main
   and selected memory overlay together, then an interleaved ≥2-pair A/B with the incumbent present
   in that invocation. The selected `e2e.tool_read.first_wall_s` must be answer/tool-checked, cold,
   non-VOID and ≤10 s; all MUST and preregistered memory/session gates must pass. Inspect A/A
   spread and success-conditioned wall before "faster"; a wide band means UNKNOWN, not KEEP.
2. **Fix the evidence boundary before a profile change.** Isolated changing-value read/edit
   trials must validate the tool event and post-task file state, reject a memorized answer and
   distinguish wrong answer, tool error, timeout and aborted memory calls. Use independent task
   seeds and repetitions under fixed prompt/context/overlay pins. Record full trajectories but
   grade final state; publish task-specific success probability/uncertainty, wall **on successful
   trials** and full time to verified completion with failures/timeouts visible. The screen only
   eliminates bad candidates; sample size/precision and the quality noninferiority condition are
   declared before stage 2. Within the selected model, pair memory off/on with the child side
   route on that model; separately compare pinned incumbent against candidate with the same
   memory overlay, tasks and grader, one resident artifact per leg. Neither contrast substitutes
   for the other. Token throughput is a diagnostic, not the product verdict. These capability
   checks do not weaken the packet's original MUST, ≤10 s or SWE acceptance.
3. **Repair consumer-visible facts and isolation failures.** Make gateway text status agree with
   its JSON payload and handle unknown Ollama state; relabel `report` traffic as bytes and stop
   treating sample-time residency as per-request attribution. The synthetic unknown-residency
   path now rejects preflight and makes during-run unknown samples non-proof, with known-good,
   app-only and known-competitor controls; confirm the same boundary during a safely parked live
   window before a release claim. For the chosen model, show installed digest/commit and sourced
   publication/update date or explicitly "unavailable"; never call file mtime a commit proof.
4. **Use the existing real-agent lane.** Run `lb-08` only after B1 and the OrbStack prerequisite,
   separate agent wall from image/evaluation wall, and preserve the five-instance limitation. For
   any adopted user profile, perform §9's *post*-cutover readback in an existing or explicitly
   restarted OMP session. Unavailable busy-GPU windows, missing permission to restart, and
   undecided oMLX comparator scope leave these items open rather than creating a synthetic PASS.

The external methods above are design prompts, not Mac results: Anthropic/OpenAI separate traces
from outcome graders; MLCommons separates functional accuracy and agentic replay; the pinned
`gpu-optimization` work rejected optimizations that lost actual throughput, and `local-agents`
explicitly executed tests after edits. This packet preserves that separation. The 2026-09-29
initial inventory exposed untracked V1/V2/V3/V6/V7 work; the new Beads named in the table now
carry their own pre-state, final state, counterexample and no-claim line. Implementing only the
*previously* open/in-progress Beads would have left those gaps. No new Bead is itself completion
evidence; only a real trial and the existing mission gate can justify a selected profile.

The stronger evaluation lane has one frozen **exploratory / held-out** boundary. A changed prompt,
fixture, grader, tool set or overlay starts a new campaign identity rather than laundering a previous
failure into PASS. When the task requires generated tests, missing/skipped tests are failures,
not successful code edits (`local-agents` had a skip on missing generated tests). Report completed
trials by failure stage: model answer, tool execution, aborted memory, timeout, grader, or
independently observed infrastructure error. For repeated use, pass@k (one of k tries succeeds)
is not pass^k (every one of k jobs succeeds); never use the easier statistic to claim dependable
daily work. Include matched task seeds and observed CPU/app load per leg, but keep
everyday app activity as recorded
context rather than a new veto. These are stricter interpretations of the existing product
question, not imported H200/RTX performance targets or model-quality leaderboard scores.

Adversarial checks before any selection: an edit whose starting bytes already equal the target
is not a completed job; a memorized answer without a matching read is not a tool success; zero
successful memory extractions cannot pass by having zero *failed* extractions; a missing resident
API sample is not evidence of one-model isolation; and the short `OK` task cannot win a
tool-using-speed comparison. A noninferior candidate against a weak incumbent can still be unusable:
require every preregistered deterministic held-out MUST job to reach its verified state, expose
all failures, and report sampling uncertainty. If the sample cannot justify a broad reliability
statement, retain only the exact task-level result and leave the mission decision open.

**Independent pane-2 read-only review:** The paired live study must not run the candidate
Qwen3.6 main alongside incumbent qwen3.8 smol and call it sound: the standing rule voids any
concurrent *distinct* resident models. `workloads.child_flags` already pins both child roles to
the model under test; the new paired study must preserve that wiring and change one factor per
contrast. A different-model user profile is descriptive CONTENDED/VOID, not a qualifying leg;
do not switch profiles to manufacture proof. The paired-study Bead now depends
on the corrected memory oracle as well as varied task outcomes. `lb-07` and `lb-09` closed on their
historical backend/monitor evidence before `lb-10` reopened; neither status proves current B1
exit. The mlx-serve Qwen3.6 QIM condition is a branch check at mission closure, never an
unconditional blocker for Ollama. Pane 2's second pass found that a within-selected-model
memory-off/on contrast cannot establish quality noninferiority to the pinned incumbent; §4 and
`kit-paired-agent-session-decision-4a2` now require a distinct incumbent-versus-candidate outcome
contrast with each leg sound under the one-model rule. Pane 2 made no edits, tests or inference.
NTM marked pane 1's composer unsafe for a direct callback; pane 1 read both reviews from its
captured pane output and integrated these findings.

## 14. Frontier research decision (2026-09-29; no new model verdict)

The [dueling-wizards report](../../DUELING_WIZARDS_REPORT.md) records 60 first-round ideas
excluded from promotion (62 patterns total), two independently sourced 30-idea second-round
slates, blind opponent scores, and two later Luna-authored 30-idea critiques. Muse exhausted
its allowance before a third blind cross-review; both panes then routed to Luna. The user accepted
Luna's last review, which is labeled **same-model**, not disguised as a Muse vote. No
controlled localbench model trial, golden update or profile change resulted; other OMP
sessions' local-memory sidecars may still have used the GPU.

A later read-only `localbench status --json` snapshot observed OMP 18.4.4,
SHA16 `ca9b8832ea05299f`, while the reviewed incumbent index still names
18.4.3, SHA16 `b72ee39b7feb2d59`. Reconcile the pin and exercised receipt
generation before any model/recipe promotion; this observation alone is not
a model regression and does not authorize golden regeneration.

Luna's later operator slate also produced no present implementation Bead: Pi would need
separate gateway route ownership and an actual user target; local LoRA, offline packaging
and project-locked tools have no evidenced tie to the current OMP mission. The only
conditional Modelfile recipe idea duplicates upstream Ollama semantics and lacks a
named setting/task that would remedy the 73,779-token **historical OMP 18.2.11
full-request fixture** (not a current 18.4.3 measurement). `localbench create`
already builds controlled Safetensors derivatives. Do not invent a recipe
benchmark to create a winner.

A subsequent source audit rejected a new prompt/profile Bead. Historical
OMP 18.2.11 `full`/`lean` fixtures record 73,779/11,433 prompt tokens, but lean
also disables skills, rules, LSP and title and narrows tools; correctness with
memory on was not established. `lb-04`/`lb-05` already own replay/cold lean
measurement. OMP Code Mode is not wired for localbench's local provider, and
cache warming cannot satisfy a cold first turn. Keep the existing cold
memory-on verified tool-read ≤10-second acceptance in the G73/IAE → paired
path; request-shape ideation retries only after a sound failure and a
local-provider-compatible seam preserving the task's required capabilities.

The current `eval varied --mem-config fixtures/omp/child-config-mem.yml`
read-case path records same-trial trajectory/final state, `wall_s`, and proxy
`cached_tokens`, but its behavior `PASS` does not enforce coldness or ≤10 s.
The run was not exercised in this study. One acceptance item was added to
the existing paired-study Bead to join these same-leg fields with pin and
residency evidence: correct-but-cached is VOID for cold speed,
correct-and-cold-but-slow fails latency, and missing cache telemetry is
UNKNOWN without independent no-cache proof. No new scoring Bead or relaxed
performance gate.

An adversarial source review found that the varied read case can pass with
**no useful memory recall**, and OMP's bounded async shutdown may detach
retention work. A second acceptance item on that same Bead now limits the
≤10-second result to memory-*enabled* first-turn read latency, enumerates
required in-window memory calls and completion/abort evidence, and keeps
deferred retention outside that timing claim. Useful/reliable memory still
requires G73 plant/control/extraction, completed `sess` retention boundaries,
and a memory-dependent paired outcome under the same selected artifact,
overlay and pins. No live result or evaluator implementation was produced.

The separate B3 `lb-08` remains blocked by B1 (`lb-02` blocked, `lb-06` open,
`lb-10` open). A read-only OrbStack check found its daemon Running at 2.2.3,
but `swebench` 5.0.2 is not installed in the inspected Python environments and
the pinned dataset is absent from the configured cache. Docker evaluation and
all five live instances remain NOT RUN; the prior GPU/client sample is only a
point-in-time contention warning, not evidence about a future preflight.

The product boundary is operator action, not a second scoring flag. Contingent code-review
and cross-job procedural-memory tasks refine existing IAE/G73/paired work; they are not new
Beads. Saved-session resume is only an OMP interface: the installed source warns on pending
tools and can fall back to the launch directory if the saved workspace is missing, so it
cannot yet be called safe cross-process recovery. An inspected **new** action would let the
user revoke one explicitly user-saved project source and its proven descendants while
preserving unrelated sources and banks; extracted fact IDs and approved procedures are
not yet established user-owned targets.

OMP's agent-facing `memory_edit` does not directly edit fact rows (a colliding ID can
still match an eligible row in another bank); localbench only prunes harness-owned banks.
`memory save` requests asynchronous fact extraction, so its returned working-row ID is
not proof that a durable fact row exists. This is an API-gap hypothesis, not permission
to bypass the fact guard or assume a working-row forget preserves linked fact artifacts.

The same returned working-row ID is eligible for agent-callable `memory_edit`
update/forget at approval tier `read`; a new TUI-only revoke would not block that
existing path. Upstream must deny such tool mutations for protected explicit-save
sources at the storage boundary and prove trusted user origin; the current TUI
handler alone is not human authentication. Symlink aliases can produce distinct
project-bank hashes, so bank identity needs an explicit contract.

The pinned source audit locates authorization and project-bank resolution upstream in
OMP/Mnemopi, not in a direct localbench database wrapper. Existing source deletion
removes some linked derivatives, but its inspected cascade omits graph triples and a
pending asynchronous extraction can reinsert derived facts; these are unexecuted
code-path risks. Any user-only upstream contract must fence the writer and verify
every enabled recall surface while preserving unrelated banks and records.

Mem0, Zep and Letta document scoped memory update/delete operations, establishing
product precedent but not OMP user authorization or safe bank cleanup. Their live
documentation was not exercised; the dated comparison and limits are in the report.

Do not add a localbench implementation dependency before a supported upstream
user-only operation and bank/source identity contract exist. Its eventual acceptance
requires controlled good/bad/alternative-valid recall-plus-store-state proof; see
the report's execution map. Neither candidate weakens the B1/MUST/≤10-second/
one-resident-model gates or blocks the existing mission graph.

Pane 2's final source-checked product-fit audit found no implementable
**local-model-plus-memory benchmarking** winner: user-owned source revocation
is an adjacent, unimplemented control, not a measured model/memory outcome.
Pane 3 amended its unsent upstream proposal to deny `memory_edit` mutations
of eligible explicit-save rows at the storage boundary and to reject synthetic
TUI confirmation without a trusted host capability. That written contract is
not an implemented or authenticated endpoint. No speculative Bead or upstream
issue was filed; continue the existing G73/IAE → paired-study path while the
distinct operator-action search remains open.

A separate, unimplemented operator candidate would apply one receipt-backed
main+smol role tuple to one user-selected OMP profile, preserve an already
proven effective memory setup, and offer conflict-refusing rollback. Read-only
installed-package source suggests one `modelRoles` record can be written in
one atomic YAML replacement under a profile-specific root. This is not a
verified 18.4.4 binary contract: the incumbent index still pins 18.4.3,
YAML comments/formatting may change, complete role-map preservation and
CAS rollback are unproven, and no selected joint recipe or live-session
adoption proof exists. An independent same-model source challenge found no
eligible joint recipe or named user-profile deployment request, and no safe
cross-setting transaction/CAS rollback. Do not mutate a profile or create
an implementation Bead merely for a command sketch. Retry **one** post-mission
Bead only after stage-2 joint KEEP, a specific user-selected profile, a
disposable-root transaction/conflict/readback proof, and reconciled recipe
provenance; the sole direct edge would be `kit-mission-gate-ad7`. An OMP-only
version bump does not automatically stale unrelated goldens. An authorized
existing session or user-restarted session must separately prove the live
local route without fallback before any deployment claims VERIFIED.
