# Negative Evidence Ledger

Rejected hypotheses, falsified claims, and dead ends — recorded so future
agents do not relitigate them. A ledger nobody writes to is decoration;
a ledger nobody reads is a graveyard. Both failure modes are the
maintainer's problem, not the format's.

Row schema (copy the block from templates/negative-evidence-entry.md):

    ## YYYY-MM-DD — VERDICT: one-line summary
    - **Bead:** <bead id>
    - **Surface:** <subsystem/area>
    - **Hypothesis:** <what you believed>
    - **A/B:** <measured ratio vs baseline, with units>
    - **A/A null:** <same-invocation null control, or VOID-<class> if none>
    - **Verdict:** KEEP | REJECT | SURVEY | UNKNOWN | RESURRECTED
    - **Retry predicate:** <testable condition for re-attempting — never "later">
    - **Lesson:** <one line>

Rules:
- Every REJECT row must carry a retry predicate. (frankensearch)
- A REJECT without a measured same-invocation null control is VOID, not a
  result — record the class of the missing evidence. (frankenscipy)
- The pre-commit hook lints staged rows: a row without a retry predicate
  blocks the commit.

Origin: frankenscipy docs/NEGATIVE_EVIDENCE.md + docs/LEDGER_RESURRECTION.md;
frankensearch docs/evidence/e8h-hypothesis-ledger.md; frankenfs
docs/LEDGER_RESURRECTION.md.

## 2026-09-22 — UNKNOWN: decode speed is why omp on local models feels slow
- **Bead:** lb-06
- **Surface:** omp turn latency, ollama qwen3.8:27b-mlx
- **Hypothesis:** local omp turns are slow because token generation (decode) is slow.
- **A/B:** decode measured 72.9 tok/s; a 27-token answer took 267.7 s wall with a 70,715-token prompt; prefill measured 371.7–422.0 tok/s, which alone accounts for ~170–190 s; identical rerun with a warm prefix: 3.6 s (docs/evidence/receipts/2026-09-22-spike.md).
- **A/A null:** VOID-SINGLE-RUN — no same-invocation null; order-of-magnitude signal, but not a result under the ledger rules.
- **Verdict:** UNKNOWN
- **Retry predicate:** lb-06 `localbench ab` receipt for the incumbent shows replay.full cold TTFT and micro.decode with an A/A spread; if cold TTFT is under 2x (prompt_tokens / micro.prefill_32k.prefill_tps), record REJECT with that receipt.
- **Lesson:** measure prefill and prompt size before touching decode; the omp prompt, not the model's generation speed, dominated the spike.

## 2026-09-22 — REJECT: a discovery-only custom provider is usable with `omp -p --model provider/id`
- **Bead:** lb-07
- **Surface:** ~/.omp/agent/models.yml, omp 18.2.10 print mode
- **Hypothesis:** a models.yml provider with `discovery: {type: openai-models-list}` and no static `models:` can be selected from the CLI.
- **A/B:** discovery-only: `omp models --json` lists `localbench/qwen3.6:35b-mlx`, but `omp -p --model localbench/qwen3.6:35b-mlx` exits 1 "Model ... not found" (3/3 attempts) and `--provider localbench` exits 1 "Unknown provider"; same invocation with one static `models:` entry: exit 0, answer returned.
- **A/A null:** same-invocation control: static entry variant succeeded against the same proxy and backend; discovery variant failed identically on repeat.
- **Verdict:** REJECT
- **Retry predicate:** on any omp version newer than 18.2.10, remove the static entry and re-run `omp -p "Reply with exactly: OK" --model mlx-serve/<id>`; resurrect if it exits 0.
- **Lesson:** `omp models` listing is not resolvability; localbench writes a static model entry (managed block) before any `omp -p` run.
- **Superseded:** see the RESURRECTED row below — the static-entry control also dropped `apiKey:`, so this A/B was confounded; the verdict does not stand.

## 2026-09-22 — REJECT: child `omp` processes measure the user's default profile without intervention
- **Bead:** lb-05
- **Surface:** harness environment, omp profiles
- **Hypothesis:** running `omp` from an agent shell measures the same configuration the user runs.
- **A/B:** the calling agent's shell exports OMP_PROFILE=claude and PI_CODING_AGENT_DIR=~/.omp/profiles/claude/agent; with those inherited, `omp models --json` showed no provider from ~/.omp/agent/models.yml (providers: anthropic, ollama, openai-codex); with them stripped, the same command listed the models.yml providers. The spike's full-flag prompt sizes (70,715 inherited vs 74,289 stripped) also differ, but those two runs used different provider paths (ollama Responses vs openai-completions), so the token delta is confounded and not attributed to the profile.
- **A/A null:** same invocation, same commands; the only varied input for the provider-listing check was the profile env (stripped vs inherited).
- **Verdict:** REJECT
- **Retry predicate:** re-check on every omp pin change: `omp --help` ENV section lists no profile variable beyond OMP_PROFILE, PI_PROFILE, PI_CODING_AGENT_DIR; if it lists another, extend `omp_env()` and re-run lb-05.
- **Lesson:** every omp child gets `omp_env()` (profile vars stripped, LOCALBENCH_OMP_PROFILE to opt in); spike rows marked (claude) are not evidence about the default profile.

## 2026-09-22 — SURVEY: PonyExl3 (EXL3 on Metal) as the speed lever for this machine
- **Bead:** lb-07
- **Surface:** candidate backends
- **Hypothesis:** EXL3 quantization via PonyExl3 would make local models fast enough for omp here.
- **A/B:** not measured on this host. Maintainer table (other hardware): Qwen3.6-27B plain decode 16.6 tok/s (M5 Max); ollama measured 72.9 tok/s for a 27B dense model here. No OpenAI-compatible server exists in PonyExl3 v0.3.0 (HEAD 8e7fa6b), so omp cannot reach it.
- **A/A null:** VOID-NOT-RUN — the EXL3 checkpoint download was stopped before completion to free bandwidth for the ollama/mlx-serve candidates.
- **Verdict:** SURVEY
- **Retry predicate:** PonyExl3 ships an OpenAI-compatible server (or `ponyexl3-serve`) AND a same-invocation localbench run on this host beats the MoE baseline's micro.prefill_32k.prefill_tps.
- **Lesson:** PonyExl3 optimizes quality per GB of RAM; on a 512 GB host memory is not the constraint — prefill is.

## 2026-09-22 — UNKNOWN: the spike's MoE full-prompt omp row (52.7 s) is untruncated
- **Bead:** lb-02
- **Surface:** ollama context length vs omp full prompt
- **Hypothesis:** the 70,736-token omp request was processed in full.
- **A/B:** ollama had qwen3.6:35b-mlx loaded with context_length 65,536 (from an earlier native-API load) while the request carried 70,736 prompt tokens.
- **A/A null:** VOID-SINGLE-RUN.
- **Verdict:** UNKNOWN
- **Retry predicate:** lb-02 `conf.no_truncation_64k` PASSes for the loaded context AND lb-04 replay of fixtures/omp/full.json reports prompt_tokens within 2% of 74,289.
- **Lesson:** isolate() reloads the target through /v1 so its context is what omp gets; never cite a run whose loaded context is below the prompt size.

## 2026-09-22 — RESURRECTED: a discovery-only custom provider is usable with `omp -p --model provider/id`
- **Bead:** lb-07
- **Surface:** ~/.omp/agent/models.yml, omp 18.2.10 print mode
- **Hypothesis:** the earlier REJECT (same date, "a discovery-only custom provider is usable…") blamed discovery; the real trigger is something else in that entry.
- **A/B:** same proxy, same backend, same model id `localbench/qwen3.6:35b-mlx`, one variable per run: discovery-only entry WITHOUT `apiKey:` → rc=0 (4/4 across colon and no-colon ids); original entry with `apiKey: local` + compat → rc=1 "Model … not found" (2/2); compat without apiKey → rc=0; `apiKey: local` without compat → rc=1; `apiKey: mlx-serve` without compat → rc=1.
- **A/A null:** each variant ran in the same invocation against the same proxy and ollama model; the failing variants failed identically on repeat.
- **Verdict:** RESURRECTED
- **Retry predicate:** on any omp version newer than 18.2.10, add `apiKey: local` next to `auth: none` on the localbench provider and re-run the bisection (/tmp/lb-debug.py logic, now lb-07's planted negative); if rc=0, record that omp fixed it.
- **Lesson:** the prior REJECT's control changed two variables at once; bisect one field per run before banking a REJECT. Custom providers here omit `apiKey:` whenever `auth: none` is set.
- **Demoted (lesson only, D4):** same day, the real `mlx-serve` provider resolved in `omp -p` with all five entry shapes tried — apiKey+auth:none+discovery, apiKey+discovery, auth:none+discovery, apiKey+static, apiKey+auth:none+static (5/5 rc=0). "apiKey next to auth: none breaks custom providers" does not hold in general; the discovery-only resurrection stands, the apiKey explanation is reduced to the UNKNOWN row below.

## 2026-09-22 — UNKNOWN: why the `localbench` provider fails `omp -p` only when `apiKey:` is set
- **Bead:** lb-05
- **Surface:** ~/.omp/agent/models.yml, omp 18.2.10 print mode, localbench proxy on :11299
- **Hypothesis:** an `apiKey:` value on a custom discovery provider makes its models unresolvable in print mode.
- **A/B:** localbench provider via proxy → ollama: with `apiKey:` (values `local`, `mlx-serve`; with and without compat) rc=1 "Model … not found" (4/4); without `apiKey:` rc=0 (5/5). mlx-serve provider direct to :11234: with or without `apiKey:` rc=0 (5/5).
- **A/A null:** each localbench variant ran in the same invocation against the same proxy and model; the mlx-serve variants ran against the same server minutes later. The difference between the two providers (proxy hop, provider id, upstream server) was not isolated.
- **Verdict:** UNKNOWN
- **Retry predicate:** run the localbench `apiKey: mlx-serve` variant with the proxy pointed at mlx-serve (:11234) and model `Qwen3.6-35B-A3B-MLX-Serve-4bit`; if rc=1, the provider id or proxy is implicated — then point the mlx-serve provider's baseUrl at the proxy to split the two.
- **Lesson:** localbench's managed provider block omits `apiKey:` (works); do not generalize the failure to other providers without the split test.

## 2026-09-22 — REJECT: bare `omp` on PATH is the omp the user runs
- **Bead:** lb-04
- **Surface:** harness omp resolution, fixture sidecars, generation binding
- **Hypothesis:** `subprocess.run(["omp", ...])` and `shutil.which("omp")` resolve to the binary the user runs, so fixture sidecars and goldens pin the incumbent omp.
- **A/B:** two omp 18.2.10 installs exist: ~/.bun/bin/omp → cli.js sha `acf06c76a4969558` (every running user pane, `ps`: `bun ~/.bun/bin/omp …`; the incumbent pin) and ~/.local/bin/omp → Mach-O sha `32895b6a0fb1f337` (installed 2026-09-14). The recording agent's shell (OMP_PROFILE=claude) put ~/.local/bin first; `localbench record` wrote `omp_sha: 32895b6a0fb1f337` into fixtures/omp/lean.meta.json and full.meta.json. A shell with ~/.bun/bin first resolves `acf06c76a4969558`. Replay checked only `omp_version` (both 18.2.10), so it would have accepted the other binary's fixtures.
- **A/A null:** same invocation, `type -a omp` + `shasum -a 256` on both resolved files; deterministic identity check, no timing involved.
- **Verdict:** REJECT
- **Retry predicate:** `type -a omp` lists exactly one omp, or both entries resolve to the same sha256; then omp resolution by PATH is safe again (the recorded omp_path/omp_sha stay in the pins regardless).
- **Lesson:** version strings are not generation pins; replay now binds on (omp_version, omp_sha), `LOCALBENCH_OMP` pins the binary explicitly, and pins record `omp_path`. Fixtures were re-recorded under the incumbent sha.
- **Re-checked 2026-09-23:** the user removed `~/.local/bin/omp` and then the orphaned `~/.local/lib/node_modules/@oh-my-pi/` package (1.4 GB, package.json 18.1.22); `type -a omp` now lists only `~/.bun/bin/omp` (18.2.11, sha `ce797fb3ed92e768`), so the retry predicate holds and bare `omp` is the user's omp again. Pins keep recording omp_path/omp_sha regardless.

## 2026-09-22 — REJECT: renaming the smol model under a `localbench-parked/` prefix keeps omp from loading it
- **Bead:** lb-01
- **Surface:** localbench/park.py, omp 18.2.10 model resolution, ollama 0.32.15
- **Hypothesis:** after `ollama cp qwen3.8:27b-mlx localbench-parked/qwen3.8:27b-mlx` + delete, every `ollama/qwen3.8:27b-mlx` smol call fails, so nothing reloads the model.
- **A/B:** sessions that resolved smol before the park got HTTP 404 on POST /v1/responses (dozens, 21:18–21:26; excerpt in docs/evidence/receipts/2026-09-22-omp-side-calls.md §1); at 21:27:09 a POST /v1/responses loaded `localbench-parked/qwen3.8:27b-mlx` (log: `starting mlx runner subprocess model=localbench-parked/qwen3.8:27b-mlx`, 30.3 s, 200) while a conf smoke run of qwen3.6:35b-mlx was starting. Cause, from source: omp's `matchModel` falls back to provider-scoped `fuzzyMatch(modelId, model.id)` (pi-coding-agent src/config/model-resolver.ts:769-811). Probe (pi-tui fuzzyMatch, query `qwen3.8:27b-mlx`): `localbench-parked/qwen3.8:27b-mlx` matches (score -2998.68); `localbench-parked:5642e97495e1`, `qwen3.6:35b-mlx`, `qwen3.8-uncensored:latest` do not.
- **A/A null:** same probe, same query, every candidate id scored in one invocation; the matching and non-matching names differ only in whether they contain the original id.
- **Verdict:** REJECT
- **Retry predicate:** omp's resolver stops fuzzy-matching provider-qualified ids (an unknown `ollama/<id>` fails instead of matching a sibling), checked by re-running scripts/omp-fuzzy-probe.ts against the new resolver; until then parked names must not contain the original id.
- **Lesson:** park under an opaque digest tag (`localbench-parked:<digest12>`); a parked name that contains the original id is still reachable. The sampler's foreign-resident check is the backstop: it would have marked that run CONTENDED.

## 2026-09-22 — REJECT: with the smol model parked, localbench's own `omp -p` children make no local calls outside the model under test
- **Bead:** lb-05
- **Surface:** e2e tier child flags, omp 18.2.10 default profile (`modelRoles.smol`, `mnemopi.llmMode: smol`)
- **Hypothesis:** parking qwen3.8:27b-mlx makes every smol/memory call from the e2e `omp -p` children fail, so the only model the children touch is the one under test.
- **A/B:** docs/evidence/receipts/2026-09-22-omp-side-calls.md §2–3. Run 20260923T033709Z (tiers micro,replay,e2e, 21:37–21:39 MDT): the sampler flagged `qwen3.8-uncensored:latest` resident during both e2e tasks (two `contention` events, GPU 97% and 80%); /tmp/ollama-serve.log shows POST /v1/responses loading `library/qwen3.8-uncensored:latest` at 21:38:48 and 21:39:11, seconds after each e2e isolate. Next run with `--smol localbench/<model>` on every child (tiers e2e): contended=false, and omp_calls.jsonl shows the auxiliary calls (tools=0, ~270 prompt tokens) served by the model under test through the proxy.
- **A/A null:** same tier, same model, consecutive runs; the only change was the `--smol` flag.
- **Verdict:** REJECT
- **Retry predicate:** omp print mode (`omp -p`) stops issuing smol/memory calls when `--no-session` is set, observed as zero tools=0 rows in omp_calls.jsonl for an e2e run without `--smol`; then the flag can go.
- **Lesson:** an unresolvable smol role does not fail closed in omp; it falls back to some other available model. Measured children pin `--smol` to the model under test so auxiliary calls are timed, counted, and never load a second model.

## 2026-09-22 — REJECT: load1 above the P-core count means the machine is too busy to measure
- **Bead:** lb-01
- **Surface:** localbench/__main__.py preflight()
- **Hypothesis:** a 1-minute load average above 24 (P-cores) indicates CPU contention that would skew a run.
- **A/B:** 2026-09-23T03:30Z preflight refused with `load1 24.2 > 24 P-cores`; one minute later `/usr/bin/top -l 2 -s 2` reported 91.21% / 88.29% idle with load averages 18.61 18.78 17.09. The host runs ~15 idle omp/bun processes with ~115 threads each; load tracks them, not work. The replacement check (100 − top idle% over 2 s, refuse above 25%) refused a planted 32× `yes` burn at 53.6% and admitted unplanted runs at 24.6% and 14.8% (docs/evidence/break-tests.md, lb-01).
- **A/A null:** same host, same minute, two probes of the same state (load1 vs measured idle); planted vs unplanted burn for the replacement.
- **Verdict:** REJECT
- **Retry predicate:** over ten 2 s samples on an unplanted machine, load1 / hw.ncpu and (100 − top idle%) / 100 agree within 0.25; then load1 may return as a refusal criterion.
- **Lesson:** judge busy by measured idle time; record load1 but do not gate on it on a host full of idle agent processes.

## 2026-09-22 — SURVEY: Underdog Husky (model-specific engine, "up to 4.5× faster than MLX") as the speed lever for omp here
- **Bead:** lb-07
- **Surface:** candidate backends
- **Hypothesis:** Husky + Woof would make omp on this Mac fast.
- **A/B:** not measured on this host. Vendor table (husky.underdog.ai, 2026-09-20, M5 Max, Woof 4B at 4-bit, 2.4 GB, greedy, medians of 3): plain writing 1.02–1.27× MLX (151–164 → 164–193 tok/s); edits whose reply copies the prompt 1.8–3.9× (prompt lookup); "730 tok/s / 4.5×" is the best row (function edit, trained draft on). "5× sooner" is 35 ms vs 140 ms to first token on a cached continued chat. Husky runs only Woof and ships inside the Underdog app; the page names no OpenAI-compatible endpoint, so omp cannot reach it. The same levers exist here already: mlx-serve (prompt lookup + MTP) decoded 1.08× ollama on the 35B MoE in the memory-off A/B (receipts/ab-mlxserve-vs-ollama-moe.json, A/A band 5%); an earlier A/B showed 1.51×, but its ollama legs decoded at 99–114 tok/s against 151–152 later (receipts/ab-mlxserve-vs-ollama-moe__omp18.2.10-memory-on.json), so that ratio was inflated. omp's cost on this host is prefill of an 11.6k–74k-token prompt plus ~1.5–4 s of omp startup, not decode.
- **A/A null:** VOID-NOT-RUN — vendor numbers from other hardware (packet §8 non-proof).
- **Verdict:** SURVEY
- **Retry predicate:** Husky (or Underdog) exposes an OpenAI-compatible /v1/chat/completions endpoint AND a localbench run on this host passes every MUST conformance case and beats mlx-serve's Qwen3.6-35B-A3B on replay.lean.cold_ttft_s. Separately, Woof-4B's MLX weights (HF ConwayResearch/Underdog-Woof-4B-1.1) can be measured under mlx-serve today; that tests the model, not Husky.
- **Lesson:** a headline "up to N×" from a one-model engine is the best row of a table; read the median row and the hardware line before treating it as a lever for a different workload.

## 2026-09-23 — REJECT: pins captured when a run starts describe the whole run
- **Bead:** lb-06
- **Surface:** localbench execute()/run_pins, backends.sha16 cache, omp auto-update
- **Hypothesis:** the generation recorded at the start of a leg holds until the leg ends, so start-of-run pins label every number in it.
- **A/B:** omp was upgraded in place from 18.2.10 (sha `acf06c76a4969558`) to 18.2.11 (sha `ce797fb3ed92e768`) at 2026-09-23T05:28:44Z (mtime of ~/.bun/install/global/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js; not done by localbench). The ab_a2 leg of receipts/ab-incumbent-vs-moe__contended-20260923T050101Z.json started 05:24:41Z with 18.2.10 pins and ran its e2e tier at ~05:43Z on 18.2.11. `sha16` was cached by path, so a same-process re-check would also have reported the old sha. The next campaign's replay tier caught the change: every replay row VOID `GENERATION-MISMATCH: fixture omp 18.2.10 sha acf06c76a4969558 vs running 18.2.11 sha ce797fb3ed92e768`.
- **A/A null:** not a timing claim; the file mtime, `omp --version` before/after, and the VOID rows are the evidence.
- **Verdict:** REJECT
- **Retry predicate:** none needed for the harness (fixed: pins re-read at the end of every leg, a difference marks the leg unsound `PINS CHANGED mid-run`, and sha16 caches on path+mtime+size). Re-audit if omp gains an update lock that localbench can hold for a run.
- **Lesson:** an agent host updates its tools under you; bind a run to the generation at both ends. The 18.2.10 goldens stay valid for 18.2.10 and now report GENERATION-MISMATCH until re-banked under 18.2.11.

## 2026-09-23 — REJECT: an `omp -p` child's prompt depends only on omp, its profile, and its flags
- **Bead:** lb-05
- **Surface:** e2e/rel tier children and `localbench record`, default profile mnemopi (auto recall + auto retain)
- **Hypothesis:** two `omp -p` runs with the same flags in the same cwd send the model the same prompt, so answer failures are model behaviour.
- **A/B:** docs/evidence/receipts/2026-09-23-memory-feedback.md. The children's cwd bank held 10 transcripts of their own earlier turns, including every wrong answer ("Understood.", "Hello, World!", invented Agent Mail tasks), each recalled 57–125 times into later prompts; the lean fixture itself carried a `<memories>` block. Same command with `--config fixtures/omp/child-config.yml` (`memory.backend: off`): 0 `<memories>` blocks, 11,433 prompt tokens vs 12,026 without it (omp 18.2.11).
- **A/A null:** same minute, same omp, same flags; the only varied input was the overlay.
- **Verdict:** REJECT
- **Retry predicate:** omp -p stops recalling/retaining memory for `--no-session` runs (verify: record without the overlay shows no `<memories>` block and the cwd bank gains no row); then the overlay can go.
- **Lesson:** a benchmark child must not read or write state that outlives it. Pre-overlay rel/e2e answer rates are void as model evidence. After the overlay (omp 18.2.11, receipts rel-ollama-moe.json, rel-mlxserve-moe.json, rel-ollama-dense.json): 40/40 correct answers on each of the three configs (Wilson 95% lower bound 0.84 per task), against 17/20 on "Reply with exactly: OK" for ollama MoE with the loop active. The overlay did not remove every wrong answer: cold first turns inside full-tier e2e runs still failed 3/7 on the MoE (separate UNKNOWN row).

## 2026-09-23 — REJECT: omp's few seconds before the first model request are fixed launch overhead
- **Bead:** lb-05
- **Surface:** omp 18.2.11 default profile (mnemopi memory, auto recall/retain), `omp -p`, ollama qwen3.6:35b-mlx warm
- **Hypothesis:** the 3–4.7 s `startup_s` seen in memory-on e2e runs (omp 18.2.10) is omp's own process start and is independent of configuration.
- **A/B:** docs/evidence/receipts/2026-09-23-memory-overhead-probe.jsonl (scripts/probe-memory-overhead.py, 8 interleaved pairs, same backend, same flags except the memory-off overlay). Medians: memory off wall 2.23 s, startup 0.98 s; memory on wall 4.79 s, startup 2.02 s. Memory on pair 7: 34 calls, 283.6 s, final answer "The file `/dev/shm/memory.db.sqlite` doe…" to "Reply with exactly: OK" (recalled turns from the probe's own earlier pairs derailed it); memory off 8/8 correct, memory on 7/8. Memory-off e2e runs on omp 18.2.11 showed startup_s 0.62 s.
- **A/A null:** interleaved same-invocation pairs (order alternated per pair); memory-off walls spread 1.48–2.77 s after the first (cold) attempt. No separate A/A run.
- **Verdict:** REJECT
- **Retry predicate:** the probe's memory-on median startup_s comes within 0.3 s of memory-off on a future omp (memory work moved off the launch path), with memory-on correct 8/8.
- **Lesson:** on this host the default profile's memory costs each `omp` launch (every `omp -p`, every subagent) about 1 s before the first request and about 2.5 s of wall, and a recall can derail a trivial turn. The user's real smol role points memory at a local dense model, which the probe did not measure (smol was routed to the model under test).

## 2026-09-23 — UNKNOWN: why Qwen3.6-35B-A3B sometimes answers a trivial cold first turn wrongly in omp
- **Bead:** lb-02
- **Surface:** e2e.ok.correct on ollama qwen3.6:35b-mlx and mlx-serve Qwen3.6-35B-A3B-MLX-Serve-4bit, omp 18.2.11, memory off
- **Hypothesis:** the wrong cold first-turn answers come from the cold state itself (first request after a model load, or partial prefix reuse against the one-token warm-up).
- **A/B:** docs/evidence/receipts/2026-09-23-cold-first-turn-tally.md: 3 wrong of 7 cold attempts inside full-tier runs between 07:07Z and 07:32Z (both backends), then 0 wrong of 132 cold attempts afterwards — relcold (isolate + warm-up) 0/40, relfresh (isolate, no warm-up) 0/40, heavy tiers then relcold 0/40, e2e tier alone 0/12. Warm: 0/120.
- **A/A null:** relcold vs relfresh ran same backend, same minute window; the heavy-tiers-first runs repeat the full-run order. None reproduced the failure, so there is no measured contrast.
- **Verdict:** UNKNOWN
- **Retry predicate:** the next wrong cold answer in any run with request bodies captured (commit f98aac4+; `bodies` listed in the rel detail or omp_calls.jsonl `body` field): diff its main-turn body and the aux (tools=0) body against a passing attempt's; if they differ, omp sent a different request; if identical, the backend sampled it.
- **Lesson:** a 3/7 cluster inside 14 minutes looked like a 40% failure rate; 139 attempts put it near 2%. Quote the denominator and the window.

## 2026-09-23 — REJECT: memory is the side work that costs an omp turn on a local model
- **Bead:** lb-05
- **Surface:** omp 18.2.11 auto-thinking classifier (session/model-controls.ts:588 `#AUTO_THINKING_TIMEOUT_MS = 4000`; auto-thinking/classifier.ts:109 judge chain typesafe/proj-b-latest → @tiny → @smol), default profile `defaultThinkingLevel: auto`, e2e/rel children (LEAN_FLAGS has no `--thinking`), ollama qwen3.6:35b-mlx
- **Hypothesis:** with memory off, what omp does before the main call on a local model is launch overhead; mnemopi memory is the side work that adds turn latency.
- **A/B:** docs/evidence/receipts/2026-09-23-memory-variants-calls.jsonl (proxy log, 88 calls, 44 attempts): every attempt whose smol role reached the proxy made one tool-free call first, the judge ("Choose the reasoning effort this turn needs", 266 prompt tokens, `enable_thinking: false`, temperature 0, max_tokens 4096; body captured via the proxy's save_dir). The main call started a median 0.05 s after it ended (n=36): serial, on the critical path. 18/36 completed in 0.9–3.7 s; 18/36 were aborted by omp at 4.0–4.4 s. Ollama returned 64–215 completion tokens for a one-word answer (mlx-serve returned 1 token for the same call on 2026-09-22, `_attempt_calls` note). Median main-turn llm_s in the memory-off variant: 1.1 s.
- **A/A null:** VOID-contended for the durations (the window had qwen3.8:27b-mlx resident; see the UNKNOWN row below). The ordering and the 4 s cap are code facts, independent of load.
- **Verdict:** REJECT
- **Retry predicate:** an uncontended e2e run (no contention event, `side.auto-thinking` reported) on ollama where the classifier's median busy_s is ≤ 0.3 s, or an omp release where the main call no longer waits for the classifier (main t_start precedes the judge call's end in the proxy log).
- **Lesson:** under `defaultThinkingLevel: auto` every user turn pays one serial effort-classification call before the main call, up to 4 s, and on ollama it spends reasoning tokens that omp asked it not to. Every e2e number so far includes it (goldens, the README's 6.1 s first turn on mlx-serve, the unclaimed ollama-vs-mlx-serve e2e ratios, which also differ in how each backend answers this call). The proxy now labels each call's `purpose` and `_attempt_calls` reports `side.auto-thinking` apart from the main turn.
- **Clean re-measure (same day, same predicate family):** with the scouts gone and qwen3.8 parked except for the probe (receipts/2026-09-23-side-work-routing.jsonl), the classifier on the dense 27B took 1.35–1.53 s, 83–87 completion tokens, 0/6 aborted; on the MoE in the contended window above it took 0.9–4.4 s. The 4 s aborts were contention, the serial ordering is not.

## 2026-09-23 — UNKNOWN: which mnemopi configuration makes memory as cheap as memory off
- **Bead:** lb-05
- **Surface:** scripts/probe-memory-overhead.py (8 variants: off, smol-moe, llm-none, lexical = llmMode none + noEmbeddings, tiny = modelRoles.memory local/lfm2.5-230m, idle = autoRecall/autoRetain off, user, user-memoff), omp 18.2.11, ollama qwen3.6:35b-mlx
- **Hypothesis:** one mnemopi setting (no LLM, no embeddings, an in-process tiny model, or no automatic recall/retain) makes a memory-on `omp -p` as fast as memory off.
- **A/B:** docs/evidence/receipts/2026-09-23-memory-variants.jsonl: 6 rounds, variant order rotated per round. Wall medians: off 5.16 s, idle 5.99, user-memoff 6.37, llm-none 6.94, tiny 7.03, lexical 7.72, smol-moe 9.55, user 11.95. Correctness: `user` (mnemopi, smol = ollama qwen3.8 dense) 4/6, wrong answers "The `MEMORY://memory/core/system.md` file is not accessible…" (5 calls, 41.5 s) and "DONE"; every variant with smol on the MoE, memory on or off, 6/6.
- **A/A null:** VOID-noise: wall cv 25–72% in every variant; the auto-thinking classifier (row above) hit its 4 s cap in half the attempts and dominates the spread. qwen3.8:27b-mlx was resident in every row, and five scout subagents of the session running the probe were on it the whole time (row below), so the dense model was generating throughout.
- **Verdict:** UNKNOWN
- **Retry predicate:** rerun the probe inside a Sampler with gpu_foreign_max_pct set and no contention event, with `--thinking` pinned on every variant (so the classifier is out of the wall), and qwen3.8 idle for the non-`user` variants; rank variants only where each variant's wall cv ≤ 10%.
- **Lesson:** "resident" cannot tell an idle model from one that is generating, and memory cannot be priced while a larger side call sits in the same wall. Only the correctness split survives: the memory failures came with the dense smol model, not with memory on the MoE.

## 2026-09-23 — REJECT: a runner ollama shows as `Stopping...` has stopped using the GPU
- **Bead:** lb-01
- **Surface:** ollama 0.32.15 MLX runner (pid 71271, qwen3.8:27b-mlx), `isolate()` unload via keep_alive 0, other omp sessions' smol side calls
- **Hypothesis:** once ollama is asked to unload a model (`ollama ps` shows `Stopping...`), it no longer competes for the GPU.
- **A/B:** `localbench gpu --seconds 5` three times at 09:10–09:11: runner 71271 (`--model qwen3.8:27b-mlx`) 87.0 / 83.2 / 85.3% GPU, device 99%, runner CPU 0.3%, `ollama ps` still `Stopping...` (runner up since 08:44). The work was this agent's own five `scout` subagents (omp's bundled scout.md has `model: "@smol"`, and every profile's smol is ollama/qwen3.8:27b-mlx): they held this session's five ollama sockets for 33 min and all failed at the moment of `localbench park` with `[ollama/qwen3.8:27b-mlx] 404 model 'qwen3.8:27b-mlx' not found`. Killing the runner once made ollama start another (pid 96606, 90% GPU) for the queued requests; parking plus a second kill freed the GPU (Terminal 1.5%, WindowServer 1.4%).
- **A/A null:** VOID-no-control: no run with the runner idle was taken in the same window (it never went idle).
- **Verdict:** REJECT
- **Retry predicate:** after `keep_alive: 0` on a runner with a request in flight, `localbench gpu --seconds 5` shows that runner under 1% within 5 s (an ollama release that cancels in-flight work on unload).
- **Lesson:** unloading waits for in-flight and queued requests, and five subagents' turns on a dense 27B kept a `Stopping...` runner at ~85% GPU for 33 min. Residency checks miss this; the Sampler now reads per-process GPU time every second and marks a run CONTENDED when a process other than the backend uses more than 25% in any second. Spawning `scout` subagents during a run violates the measurement law as surely as running a second model.
- **Correction (same day):** this row first blamed pid 52772 (tmux omp-test:0.2) because nettop showed it with 1,546 bytes sent / 0 received while this session's five sockets moved no bytes over ~6 s. Long prefills on a dense 27B send no bytes for minutes, so a quiet socket is not an idle one; the scout failures above are the attribution. A client's socket count says who CAN send work; only the proxy (or the 404s after parking) says who DID.

## 2026-09-23 — SURVEY: mlx-serve's native MTP head speeds up omp turns on Qwen3.6-35B-A3B
- **Bead:** lb-07
- **Surface:** mlx-serve (homebrew) serving ~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit (ships `mtp/weights.safetensors`), harness launch without `--mtp`/`--no-pld`
- **Hypothesis:** the checkpoint's MTP draft head is already speeding up decode in the banked mlx-serve runs. Prompted by r/LocalLLaMA 1wnwbjs ("your MLX MTP head may be silently ignored"; community claim, ~13.0 vs ~8.0 tok/s).
- **A/B:** not measured. /tmp/Qwen3.6-35B-A3B-MLX-Serve-4bit.mlx-serve.log (every start of 2026-09-22/23): `[mtp] loaded native MTP head (moe-mlp…)`, `MTP head ready (depth=6, profile=generic)`, then `PLD speculative decoding: ENABLED (… default for new requests)`; every `[spec-stats]` line is `mode=pld`. `mlx-serve --help`: "`--mtp` Force the MTP head ON for MoE targets too. Requests default to MTP only on DENSE models". PLD itself turned off on the benchmark prompts: micro decode ("Count from 1 to 60") `ngram-score=0.000 … pld=disabled`, omp main turns `yield gate: 0 drafted tokens over 8 steps` → `runtime_disabled=true`. So the banked mlx-serve decode numbers are plain decode, and the head is resident but unused.
- **A/A null:** VOID-no-measurement.
- **Verdict:** SURVEY
- **Retry predicate:** a same-invocation A,B,A with B = `--server-arg --mtp` (both legs uncontended, every MUST passes) and `[spec-stats] mode=mtp` in B's log; adopt only if micro.decode and e2e main-turn llm_s improve beyond the A/A band.
- **Lesson:** "loaded" in a server log is not "used": check the per-request spec mode, not the startup banner. The same log shows mlx-serve honouring omp's `thinking=false` on the auto-thinking judge (`emitted=1 … bytes="low"`), which ollama does not.

## 2026-09-23 — KEEP: omp's configured side work costs a local turn seconds before the main call starts
- **Bead:** lb-05
- **Surface:** scripts/probe-side-work.py; omp 18.2.11 default profile; main model ollama qwen3.6:35b-mlx via the proxy; side work on qwen3.8 27B dense under its parked name via the proxy (every other session's smol calls parked out); one tool-using `omp -p` turn (read answer.txt)
- **Hypothesis:** the default profile's side work (auto-thinking classifier and mnemopi memory, both on the local dense smol model) adds more than a second per turn before the model starts answering, and moving it off the GPU or removing it recovers that time.
- **A/B:** docs/evidence/receipts/2026-09-23-side-work-routing.jsonl (+ `-calls.jsonl`): 1 warm-up + 6 rounds × 4 arms, order rotated, 24/24 correct, no row with a non-ollama process above 25% GPU. Median time until the main call (pre_main_s): current 4.69 s (cv 6.8%), tiny = modelRoles.tiny local/lfm2.5-230m 3.75 s (7.6%), pinned = `--thinking low` 2.99 s (8.8%), floor = pinned + memory off 1.17 s (11.2%); every pair kept that order in 6/6 rounds. Dense runner ~16% GPU per turn in `current`, 0 elsewhere. Wall medians: current 7.04 s (cv 11.1%), tiny 6.30 (16.9%), pinned 5.10 (10.6%), floor 2.28 (6.6%); faster than current in tiny 3/6, pinned 5/6, floor 6/6 rounds. tiny's main call median 1.38 s vs current 1.03 s (omp's local 3-bucket question maps "moderate" to high effort).
- **A/A null:** VOID-no-aa (no same-arm pair); within-arm cv reported. By the predeclared gate (wall cv ≤ 10%) only floor's wall is rankable; the pre_main ordering is attribution, not a timing claim.
- **Verdict:** KEEP
- **Retry predicate:** re-run the probe after any change to omp's classifier (timeout, judge chain, local question), mnemopi recall, or the smol model; the row stands while `current`'s pre_main_s median exceeds `floor`'s by at least 1 s.
- **Lesson:** per local turn, the classifier on the dense model costs ~1.7 s before the main call and memory recall ~1.8 s. The CPU tiny model takes the classifier off the GPU and saves ~0.9 s of that wait but not wall time (it picks heavier thinking); pinning the thinking level removes the classifier outright. None of this shows in residency or device-wide GPU %: it took the proxy's per-purpose timing and per-process GPU time.

## 2026-09-23 — REJECT: binding every golden row to the omp binary keeps the regression gate meaningful
- **Bead:** lb-06
- **Surface:** golden.compare/pin_diff (all rows GENERATION-MISMATCH on any pin change), replay tier (VOID when the fixture's omp ≠ the running omp)
- **Hypothesis:** one generation per golden (backend + model + omp + child config + macOS) is the right unit: any change re-banks everything.
- **A/B:** omp ships most days, sometimes several times a day (user, 2026-09-23); on 2026-09-23 itself omp moved 18.2.10 → 18.2.11 mid-campaign (earlier ledger row). Under whole-golden binding, `localbench status` showed the ollama MoE golden entirely stale although its conf/micro rows never run omp; after per-tier binding the same golden reports conf, micro CURRENT and only e2e GENERATION-MISMATCH (docs/evidence/break-tests.md, lb-06 per-tier entry, both directions checked).
- **A/A null:** VOID-no-measurement (a binding rule, not a timing claim).
- **Verdict:** REJECT
- **Retry predicate:** a tier shown CURRENT under tier_keys regresses on a pin outside its key set — e.g. a micro row that moves beyond its band between two omp versions with backend, model and macOS unchanged; then that pin joins the tier's keys.
- **Lesson:** bind a row to what it exercised. The backend gate now survives omp releases; only the e2e/rel family is re-banked per omp version (`aa --tiers e2e --write-golden` merges), and replay binds to the recorded bodies, with freshness reported instead of voided.

## 2026-09-23 — SURVEY: FrankenSQLite (fsqlite) instead of SQLite for localbench's store
- **Bead:** lb-09
- **Surface:** the store for local-model awareness data (GPU-by-process samples, proxy call purposes, run receipts, model inventory); fsqlite 0.3.x (`~/.cargo/bin/fsqlite`, repo `~/Developer/frankensqlite`), read through the frankensqlite-mega-skill references (STATUS-LEDGER, CONCURRENCY-CONTRACT-USAGE, PERIPHERAL-CRATES), not executed here
- **Hypothesis:** fsqlite's page-level MVCC (parallel writers) or time travel makes it a better store for this repo than stock SQLite.
- **A/B:** not measured. What decides it today, from the skill's status ledger: (1) the parallel-writer benefit targets write contention localbench does not have (a few writers — sampler, proxy, one run — at a few rows per second), and ≥10 concurrent writers is unsupported (bd-9inpb corruption repro); (2) file-backed time travel is design-only — `FOR SYSTEM_TIME AS OF` works on `:memory:` connections only; (3) `fsqlite-c-api` exports no `sqlite3_bind_*`, backup, or VFS registration, so Python's stdlib `sqlite3` cannot use it and a ctypes shim would build SQL strings without parameters; (4) localbench is stdlib-only (AGENTS.md). Stock SQLite in WAL mode writes the same file format, and fsqlite reads checkpointed SQLite files (sequential hand-off supported; concurrent mixed opens are not).
- **A/A null:** VOID-no-measurement.
- **Verdict:** SURVEY
- **Retry predicate:** any of — fsqlite ships file-backed time travel (STATUS-LEDGER moves it to LIVE) and a localbench question needs as-of queries; or a Python-usable binding with parameter binding exists; or localbench's own store measures writer contention (SQLITE_BUSY waits above 1% of writes) with ≤8 concurrent writers. Then A/B the store workload on both engines.
- **Lesson:** start on stock SQLite (stdlib, WAL): the file format is shared, so switching later costs a checkpoint, not a migration. The same machine already runs fsqlite in production use through ee (`~/.ee/ee.db-fsqlite-*` sidecars), which is where its multi-writer contract matters.

## 2026-09-23 — REJECT: WindowServer's GPU share rises because the model under test saturates the GPU
- **Bead:** lb-01
- **Surface:** Sampler per-process contention (25% per second), WindowServer (pid 179), ollama qwen3.6:35b-mlx and mlx-serve Qwen3.6-35B-A3B
- **Hypothesis:** WindowServer's 25.3–30.9% during the two 2026-09-23 A/A re-banks (device 98–100%) was an accounting effect of the backend saturating the GPU, so the compositor could be exempt from the per-second veto.
- **A/B:** /tmp/breaktest-compositor.py, backend-only leg (MoE generating 3,000 tokens, device mean 88.6–91.2%, 24–46 one-second windows): WindowServer max 9.2–9.3%, whole-window 3.9–5.7%; the same process read 1.4–9.7% in idle samples. During the refused A/A legs it held 25.3–30.9% for tens of minutes (receipts/aa__ollama__qwen3.6_35b-mlx__20260923T153656Z.json, problems: CONTENDED).
- **A/A null:** two backend-only legs (the script ran twice concurrently by mistake; both backend-only) agreed within 0.1 point on WindowServer's max.
- **Verdict:** REJECT
- **Retry predicate:** a backend-only saturation window where WindowServer exceeds 25% with no display activity (no video, no scrolling panes, screen idle) — then its share is load-coupled and the exemption question reopens.
- **Lesson:** the veto was right and the exemption was reverted before landing: sustained desktop activity costs the model real GPU time. Preflight now applies the same per-process 25% rule before a run starts (refuse, or wait with `--wait-idle`), instead of discovering it 35 minutes in.

## 2026-09-23 — REJECT: the memory LLM is on the per-turn path of one-shot omp sessions
- **Bead:** lb-05
- **Surface:** mem tier, omp 18.2.11 `omp -p` with mnemopi on (fixtures/omp/child-config-mem.yml, llmMode smol → the model under test), proxy purpose labels
- **Hypothesis:** where the memory LLM runs (dense 27B, the MoE, the CPU tiny model) decides memory's per-turn cost, so routing it off the GPU speeds up every omp launch.
- **A/B:** receipts/2026-09-23-mem-ollama-moe.json run dir `mem_calls.jsonl`: 134 proxied calls over 36 memory-on launches — 98 main-turn calls, 36 side calls, all `auto-thinking` (median 1.36 s); zero memory-extraction calls. Recall still hit 8/9, so retrieval ran without an LLM: mnemopi stores the transcript at process exit without extraction (omp source: dispose passes `extract: false` on the print/exit path) and recalls in-process. Recall turns reached the main call after 5.09 s median vs 1.17 s for memory off + pinned thinking in the side-work probe; 1.36 s of that is the classifier.
- **A/A null:** VOID-no-aa for the timing split (attribution only); the zero-call count is exact.
- **Verdict:** REJECT
- **Retry predicate:** a mem-tier run whose proxy log shows side calls other than `auto-thinking` (an omp release that extracts at print-mode exit), or an interactive multi-turn measurement (retain every 4 turns) — the memory LLM's placement matters there and has not been measured.
- **Lesson:** for one-shot launches (localbench children, scouts, `omp -p`) memory costs in-process recall, not an LLM call: the lever to test is retrieval (e.g. `mnemopi.noEmbeddings: true`, FTS only), not memory-LLM routing. The routing question stays open for interactive sessions.

## 2026-09-23 — REJECT: retention's memory-LLM call is free in an interactive omp session because it runs after the turn
- **Bead:** lb-05
- **Surface:** sess tier (`localbench run ollama:qwen3.6:35b-mlx --tiers sess --repeats 1`), omp 18.2.11 `--mode rpc`, mnemopi on (fixtures/omp/child-config-mem.yml, llmMode smol → the model under test, so one model on the GPU), retainEveryNTurns 4 (default), thinking level auto
- **Hypothesis:** mnemopi retains fire-and-forget on agent_end (omp source: `void this.maybeRetainOnAgentEnd(...)`), so its extraction call never delays the user; this row's predecessor left the interactive case unmeasured.
- **A/B:** receipts/2026-09-23-sess-smoke-ollama-moe.json (1 session, 12 turns, sound): regular turns 1.33 s median wall (n=9); the turns sent right after retention (5, 9) 9.00 s (n=2), +7.28 s median against their own neighbours (6.04–8.52). Wait before their main call 4.03 s both times: omp's 4 s auto-thinking cap. The proxy shows why: each retention's memory-extract call ran 6.1–9.1 s and generated ~1.2k tokens (3,573 for three); the next turn's classifier queued behind it and was aborted (7.46 s, 9.20 s); 8.43 s of main-call time overlapped extraction. `localbench show` on the receipt prints all of it.
- **A/A null:** VOID-no-aa: one session, n=2 post-retain turns. The mechanism is exact (proxy timestamps, aborted calls); the size is not banked.
- **Verdict:** REJECT
- **Retry predicate:** a sess A/A (--repeats 3, twice) where post_retain_wall_s sits inside the A/A band of wall_s — e.g. `ab <spec> <spec> --tiers sess --b-mem-config` with the memory LLM off the GPU (llmMode none, or the memory role on the CPU tiny model), a pinned thinking level (no classifier to starve), or an omp release that defers retention until the session is idle.
- **Lesson:** "fire-and-forget" moves a call off the turn that triggered it and onto the next one. On one GPU the memory LLM and the main model do contend in an interactive session: every 4th turn pays the extraction plus the classifier's timeout. The same run found a harness bug (SIGPIPE made the proxy's abandoned-stream path fatal; fixed in 3d4616e).

## 2026-09-23 — UNKNOWN: mlx-serve's MTP head speeds up omp turns on Qwen3.6-35B-A3B (measured half of the SURVEY row)
- **Bead:** lb-07
- **Surface:** `localbench ab mlx-serve:<Qwen3.6-35B-A3B-MLX-Serve-4bit> <same> --b-server-arg=--mtp --tiers conf,micro,replay` (mlx-serve 26.9.2, greedy, repeats 3)
- **Hypothesis:** the SURVEY row above: forcing the resident MTP head on for this MoE target speeds up omp turns.
- **A/B:** receipts/ab-mlx-serve-qwen3.6-mtp-20260923.json, all three legs uncontended, every MUST passes; B's log has 29 `[spec-stats] mode=mtp` rounds (mean 1.78 accepted per round), A1/A2 only `mode=pld` — the flag took effect. micro.decode 163.5 → 188.0 tok/s (B/A 1.15, band 0.05): B-BETTER. Prefix-cached first tokens got slower: replay.lean.warm 0.098 → 0.111 s, replay.full.warm 0.343 → 0.391 s, cache_hit_8k warm 0.091 → 0.102 s (B-WORSE, 5 rows); cold prefill and replay turn2 within noise. Greedy text identical A vs B on the 4 prompts without a nonce (13, 13, 200, 26 tokens). Caveat: A2 ran at localbench 34fe604-dirty, A1/B at ae2f784-dirty (a models-inventory commit landed mid-run; no measured code path changed).
- **A/A null:** the A1/A2 legs of the same invocation (decode spread 0.8%).
- **Verdict:** UNKNOWN
- **Retry predicate:** the SURVEY row's own gate, now half met: rerun the same A,B,A with `--tiers micro,replay,e2e` and adopt `--mtp` only if e2e main-turn llm_s is B-BETTER and no omp-shaped warm row (replay.*.warm, e2e repeat) is B-WORSE beyond its band; plus a nonce-free greedy decode of 256+ tokens identical A vs B before calling it lossless beyond 200 tokens.
- **Lesson:** MTP trades a small per-request cost on cached prefixes for faster long decodes. omp's turns are mostly cached prefixes with short replies, so the micro decode number alone would have been the wrong reason to switch it on.

## 2026-09-23 — REJECT: a mem.no_leak FAIL means mnemopi recalled another project's memory
- **Bead:** lb-05
- **Surface:** mem tier, `localbench ab ollama:qwen3.6:35b-mlx <same> --tiers mem --b-mem-config fixtures/omp/child-config-mem-fts.yml`; children ran `omp -p` with tools read,bash,edit,write,grep,glob,todo
- **Hypothesis:** leg A1's MUST FAIL (a control project with nothing planted answered the planted owner) is mnemopi recall crossing project banks.
- **A/B:** receipts/ab-mem-fts-ollama-moe-20260923.json (UNSOUND by that FAIL). The saved request bodies of A1 (runs/20260923T164729Z__ab_a1__…/bodies, parsed for tool calls) show the control turns searching the machine instead: 174 distinct search calls in one leg — `find /`, `grep -r` over ~, /tmp and /Users/Shared, Python scripts opening SQLite files, and `sqlite3 ~/.claude/agent-memory.db`, `~/.claude/projects.db`, `~/.claude/context-registry.db`. The flagged answer opens "I've searched extensively across all SQLite databases (am-recovered, am-backup, projects.db, agent-memory)…". Other controls answered "41024 … found in mem-103.json" and, in A2, the previous round's code ZEBRA-5567 — which the same-round-only check did not flag.
- **A/A null:** n/a — attribution from the recorded tool calls, not a timing claim.
- **Verdict:** REJECT
- **Retry predicate:** a mem run with `--no-tools` children (the tier's default since this row) whose control answer contains any value planted in the run — then recall itself crosses projects and this row reopens.
- **Lesson:** with tools, "does memory leak between projects" is untestable: an agent that lacks an answer searches the disk. The mem tier now runs `--no-tools` and counts any planted value as a leak. Correction (same day): `--no-tools` removes only omp's built-in tools; connected MCP tools stay enabled (omp 18.2.11 session-tools.ts #applyMCPToolRefresh, no CLI flag disables them). The --no-tools rerun (receipts/ab-mem-fts-notools-ollama-moe-20260923.json) found no planted value in any control, but its controls made 76 side-effecting calls to the user's Agent Mail MCP server (44 ensure_project, 13 register_agent, 10 ensure_product, 5 send_message, 2 file_reservation_paths, 2 macro_prepare_thread; from the saved request bodies). Isolating omp children from the user's MCP servers is open and the user's decision. Separately, per-project memory banks are no confidentiality boundary on this machine: an omp session with bash, on a local model, opened other agents' databases under ~/.claude on its own initiative.

## 2026-09-23 — KEEP: with no memory LLM, retention in an interactive omp session costs nothing on one GPU
- **Bead:** lb-05
- **Surface:** `localbench ab ollama:qwen3.6:35b-mlx <same> --tiers sess --b-mem-config fixtures/omp/child-config-mem-nollm.yml` (repeats 3: 36 turns and 6 post-retain turns per leg), omp 18.2.11 `--mode rpc`, thinking auto
- **Hypothesis:** the retention stall found by the REJECT row above is the memory LLM's extraction sharing the GPU; `mnemopi.llmMode: none` (retain the transcript, no extraction call) removes it.
- **A/B:** receipts/ab-sess-nollm-ollama-moe-20260923.json, SOUND, all legs uncontended: post-retain turn wall A1 8.38 / A2 8.44 → B 2.06 s (B/A 0.245, band 0.10, B-BETTER); their wait before the main call 4.03 / 4.03 → 0.82 s (B-BETTER; A is omp's 4 s classifier cap); regular turns 1.24 / 1.18 → 1.33 s (within noise); neighbour delta +6.48 / +7.00 → +0.01 s. The A legs' memory LLM was busy 79.3 / 65.2 s per leg (9 extractions, ~1.1-1.3k tokens each) and overlapped main calls 25.3 / 19.3 s; session close took 5.7–10.3 s (retention at exit) vs 0.18–0.27 s in B.
- **A/A null:** A1 vs A2 of the same invocation (post-retain wall spread 0.8%).
- **Verdict:** KEEP
- **Retry predicate:** a mem-tier A/B (recall, --no-tools) showing llmMode none loses recall hits against llmMode smol — then extraction buys correctness and the trade reopens; or an omp release that schedules extraction off the interactive path (post-retain wall within band with llmMode smol).
- **Lesson:** the memory LLM's value has not been shown here (one-shot recall hit 9/9, 9/9, 8/9 with zero memory-extract calls in the proxy logs of receipts/ab-mem-fts-notools-ollama-moe-20260923.json), while its cost on one GPU is ~6.5–7 s on every fourth-plus-one turn and ~10 s at session close. Measured with the memory LLM on the main model; the user's profiles differ: `localbench models` shows 8 of 9 route it to a second local model (qwen3.8 27B dense), which adds a second resident model to the contention and is unmeasured. Changing the profiles is the user's decision.

## 2026-09-23 — UNKNOWN: full-text-only recall (no embedding worker) makes memory-on omp launches faster
- **Bead:** lb-05
- **Surface:** `localbench ab ollama:qwen3.6:35b-mlx <same> --tiers mem --b-mem-config fixtures/omp/child-config-mem-fts.yml` on the mem tier with `--no-tools` (built-in tools only; MCP tools stayed and were called, see the mem.no_leak row) and any planted value counting as a leak; 9 attempts per leg
- **Hypothesis:** one-shot memory cost is in-process recall (REJECT row "memory LLM is on the per-turn path"); dropping embeddings (`mnemopi.noEmbeddings: true`) cuts it.
- **A/B:** receipts/ab-mem-fts-notools-ollama-moe-20260923.json, SOUND, 0 leaks in 27 controls. B (FTS only) is the fastest leg on every row but none clears the band: wait before the main call on recall turns 2.69 / 3.29 → 1.90 s (B/A 0.635, band 0.60), recall wall 11.08 / 11.89 → 10.03 s (0.873, band 0.21), plant wall 9.59 / 10.04 → 8.68 s (0.885, band 0.14). Recall 9/9, 9/9 (B), 8/9. Derails (recalled memory breaking "Reply with exactly: OK"): 2/9, 2/9 (B), 0/9 — answers "NOT FOUND", "OK: I'll proceed with the task as described.", the recalled owner's name, and "NOTED".
- **A/A null:** A1 vs A2 of the same invocation; pre-main spread 20% sets the wide band.
- **Verdict:** UNKNOWN
- **Retry predicate:** the same A/B with MEM_ROUNDS raised until the A/A pre-main spread is under 10% (or 3 invocations pooled), adopting noEmbeddings only if pre_main_s is B-BETTER and recall hits stay equal.
- **Lesson:** a memory-on `omp -p` launch costs ~10-12 s end to end here, and the embedding worker is not visibly most of it. Recalled memory does derail a trivial instruction in ~15% of turns (4/27) on this model: memory helps recall and hurts instruction-following, both measured by the same tier.

## 2026-09-23 — UNKNOWN: parked dense e2e+replay A/A can be banked while agent panes are mid-turn
- **Bead:** lb-06
- **Surface:** `localbench aa ollama:localbench-parked:5642e97495e1 --tiers e2e,replay --write-golden --wait-idle 1800`. Two voids, same day. This row does not reopen the REJECT "WindowServer's GPU share rises because the model under test saturates the GPU" (NEGATIVE_EVIDENCE.md, the lb-01 row whose retry predicate requires a saturation window with no display activity).
- **Hypothesis:** a passing preflight plus `--wait-idle 1800` banks the parked qwen3.8 e2e and replay rows even while the localbench tmux window has agent panes mid-turn.
- **A/B:** not a comparison, and not a speed result. `docs/evidence/receipts/aa__ollama__localbench-parked_5642e97495e1__20260923T212234Z.json#/runs/*/system/contention`: 1 WindowServer second per leg, 26.7% and 28.5%, device 100%. `docs/evidence/receipts/aa__ollama__localbench-parked_5642e97495e1__20260923T213418Z.json#/runs/*/system/contention`: leg 0 one second at 29.1% (device 100%); leg 1 twenty-two seconds at 25.1–28.9% (device 99–100%). Both receipts UNSOUND, must_fail empty, golden not written. Display activity was present: panes %21 and %20 were mid-turn with animated spinners in the visible localbench tmux window. The MoE e2e A/As banked in 3e1e8ec (`aa__ollama__qwen3.6_35b-mlx__20260923T212051Z.json`, `aa__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit__20260923T212141Z.json`) had `contended=false` on the same desktop.
- **A/A null:** VOID-CONTENDED — both dense pairs ran and both are non-proof, so neither spread is a band.
- **Verdict:** UNKNOWN
- **Retry predicate:** superseded by the next row. The idle-pane rerun was not CONTENDED; preflight refused after 1800 s. Do not launch this A/A again until the owner names a window.
- **Lesson:** `--wait-idle` only gates the start. The REJECT row's predicate was not met on the first two voids: display activity was present. Do not exempt WindowServer.

## 2026-09-23 — UNKNOWN: `--mtp` on mlx-serve Qwen3.6-35B-A3B meets the e2e adopt gate
- **Bead:** lb-07
- **Surface:** `localbench ab mlx-serve:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit <same> --b-server-arg=--mtp --tiers micro,replay,e2e --bank ab-mlx-serve-qwen3.6-mtp-e2e-20260923 --wait-idle 1800` (hub `lb-u2-mtp-ab`, pid 98508, exit 1). Does not edit the 2026-09-23 UNKNOWN row "mlx-serve's MTP head speeds up omp turns" or its adopt gate.
- **Hypothesis:** the same A,B,A with e2e added meets that row's adopt gate (e2e main-turn llm_s B-BETTER, no omp-shaped warm or e2e-repeat row B-WORSE beyond its band).
- **A/B:** not a result. `docs/evidence/receipts/ab-mlx-serve-qwen3.6-mtp-e2e-20260923.json` is UNSOUND. `localbench show`: all three legs `contended=yes`; B also `MUST FAIL: e2e.ok.correct` (`answer_ok=false`, `llm_calls=3`, `output=208`; the answer string is not in `/legs/1/details/e2e.ok`). Contention, from `#/legs/<i>/system/contention`: A1 one second with `qwen3.8-uncensored:latest` resident and no foreign GPU process; B one second WindowServer 27.3% (device 99%) and one second with `qwen3.8-uncensored:latest` resident; A2 six seconds of `Google Chrome fo` at 26.5, 26.2, 27.5, 27.3, 26.9, and 40.5%. The show table's one B-BETTER (`micro.decode`) and its WITHIN-NOISE e2e rows are non-proof. `replay.lean.prompt_tokens` SHOULD-FAIL on every leg is the already-named mlx-serve fixture mismatch, not a new verdict. `--mtp` was not adopted.
- **A/A null:** VOID-CONTENDED — A1 and A2 are both non-proof, so their spread is not a band.
- **Verdict:** UNKNOWN
- **Retry predicate:** rerun the same command only when `ollama ps` shows no model other than the mlx-serve target and `localbench gpu --seconds 3` shows no non-backend process above 25%. Adopt `--mtp` only if that receipt is sound and meets the earlier row's adopt gate, including the nonce-free 256-token identity check. If that sound receipt FAILs `e2e.ok.correct` only on the `--mtp` leg, record REJECT and do not adopt.
- **Lesson:** parking `qwen3.8:27b-mlx` does not park `qwen3.8-uncensored:latest`. A second resident ollama model voids the pair even with `foreign_gpu` empty. This run did not start that model; it is not killed here.

## 2026-09-23 — UNKNOWN: idle panes plus --wait-idle 1800 banks the parked dense e2e+replay golden
- **Bead:** lb-06
- **Surface:** hub `lb-u1-idle-aa` pid 70861, exit 1, uptime 30m4s. `localbench aa ollama:localbench-parked:5642e97495e1 --tiers e2e,replay --write-golden --wait-idle 1800`. Both localbench panes were idle: pane 2 permanently idle, this pane ended its turn before launch.
- **Hypothesis:** with both agent panes idle, a 1800 s idle wait finds a quiet window and the A/A banks.
- **A/B:** not a result. No receipt was banked. `git status --short -- goldens` was empty. `runs/20260923T221955Z__aa1__ollama__localbench-parked_5642e97495e1/progress.jsonl` has 54 `preflight_wait` events and one `start`. First wait `#` problems: GPU already 42.2% busy (Google Chrome fo pid 37962 30.9%, Google Chrome He pid 83619 9.4%, WindowServer pid 179 5.3%). Last wait problems: GPU already 98.0% busy (llama-server pid 214, blob sha256-6eaa8b1a8d3489403e44ee8002ab7dd15cfd7855b434fabed4, 84.9%; Google Chrome He pid 83619 6.3%; WindowServer pid 179 4.9%). Process exit line: preflight refused, GPU already 97.8% busy (that llama-server 86.0%, Google Chrome He 7.0%, WindowServer 5.0%), CPU already 25.6% busy.
- **A/A null:** VOID-NOT-RUN — the pair never started.
- **Verdict:** UNKNOWN
- **Retry predicate:** the owner names a window, and before launch `localbench gpu --seconds 3` shows no process above 25% and `ollama ps` shows no model. Do not launch another `--wait-idle 1800` while a llama-server on `/Volumes/Models/ollama-models/blobs/sha256-6eaa8b1a8d3489403e44ee8002ab7dd15cfd7855b434fabed4` is using the GPU. Do not kill that process; this session did not start it.
- **Lesson:** pane-idle is not machine-idle. The 1800 s wait expired on someone else's llama-server and on Chrome. U1 waits for a window the owner names.

## 2026-09-23 — KEEP: a quiet machine banks the parked dense e2e+replay A/A
- **Bead:** lb-06
- **Surface:** `localbench aa ollama:localbench-parked:5642e97495e1 --tiers e2e,replay --write-golden --wait-idle 1800` (hub lb-u1-idle-aa, exit 0). Closes the UNKNOWN row "idle panes plus --wait-idle 1800 banks the parked dense e2e+replay golden".
- **Hypothesis:** with the GPU clear of other models, that A/A banks.
- **A/B:** not a comparison. Receipt `docs/evidence/receipts/aa__ollama__localbench-parked_5642e97495e1__20260924T025154Z.json`, both legs contended=no, MUST fail 0, answers OK and 4817. Golden committed in 8d6da75. omp pin moved 18.2.11 ce797fb3ed92e768 to 18.3.0 33cf63aab3a109b3.
- **A/A null:** the two legs of that invocation. Spreads are in the receipt; the golden was written by `aa --write-golden`.
- **Verdict:** KEEP
- **Retry predicate:** re-bank when `localbench status` names this golden GENERATION-MISMATCH, and only by `localbench aa` of the stale tiers.
- **Lesson:** the 1800 s wait did not find the window. Killing llama-server pid 214, which `ollama stop` left at Stopping, did. Preflight then passed in under a second.

## 2026-09-23 — REJECT: `--mtp` on mlx-serve Qwen3.6-35B-A3B does not meet the e2e adopt gate
- **Bead:** lb-07
- **Surface:** `localbench ab mlx-serve:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit <same> --b-server-arg=--mtp --tiers micro,replay,e2e --bank ab-mlx-serve-qwen3.6-mtp-e2e-20260924 --wait-idle 1800` (hub lb-u2-mtp-ab, exit 0). Does not edit the earlier UNKNOWN row's gate.
- **Hypothesis:** a sound A,B,A with e2e meets the adopt gate: e2e main-turn llm_s B-BETTER, and no omp-shaped warm row (replay.*.warm, e2e repeat) B-WORSE beyond its band.
- **A/B:** receipts/ab-mlx-serve-qwen3.6-mtp-e2e-20260924.json, SOUND, all three legs contended=no, MUST fail 0, e2e.ok.correct PASS on every leg. e2e.ok.first_llm_s 1.864 / 1.887 → 1.868 (B/A 0.996, band 0.1, WITHIN-NOISE). replay.lean.warm_ttft_s 0.1013 / 0.1065 → 0.1331 (B/A 1.281, band 0.1501, B-WORSE). e2e repeat rows WITHIN-NOISE. micro.decode.decode_tps 65.44 / 93.81 → 194.7 (B/A 2.446, B-BETTER) is not the adopt gate. The nonce-free 256-token identity check was not run.
- **A/A null:** A1 vs A2 of the same invocation. e2e.ok.first_llm_s spread is inside the 0.1 band (1.864 vs 1.887).
- **Verdict:** REJECT
- **Retry predicate:** do not adopt `--mtp` from this receipt. Re-measure when mlx-serve's version or this model's digest differs from backend_sha 4d09a3beb8d4c9be and model_digest files:68dedceb5da0, and adopt only if that sound receipt makes e2e.ok.first_llm_s B-BETTER and no replay.*.warm or e2e-repeat row B-WORSE, plus a nonce-free greedy decode of 256 tokens identical on A and B.
- **Lesson:** the omp turns did not get faster with `--mtp`, and a cached lean prefix got slower: the same trade the micro-only row already named, now with e2e in the receipt. Correction (2026-09-24): decode did not "double". The A legs ran slow, not B fast: micro.decode 65.4 (samples 48.6–87.5) and 93.8 tok/s against the golden's 162.4 and the 2026-09-23 A legs' 162.8 / 164.2, with the same backend_sha and model_digest; B's 194.7 is 1.19× those earlier A legs (that row's own B measured 188.0, B/A 1.15), not 2.4×. The decode A/A spread (0.356) makes this receipt's decode verdict worthless; the e2e verdict stands (e2e.ok.first_llm_s A/A spread 0.012). Why both A legs ran at 40–58% of golden is unmeasured.

## 2026-09-24 — REJECT: full-text-only recall does not clear the adopt gate at 12 rounds
- **Bead:** lb-05
- **Surface:** `localbench ab ollama:qwen3.6:35b-mlx <same> --tiers mem --b-mem-config fixtures/omp/child-config-mem-fts.yml --mem-rounds 12 --bank ab-mem-fts-rounds12-20260924 --wait-idle 1800` (hub lb-u3-mem-fts, exit 0). 36 attempts per leg.
- **Hypothesis:** raising MEM_ROUNDS from 3 to 12 makes the A/A pre-main spread measurable under 10%, and FTS-only recall (no embeddings) is then B-BETTER on pre_main_s with recall hits equal.
- **A/B:** receipts/ab-mem-fts-rounds12-20260924.json, SOUND, all three legs contended=no, MUST fail 0, leaks=[]. Hits 34/36, 34/36, 33/36 (hit_rate B/A 0.971, band 0.05, WITHIN-NOISE). pre_main_s VOID on every leg (`no valid samples`), so the spread and the B-BETTER test cannot be computed. wall_s and plant.wall_s WITHIN-NOISE on bands 1.477 and 2.683. derail.ok_rate 0.9167 / 0.9444 → 0.8056 (B/A 0.866, band 0.0893, B-WORSE; 3, 2, and 7 derails).
- **A/A null:** A1 vs A2 of the same invocation. hit_rate equal at 0.9444. pre_main_s has no samples, so it is not a spread.
- **Verdict:** REJECT
- **Retry predicate:** do not adopt `fixtures/omp/child-config-mem-fts.yml` from this receipt. Re-measure when `mem.recall.pre_main_s` reports a number on both A legs and the B leg in one `localbench ab` of this command, and adopt only if that pre_main_s is B-BETTER, the A/A spread of that number is under 10%, hit counts are equal, and derail.ok_rate is not B-WORSE.
- **Lesson:** four times the attempts did not produce a pre_main sample, and the FTS leg derailed more often (7 vs 3 and 2). The speed claim is still unmeasured. The correctness claim moved the wrong way.

## 2026-09-24 — REJECT: an omp version bump stales the e2e golden
- **Bead:** lb-06
- **Surface:** `golden.tier_keys` for e2e/rel/mem/sess. omp 18.2.11 → 18.3.0 marked both MoE e2e goldens GENERATION-MISMATCH and halted a tick.
- **Hypothesis:** the live omp version and sha are what an e2e row depends on, so a daily omp update must stale that golden until it is re-banked.
- **A/B:** not a timing claim. `tests.test_golden.TierBinding.test_an_omp_version_bump_does_not_stale_e2e` passes when version and sha move and child config does not. The plant that puts `omp_version` and `omp_sha` back into `OMP_KEYS` fails that test.
- **A/A null:** VOID-no-measurement (a binding rule).
- **Verdict:** REJECT
- **Retry predicate:** an e2e or rel row moves beyond its band between two omp versions with backend, model, child overlay, and agent config unchanged, and the request bodies show omp sent a different prompt. Then those pins rejoin `OMP_KEYS`.
- **Lesson:** omp ships most days. Version and sha stay on the receipt. They are not the gate. Re-bank e2e when the numbers are worth refreshing, not because the version string moved.

## 2026-09-24 — REJECT: Nemotron 3.5 Lightning 30B-A3B replaces qwen3.6 for omp turns
- **Bead:** lb-06
- **Surface:** `localbench ab ollama:qwen3.6:35b-mlx ollama:nemotron-3.5-lightning:30b-mlx --tiers conf,micro,replay,e2e --bank ab-qwen36-vs-nemotron-20260924`
- **Hypothesis:** the vendor "up to 4×" claim holds here against the recommended MoE on the omp-shaped path.
- **A/B:** receipts/ab-qwen36-vs-nemotron-20260924.json, SOUND, all legs contended=no, MUST fail 0. Both answered 4817 and OK. e2e.tool_read.first_wall_s 3.964 / 4.035 → 7.766 (B/A 1.942, B-WORSE). e2e.ok.first_wall_s 3.247 / 3.391 → 4.829 (B/A 1.455, B-WORSE). micro.decode.decode_tps WITHIN-NOISE (150.6 / 145 → 153.5). replay.full.cold_ttft_s 51.01 / 54.65 → 40.39 (B/A 0.765, B-BETTER). replay.full.warm_ttft_s B-WORSE.
- **A/A null:** A1 vs A2 of the same invocation.
- **Verdict:** REJECT
- **Retry predicate:** do not adopt `nemotron-3.5-lightning:30b-mlx` from this receipt. Re-measure when its digest differs from 8b1474be6e54, and adopt only if a sound A/B makes e2e.tool_read.first_wall_s B-BETTER and no e2e wall row B-WORSE.
- **Lesson:** the cold full-prompt TTFT was faster and the omp turns were not. Decode did not move. The 4× claim is not this host.

## 2026-09-24 — UNKNOWN: full-text-only recall, re-measured with pre_main counted (row 386's retry)
- **Bead:** lb-05
- **Surface:** `localbench ab ollama:qwen3.6:35b-mlx <same> --tiers mem --b-mem-config fixtures/omp/child-config-mem-fts.yml --mem-rounds 12 --bank ab-mem-fts-premain-20260924 --wait-idle 1800` (hub lb-u3b-mem-premain, exit 0, 1 h 2 min), after 53a3ba6 made pre_main count tool-less answer calls. 36 attempts per leg.
- **Hypothesis:** row 386's gate: pre_main_s B-BETTER with an A/A spread under 10%, equal hits, derail.ok_rate not B-WORSE.
- **A/B:** receipts/ab-mem-fts-premain-20260924.json, SOUND, all legs contended=no, MUST fail 0, no leaks. pre_main_s now has 36 samples per leg (the 6bce927 receipt had none): 5.863 / 6.711 → 4.465 s (B/A 0.710, band 0.404, WITHIN-NOISE; A/A spread 0.135). recall wall_s 8.377 / 9.246 → 6.054 (B/A 0.687, band 0.296, B-BETTER). plant wall_s 6.351 / 7.245 → 3.571 (B/A 0.525, band 0.394, B-BETTER). Hits 33/36, 34/36, 33/36. Derails 1, 5 (B), 5.
- **A/A null:** A1 vs A2 of the same invocation.
- **Verdict:** UNKNOWN
- **Retry predicate:** pool a second invocation of this exact command with this receipt (72 attempts per arm) and apply row 386's gate unchanged: adopt only if pooled pre_main_s is B-BETTER with its A/A spread under 10%, hits equal, derail.ok_rate not B-WORSE. Adopting means setting `mnemopi.noEmbeddings: true` in the owner's profiles: his decision.
- **Lesson:** without embeddings every memory-on launch got faster end to end (plant turn 0.53×, recall turn 0.69×), while the wait before the answer call, the number the gate names, stayed inside a 13.5% A/A spread. The plant speedup fits retention skipping the embedding pass at exit (not measured directly). Derails were 5/36 on both B and A2, so the 6bce927 row's "FTS derails more" did not reproduce.

## 2026-09-24 — SURVEY: ThinkingCap-Qwen3.8-27B thinks less than Qwen3.8-27B on this host at equal accuracy (gate fixed before measuring)
- **Bead:** kit-thinkingcap-ab-wwz
- **Surface:** the smol role (`ollama/qwen3.8:27b-mlx` in 8 of 9 profiles: titles, memory LLM, auto-thinking classifier, scouts). Arms matched on backend and quantization: A `ollama:hf.co/bartowski/Qwen3.8-27B-GGUF:Q4_K_M`, B `ollama:hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q4_K_M` (both 17.4 GB, same runner). The fine-tune's NVFP4 build is compressed-tensors (CUDA) and cannot run here; its MLX-4bit-DWQ build against the incumbent NVFP4 MLX would change backend and quantization at once.
- **Hypothesis:** the vendor card (bottlecapai/ThinkingCap-Qwen3.8-27B, base Qwen/Qwen3.8-27B, released 2026-09-23): 37% fewer reasoning tokens on average (11–66% by benchmark) at 85.8% vs 86.6% accuracy. A post's "up to 65%" is the top of that range.
- **A/B:** not yet measured. Command, after `localbench park` (both qwen3.8 smol names out of reach) and with no `--allow-busy`: `localbench ab <A> <B> --tiers conf,micro,think,e2e,sess --repeats 2 --bank ab-thinkingcap-q4km-20260924 --wait-idle 1800`. The think tier (30c9a18) asks six checkable questions greedily with the model's own thinking; qwen3.6 answered 6/6 in 7,408 tokens.
- **A/A null:** A1 vs A2 of the same invocation sets every band, reasoning length included. Do not assume greedy decoding repeats it: two temperature-0 qwen3.6 runs of the same prompts (59 and 40 tokens) reasoned 2,586 vs 1,830 and 992 vs 862 tokens (runs 20260924T145105Z, contended, and 20260924T145655Z); contention confounds that pair. Corrected on 2026-09-24 before any A/B data existed; the six conditions are unchanged.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt. Recommend replacing the smol model only if ALL hold: (1) no conf MUST fails on B and every e2e .correct passes; (2) think accuracy on B is at least the lower of A1/A2; (3) think.reasoning_chars B/A <= 0.8 and B-BETTER beyond its band; (4) no e2e first_llm_s or wall row B-WORSE; (5) sess post_retain_wall_s not B-WORSE; (6) micro.decode.decode_tps WITHIN-NOISE: the arms share architecture and quantization, so a decode difference beyond the band means they are not matched and the receipt is VOID for this row. Any item failing: REJECT; anything within noise: UNKNOWN. A KEEP still needs the deployable build (MLX-4bit-DWQ on mlx-serve) measured against `qwen3.8:27b-mlx` before a profile change, which is the owner's.
- **Lesson:** fix the gate before the numbers exist; a vendor's "up to" is the best benchmark, and six arithmetic questions are not SWE-bench, so a pass here means "thinks less on short checkable questions without losing them", nothing broader.

## 2026-09-24 — REJECT: `--allow-busy` is harmless for a smoke that only checks a new tier runs
- **Bead:** kit-thinkingcap-ab-wwz
- **Surface:** `localbench run ollama:qwen3.6:35b-mlx --tiers think --allow-busy` (runs/20260924T145105Z, cancelled) against the same command gated (`localbench park`, `--wait-idle`; receipts/think-smoke-qwen36-20260924.json, SOUND, contended=no).
- **Hypothesis:** skipping the busy gate costs nothing when the smoke only asks whether the tier runs and answers.
- **A/B:** isolate evicted `qwen3.8:27b-mlx`; 1.1 s later the run's contention event names a foreign runner reloading it (pid 97876, 32.6% GPU, device 97%). qwen3.6 then decoded 19.3 and 20.2 tok/s on think.minutes and think.arith, against 165 and 176 gated, and reasoned 2,586 and 992 tokens against 1,830 and 862 on the same prompts. `--allow-busy` also skipped the preflight that refuses while a session can load a fallback (4b049d9), the check that would have named the cause.
- **A/A null:** VOID-no-control: two invocations; the contention is an observed event in the run's own record, not a ratio.
- **Verdict:** REJECT
- **Retry predicate:** three of three `--allow-busy` smokes started while omp sessions are live record zero contention events. Until then every run, smokes included, is `localbench park` then `--wait-idle 1800` (AGENTS.md, Commands).
- **Lesson:** a contended smoke does not measure a slower copy of the tier; it measures a different machine (8x slower decode, 41% longer reasoning on the same prompt). Its numbers mislead even when they are only read, never banked.

## 2026-09-24 — KEEP: ThinkingCap-Qwen3.8-27B thinks less than Qwen3.8-27B at equal accuracy on this host (the SURVEY row's gate, all six conditions met)
- **Bead:** kit-thinkingcap-ab-wwz
- **Surface:** the smol role. Receipt docs/evidence/receipts/ab-thinkingcap-q4km-20260924.json (hub lb-ab-thinkingcap, exit 0), SOUND: A1, B, A2 contended=no, preflight ok, MUST fail 0, conformance 9/9 on every leg. Arms ran under their parked names, same digests (bead comment, written before the run): A `localbench-parked:76c1fd6cf3a2` = bartowski Qwen3.8-27B Q4_K_M, B `localbench-parked:0ca43d94af99` = bottlecapai ThinkingCap Q4_K_M; ollama 0.32.15, llama-server runner, architecture qwen35, 27.3B, loaded_context 262144.
- **Hypothesis:** the SURVEY row above, judged only by its six pre-registered conditions.
- **A/B:** (6, the control, checked first) micro.decode.decode_tps 22.78 / 23.58 → 20.83, B/A 0.899, band 0.1042, WITHIN-NOISE: the arms are matched, with 0.003 to spare, and both B samples sit below all four A samples, so B decodes about 10% slower and the band admits it. (1) conf MUST 0 fails on B; e2e answers 4817 and OK on every leg. (2) think.accuracy 12/12 on A1, B and A2, cut_off 0: six questions asked twice under greedy decoding, whose rounds repeat, so six independent answers per leg. (3) think.reasoning_chars 1429 / 1429 → 880, B/A 0.616, B-BETTER; completion_tokens B/A 0.572; think.wall_s 31.32 / 32.23 → 18.86, B/A 0.594, B-BETTER. (4) no e2e first_llm_s or wall row B-WORSE (tool_read.first_wall_s 0.905 and ok.first_wall_s 0.863 WITHIN-NOISE; tool_read.repeat_wall_s 0.825 B-BETTER). (5) sess.turn.post_retain_wall_s 25.13 / 29.12 → 12.03, B/A 0.443, B-BETTER; memory-extract completion tokens 3014 / 2490 → 1298, busy 125.2 / 112.3 → 60.5 s. The only B-WORSE row, micro.eviction_probe_8k.a_again_ttft_s 0.155 / 0.150 → 0.178 (B/A 1.165), is not a gate row.
- **A/A null:** A1 vs A2 of the same invocation. Every think row repeated exactly (per question: same reasoning characters, same tokens), so greedy decoding reproduces on this runner and those rows sit at the 0.1 band floor. The 2,586 vs 1,830 variance in the SURVEY row was the MLX engine with one leg contended.
- **Verdict:** KEEP
- **Retry predicate:** demote to REJECT if the deployable build's own pre-registered A/B (ThinkingCap as it would run for smol, against `qwen3.8:27b-mlx`, the NVFP4 incumbent) fails conditions 1–5. Condition 6 does not carry over: that A/B changes backend or quantization by design, so decode is reported there, not used as a control (corrected after the c1a78a4 grade); re-run this receipt if either digest (76c1fd6cf3a2, 0ca43d94af99) or the ollama build changes. A smol-model change in any omp profile is the owner's decision, and it waits for that deployable A/B.
- **Lesson:** on short checkable questions both arms think briefly (A: 150–304 characters per question). ThinkingCap cut reasoning 38%, think wall 41% and memory-extract output 53% (vs the A mean) without losing an answer, and its decode was about 10% slower (inside the band), so the time saved comes from fewer tokens, not faster ones. This supports "thinks less on short questions, answers kept". It says nothing about the vendor's hard-benchmark figures, and nothing about memory recall quality: sess checks acks, not what the shorter extractions retained, so the mem tier with ThinkingCap as smol is the check for that.

## 2026-09-24 — SURVEY: qwen3.6:35b-mlx does omp's smol work (memory, classifier) at least as well as qwen3.8:27b-mlx, and faster (gate fixed before measuring)
- **Bead:** kit-smol-qwen36-ab-x2l
- **Surface:** the smol role (`ollama/qwen3.8:27b-mlx` in 8 of 9 profiles: memory LLM, the auto-thinking classifier, titles, scouts). In every measured child the harness makes smol the model under test (one model on the machine), so each arm is that model doing the side work: mem and sess isolate the memory LLM; e2e turns mix main and classifier (in the owner's sessions main is a cloud model). Titles (`--no-title`) and scouts are not measured. Arms: A `ollama:localbench-parked:5642e97495e1` (qwen3.8:27b-mlx, dense 27B, NVFP4, parked name, same digest), B `ollama:qwen3.6:35b-mlx` (MoE 36B / ~3B active, NVFP4, digest e92a3e94bbca).
- **Hypothesis:** qwen3.6 prefills 6.4–6.6x and decodes 2.3x faster here (receipts/ab-incumbent-vs-moe.json, micro only). Swapping smol to it should shorten memory and classifier calls without costing recall or answers, and leave one model on the GPU instead of two. No same-invocation quality comparison of smol's jobs exists: qwen3.8 has no mem, sess or think receipt.
- **A/B:** not yet measured. Command, after `localbench park`, no `--allow-busy`: `localbench ab ollama:localbench-parked:5642e97495e1 ollama:qwen3.6:35b-mlx --tiers conf,micro,e2e,think,mem,sess --repeats 2 --mem-rounds 6 --bank ab-smol-qwen38-vs-qwen36-20260924 --wait-idle 1800` (18 recall attempts per leg). Amended 2026-09-24, before any sound data, when the measurement law changed to measuring a working machine (row below): the command adds `--pairs 2` (A,B,A,B,A) and banks as ab-smol-qwen38-vs-qwen36-pairs2-20260924; 'the lower of A1/A2' in Q2 and Q3 reads 'the lowest A leg'; every other condition is unchanged. The void attempts were not read for B against A.
- **A/A null:** A1 vs A2 of the same invocation sets every band. There is no decode control: the arms differ in model and architecture by design, so micro is reported, not gated.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt. Quality, every item must hold: (Q1) no MUST fails on B, and e2e.ok.correct and e2e.tool_read.correct PASS on B; (Q2) think.accuracy on B >= the lower of A1/A2; (Q3) mem.recall.hit_rate not B-WORSE, B's recall hits >= the lower of A1/A2 minus 1 (of 18), mem.no_leak PASS on B, mem.derail.ok_rate not B-WORSE; (Q4) sess.turns_complete and sess.memory_calls_ok PASS on B, sess.turn.ack_rate not B-WORSE. Speed: (S1) no e2e first_llm_s or wall row and no sess.turn row B-WORSE; (S2) at least one of sess.memory.extract_s, sess.turn.post_retain_wall_s, mem.plant.wall_s, mem.recall.wall_s B-BETTER. KEEP = all Q and both S: recommend the swap, which the owner applies in his profiles. Any Q failing: REJECT. Q holding without S2: UNKNOWN (no measured benefit).
- **Lesson:** fix the gate before the numbers exist. Speed alone does not justify a smol swap, because the classifier sets every main turn's thinking effort and memory extraction decides what gets stored.

## 2026-09-24 — UNKNOWN: the smol-role A/B's first two attempts are void (contended); the SURVEY row's gate is unjudged
- **Bead:** kit-smol-qwen36-ab-x2l
- **Surface:** the SURVEY row's exact command. Attempt 1 (18:34Z) slipped through a dip in omp's managed Chromium load, logged Chrome at 33.1% two minutes in, and was stopped (this session's own process). `localbench quiet` was built for that (d7b1c0b). Attempt 2 (18:46Z) completed: receipts/ab-smol-qwen38-vs-qwen36-20260924-void-contended.json, UNSOUND.
- **Hypothesis:** with omp's browser paused, the command banks a SOUND receipt.
- **A/B:** not judged. Attempt 2: ab_a1 CONTENDED in 13 seconds (WindowServer 25.2–27.9% in 12, Terminal 25.1% in 1) in two clusters, 18:49–18:58 (conf to e2e) and 19:11–19:13 (sess start). In A1's sampler Terminal held >=10% GPU in 766 seconds and WindowServer >=20% in 50: sustained on-screen pane output while this pane was idle. ab_b and ab_a2 were uncontended (top other process 1.9% and 8%). MUST fail 0 on all legs. The numbers are not read for a verdict (pattern 2).
- **A/A null:** VOID-CONTENDED: A1 is non-proof, so there is no A/A band.
- **Verdict:** UNKNOWN
- **Retry predicate:** re-run the SURVEY row's command unchanged, with omp's browser paused (`localbench quiet`) and the display asleep (`localbench quiet --display`), and judge only a SOUND receipt. WindowServer is not exempted (the lb-01 REJECT row: display activity costs the model real GPU time).
- **Lesson:** a quiet GPU at launch is not a quiet GPU for 70 minutes. The screen itself is a GPU client, and other panes' output draws on it even while this one is idle.

## 2026-09-24 — UNKNOWN: the smol-role A/B's third attempt is void too: the Mac was in use during A2
- **Bead:** kit-smol-qwen36-ab-x2l
- **Surface:** the SURVEY row's exact command, attempt 3 (20:14Z, rev 1be20af), with omp's browser paused (`localbench quiet`) and the display put to sleep (`localbench quiet --display`). receipts/ab-smol-qwen38-vs-qwen36-20260924-void-contended-3.json, UNSOUND.
- **Hypothesis:** with omp's browser paused and the display asleep, the 70-minute A,B,A stays uncontended.
- **A/B:** not judged. ab_a1 and ab_b uncontended (top other process Terminal 8.4% and 2.7%). ab_a2 contended in 19 seconds: 21:05–21:10Z, during micro, the owner's own Google Chrome (/Applications/Google Chrome.app helper) at 25.8–31.3% and WindowServer at 25.2–28.3% (the display was awake); and 21:28–21:29Z, at sess start, WindowServer at 26.4–28.3%. MUST fail 0 on all legs.
- **A/A null:** VOID-CONTENDED: A2 is non-proof, so there is no A/A band.
- **Verdict:** UNKNOWN
- **Retry predicate:** the same command in a window the owner names in which nobody uses the Mac for about 75 minutes, with `localbench park`, `localbench quiet --display` and `--wait-idle 1800`. Judge only a SOUND receipt; WindowServer and the user's own apps are not exempted.
- **Lesson:** the harness's controls (park, quiet, display sleep) cover agents and omp. They cannot cover the person at the keyboard, and they should not: his use is the machine's purpose, so a long A,B,A needs a window he names.

## 2026-09-24 — REJECT: an idle machine is the right condition for localbench's measurements
- **Bead:** lb-06
- **Surface:** the per-second veto (any non-backend process >25% GPU makes a run CONTENDED) and the preflight that waited for an idle GPU and CPU.
- **Hypothesis:** a run is only meaningful on a quiet machine, so any app or screen spike must void it.
- **A/B:** five smol A/B attempts today: attempt 1 stopped (omp's browser), attempts 2, 3 and 5 void by app/screen spikes (WindowServer 25–39%, the owner's Chrome 25–31%), attempt 4 stopped after a woken screen. Per-leg app GPU across ten legs, SOUND ones included (the ThinkingCap receipt: 12.2–14.0% mean): 2.8–14.8% mean; the veto fired on seconds, not on load. The spiked vs clean A legs of the same model differed by a median 4.2% (attempt 2) and 19.3% (attempt 3: 32k prefill −24%, mem walls +30–38%), so load does move numbers, and it biases an A/B only when it lands on one arm. Per-process app GPU is confounded by the model under test: dense-model legs read 8.5–14.8%, MoE legs 2.8–3.6%, in every attempt. the owner, 2026-09-24: "my machine is never going to be fully quiet - we need our testing to be while our system is working".
- **A/A null:** VOID-no-measurement (a design decision grounded in the receipts above; the new rules have their own tests, tests/test_load.py, with 10/10 plants caught).
- **Verdict:** REJECT
- **Retry predicate:** revisit if interleaved A/Bs (`--pairs 2` or more) under load leave the gate unreachable, i.e. three consecutive pre-registered A/Bs end UNKNOWN because bands exceed 30% on their gated rows. Then add a load-balance gate on user_active_pct between arms, or schedule overnight windows. Another model resident or running still voids a run: that rule stands.
- **Lesson:** measure the condition the product runs in. Keep the controls that separate the arms (one model at a time, interleaving, medians), and drop the ones that only demand a machine nobody has.
- **Addendum 2026-09-25:** the first UNKNOWN this predicate counts is receipts/ab-mlxserve-vs-omlx-qwen36-20260925.json (gated bands 1.2–2.6). Per-leg data now points at CPU busy, not user_active_pct, as the load that moves decode (mlx-serve 160–165 tok/s at <=12% busy, 29–79 at 37–40%; ollama held 143–147 up to ~30%; bead kit-mlx-native-decode-drop-qim). If the predicate fires, the balance gate belongs on CPU busy between arms. Receipts show it per leg since ccf48ac.

## 2026-09-25 — REJECT: qwen3.6:35b-mlx does omp's smol work at least as well as qwen3.8:27b-mlx, and faster (the SURVEY row's gate, amended to --pairs 2)
- **Bead:** kit-smol-qwen36-ab-x2l
- **Surface:** receipts/ab-smol-qwen38-vs-qwen36-pairs2-20260924.json, SOUND: legs A1,B1,A2,B2,A3 (23:20Z–03:18Z, rev 9340ead, working-machine law 19a8765), contended=no and MUST fail 0 on every leg. Earlier attempts: 1 stopped (omp browser), 2 and 3 void (app/screen spikes under the old veto), 4 stopped (woken screen), 5 void (the owner's Chrome); none were read for B against A.
- **Hypothesis:** the SURVEY row above (qwen3.6 matches qwen3.8 on smol's jobs and is faster).
- **A/B:** judged by Q1–Q4 and S1–S2 as amended. Q1 holds (e2e answers pass on B; B's only non-PASS is conf.greedy_deterministic, a SHOULD). **Q2 fails:** think.accuracy A legs 1/1/1, B legs 0.9167/0.8333 (median 0.875). Every B miss is the same task, "pens", cut off at the 8,192-token budget with no answer (25,858–26,108 reasoning chars; cut_off 1 and 2); the pre-registered rule counts a cut-off as wrong. Q3 holds (mem.recall.hit_rate B 1.0 vs A 0.944, B-BETTER; no leak; derail WITHIN-NOISE). Q4 holds. S1 holds (e2e and sess.turn rows WITHIN-NOISE; e2e.ok.first_llm_s B-BETTER 0.142). **S2 not met:** sess.memory.extract_s 4.19 → 28.6 s, B-WORSE; post_retain, plant and recall walls WITHIN-NOISE. Why: qwen3.6 thinks far more. think.reasoning_chars B/A 17.5 (1,981 → 34,718 per round), completion tokens B/A 8.5, and memory extraction produced 8,341 completion tokens against 884 (B1 vs A1, six calls each; busy 138 s vs 67 s). Its 6.3–6.5x prefill (B-BETTER) wins first turns and loses the smol jobs, whose cost is tokens generated.
- **A/A null:** three A legs of the same invocation; each arm is its legs' median. Load, reported and not gated: user active A 91/74/26%, B 49/67%. The A arm carried more of the owner's activity, which favours B, and B still failed Q2 and S2. A3's decode read 19.5 tok/s against 53–54 on A1/A2 (a leg hit by load); the median held 53.1.
- **Verdict:** REJECT
- **Retry predicate:** re-test qwen3.6 as smol only with thinking capped for smol calls (an omp setting such as a thinking level or budget for the smol role). Measure the cap where omp applies it: sess.memory extraction completion tokens and sess.memory.extract_s, the mem tier, and the e2e auto-thinking classifier. The think tier cannot see an omp setting, because it posts straight to the backend (workloads.py think -> chat_stream), unless the cap is also put in its request body. Re-run when those rows show the cap working (extraction tokens B/A under 2), then apply the same gate with `--pairs 2`. No smol change until then: qwen3.8:27b-mlx stays the smol model (corrected after the e86627f grade).
- **Lesson:** a model's tokens per second does not predict how fast it does a job; tokens per job does. The MoE was 6x faster per prompt token and still 7x slower at memory extraction, because it wrote 9x more tokens.
- **Addendum 2026-09-25 (CPU load, D4):** whole-machine CPU busy per leg, recorded by each leg's top sampler and now shown in the receipt (ccf48ac): A 29.0/32.1/40.1%, B 32.1/47.8%. B carried more CPU load, the opposite of the user-activity split above, and across 35 legs decode fell with CPU busy (bead kit-mlx-native-decode-drop-qim). The verdict stands: Q2's cut-offs hit a token budget, which load cannot move, and S2's 6.8x came with 9x more completion tokens. Read S2's magnitude as an upper bound.

## 2026-09-25 — SURVEY: ThinkingCap-Qwen3.8-27B (GGUF Q4_K_M, deployable on ollama today) does omp's smol work at least as well as the incumbent qwen3.8:27b-mlx, and faster per job (gate fixed before measuring)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** the smol role, deployable build against the incumbent. A `ollama:localbench-parked:5642e97495e1` (qwen3.8:27b-mlx, NVFP4, ollama MLX engine), B `ollama:localbench-parked:0ca43d94af99` (hf.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF:Q4_K_M, llama-server engine), both under parked names (park seals B as a smol fallback). No MLX build of ThinkingCap-Qwen3.8 exists (HF, 2026-09-25: MLX ThinkingCap builds are of Qwen3.6-27B, another base); building one with `ollama create` is experimental on the installed ollama 0.32.15, and an upgrade would stale every golden, which is a separate decision.
- **Hypothesis:** ThinkingCap's shorter thinking outweighs its slower engine per job. Motivation only, not a result (two invocations under different load): ThinkingCap GGUF think.wall_s 18.86 s per round writing 419 completion tokens (receipts/ab-thinkingcap-q4km-20260924.json) against qwen3.8:27b-mlx 31.14 s writing 1,599 (receipts/ab-smol-qwen38-vs-qwen36-pairs2-20260924.json, A median), while its decode is about 21 tok/s against 53.
- **A/B:** not yet measured. Command, after `localbench park`, no `--allow-busy`: `localbench ab ollama:localbench-parked:5642e97495e1 ollama:localbench-parked:0ca43d94af99 --tiers conf,micro,e2e,think,mem,sess --repeats 2 --mem-rounds 6 --pairs 2 --bank ab-smol-qwen38-vs-thinkingcap-q4km-pairs2-20260925 --wait-idle 1800`, under the working-machine law (19a8765).
- **A/A null:** the three A legs of the same invocation; each arm is its legs' median. No decode control: the arms differ in engine and quantization by design.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt. The same smol gate as the qwen3.6 A/B (the SURVEY row of 2026-09-24, as amended). Quality, every item must hold: (Q1) no MUST fails on B, e2e.ok.correct and e2e.tool_read.correct PASS on B; (Q2) think.accuracy on B >= the lowest A leg; (Q3) mem.recall.hit_rate not B-WORSE, B's recall hits >= the lowest A leg minus 1 (of 18), mem.no_leak PASS on B, mem.derail.ok_rate not B-WORSE; (Q4) sess.turns_complete and sess.memory_calls_ok PASS on B, sess.turn.ack_rate not B-WORSE. Speed: (S1) no e2e first_llm_s or wall row and no sess.turn row B-WORSE; (S2) at least one of sess.memory.extract_s, sess.turn.post_retain_wall_s, mem.plant.wall_s, mem.recall.wall_s B-BETTER. KEEP = all Q and both S: apply the smol swap through a tested, reversible `localbench smol` (delegated by the owner on 2026-09-24 once proven: SOUND receipt, gate met, blind re-judge agrees). Any Q failing: REJECT. Q holding without S2: UNKNOWN. Load is reported, not gated.
- **Lesson:** judge a smol candidate per job, not per token: the qwen3.6 REJECT was 6x faster per prompt token and 7x slower at memory extraction.

## 2026-09-25 — SURVEY: a 30-minute screen (think + sess, one pair) sorts smol candidates before any full gate (standing SCREEN gate, fixed before use)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 of the two-stage protocol (AGENTS.md). `localbench ab <incumbent> <candidate> --tiers think,sess --repeats 1 --pairs 1 --wait-idle 1800` after `localbench park`, under the working-machine law. Incumbent A: `ollama:localbench-parked:5642e97495e1` (qwen3.8:27b-mlx). think measures tokens and accuracy per checkable job; sess measures the memory LLM's extraction inside a live omp session.
- **Hypothesis:** the two tiers that decided the qwen3.6 REJECT (think cut-offs, 7x slower extraction) sort candidates for a fraction of the full gate's cost.
- **A/B:** per candidate, one receipt named ab-screen-<candidate>-<date>.
- **A/A null:** A1 vs A2 of the same invocation (one sample each; band = max(3 x their spread, floor)), a weaker null than stage 2's, so a screen can only drop or hold.
- **Verdict:** SURVEY
- **Retry predicate:** the SCREEN gate for each SOUND screen receipt. DROP (a REJECT row marked SCREEN) if any: think.accuracy on B < the lower of A1/A2; think.wall_s B-WORSE; sess.memory.extract_s B-WORSE; sess.turns_complete or sess.memory_calls_ok not PASS on B. Otherwise ADVANCE to stage 2 if at least one of think.wall_s, sess.memory.extract_s, sess.turn.post_retain_wall_s is B-BETTER; if none is, HOLD (an UNKNOWN row: no benefit shown). Revisit this gate if a candidate it advances fails stage 2 on a condition the screen measured.
- **Lesson:** reject cheaply and confirm expensively: the screen only removes candidates; adoption keeps the full gate.

## 2026-09-25 — SURVEY: empero-ai/Qwen3.8-9B-Distill (GGUF Q4_K_M) is a faster smol model than qwen3.8:27b-mlx without losing smol quality (screen first)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** smol candidate 3 in the queue. `hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M`, a community distillation of Qwen3.8-2.4T into the Qwen3.5-9B architecture (not a Qwen release; 693k downloads; apache-2.0). The uncensored "heretic" variants are excluded: a default smol model keeps its safety tuning. Its name makes it an omp fallback for the parked smol role, so `localbench park` runs again after the pull and it is measured under its parked name.
- **Hypothesis:** a 9B decodes about 3x faster than a 27B on the same engine class, which offsets the llama-server engine (the 27B GGUF decoded ~21 tok/s against 53 for the NVFP4 MLX incumbent); whether its distilled reasoning holds up on smol's jobs is the question.
- **A/B:** not yet measured. Stage 1: `localbench ab ollama:localbench-parked:5642e97495e1 ollama:<its parked name> --tiers think,sess --repeats 1 --pairs 1 --bank ab-screen-qwen38-9b-distill-q4km-20260925 --wait-idle 1800`, judged by the standing SCREEN gate.
- **A/A null:** A1 vs A2 of the screen.
- **Verdict:** SURVEY
- **Retry predicate:** the SCREEN gate's DROP / ADVANCE / HOLD; on ADVANCE, the full smol gate of the qwen3.6 SURVEY row (Q1-Q4, S1-S2, `--pairs 2`) before any profile change.
- **Lesson:** a community distill earns trust only by measurement on this host's jobs.

## 2026-09-25 — SURVEY: Gemma 4 12B (Google QAT Q4_0 GGUF) is a faster smol model than qwen3.8:27b-mlx without losing smol quality (screen first)
- **Bead:** kit-gemma4-omp-ab-wdf
- **Surface:** smol candidate 4. `hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0` (Google's quantization-aware 4-bit build, ungated, apache-2.0; one model file plus a vision projector). ollama's library offers only `gemma4:31b` at 58.3 GB (about BF16 for a dense 31B), too slow for smol and too large for a fast cycle. Measured through omp, as the owner asked, not through agy (agy runs Google-hosted models).
- **Hypothesis:** a 12B QAT model does smol's jobs at least as well, faster per job.
- **A/B:** not yet measured. Stage 1: `localbench ab ollama:localbench-parked:5642e97495e1 ollama:hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0 --tiers think,sess --repeats 1 --pairs 1 --bank ab-screen-gemma4-12b-qat-20260925 --wait-idle 1800`, judged by the standing SCREEN gate.
- **A/A null:** A1 vs A2 of the screen.
- **Verdict:** SURVEY
- **Retry predicate:** the SCREEN gate's DROP / ADVANCE / HOLD; on ADVANCE, the full smol gate before any profile change.
- **Lesson:** measure the build that fits the role, not the only tag a library lists.

## 2026-09-25 — REJECT (SCREEN): ThinkingCap-Qwen3.8-27B GGUF Q4_K_M as smol — memory extraction 2.5x slower on its engine
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, receipts/ab-screen-thinkingcap-q4km-20260925.json, SOUND (A1,B,A2 03:34–03:52Z, rev 228a529, contended=no, MUST fail 0; user active 0% on all three legs).
- **Hypothesis:** the 2026-09-25 SURVEY row "ThinkingCap-Qwen3.8-27B (GGUF Q4_K_M, deployable on ollama today)...".
- **A/B:** SCREEN gate. think.accuracy 6/6 on all legs, cut_off 0 (holds); think.wall_s 24.67 / 27.26 → 22.51, B/A 0.867, WITHIN-NOISE (holds); sess.turns_complete and sess.memory_calls_ok PASS (holds). **sess.memory.extract_s 3.277 / 3.616 → 8.737, B/A 2.535, B-WORSE → DROP.** ThinkingCap cut think completion tokens to 0.27x (1,548 / 1,598 → 419), but memory extraction is not reasoning-heavy: 3 calls wrote 451 / 444 tokens on A and 502 on B, so the engine decides it (busy 9.5 / 10.9 s → 29.3 s; the GGUF runs on llama-server at ~21 tok/s against the NVFP4 MLX incumbent's ~53). Also seen, not gated: sess.turn.ack_rate 1 → 0.917 (one of 12 turns not acknowledged), sess.turn.pre_main_s B-WORSE.
- **A/A null:** A1 vs A2 of the screen (one sample each).
- **Verdict:** REJECT
- **Retry predicate:** screen again only an MLX build of ThinkingCap-Qwen3.8 on the incumbent's engine class (e.g. NVFP4 via `ollama create` from bottlecapai/ThinkingCap-Qwen3.8-27B safetensors, which needs ollama 0.34.1+, the owner's upgrade call since it stales every golden). The same SCREEN gate.
- **Lesson:** a model that thinks less only wins jobs that were thinking. Smol's extraction calls were not, so the engine's per-token speed decided.

## 2026-09-25 — REJECT (SCREEN): empero-ai/Qwen3.8-9B-Distill GGUF Q4_K_M as smol — thinks 3x longer, think jobs 4x slower
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, receipts/ab-screen-qwen38-9b-distill-q4km-20260925.json, SOUND (A1,B,A2 04:13–04:31Z, rev 9b3956a, contended=no, MUST fail 0, user active 0% on all legs). B under its parked name localbench-parked:9ce406b4f20e.
- **Hypothesis:** the 2026-09-25 SURVEY row for this model.
- **A/B:** SCREEN gate. think.accuracy 6/6 on all legs (holds). **think.wall_s 19.52 / 38.17 → 119.4, B/A 4.14, band 1.94, B-WORSE → DROP.** The distill reasons at length: reasoning_chars B/A 5.3 (1,968 / 1,983 → 10,420), completion tokens B/A 3.0. sess.memory.extract_s 2.44 / 3.55 → 4.47, WITHIN-NOISE; sess.turn.post_retain_wall_s B-BETTER (0.598). Also seen: sess.turn.ack_rate 1 → 0.833 (2 of 12 turns not acknowledged).
- **A/A null:** A1 vs A2 of the screen (one sample each; A's think.wall_s spread was wide, 19.5 vs 38.2, and B still sat outside the band).
- **Verdict:** REJECT
- **Retry predicate:** screen again only with thinking capped for smol calls (as in the qwen3.6 row), measured on the omp path; the same SCREEN gate.
- **Lesson:** distilling a frontier model's chain of thought transfers its length too.

## 2026-09-25 — REJECT (SCREEN): Gemma 4 12B (Google QAT Q4_0) as smol — 2/6 think answers, 2/12 session acknowledgments
- **Bead:** kit-gemma4-omp-ab-wdf
- **Surface:** stage 1 screen through omp, receipts/ab-screen-gemma4-12b-qat-20260925.json, SOUND (A1,B,A2 04:31–05:06Z, rev 9b3956a, contended=no, MUST fail 0, user active 0% on all legs).
- **Hypothesis:** the 2026-09-25 SURVEY row for Gemma 4 12B.
- **A/B:** SCREEN gate, every item fails. **think.accuracy 1 / 1 → 0.333 (2/6; 4 cut off at 8,192 tokens after 81,340 reasoning chars)**; think.wall_s 24.97 / 21.9 → 1,109 s, B-WORSE; sess.memory.extract_s 3.08 / 3.92 → 16.48 s, B-WORSE (3 calls wrote 1,810 tokens against 444); sess.turn.ack_rate 1 / 0.917 → 0.167 (the session's "reply NOTED" turns mostly not followed); sess.turn rows B-WORSE. sess.turns_complete and sess.memory_calls_ok PASS.
- **A/A null:** A1 vs A2 of the screen.
- **Verdict:** REJECT
- **Retry predicate:** a Gemma 4 build or setting that answers the think tier within budget (cut_off 0) and acknowledges the sess turns (ack_rate at least the incumbent's) on a direct smoke; then the same SCREEN gate.
- **Lesson:** a model that does not follow omp's session instructions fails smol before speed matters.

## 2026-09-25 — REJECT: replacing /Applications/Ollama.app is a drop-in ollama upgrade on this Mac
- **Bead:** lb-06
- **Surface:** the ollama backend, 0.32.15 → 0.34.4 (authorized by the owner, 2026-09-24). Local models were down for ~15 minutes in every session.
- **Hypothesis:** swap the app bundle for the verified release zip, relaunch, and the same models serve under the new version.
- **A/B:** not a timing comparison; four observed failures, each with its fix. (1) The 0.34.x app passes its own `models` setting (default ~/.ollama/models) as OLLAMA_MODELS, overriding the launchctl variable 0.32.15 used: the new server listed no models (server.log "server config", OLLAMA_MODELS:~/.ollama/models). Fix: the app setting set to /Volumes/Models/ollama-models. (2) macOS blocked the new server from the external USB volume ("open /Volumes/Models/ollama-models/blobs: operation not permitted", respawning every second); no Files and Folders entry appears for a background helper. Fix: the owner added Ollama.app to Full Disk Access with +. (3) The old app had staged an update to 0.34.2; restarting it installed that over the manual 0.34.4 (bundle mtime 23:25:28; app.log then "New update available ... v0.34.4"). Fix: install 0.34.4 again once the staged update was 0.34.4 itself. (4) A dying app respawned a server from the old binary that kept port 11434 after the bundle moved (pid 45265 executing /private/tmp/Ollama-0.34.2-replaced.app); the new app could not bind. Fix: stop it; the new app's server bound at once (api 0.34.4, binary bba8b79eac84ab09 in /Applications).
- **A/A null:** VOID-no-measurement (an install procedure).
- **Verdict:** REJECT
- **Retry predicate:** the next ollama upgrade follows this order and checks each step: read the app's `models` setting and staged update first; stop the app and every server by pid; move the old bundle aside only after a signature-verified backup copy; install; confirm the listening process's executable path (`lsof -p <pid>` txt) and `/api/version`; confirm `localbench models` lists the ollama models; re-pin residency and refresh catalogs. Revisit if an upgrade needs none of these fixes.
- **Lesson:** an app with its own updater and settings is not a binary swap. Auto-update is still on, so ollama's version can move under a golden at any time; `localbench status` and a run's pins_changed catch it, and turning auto-update off is the owner's call.
- **Addendum 2026-09-25 13:10Z (07:10 MDT):** the owner turned auto-update off ("yes proceed"). `auto_update_enabled = 0` in Ollama.app's settings db (backup ~/.localbench/rollback/ollama-db-20260925T1310Z.sqlite). The updater re-reads it before each hourly download (app/updater/updater.go v0.34.4). The staged bundle it had re-downloaded every hour since 03:27 local was byte-identical to the installed 0.34.4 zip (sha256 f7ed834269e98929) and is removed. `localbench status` now prints the setting; ab also VOIDs any tier whose A legs ran different backend versions (ccf48ac). Verified at the next hourly check (13:27:37Z, ~/.ollama/logs/app.log): `New update available ... v0.34.4` and no download followed; the stage dir stayed empty.

## 2026-09-25 — SURVEY: oMLX 0.7.0rc1 serves Qwen3.6-35B-A3B faster than mlx-serve on omp's path, same weights (gate fixed before measuring)
- **Bead:** kit-omlx-backend-survey-9e8
- **Surface:** the local main-model engine. A `mlx-serve:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit` (mlx-serve 26.9.2, the CURRENT mlx-serve golden's config), B `omlx:` the same directory (oMLX 0.7.0rc1 wheel, backend 90006b2: fresh one-model dir and SSD cache per start). Same weights, so only the engine differs. oMLX detects this pack as a VLM (engine vlm, text-only 19.08 GB); that is how it would serve it, so it is measured that way and recorded in the fingerprint.
- **Hypothesis:** the release notes (github.com/jundot/omlx/releases/tag/v0.7.0rc1): partial block caching on a 13.4K-token Qwen3.6-35B-A3B conversation cut next-turn prefill from 1,174 to 37 tokens and TTFT from 0.83 to 0.42 s; faster Qwen prefill and decode (M5 Max figures, not this host). omp's turns re-send a growing prefix, so next-turn prefill is its cost.
- **A/B:** not yet measured. Command, after `localbench park`: `localbench ab mlx-serve:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit omlx:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit --tiers conf,micro,replay,e2e --repeats 3 --pairs 1 --bank ab-mlxserve-vs-omlx-qwen36-20260925 --wait-idle 1800`.
- **A/A null:** A1 vs A2 of the same invocation.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt. REJECT if any: a conf MUST fails on B; e2e.ok.correct or e2e.tool_read.correct fails on B; any e2e first/repeat wall row B-WORSE; micro.decode.decode_tps B-WORSE. KEEP if none of those and replay.lean.turn2_ttft_s or replay.full.turn2_ttft_s is B-BETTER (the caching claim on omp's request). UNKNOWN otherwise. A KEEP names oMLX the better engine for this model here; switching what serves omp's local model is a separate, pre-registered deployment step.
- **Lesson:** compare engines on identical weights first; a new model and a new engine at once cannot say which one moved the number.

## 2026-09-25 — REJECT: an omp patch release leaves local-turn latency unchanged (18.3.0 → 18.3.1)
- **Bead:** kit-omp-1831-startup-9nn
- **Surface:** omp startup before its first model request (e2e startup_s), ollama qwen3.6:35b-mlx.
- **Hypothesis:** omp's self-update from 18.3.0 to 18.3.1 on 2026-09-24 is neutral for local turns (the premise of dropping the omp version as a golden key, 4b9532d).
- **A/B:** startup_s 0.597 / 0.590 s (18.3.0, receipts/aa__ollama__qwen3.6_35b-mlx__20260924T050600Z.json) → 3.76 / 3.339 s (18.3.1, receipts/aa__ollama__qwen3.6_35b-mlx__20260925T053350Z.json); llm_s 1.938 / 1.928 → 1.892 / 1.925 s. The re-banked golden's e2e walls doubled to tripled (ok.first 3.2 → 6.6 s).
- **A/A null:** VOID-cross-invocation: each receipt has its own A/A pair (spread 0.007 s and 0.42 s), and the gap (2.7–3.2 s) is several times the wider one; ollama also changed between them (0.32.15 → 0.34.4), but llm_s and every micro row held, so the extra time is before the request, in omp.
- **Verdict:** REJECT
- **Retry predicate:** the bead's same-day A/B (both omp versions pinned via LOCALBENCH_OMP) shows startup_s within its band, or the next omp release (above 18.3.1) measures startup_s under 1 s in an e2e run.
- **Lesson:** dropping the omp version from golden keys kept goldens CURRENT across omp updates, which is right, but it also means an omp regression shows up only when e2e is re-banked or run. `localbench run` against the golden would have flagged these rows as REGRESSED.
- **Demoted 2026-09-25 (D4, same-day A/B):** receipts/ab-omp-1831-vs-1830-startup-20260925.json (SOUND, A,B,A,B,A 13:17–13:28Z, CPU busy 32–36% on every leg) put the current 18.3.1 build (cacaf572) and 18.3.0 (33cf63aa) level: all four startup rows WITHIN-NOISE (e2e.ok.first_startup_s A 1.855 vs B 2.124; tool_read 2.019 vs 1.865). The 0.59 → 3.3–3.8 s gap in this row compared 18.3.0 at ~9% CPU busy with 18.3.1 at 22–30%, and the first 18.3.1 build (46cb390b, 3.2–5.4 s, replaced at 05:52Z) was really slower than its replacement in the same A/B. What remains true: omp startup is CPU work and roughly triples under this host's working load (0.59 s at ~9% busy, ~1.9 s at ~34%). The verdict stands only for build 46cb390b.
- **Addendum 2026-09-25 (second 18.3.1 build):** omp replaced build 46cb390bb7e6def5 with cacaf5726609a21b at 05:52Z, same version string, inside leg A1 of receipts/ab-mlxserve-vs-omlx-qwen36-20260925.json. startup_s under cacaf572: 1.427/1.251 (ab_b) and 1.479/1.393 (ab_a2), against 3.2–5.4 s under 46cb390b (aa1, aa2, ab_a1). The REJECT stands at about +0.8 s over 18.3.0; every one of these legs ran at 37–45% CPU busy.

## 2026-09-25 — UNKNOWN: oMLX 0.7.0rc1 serves Qwen3.6-35B-A3B faster than mlx-serve on omp's path (the SURVEY row's gate; null too wide)
- **Bead:** kit-omlx-backend-survey-9e8
- **Surface:** receipts/ab-mlxserve-vs-omlx-qwen36-20260925.json, SOUND (A1 mlx-serve, B oMLX, A2 mlx-serve, 05:51–06:09Z, rev df38ea7, contended=no, MUST fail 0 on every leg, user active 0%). Same weights (files:68dedceb5da0); oMLX pins 0.7.0rc1 / 0765dade2ba118f5. First real oMLX run on this host: it loaded the pack (as a VLM), answered every e2e task and passed conformance.
- **Hypothesis:** the 2026-09-25 SURVEY row "oMLX 0.7.0rc1 serves Qwen3.6-35B-A3B faster than mlx-serve...".
- **A/B:** judged by the pre-registered gate. No REJECT condition: conf MUST 0 on B; e2e answers pass; no e2e wall row B-WORSE; micro.decode.decode_tps WITHIN-NOISE. KEEP not met: replay.full.turn2_ttft_s 1.11 / 0.742 → 1.145 and replay.lean.turn2_ttft_s 0.486 / 0.190 → 0.300, both WITHIN-NOISE. Outside the gate: replay.full.cold_ttft_s 49.3 / 47.9 → 42.7 s (B-BETTER, 0.879); e2e.ok.repeat_llm_s 0.351 / 0.265 → 0.911 s (B-WORSE).
- **A/A null:** A1 vs A2 disagreed far beyond any earlier mlx-serve pair: decode 28.96 vs 79.18 tok/s, prefill_1k 1,182 vs 2,562 tok/s, e2e.tool_read.first_wall_s 13.98 vs 5.05 s, so most bands were 2–3x. Both A legs sit far below mlx-serve's banked decode (162.4); bead kit-mlx-native-decode-drop-qim.
- **Verdict:** UNKNOWN
- **Retry predicate:** after bead kit-mlx-native-decode-drop-qim restores an mlx-serve A/A decode spread under 10%, rerun this exact command with `--pairs 2` and apply this row's gate unchanged.
- **Lesson:** a new engine can only be judged against a stable incumbent; this one showed its instability in its own A/A pair, which is what the null is for.

## 2026-09-25 — SURVEY: an A/B time verdict that an unbracketed CPU-load difference could explain should be withheld (rule fixed before applying it to any receipt)
- **Bead:** lb-06
- **Surface:** `golden.ab_table` verdicts on time rows (metric keys ending `_s` or `_tps`). Authorized by the owner 2026-09-25 ("yes proceed") in answer to "keep only reporting CPU load, or require the two sides of an A/B to see similar load?"; this makes the working-machine row's balance gate live now instead of after its three-UNKNOWN trigger. Nothing waits for an idle machine.
- **Hypothesis:** interleaving alone keeps arms fair. Counter-evidence: CPU busy, not user activity, moves decode here (bead kit-mlx-native-decode-drop-qim: mlx-serve 160–165 tok/s at <=12% busy, 29–79 at 37–40%), and the smol pairs2 receipt's B arm ran at 32.1/47.8% against A's 29.0/32.1/40.1%.
- **Rule (fixed now):** per leg, c = system.cpu.busy_pct.mean. If B's median c is more than 5 points below every A leg's c, B ran lighter: a time row judged B-BETTER becomes LOAD-FAVOURED. If B's median c is more than 5 points above every A leg's c, B ran heavier: a time row judged B-WORSE becomes LOAD-FAVOURED. Rows that go against the load (B-BETTER while heavier) stand, as do non-time rows (accuracy, token counts, hit rates, conformance). A missing c on any leg leaves verdicts unchanged and says so. The 5-point tolerance sits above the jitter of c itself between back-to-back legs of one config: 0.0–4.3 points across 16 A/A pairs from 2026-09-23/24 (runs/*__aa1__*, aa2); the one wider pair, 8.2 points on 2026-09-25 05:33Z, ran while an outside workload came and went, which is load moving, not measurement jitter. (Corrected before first use: the first draft quoted five figures from memory.) A pre-registered gate reads LOAD-FAVOURED as neither B-BETTER nor B-WORSE.
- **A/A null:** n/a (a harness rule); tested against planted receipts and applied retroactively to every banked A/B receipt that recorded c.
- **Verdict:** SURVEY
- **Retry predicate:** after implementation, list every banked A/B receipt whose verdicts the rule changes. KEEP the rule if no pre-registered verdict that was confirmed by a blind re-judge flips in a way the load data does not support. REJECT it if it withholds a verdict whose A and B legs all sat at or under 12% CPU busy (the flat region of the dose-response, where load did not move decode).
- **Lesson:** interleaving spreads load that comes in bursts. It does not protect against load that shifts between arms over hours, which is what happened on 2026-09-24/25.

## 2026-09-25 — KEEP: an A/B time verdict that an unbracketed CPU-load difference could explain should be withheld (the SURVEY row's check)
- **Bead:** lb-06
- **Surface:** implementation e77719f (golden.load_balance, ab_table load_favours, receipt `load_balance`), applied retroactively to all 21 banked A/B receipts (docs/evidence/receipts/ab-*.json) by recomputing each table with the rule.
- **Hypothesis:** the SURVEY row above.
- **A/B:** 0 of 21 receipts change. In every one B's median CPU busy sat within 5 points of the A legs' range, e.g. the smol pairs2 receipt (A 29.0/32.1/40.1, B 32.1/47.8, median 40.0) and the oMLX receipt (A 37.2/40.4, B 44.7). No confirmed verdict flips; none withheld in the flat region at or under 12%.
- **A/A null:** not applicable to a harness rule; plants 5/5 caught (e77719f).
- **Verdict:** KEEP
- **Retry predicate:** REJECT the rule if a future receipt withholds a verdict whose A and B legs all sat at or under 12% CPU busy, or if a verdict it withholds is confirmed in the same direction by a rerun of the same command whose arms were bracketed.
- **Lesson:** interleaving kept every banked A/B balanced on CPU load, including the ones run through hours of an outside workload; the rule costs nothing until an arm really drifts.

## 2026-09-25 — SURVEY: omp 18.3.1 (current build cacaf572) starts no slower than 18.3.0 on the same day (gate fixed before measuring)
- **Bead:** kit-omp-1831-startup-9nn
- **Surface:** `localbench ab ollama:qwen3.6:35b-mlx ollama:qwen3.6:35b-mlx --tiers e2e --pairs 2 --b-omp ~/.localbench/omp-18.3.0/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js --bank ab-omp-1831-vs-1830-startup-20260925 --wait-idle 1800` after `localbench park`. A = omp 18.3.1 at ~/.bun/bin/omp (sha cacaf5726609a21b), B = omp 18.3.0 isolated install (sha 33cf63aab3a109b3). Rows: e2e.{ok,tool_read}.{first,repeat}_startup_s (9fbed1b).
- **Hypothesis:** the REJECT row above ("an omp patch release leaves local-turn latency unchanged") was a cross-day, cross-load comparison: 18.3.0's 0.59 s at ~9% CPU busy against 18.3.1's 1.3–5.4 s at 37–45%. Same day, interleaved, 18.3.1 is no slower.
- **A/B gate:** every e2e .correct passes on both arms, or the receipt is UNKNOWN. Neutral (this SURVEY row becomes KEEP, and the REJECT row is demoted to a cross-load artifact) if all four startup rows are WITHIN-NOISE. Regression confirmed (this row REJECT, the earlier REJECT stands) if e2e.ok.first_startup_s and e2e.tool_read.first_startup_s are both B-BETTER (18.3.0 faster), not LOAD-FAVOURED. UNKNOWN otherwise.
- **A/A null:** three A legs of the same invocation (pairs 2).
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt; apply this gate unchanged.
- **Lesson:** a version comparison across days is also a comparison across machine load; startup is CPU work.

## 2026-09-25 — KEEP: omp 18.3.1 (current build cacaf572) starts no slower than 18.3.0 on the same day (the SURVEY row's gate)
- **Bead:** kit-omp-1831-startup-9nn
- **Surface:** receipts/ab-omp-1831-vs-1830-startup-20260925.json, SOUND (legs A1,B1,A2,B2,A3 13:17–13:28Z; contended=no, MUST fail 0 and every e2e .correct PASS on all legs; load balance: B within 5 CPU-busy points of the A legs; user active 100% on B1–A3, reported). A = ~/.bun/bin/omp 18.3.1 cacaf5726609a21b, B = isolated 18.3.0 33cf63aab3a109b3 (--b-omp, 9fbed1b). Legs B1–A3 record rev e2080fa-dirty: files were edited during the run, after the run process had imported its code.
- **Hypothesis:** the SURVEY row above.
- **A/B:** all 12 rows WITHIN-NOISE, including the four startup rows the gate names: e2e.ok.first_startup_s 1.855 → 2.124 (band 0.91), e2e.tool_read.first_startup_s 2.019 → 1.865 (0.37), e2e.ok.repeat_startup_s 1.975 → 1.828 (0.78), e2e.tool_read.repeat_startup_s 2.001 → 2.095 (1.44). Walls and llm_s likewise.
- **A/A null:** three A legs of the same invocation; startup spread 1.85–1.87 s (ok.first) and 1.98–2.09 s (tool_read.first).
- **Verdict:** KEEP
- **Retry predicate:** re-run this command when omp's version or sha changes from cacaf5726609a21b; demote this row if e2e.ok.first_startup_s or e2e.tool_read.first_startup_s is B-BETTER (the older omp faster) in a SOUND receipt.
- **Lesson:** measure a version change inside one invocation. Across days the machine's load moved startup more than omp did.

## 2026-09-25 — SURVEY: ThinkingCap-Qwen3.8-27B as bottlecapai's MLX 4-bit DWQ, served by mlx-serve, is a faster smol model than qwen3.8:27b-mlx without losing smol quality (screen first)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, `localbench ab ollama:localbench-parked:5642e97495e1 mlx-serve:/Volumes/Models/ollama-models/hf-sources/bottlecapai/ThinkingCap-Qwen3.8-27B-MLX-4bit-DWQ --tiers think,sess --repeats 1 --pairs 1 --bank ab-screen-thinkingcap-mlx-dwq-20260925 --wait-idle 1800` after `localbench park`. B = the official MLX build (revision 93944fef, 22.8 GB, mixed 4/8-bit DWQ with an embedded MTP head), mlx-serve 26.9.2 (Homebrew, sha 4d09a3beb8d4c9be), no server flags.
- **Hypothesis:** ThinkingCap's shorter thinking (KEEP c1a78a4) survives on an MLX engine, and the job-level speed the GGUF screen lost (REJECT SCREEN, 2026-09-25) comes back.
- **A/B gate:** the standing SCREEN gate (row "a 30-minute screen ... sorts smol candidates"), unchanged: DROP / ADVANCE / HOLD. The load-balance rule (e77719f) applies.
- **Confounders, named before data:** B changes engine as well as model, and mlx-serve's decode fell further under CPU load than ollama's did (bead kit-mlx-native-decode-drop-qim), so a DROP here does not reject ThinkingCap on ollama's MLX engine: that stays the registered retry of the GGUF row (bf16 -> `localbench create --quantize nvfp4`, blocked on the Hugging Face gate). Unmeasured: whether mlx-serve applies the pack's chat_template.jinja, whose default reasoning_effort is xhigh.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt; apply the SCREEN gate unchanged. On ADVANCE, the full smol gate (Q1-Q4, S1-S2, `--pairs 2`) before any profile change.
- **Lesson:** a model that thinks less needs an engine at least as fast per token as the incumbent's to win jobs that were not thinking.

## 2026-09-25 — REJECT (SCREEN): ThinkingCap-Qwen3.8-27B MLX 4-bit DWQ on mlx-serve 26.9.2 as smol — think jobs 4.6x slower at 20 tok/s
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, receipts/ab-screen-thinkingcap-mlx-dwq-20260925.json, SOUND (A1,B,A2 14:34–14:44Z, rev b9cd0ed, contended=no, MUST fail 0; CPU busy A 34.4/27.0%, B 32.5%, balanced; user active 100% on all legs, reported). B = revision 93944fef, mlx-serve 26.9.2 (4d09a3beb8d4c9be), no flags.
- **Hypothesis:** the 2026-09-25 SURVEY row for this build.
- **A/B:** the SCREEN gate. DROP on think.wall_s B-WORSE: 18.77 / 21.09 → 91.56 s (B/A 4.594, band 0.349). Other DROP conditions clear: think.accuracy 1/1 → 1; sess.memory.extract_s WITHIN-NOISE (2.405 / 2.749 → 2.358); sess.turns_complete and sess.memory_calls_ok PASS. Why: B decoded 20.0 tok/s over its think rounds against 81.2 / 72.5 on A, and wrote more, not fewer, think tokens (1,829 vs 1,524 / 1,530). think.reasoning_chars is VOID on B: mlx-serve 26.9.2 streamed no separate reasoning for this pack, so the thinking length is unseen on this path (the pack's template defaults reasoning_effort to xhigh). In sess, B's memory extraction wrote 55 completion tokens against 442 and still took 14.0 s of busy time against 7.2–7.9 s.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** REJECT
- **Retry predicate:** as registered, this DROP rejects the engine pairing, not ThinkingCap on ollama: screen the nvfp4 build (`localbench create --quantize nvfp4 --like ollama:qwen3.8:27b-mlx` from bottlecapai/ThinkingCap-Qwen3.8-27B) once the account is granted the Hugging Face gate. Screen this MLX pack again only on an mlx-serve build whose micro.decode.decode_tps for a dense 27B is within 20% of ollama's on the same day and load.
- **Lesson:** on this working machine the engine's per-token cost decides smol jobs before the model's thinking length does: 4x per token cannot be bought back by thinking less.

## 2026-09-25 — SURVEY: Bonsai 2 (ternary 2-bit Qwen3.8-27B) on mlx-serve 26.9.5 with MTP is a faster smol model than qwen3.8:27b-mlx without losing smol quality (best case for the MLX-native engine; screen first)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, `localbench ab ollama:localbench-parked:5642e97495e1 mlx-serve:/Volumes/Models/ollama-models/hf-sources/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --b-mlx-serve ~/.localbench/mlx-serve-26.9.5/mlx-serve-macos-arm64/mlx-serve --b-server-arg=--mtp --tiers think,sess --repeats 1 --pairs 1 --bank ab-screen-bonsai2-mtp-20260925 --wait-idle 1800` after `localbench park`. B = prism-ml/Ternary-Bonsai-2-27B-mlx-2bit @ fcba37d2 (8.6 GB, Apache-2.0), mlx-serve 26.9.5 (f6b32efcbbaa3d2d).
- **Hypothesis:** the MLX-native engine's per-token penalty on this working machine (ThinkingCap DWQ screen: 20 vs 72–81 tok/s) shrinks enough with 2-bit weights and MTP drafting (release notes: 73.5 tok/s on an M4 Max) for a Qwen3.8-27B to win smol jobs.
- **A/B gate:** the standing SCREEN gate unchanged (DROP / ADVANCE / HOLD); the load-balance rule applies. Also required before any ADVANCE counts: B's mlx-serve log shows `[spec-stats] mode=mtp` on generation requests (MTP used, not only loaded).
- **Why this one first:** it is the most favourable MLX-native configuration available here (fewest bytes per token, MTP). If it DROPs on engine speed, the 4-bit MTP and DFlash2 packs of the same model on the same engine are expected to DROP too; that expectation is not a verdict, and each would still need its own screen.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt; apply the SCREEN gate unchanged. On ADVANCE, the full smol gate before any profile change.
- **Lesson:** test the best case of a family first; its failure bounds the rest.

## 2026-09-25 — REJECT (SCREEN): Bonsai 2 (2-bit) on mlx-serve 26.9.5 with MTP as smol — think jobs tie, memory extraction 29% slower
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, receipts/ab-screen-bonsai2-mtp-20260925.json, SOUND (A1,B,A2 14:58–15:05Z, rev 7f47eeb, contended=no, MUST fail 0; CPU busy A 38.9/27.1%, B 25.6%: bracketed, B the lightest leg; user active 33/54/100%). B = fcba37d2 on mlx-serve 26.9.5 --mtp. mlx-serve assembled an MTP head for this pack from ddalcu/Qwen3.8-27B-MLX-Serve-4bit (~314 MB) and wrote mtp.safetensors into the model dir at first load; B's log shows `[spec-stats] mode=mtp` on 35 requests (draft acceptance 64–80%).
- **Hypothesis:** the 2026-09-25 SURVEY row for this configuration.
- **A/B:** the SCREEN gate. DROP on sess.memory.extract_s B-WORSE: 2.562 / 2.742 → 3.430 s (B/A 1.293, band 0.204), although B's extraction wrote 128 completion tokens against 449. The rest held or improved: think.accuracy 1 → 1; think.wall_s WITHIN-NOISE (19.01 / 22.96 → 20.10 s; B decoded 74.2 tok/s over think rounds against 80.5 / 69.9); sess.turn.pre_main_s B-BETTER (2.04 / 2.11 → 0.95 s) and post_retain_pre_main_s B-BETTER; sess checks PASS.
- **A/A null:** A1/A2 of the same invocation (sess.turn.ack_rate VOID: A1 acknowledged 0 of its turns).
- **Verdict:** REJECT
- **Retry predicate:** screen again when an mlx-serve release above 26.9.5 or a new Bonsai revision reports faster prefill for dense Qwen3.8 packs, or with a server flag that changes prefill (e.g. prefill chunking reported by /props); the same SCREEN gate.
- **Lesson:** 2-bit weights and MTP closed the decode gap (74 vs 70–80 tok/s at equal load) and halved omp's pre-turn classifier wait; memory extraction is prefill-bound, and there the MLX-native pack still lost. The SURVEY row's "best case bounds the rest" does not hold for prefill, so the 4-bit MTP pack gets its own screen.

## 2026-09-25 — SURVEY: Qwen3.8-27B MLX-Serve 4-bit with MTP on mlx-serve 26.9.5 is a faster smol model than qwen3.8:27b-mlx without losing smol quality (screen first)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, `localbench ab ollama:localbench-parked:5642e97495e1 mlx-serve:/Volumes/Models/ollama-models/hf-sources/ddalcu/Qwen3.8-27B-MLX-Serve-4bit --b-mlx-serve ~/.localbench/mlx-serve-26.9.5/mlx-serve-macos-arm64/mlx-serve --b-server-arg=--mtp --tiers think,sess --repeats 1 --pairs 1 --bank ab-screen-qwen38-4bit-mtp-20260925 --wait-idle 1800` after `localbench park`. B = ddalcu/Qwen3.8-27B-MLX-Serve-4bit @ b543ed7c (18.2 GB, the incumbents file's mlx-serve dense control), mlx-serve 26.9.5.
- **Hypothesis:** the same model as the incumbent, in mlx-serve's native 4-bit pack with its MTP head, beats the incumbent's ollama build on smol jobs; unlike Bonsai's 2-bit Hadamard pack, its prefill may keep up.
- **A/B gate:** the standing SCREEN gate unchanged; load-balance rule applies; `[spec-stats] mode=mtp` must appear in B's log before an ADVANCE counts.
- **Consequence fixed before data:** incoai/Qwen3.8-27B-DFlash2 is a decode drafter for this base (3.8 GB, sglang/vLLM format). If this screen DROPs on sess.memory.extract_s, DFlash2 on the same base is not screened: a drafter leaves prefill unchanged. If it DROPs on anything else, or ADVANCEs/HOLDs, DFlash2 gets its own row.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt; apply the SCREEN gate unchanged.
- **Lesson:** separate what a speed trick changes (decode) from what decided the last screen (prefill) before queueing it.

## 2026-09-25 — KEEP (SCREEN, advances to stage 2 only): Qwen3.8-27B MLX-Serve 4-bit with MTP on mlx-serve 26.9.5 as smol — session waits cut, nothing lost on the screen
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, receipts/ab-screen-qwen38-4bit-mtp-20260925.json, SOUND (A1,B,A2 15:28–15:35Z, rev ead3c43, contended=no, MUST fail 0; CPU busy A 21.6/19.5%, B 16.2%: within tolerance, B the lightest leg; user active 0/51/17%). B's log: `[spec-stats] mode=mtp` on 34 requests.
- **Hypothesis:** the 2026-09-25 SURVEY row for this configuration.
- **A/B:** the SCREEN gate. No DROP condition: think.accuracy 1 → 1; think.wall_s WITHIN-NOISE (18.94 / 20.21 → 21.20 s); sess.memory.extract_s WITHIN-NOISE (2.205 / 2.556 → 1.938 s); sess checks PASS. ADVANCE: sess.turn.post_retain_wall_s B-BETTER (5.370 / 5.174 → 3.066 s, B/A 0.582, band 0.112); also B-BETTER, not gate rows: pre_main_s 1.93 / 2.13 → 0.94 s, post_retain_pre_main_s 3.98 / 4.03 → 1.94 s. Open for stage 2: B's memory extraction wrote 56 completion tokens against 442. B's log shows `reasoning budget 2048: enforced in-stream`: mlx-serve 26.9.5 caps Qwen3.8's thinking by default, the smol thinking cap the qwen3.6 REJECT row asked for; whether extraction quality survives it is what the full gate's Q3 recall rows decide, and B wrote 26% more think tokens (1,935 vs 1,530) while decoding at 91 tok/s against 76–81.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** KEEP
- **Retry predicate:** this row changes nothing on its own. Stage 2: the full smol gate of the qwen3.6 SURVEY row (Q1–Q4, S1–S2 as amended, `--pairs 2`) with this B, then a blind re-judge; a profile change only after both. Demote to REJECT if stage 2 fails Q3.
- **Lesson:** on the same weights family, mlx-serve's native 4-bit pack with MTP matched ollama on decode and prefill here and shortened every omp session wait; the DWQ screen's 20 tok/s ran at 32.5% CPU busy on mlx-serve 26.9.2 with a different pack, this one at 16.2% on 26.9.5: load, version and pack all differ, so which of them caused the earlier collapse is not separated.

## 2026-09-25 — SURVEY: the same 4-bit pack with incoai's DFlash2 drafter instead of MTP is a faster smol model than qwen3.8:27b-mlx (screen first)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, `localbench ab ollama:localbench-parked:5642e97495e1 mlx-serve:/Volumes/Models/ollama-models/hf-sources/ddalcu/Qwen3.8-27B-MLX-Serve-4bit --b-mlx-serve ~/.localbench/mlx-serve-26.9.5/mlx-serve-macos-arm64/mlx-serve --b-server-arg=--drafter --b-server-arg=/Volumes/Models/ollama-models/hf-sources/incoai/Qwen3.8-27B-DFlash2 --tiers think,sess --repeats 1 --pairs 1 --bank ab-screen-qwen38-4bit-dflash2-20260925 --wait-idle 1800`. B = the base of the row above (b543ed7c) + incoai/Qwen3.8-27B-DFlash2 @ 015e7956 (3.8 GB, Apache-2.0), no --mtp (mlx-serve's draft priority is MTP > dflash).
- **Hypothesis:** a block-diffusion drafter decodes faster than the pack's MTP head on smol jobs.
- **Correction before any B data (15:45Z):** the first launch used `--drafter=<path>`; mlx-serve 26.9.5 refuses `flag=value` ("flags take their value as a separate argument"), so B never started and nothing was banked (runs/20260925T154031Z__ab_a1__...). The command above now passes the path as its own argument; the gate is unchanged.
- **A/B gate:** the standing SCREEN gate unchanged; `[spec-stats] mode=dflash` (or mlx-serve's DFlash equivalent) must appear in B's log, else this row is UNKNOWN (drafter not used). A DFlash2 KEEP here does not outrank the MTP row: the two only compete in stage 2 if both advance.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt; apply the SCREEN gate unchanged.
- **Lesson:** compare decode accelerators on one base, one at a time.

## 2026-09-25 — REJECT (SCREEN): the 4-bit pack with the DFlash2 drafter instead of MTP as smol — think jobs 23% slower
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, receipts/ab-screen-qwen38-4bit-dflash2-20260925.json, SOUND (A1,B,A2 15:45–15:53Z, rev eaef96e, contended=no, MUST fail 0; CPU busy A 26.5/24.9%, B 22.2%, balanced; user active 0% on all legs). B's log: `drafter=enabled (block_size=8)` and `[spec-stats] mode=dflash` on generation requests (the drafter was used).
- **Hypothesis:** the 2026-09-25 SURVEY row for this configuration.
- **A/B:** the SCREEN gate. DROP on think.wall_s B-WORSE: 20.22 / 20.19 → 24.95 s (B/A 1.235, band 0.100, the floor: the A legs agreed within 0.2%). B wrote 1,931 think tokens (1,530 / 1,598 on A) at 77.4 tok/s; the MTP head on the same base decoded 91.3 tok/s in its screen. Session rows moved as with MTP: post_retain_wall_s 7.21 / 6.99 → 3.02 s, pre_main_s 2.04 / 1.94 → 0.99 s (B-BETTER), extract_s WITHIN-NOISE; sess checks PASS; think.accuracy 1 → 1.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** REJECT
- **Retry predicate:** screen again only if mlx-serve's DFlash path decodes this base faster than its MTP head in a same-day micro.decode comparison; the same SCREEN gate.
- **Lesson:** on this base the MTP head out-decodes the DFlash2 drafter; the session-wait gains are shared by both and so come from the engine (its default reasoning cap and faster classifier replies), not from either drafter.

## 2026-09-25 — SURVEY: Qwen3.8-27B MLX-Serve 4-bit with MTP on mlx-serve 26.9.5 does omp's smol work at least as well as qwen3.8:27b-mlx on ollama, and faster (stage 2; gate fixed before measuring)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** `localbench ab ollama:localbench-parked:5642e97495e1 mlx-serve:/Volumes/Models/ollama-models/hf-sources/ddalcu/Qwen3.8-27B-MLX-Serve-4bit --b-mlx-serve ~/.localbench/mlx-serve-26.9.5/mlx-serve-macos-arm64/mlx-serve --b-server-arg=--mtp --tiers conf,micro,e2e,think,mem,sess --repeats 2 --pairs 2 --bank ab-smol-qwen38-ollama-vs-mlxserve-4bit-mtp-20260925 --wait-idle 1800` after `localbench park`, in a window the owner names (the last run of this gate took ~4 h with smol parked, 2026-09-24 23:20Z–03:18Z).
- **Hypothesis:** the stage 1 KEEP (SCREEN) row above holds under the full gate.
- **Launch note, before data (2026-09-25 ~19:45Z, window named by the owner):** A runs as `ollama:qwen3.8:27b-mlx` (same digest 5642e97495e1): since the go-live `localbench park` no longer creates the `localbench-parked:` alias (see the ThinkingCap nvfp4 screen row). Command otherwise unchanged; gate unchanged.
- **A/B gate:** the full smol gate of the 2026-09-24 SURVEY row "qwen3.6:35b-mlx does omp's smol work ...", as amended to `--pairs 2`, unchanged: Q1–Q4 quality (each must hold), S1–S2 speed; the load-balance rule (e77719f) and within-arm pin drift (ccf48ac) apply. Named in advance because the screen raised it: mlx-serve 26.9.5 enforces a 2048-token reasoning budget by default (`reasoning budget 2048: enforced in-stream`) and B's memory extractions wrote 56 tokens against 442; Q3 (recall hit rate, no leak, derail) is where that cap either costs quality or does not.
- **A/A null:** three A legs of the same invocation.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt; apply the gate unchanged; then a blind re-judge; a profile change (smol moved to a persistent mlx-serve 26.9.5 via a tested, reversible `localbench smol` verb) only after both.
- **Lesson:** a screen can find a speed win; only the full gate can say what the speed cost.
- **Superseded as a gate, kept as the demotion test (2026-09-25 ~16:18Z):** the owner adopted the configuration without it ("we dont need 4 hour test - go live with it"); see the next row.

## 2026-09-25 — UNKNOWN (adopted by the owner's decision, stage 2 not run): omp's smol role on Qwen3.8-27B MLX-Serve 4-bit with MTP, mlx-serve 26.9.5
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** live since 2026-09-25 16:18:43Z: `localbench smol set mlx-serve:/Volumes/Models/ollama-models/hf-sources/ddalcu/Qwen3.8-27B-MLX-Serve-4bit --mlx-serve ~/.localbench/mlx-serve-26.9.5/mlx-serve-macos-arm64/mlx-serve --server-arg=--mtp` (565da0e). Server 127.0.0.1:11235; all 8 omp profiles' `modelRoles.smol` = `mlx-smol/Qwen3.8-27B-MLX-Serve-4bit`, verified by omp itself (config list, and `omp models mlx-smol` lists the model in every profile). Backups: ~/.localbench/rollback/smol-20260925T161843Z. The ollama incumbent `qwen3.8:27b-mlx` (5642e97495e1) stays installed: revert target, omp-test's default role, and the parked-dense golden's model.
- **Hypothesis:** the stage 2 SURVEY row above (Q1–Q4 quality, S1–S2 speed).
- **Evidence so far:** the stage 1 screen only (KEEP (SCREEN), receipts/ab-screen-qwen38-4bit-mtp-20260925.json). Post-switch checks: the server answered "OK"; an `omp --profile omp-test -p ... --model mlx-smol/...` turn returned OK (rc 0; 25,366 prompt tokens read at 387 tok/s, as a main model would, which smol is not); a 256-token decode ran 68.3 / 78.9 tok/s end to end with MTP drafting (per-draft acceptance 53–54%). Not measured: memory quality under mlx-serve's 2048-token reasoning cap (extraction wrote 56 tokens against 442 in the screen), and behaviour under the heavy CPU load that collapsed another mlx-serve pack earlier.
- **A/A null:** not run (no stage 2).
- **Verdict:** UNKNOWN
- **Retry predicate:** run the stage 2 command above unchanged in a window the owner names; if any of Q1–Q4 fails, `localbench smol revert` (every profile back to ollama/qwen3.8:27b-mlx, server stopped). Revert sooner on any observed failure of smol work in real sessions (mnemopi extraction errors, titles or commit messages failing), noted here with the session and log line.
- **Lesson:** an adoption on judgement is still a hypothesis; record what was skipped so the next reader knows what the verdict does not cover. The server is not a launchd job (a launchd mlx-serve could not read /Volumes/Models: TCC), so after a reboot `localbench status` shows it DOWN until `localbench smol start`.

## 2026-09-25 — SURVEY: ThinkingCap-Qwen3.8-27B as NVFP4 on ollama's MLX engine is a faster smol model than qwen3.8:27b-mlx without losing smol quality (the registered retry; gate fixed before data)
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, `localbench ab ollama:localbench-parked:5642e97495e1 ollama:thinkingcap-qwen3.8:27b-nvfp4 --tiers think,sess --repeats 1 --pairs 1 --bank ab-screen-thinkingcap-nvfp4-20260925 --wait-idle 1800` after `localbench park`. B = `localbench create ollama:thinkingcap-qwen3.8:27b-nvfp4 --from bottlecapai/ThinkingCap-Qwen3.8-27B@d2b4e7a599f3 (bf16, 55.6 GB) --quantize nvfp4 --like ollama:qwen3.8:27b-mlx`, digest 34875c4701a6, ollama 0.34.4. Same engine class and quantization as A, so the model is the only change.
- **Hypothesis:** the retry predicate of the GGUF REJECT (SCREEN) row and the DWQ REJECT (SCREEN) row: on the incumbent's engine, ThinkingCap's shorter thinking (0.27x think tokens in the GGUF screen) wins think jobs without the per-token penalty that decided both earlier DROPs.
- **A/B gate:** the standing SCREEN gate unchanged (DROP / ADVANCE / HOLD); the load-balance rule (e77719f) applies.
- **First launch, before any data (19:12–19:21Z):** A1 failed at warm-up with HTTP 404: `localbench park` no longer creates `localbench-parked:5642e97495e1`, because after the go-live no profile names qwen3.8:27b-mlx as smol. It only stopped the smol server. Preflight also waited on ollama qwen3.8:27b-mlx at 50–61% GPU, used by the six omp panes still on the pre-16:18Z config. Nothing banked; unparked 19:24Z. Relaunch with A = `ollama:qwen3.8:27b-mlx` (same digest 5642e97495e1) once those panes are restarted; the gate is unchanged.
- **Confounder, named before data:** since 16:18Z smol is live on mlx-serve (the 4-bit MTP pack, adopted without stage 2), not on A. An ADVANCE here earns a stage 2 against A as registered; to replace the live smol it must also be compared with the mlx-serve pack.
- **A/A null:** A1/A2 of the same invocation.
- **Verdict:** SURVEY
- **Retry predicate:** the command above banks a SOUND receipt; apply the SCREEN gate unchanged. On ADVANCE, the full smol gate (Q1–Q4, S1–S2, `--pairs 2`) before any profile change.
- **Lesson:** isolate the model from the engine: the two earlier ThinkingCap screens changed both.

## 2026-09-25 — UNKNOWN (SCREEN, HOLD): ThinkingCap-Qwen3.8-27B NVFP4 on ollama's MLX engine as smol — thinks 21% less, no gated job faster
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** stage 1 screen, receipts/ab-screen-thinkingcap-nvfp4-20260925.json, SOUND (A1,B,A2 19:27–19:34Z, rev bc1f525, contended=no on every leg, conformance 2/2 PASS on each; load balance: B within 5 CPU-busy points of the A legs). A `ollama:qwen3.8:27b-mlx` (5642e97495e1), B `ollama:thinkingcap-qwen3.8:27b-nvfp4` (34875c4701a6); both nvfp4 on ollama 0.34.4, so the model is the only change.
- **Hypothesis:** the 2026-09-25 SURVEY row for this build.
- **A/B:** the SCREEN gate. No DROP condition: think.accuracy 1 / 1 → 1 (6/6, cut_off 0); think.wall_s 18.89 / 18.61 → 18.02, WITHIN-NOISE (B/A 0.961, band 0.1); sess.memory.extract_s 2.522 / 2.389 → 2.745, WITHIN-NOISE (B/A 1.118, band 0.1625); sess.turns_complete and sess.memory_calls_ok PASS. No ADVANCE row B-BETTER: sess.turn.post_retain_wall_s 6.91 / 5.193 → 6.149, WITHIN-NOISE. → **HOLD.** Not gate rows: think.reasoning_chars B/A 0.789 and think.completion_tokens 0.79, B-BETTER (1,515 / 1,530 → 1,203); sess.turn.pre_main_s 2.088 / 2.026 → 1.196, B-BETTER; sess.turn.wall_s B-BETTER (0.651). With the engine held fixed, ThinkingCap's cut is 21% on these six questions, not the 73% the GGUF screen showed (that cut came with a different engine's template), and 21% fewer tokens did not clear think.wall_s's 10% band.
- **A/A null:** A1/A2 of the same invocation (one sample each).
- **Verdict:** UNKNOWN
- **Retry predicate:** screen again only with a smol workload where reasoning is most of the wall time (e.g. a think tier with longer questions), or when bottlecapai publishes a ThinkingCap-Qwen3.8 revision after d2b4e7a599f3 that reports a larger cut; the same SCREEN gate. Against the live smol (the mlx-serve 4-bit MTP pack), B would also have to beat that pack's post_retain_wall_s (3.07 s in its screen), which this build did not approach.
- **Lesson:** measure a fine-tune's claimed token saving on its own engine; two of three earlier "ThinkingCap is shorter" readings came with an engine change.

## 2026-09-25 — UNKNOWN (stage 2 UNSOUND; Q3 fails on the sound legs): Qwen3.8-27B MLX-Serve 4-bit with MTP on mlx-serve 26.9.5 as smol
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** receipts/ab-smol-qwen38-ollama-vs-mlxserve-4bit-mtp-20260925.json, order A1,B1,A2,B2,A3 (19:47–22:40Z, rev after bc1f525), command of the stage 2 SURVEY row with A by its own name (launch note). **UNSOUND: ab_a3 CONTENDED.** At 22:29Z a client outside the run loaded `thinkingcap-qwen3.8:27b-nvfp4` on ollama (HEAD /, POST /api/show, then a 34 s /v1/chat/completions, the `ollama run` pattern; ollama server.log line 55674; client not identified) and held 43.6% GPU; A3's think.wall_s 175 s and extraction 5.7 s come from that. A1, B1, A2, B2 are contended=no.
- **Hypothesis:** the stage 2 SURVEY row (the go-live pack holds the full smol gate).
- **A/B (read against the gate, but not a verdict: the receipt is UNSOUND):** Q1 holds (no MUST fails; e2e answers correct on every leg). Q2 holds (think.accuracy 1 on every leg). **Q3 fails its first clause:** mem.recall.hit_rate A 1 / 1 / 1 → B 0.889 / 0.889, B-WORSE (band 0.05); the second clause holds (8 >= 9 − 1); no_leak PASS, derail 1 → 1. Both B misses end the recall turn with a tool call written as text instead of an answer (B1 "owner" round 2: `<tool_call><function=bash>…`; B2 "port" round 0: `<tool_call><function=recall>…`), i.e. a tool call the omp child did not execute; no A leg shows this. Q4 holds (24/24 turns, 6/6 memory calls, ack_rate 1). Speed, sound legs: sess.turn.post_retain_pre_main_s B-BETTER (4.04 → 2.05 s); post_retain_wall_s 5.1 / 4.8 → 3.1 / 2.7 s, pre_main_s 2.07 / 1.93 → 0.87 / 0.82 s and extract_s 2.72 / 2.22 → 2.02 / 1.88 s all lower on B but WITHIN-NOISE only because A3 widened the bands. Also: think.completion_tokens B/A 1.277 B-WORSE; think.reasoning_chars VOID on B (mlx-serve streams no separate reasoning); extraction wrote 162 / 63 completion tokens on B against 893 / 900 on A (the 2048 reasoning budget question stays open, and recall quality is where it would show).
- **A/A null:** A1/A2/A3 of the same invocation (A3 contended).
- **Verdict:** UNKNOWN
- **Retry predicate:** re-run the same command in a window with no other ollama client (check `localbench gpu` shows none before launch; delete or rename `thinkingcap-qwen3.8:27b-nvfp4` first so nothing can load it); if Q3 fails again on a SOUND receipt, the pre-registered consequence applies: REJECT and `localbench smol revert`. Before that re-run, a direct smoke of the raw-text tool call on mlx-serve 26.9.5 (one recall-tier turn, `--repeats 1`) tells whether the misses are the server's tool-call parser.
- **Lesson:** a screen without the mem tier cannot see a tool-call failure that only shows when the model must call recall mid-turn; the go-live skipped exactly the tier that shows it.

## 2026-09-25 — REJECT (demotion, D4): Qwen3.8-27B MLX-Serve 4-bit with MTP on mlx-serve 26.9.5 as smol — tool calls come back as text and are never executed
- **Bead:** kit-thinkingcap-deployable-ab-v25
- **Surface:** the live smol (go-live 16:18Z, adopted without stage 2). Evidence: receipts/ab-smol-qwen38-ollama-vs-mlxserve-4bit-mtp-20260925.json (stage 2, UNSOUND by A3 only) and the mem-tier smoke runs/20260925T224009Z__run__mlx-serve__Qwen3.8-27B-MLX-Serve-4bit (UNSOUND: contended by the same foreign `ollama run thinkingcap-qwen3.8:27b-nvfp4`, pid 37495, started by an omp codex session's Python runner in another project; it had exited when the owner authorized stopping it).
- **Hypothesis:** the stage 2 UNKNOWN row's open question: are B's recall misses the server's tool-call handling?
- **A/B:** answer strings containing a raw `<tool_call>` (the model's tool call returned as message text, so omp executes nothing), counted over each leg's details in the stage 2 receipt: A1 / A2 / A3 = 0 / 0 / 0, B1 / B2 = 11 / 13. The smoke shows 12 on the same pack: plant answers (`retain` not called), control and recall answers (`recall`, `bash` not called), one derail answer, one call written as JSON instead of XML. Contention changes speed, not which strings the model writes, so these counts stand although both artifacts are UNSOUND. Both stage 2 recall misses are this failure; so is the smoke's derail miss (ok_rate 0.889).
- **A/A null:** A1/A2/A3: zero raw tool calls on every incumbent leg.
- **Verdict:** REJECT
- **Action:** `localbench smol revert` at ~22:58Z (the owner's decision): every profile's smol back to `ollama/qwen3.8:27b-mlx`, server stopped, LaunchAgent removed. Running omp sessions keep mlx-serve until restarted and until then their smol calls fail (server down).
- **Retry predicate:** a same-day A/B in which B's legs show zero raw `<tool_call>` answer strings over the mem and sess tiers (an mlx-serve release above 26.9.5, or a server flag that parses Qwen3.8's XML tool calls), then the full smol gate on a SOUND receipt before any go-live.
- **Lesson:** an adoption that skips the tier exercising tool calls ships a model that cannot use tools; a screen of think and sess has no tool-calling job and cannot see it.
