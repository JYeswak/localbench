<!-- Working copy of templates/agents.md. -->
<!--
  Anti-ceremony (docs/CHECKLIST.md A12):
  - Consumer: every agent working in this repo, and the reviewer grepping for the patterns (A9).
  - Gate: A9 anti-reward-hacking law — the patterns must appear here verbatim.
  - Defect class: agents under pressure optimizing the metric instead of the work (29/44 suite projects needed this written down).
  - Delete when: the project adopts a machine-enforced equivalent (none exists in this kit; keep this file).
-->

# AGENTS.md — agent operating instructions for this project

localbench gives awareness and control of local model use on THIS Mac Studio: what uses the GPU and why, which models
serve which features, and regression gates for each. Since 2026-09-30 (the owner) local models are not evaluated as main
coding agents (contention); they serve only side features proven better than today's route: no answer-quality loss
beyond A/A noise on the same items, plus at least one measured win in p95 latency, availability, GPU cost or privacy.
Retained uses: proj-b-like decision work via Ollama 0.35 `/v1/systemone` (nimble 9B, tev1 4B, tev1 0.8B) in omp's judge
role and the proj-b CLI, and memory (mnemopi extraction). Epic: `kit-side-model-mission-hgy`. All tiers and harness code
stay as the regression suite. omp is the first harness it drives; others can follow.
The plan of record is docs/planning/packet.md; the pinned incumbent is docs/evidence/incumbents.md.
The 12 forbidden patterns below stay verbatim (docs/CHECKLIST.md A9).

## The one rule

"A step you can satisfy by believing you did it is not a step."

## The 12 forbidden reward-hacking patterns

Quoted verbatim from the suite-wide agent law (AGENTS.md Rule 0.5), as
observed in franken_markdown and frankenfs:

1. gate self-weakening ("temporarily" loosening a check to land a change)
2. proof-class inflation (relabeling weak evidence as strong)
3. golden regeneration reflex (regenerating goldens until they pass)
4. commit-stream pumping (many trivial commits to look productive)
5. tautological tests (tests that assert what the code does, not what it should do)
6. easy-lever cherry-picking (only attempting levers known to pass)
7. close-pump abuse (closing beads to move a metric)
8. scope-splitting (splitting work to dodge gate thresholds)
9. spec-editing as progress (editing the spec instead of the code)
10. conformance metastasis (growing the conformance suite instead of fixing failures)
11. dependency smuggling (undeclared dependencies)
12. bench-path hardcoding (special-casing the benchmark input)

Three load-bearing rules sit above the list:
- Never weaken a gate to land a change (docs/CHECKLIST.md B7).
- No self-grading without independent verification.
- Demotions are always allowed (see docs/evidence/demotion-rules.md, rule D4).

## Project-specific instructions

### Commands

- Install/refresh the CLI: `uv tool install -e .` (stdlib only; Python >= 3.12). `localbench doctor [--fix]` checks
  every subsystem (PASS/WARN/FAIL with the fix command); exit codes: 0 ok, 1 unsound or refused, 2 usage, 141 pipe.
- Machine snapshot: `localbench stats`.
- Every state-changing verb takes `--dry-run` (prints the plan, changes nothing) and `--explain`, and appends a row to
  `~/.localbench/audit.jsonl`: `localbench audit [--since 24h]`, `localbench why <id>`. `localbench validate <file>`
  checks a receipt or golden without running anything.
- Which configs have a live regression gate right now: `localbench status` (each golden CURRENT /
  GENERATION-MISMATCH / UNAVAILABLE against current pins, park state, GPU users; exit 1 on any mismatch).
- Heavy slot holder, its expected remaining time and the --wait-slot queue with ETAs: `localbench slot [--json]`.
- Who is using the GPU now, and which sessions/omp features can send it work: `localbench gpu` (per-process GPU %,
  resident models, clients of ollama/mlx-serve/the proxy, local-routed features per omp profile). Runs record the same
  per second and are CONTENDED when a non-backend process uses >25% GPU.
- History of local-model use: `localbench watch` (supervised as hub process `lb-watch`, one sample a minute into
  runs/observe.db) and `localbench report --since 24h` (GPU-seconds by process/model, residency, clients).
- Models and releases: `localbench models` (installed models, freshness vs registry/HF, which omp features route to
  each, upstream releases). Memory store: `localbench memory [--prune]`.
- omp updates (uca, about every 3 h): `localbench omp watch install` adds a launchd WatchPaths job on omp's
  package.json that runs `localbench omp refresh`. That command drives the real omp against a local mock (rpc legs, plus a
  PTY leg for titles), captures each side feature's request, and normalizes volatile spans (timings, ids, dates).
  It carries a feature's proof forward only on an identical normalized request shape and module sha. Anything else goes
  STALE with its re-proof queued, and the command exits 1. Log: ~/.localbench/omp-watch/refresh.log.
- A study longer than uca's interval runs on a frozen omp: `localbench run|ab ... --omp-frozen` (or `localbench omp
  freeze`, then `LOCALBENCH_OMP=<printed entry>`). It copies omp's package, its dependency closure and bun to
  ~/.localbench/omp-frozen/<version>-<sha16>/ (read-only, ~870 MB, reused while omp is unchanged, newest 3 kept), so
  every leg pins one omp_version/omp_sha; on 2026-10-02 an 18.4.9 -> 18.4.10 update between legs voided a memory A/B.
  The frozen omp_sha hashes the snapshot's entry script, not dist/cli.js.
- Ollama upgrades are manual: auto-update is OFF (`localbench ollama-app auto-update off`), and each release is A/B'd
  on the role suites before adoption.
- Every GPU and model action goes through this CLI (rule use-localbench-cli-for-gpu-and-models), never raw ollama or curl to
  :11434: residency and finite leases appear in localbench status and localbench gateway status; GPU users via
  localbench gpu --seconds 5. localbench keep ollama:<m> [5m|30m|2h] defaults to a finite five-minute lease; 0/unload
  is refused while gateway requests, established non-gateway Ollama sockets, or uncertain activity make unloading unsafe.
  Quiet nettop/GPU counters and elapsed time do not prove an external request idle; long prefills can be silent.
- OMP Ollama requests route through localbench gateway install's loopback LaunchAgent; use gateway status, start, stop,
  and remove for lifecycle/readback. OMP requests fail closed when it is down. Profile edits do not restart already-running
  OMP sessions; restart them before assuming they use the new route. Inbound POST bodies have a 128 MiB cap and a 30-second
  total read deadline.
- Direct non-OMP callers can still bypass the gateway by calling Ollama directly and are outside this policy; do not claim
  host-wide enforcement. Existing residents without a gateway lease are unowned and are never unloaded during migration.
- Download models with localbench pull ollama:<m> (library tag or hf.co/<org>/<repo>:<quant>); remove unused models
  with uv run python scripts/prune_models.py [--delete NAME...] (lists why each is kept); optionally pause omp's managed
  browser with localbench quiet [--display] / --resume (not needed for soundness since app load is recorded, not vetoed).
  Every run, smokes included: `localbench park`, then `run|aa|ab ... --wait-idle 1800` (it now waits only for model
  conditions); never `--allow-busy` (it skipped the gate on 2026-09-24 and the smoke was CONTENDED by an omp session
  loading qwen3.8 mid-run). Before launching, `localbench gpu --seconds 5` must show no foreign ollama client: on
  2026-09-25 an `ollama run` from another project's omp session contaminated two runs. On a machine in use, A/Bs run
  `--pairs 2` or more. If the CLI lacks a verb, add it with a test.
- Model candidates cycle in two stages (the owner, 2026-09-25: "a much faster test to cycle through these"). Stage 1,
  the screen (~30 min): `localbench ab <incumbent> <candidate> --tiers think,sess --repeats 1 --pairs 1`, judged by the
  ledger's standing SCREEN gate. It can drop a candidate (a REJECT row marked SCREEN) or hold it, and never adopts one.
  Stage 2, for candidates the screen advances: the candidate's own pre-registered gate (`--pairs 2`, every gated tier).
  Only a stage-2 KEEP with a blind re-judge changes a profile.
- Backend spec: `ollama:<model>`, `mlx-serve:<model dir>`, `omlx:<model dir>` or `mlxfast:<model dir>` (mlx-serve, oMLX
  and mlxfast are started/stopped by the harness; oMLX on :11236 with a fresh one-model dir and SSD cache per start, so
  a restart is cold; mlxfast is Layr-Labs' `mlx-server` on :11237 from `LOCALBENCH_MLXFAST` (ab: `--b-mlxfast`), pinned
  by sha16 + the git commit of its build tree, `mlx.metallib` beside it; use the 88cf569 build, HEAD f616f65 emits
  gibberish on this Mac).
- Measure one model and compare to its golden: `localbench run ollama:qwen3.6:35b-mlx` (exit 1 = unsound or regressed).
- Bank a golden (only way): `localbench aa <spec> --write-golden` — two runs, band = max(3 x A/A spread, floor),
  receipt in docs/evidence/receipts/, then review with `localbench show <golden> --diff HEAD` (moved pins, changed
  rows with Δ%, dropped tiers; `git diff goldens/` is the raw view). An omp version bump does not stale a golden;
  `localbench status` names a mismatch only when a pin that tier exercised moved (backend, model, child overlay, agent config, fixtures).
- Compare two configs in one invocation: `localbench ab <A spec> <B spec> --bank <name>` (order A, B, A).
- Record omp's request shape + sidecar: `localbench record --label lean ollama:<m> -- <omp flags>`.
- Read a receipt, golden or run: `localbench show <file.json|runs/<dir>>`, not the raw JSON (a 3-leg A/B receipt:
  10.5k tokens raw, 1.6k shown). Verdict first, pins once, rows citable as `<file>#<RFC 6901 pointer>`;
  `--path <pointer>` prints one subtree unrounded; long tables and lists page with `--cursor` (the view names the
  next one). Run streams print one rounded `event k=v` line per event; progress.jsonl holds them whole.
- Memory LLM and main model on one GPU in an interactive session: `localbench run <spec> --tiers sess`
  (`omp --mode rpc`, 12 turns; retention every 4th turn overlaps the next turn).
- Monitors: docs/MONITOR.md; `sh scripts/check-monitor-pane.sh %N` before a watcher pane starts watching.
- Gates: `uv run python -m unittest discover -s tests -t .` (pure logic, ~100 s; the pre-commit hook runs it when
  anything the suite reads is staged: code, tests, fixtures, scripts, goldens, registries, receipts, docs/port, the
  packet, pyproject.toml; CI runs it on macOS with Python 3.12), `sh scripts/check-readiness.sh`,
  `sh scripts/check-claim-discipline.sh`, `uvx ruff check localbench` (same rule set as CI). A new test must fail on a
  plausible bug: `uv run --quiet python scripts/mutate.py <cases.json>` plants it, requires the test to fail, and
  restores the file by sha (it holds runs/.mutation.lock and gives each test run a fresh bytecode cache; a same-size
  plant otherwise runs stale .pyc).
- Optional root probes (powermetrics, purge): the user runs `sudo scripts/install-sudoers.sh` once; code uses `sudo -n` only.

### Layout

- `localbench/` — client (streaming timing), sysstats (machine receipts), backends (ollama, mlx-serve, oMLX, mlxfast),
  workloads (tiers: conf, micro, replay, e2e, rel, relcold, relfresh, mem, sess, think), golden (compare/update), proxy (omp traffic timing),
  render (reading views: `show`, run stream, report.md), park, smol (dedicated smol server), models, memory, quiet,
  doctor, audit (mutation ledger), `__main__` (CLI).
- `tests/` — stdlib unittest regression suite for the judging code; also the conformance oracle for a port.
- `fixtures/omp/` — recorded omp request bodies. Replay binds to their hash, not to the running omp; `localbench
  status` names the omp each fixture was recorded under. Re-record (`localbench record`) to follow a newer omp's
  request, then re-bank replay.
- `goldens/<host_id>/` — reviewed goldens; never compared across hosts.
- `docs/evidence/` — ledger, demotion rules, incumbents, banked receipts. `runs/` is gitignored scratch.
- Captured omp traffic lives only under `~/.localbench/corpora`, never in git.

### Pacing (the owner, 2026-10-02: "computer running really slow ... do this a little slower")

- One heavy job at a time across ALL localbench agents: a local-inference run (ab, run, decision run, prove, generation
  replay, judge pass) OR a full suite / mutate.py landing gate, never two at once. The orchestrator (pane %pane) grants
  the slot; workers ask before starting one. Single test modules, edits and reads need no slot.
- Before starting a heavy job: at nice 10, `uptime` load average must be under 40 and
  `localbench gpu --seconds 5` must report device <80% with either `IDLE` (device and process
  each <5%) or `ATTRIBUTED` (coverage >=90%). `UNATTRIBUTED`, `UNALIGNED`, or `UNAVAILABLE` means wait;
  instantaneous ioreg utilization is not an admission signal.
  No large filesystem scans (`find ~`, `du ~`) while another heavy job runs.
- Unload what a run loaded when it ends (`localbench keep ollama:<m> 0`); never leave a 20+ GB model resident for later.
- On the owner's "slow" signal: stop localbench inference first, unload its models, report; restart only on his go.

### Measurement law (project-specific; a violation of the model rules voids the run)

- The machine is measured while it works (the owner, 2026-09-24: "my machine is never going to be fully quiet - we need
  our testing to be while our system is working"). Apps, the screen and the person at the keyboard are recorded per
  leg (`system.during.load`: app GPU mean/p95, seconds an app passed 25%, user-active % from HIDIdleTime) and shown
  by `show`; they do not void a run. A/Bs interleave (`--pairs N`: A,B,...,A); each arm is the median of its legs
  and the band comes from the wider arm spread, so a burst lands in one leg, not one arm. App GPU % is inflated by the
  model's own saturation (8.5-14.8% on dense legs vs 2.8-3.6% on MoE legs, same desk): compare it only between legs
  that load the GPU alike; user-active % does not depend on the model. Load balance is reported, not gated (ledger).
- One model on the machine at a time for chat-model goldens and main-tier runs: another model resident or running is
  CONTENDED and voids the run. The harness unloads other ollama models and refuses to run ollama while mlx-serve
  serves; do not start other local inference during a run; `localbench park` keeps omp sessions' smol calls off the
  GPU (it parks the smol model, see below).
- Side models run under two regimes. Decision quality is deterministic (label log-probs, no sampling) and is measured
  anytime; the decision tier enforces this: a repeat disagreement or an invalid response makes the run non-proof (a
  MUST; the fix is in progress in `kit-decision-tier-j4y`). Latency, availability and GPU cost are measured live with
  co-resident models under recorded load; the load is recorded, never used to void the run. Goldens also get a parked
  baseline.
- Monitor or helper agents MUST NOT use a local model while a run is active. Every omp profile sets the `smol` role to
  `ollama/qwen3.8:27b-mlx`, and most profiles route mnemopi memory through smol: an omp session on a cloud model still
  hits the local GPU. The 2026-09-25 move of smol to an mlx-serve 26.9.5 server was reverted the same day (ledger
  REJECT: tool calls came back as text). `localbench smol set|status|start|stop|autostart|revert` manages such a server;
  `autostart on` makes it a LaunchAgent, whose binary needs Full Disk Access to read an external volume. The run's
  sampler record lists resident models and GPU users each second; outside a run, `localbench gpu`.
- omp's bundled `scout` subagent runs on `@smol`, i.e. that local server: never spawn scouts during a run or a
  probe. On 2026-09-23 five scouts held the then-smol runner at ~85% GPU for 33 min and voided a probe (ledger); on
  2026-09-25 two scouts kept a parked runner busy and delayed a run's preflight. Research during a run goes to
  `task` agents on the session's cloud model.
- omp's scout agent is disabled in all local-smol profiles by owner decision 2026-10-01
  (`task.disabledAgents: ['scout']`); sonic stays local by owner decision.
- Every measured `omp` child runs with `child_env()`: localbench's own agent dir (`runs/omp-agent`, base settings
  from `fixtures/omp/agent/config.yml`, pinned as `omp_agent_config`). None of the user's MCP servers, plugins,
  settings or memory banks reach a child. With the user's dir, children read other agents' databases under ~/.claude
  and called the user's Agent Mail MCP server (2026-09-23; `--no-tools` does not remove MCP tools). Reads of the
  user's own setup (`models`, routes) use `omp_env()`.
- Loopback endpoints only. A failed local call is a finding, never a cloud fallback. Hosted decision service is allowed only as an
  explicitly flagged comparison arm in an A/B, with its cost recorded, never as a fallback.
- Goldens are written only by `localbench aa <spec> --write-golden` (the packet's "UPDATE_GOLDENS"), which refuses
  unsound A/A pairs; follow with `localbench show <golden> --diff HEAD` review in the same commit (pattern 3, golden
  regeneration reflex).
- The incumbent smol model is parked during test windows (`localbench park`); measure it by its parked name
  (`ollama:localbench-parked:5642e97495e1`, same digest). `localbench unpark` restores it.
- Numbers go into claims only from banked, same-generation receipts (packet §6, §8).

### Tick loop (standing; one product-moving unit per tick)

The product is the harness and its sound verdicts: what uses local models here and why, which models, how fast and
reliable each path is (decision routes, memory, side work), and when a new release is worth testing — not documents about
it. A tick that changes neither scores nothing.

1. Facts, stateless: `git status --short`, `localbench status`, open beads (`jq` on `.beads/issues.jsonl`), ledger
   rows whose retry predicate may now hold.
2. Pick one unit, in this order: a GENERATION-MISMATCH golden → the user's first tasks (decision routes and memory
   proven better than today's route) → a ledger retry predicate that is now testable → an open bead
   (`lb-*` or `kit-*`).
3. Build, then verify against reality: a live run, a break-test with a known-bad AND a known-good leg, or a measured
   number. A check that only asks our own gates does not count.
4. Commit with its `[level]`; bank the receipt; a ledger row for every kept, refuted, or void hypothesis; close a
   bead only on cited evidence. A unit that cannot reach a verdict leaves an UNKNOWN row with a retry predicate,
   never a stub.
5. Halt and name the one decision when the next unit needs the user: changing their omp profiles other than a live
   switch to a preset that just passed its gate, after a heads-up, with one-command rollback (packet §9); killing a
   process this session did not start, spend, publishing. Parking qwen3.8 for a test window is standing (unpark after).
   Three ticks in a row that move nothing: stop and report.
6. When two agents share this tree, the other agent grades each `[test]`/`[mutation]` commit: plant your own
   defects through scripts/mutate.py, run the smoke, append a row (PASS, or FIX naming the defect; the fix gets
   its own row) to docs/evidence/reviews/<date>-cross-grades.md. Name agents and panes by tmux id (`%pane`), never
   by pane number. One owner per file; the hook runs the suite on the shared tree, so commit small and green.
   With no second agent, the commit still gets a row, UNGRADED, and stays unclaimed until graded. A grade by a
   fresh-context subagent on the author's own model is recorded as that, never as a cross-model grade.
   A FIX names a plausible defect: one a reasonable edit could introduce, with a consequence a user or the next
   agent would see. A planted side effect no test claims to watch is out of scope, or it shows the claim
   overreaches, and then the fix narrows the claim. Without this bar, grading never ends: some write always
   lands where a finite test does not look.
   `scripts/tick_driver.py` (a hub process, one per pane: pids in runs/tick-drivers.pids, wake texts
   runs/tick-<lane>.md) wakes an idle pane, never during a run. HEAD, working-tree and bead changes count as
   movement; three wakes that move nothing end the driver, and `--notify %<orchestrator>` reports that exit.
   `touch runs/LOOP-STOP` stops them all.
7. Queued work exists only as beads. Every item a pane is asked to do is a bead with a lane label (`lane-infra`,
   `lane-measure`, `lane-review`) and a parent, filed before the dispatch that names it. An idle pane pulls: its own
   in_progress bead, then its lane's ready beads, then any ready bead whose files it can reserve, then a review of
   another pane's diff; it never files queue-empty or marker beads. Heavy jobs run in the background with
   `--wait-slot` while the pane takes another edit-only bead. Reports go through `scripts/report.sh` (file in
   runs/inbox/, pointer verified in the orchestrator's pane); a report not delivered is not a close. (2026-10-02:
   panes idled while their queues lived in chat, parked their turns on the heavy slot under load 47-53, filed
   queue-empty marker beads, and a 9-leap report typed into a busy pane was lost while its bead was closed.)

## Honesty machinery (how to use it)

- The beads graph (`.beads/issues.jsonl`) is the executable form of the plan;
  prose documents are the rationale of record. Close beads only with a
  `close_reason` that cites evidence (commit, receipt, ledger row).
- Claims go in `registries/claims.tsv` BEFORE the README makes them; set
  `enforce=yes` once the proof exists. The pre-commit hook cross-checks them.
- Falsified hypotheses go in `docs/evidence/NEGATIVE_EVIDENCE.md` with a
  real, testable retry predicate — never "later". The hook lints every row.
- Independent verification while grok is out of credits: a fresh-context subagent on the same model (GradeFiveCommits) grades commits and blind re-judges verdicts; accepted by the owner 2026-09-25. Label every such grade "same-model fresh-context"; it is weaker than a different model, so a verdict that changes a profile still needs the blind re-judge.
- Definition of Done: `docs/definition-of-done.md`. Done needs command
  evidence; blocked needs artifact evidence (error text + command + path).

<!-- bv-agent-instructions-v3 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking and [beads_viewer](https://github.com/Dicklesworthstone/beads_viewer) (`bv`) for graph-aware triage. Issues are stored in `.beads/` and tracked in git. Current `br` workspaces normally export `.beads/issues.jsonl`; older `bd`/legacy workspaces may use `.beads/beads.jsonl`. `bv` auto-discovers the supported JSONL files, so agents should use `br`/`bv` commands instead of hard-coding a single filename.

### Using bv as an AI sidecar

bv is a graph-aware triage engine for Beads projects. Instead of parsing .beads/issues.jsonl / .beads/beads.jsonl directly or hallucinating graph traversal, use robot flags for deterministic, dependency-aware outputs with precomputed metrics (PageRank, betweenness, critical path, cycles, HITS, eigenvector, k-core).

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). `br` handles creating, modifying, and closing beads.

**CRITICAL: Use ONLY --robot-* flags. Bare bv launches an interactive TUI that blocks your session.**

#### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns everything you need in one call:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command

# Token-optimized output (TOON) for lower LLM context usage:
bv --robot-triage --format toon
```

Before claiming, verify current state with `br show <id> --json` or `br ready --json`. `recommendations` can include graph-important blocked or assigned work; only `quick_ref.top_picks` and non-empty `claim_command` fields represent claimable work.

#### Other bv Commands

| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with unblocks lists |
| `--robot-priority` | Priority misalignment detection with confidence |
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions, cycle breaks |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |

#### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work (no blockers)
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank scores
```

### br Commands for Issue Management

```bash
br ready --json                       # Show issues ready to work (no blockers)
br list --status=open --json          # All open issues
br show <id> --json                   # Full issue details with dependencies
br create --title="..." --type=task --priority=2 --json
br update <id> --status=in_progress --json
br close <id> --reason="Completed" --json
br close <id1> <id2> --reason="Completed" --json
br sync --flush-only                  # Export DB to JSONL after Beads mutations
```

### Workflow Pattern

1. **Triage**: Run `bv --robot-triage` to find the highest-impact actionable work
2. **Claim**: Use `br update <id> --status=in_progress --json`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id> --reason="Completed" --json`
5. **Sync**: Run `br sync --flush-only` after Beads mutations so the JSONL export is current

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready --json` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Git Policy

`br` never commits or pushes. Follow this repository's own git instructions before staging, committing, or pushing. If the repository says "commit only when asked," that rule overrides any generic workflow advice.

<!-- end-bv-agent-instructions -->
