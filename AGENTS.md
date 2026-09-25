<!-- Working copy of templates/agents.md (installed by scripts/init.sh). -->
<!--
  Anti-ceremony (CHECKLIST.md A12):
  - Consumer: every agent working in this repo, and the reviewer grepping for the patterns (A9).
  - Gate: A9 anti-reward-hacking law — the patterns must appear here verbatim.
  - Defect class: agents under pressure optimizing the metric instead of the work (29/44 suite projects needed this written down).
  - Delete when: the project adopts a machine-enforced equivalent (none exists in this kit; keep this file).
-->

# AGENTS.md — agent operating instructions for this project

localbench gives awareness and control of local model use on THIS Mac Studio: what uses the GPU and why, which models
serve which features, whether memory and a fast local model work reliably (together), and regression gates for each.
omp is the first harness it drives; others can follow.
The plan of record is docs/planning/packet.md; the pinned incumbent is docs/evidence/incumbents.md.
The 12 forbidden patterns below stay verbatim (CHECKLIST.md A9).

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
- Never weaken a gate to land a change (CHECKLIST.md B7).
- No self-grading without independent verification.
- Demotions are always allowed (see docs/evidence/demotion-rules.md, rule D4).

## Project-specific instructions

### Commands

- Install/refresh the CLI: `uv tool install -e .` (stdlib only; Python >= 3.12).
- Machine snapshot: `localbench stats`.
- Which configs have a live regression gate right now: `localbench status` (each golden CURRENT /
  GENERATION-MISMATCH / UNAVAILABLE against current pins, park state, GPU users; exit 1 on any mismatch).
- Who is using the GPU now, and which sessions/omp features can send it work: `localbench gpu` (per-process GPU %,
  resident models, clients of ollama/mlx-serve/the proxy, local-routed features per omp profile). Runs record the same
  per second and are CONTENDED when a non-backend process uses >25% GPU.
- History of local-model use: `localbench watch` (supervised as hub process `lb-watch`, one sample a minute into
  runs/observe.db) and `localbench report --since 24h` (GPU-seconds by process/model, residency, clients).
- Models and releases: `localbench models` (installed models, freshness vs registry/HF, which omp features route to
  each, upstream releases). Memory store: `localbench memory [--prune]`.
- Every GPU and model action goes through this CLI (rule use-localbench-cli-for-gpu-and-models), never raw `ollama`
  or curl to :11434: residency and keep-alive `localbench status`; GPU users `localbench gpu --seconds 5`; keep a model
  loaded `localbench keep ollama:<m> forever` (a run's isolate resets it; `0` unloads); download `localbench pull
  ollama:<m>` (library tag or `hf.co/<org>/<repo>:<quant>`); remove unused models `uv run python
  scripts/prune_models.py [--delete NAME...]` (lists why each is kept); optionally pause omp's managed browser
  `localbench quiet [--display]` / `--resume` (not needed for soundness since app load is recorded, not vetoed).
  Every run, smokes included: `localbench park`, then `run|aa|ab ... --wait-idle 1800` (it now waits only for model
  conditions); never `--allow-busy` (it skipped the gate on 2026-09-24 and the smoke was CONTENDED by an omp session
  loading qwen3.8 mid-run). On a machine in use, A/Bs run `--pairs 2` or more. If the CLI lacks a verb, add it with a test.
- Model candidates cycle in two stages (the owner, 2026-09-25: "a much faster test to cycle through these"). Stage 1,
  the screen (~30 min): `localbench ab <incumbent> <candidate> --tiers think,sess --repeats 1 --pairs 1`, judged by the
  ledger's standing SCREEN gate. It can drop a candidate (a REJECT row marked SCREEN) or hold it, and never adopts one.
  Stage 2, for candidates the screen advances: the candidate's own pre-registered gate (`--pairs 2`, every gated tier).
  Only a stage-2 KEEP with a blind re-judge changes a profile.
- Backend spec: `ollama:<model>`, `mlx-serve:<model dir>` or `omlx:<model dir>` (mlx-serve and oMLX are started/stopped
  by the harness; oMLX on :11236 with a fresh one-model dir and SSD cache per start, so a restart is cold).
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
- Gates: `uv run python -m unittest discover -s tests -t .` (pure logic, ~1 s; the pre-commit hook runs it when
  code, tests or fixtures are staged), `sh scripts/check-readiness.sh`, `sh scripts/check-claim-discipline.sh`,
  `uvx ruff check localbench`. A new test must fail on a plausible bug: `uv run --quiet python scripts/mutate.py
  <cases.json>` plants it, requires the test to fail, and restores the file by sha (it holds runs/.mutation.lock
  and gives each test run a fresh bytecode cache; a same-size plant otherwise runs stale .pyc).
- Optional root probes (powermetrics, purge): the user runs `sudo scripts/install-sudoers.sh` once; code uses `sudo -n` only.

### Layout

- `localbench/` — client (streaming timing), sysstats (machine receipts), backends (ollama, mlx-serve),
  workloads (tiers: conf, micro, replay, e2e, rel, relcold, relfresh, mem, sess, think), golden (compare/update), proxy (omp traffic timing),
  render (reading views: `show`, run stream, report.md), `__main__` (CLI).
- `tests/` — stdlib unittest regression suite for the judging code; also the conformance oracle for a port.
- `fixtures/omp/` — recorded omp request bodies. Replay binds to their hash, not to the running omp; `localbench
  status` names the omp each fixture was recorded under. Re-record (`localbench record`) to follow a newer omp's
  request, then re-bank replay.
- `goldens/<host_id>/` — reviewed goldens; never compared across hosts.
- `docs/evidence/` — ledger, demotion rules, incumbents, banked receipts. `runs/` is gitignored scratch.

### Measurement law (project-specific; a violation of the model rules voids the run)

- The machine is measured while it works (the owner, 2026-09-24: "my machine is never going to be fully quiet - we need
  our testing to be while our system is working"). Apps, the screen and the person at the keyboard are recorded per
  leg (`system.during.load`: app GPU mean/p95, seconds an app passed 25%, user-active % from HIDIdleTime) and shown
  by `show`; they do not void a run. A/Bs interleave (`--pairs N`: A,B,...,A); each arm is the median of its legs
  and the band comes from the wider arm spread, so a burst lands in one leg, not one arm. App GPU % is inflated by the
  model's own saturation (8.5-14.8% on dense legs vs 2.8-3.6% on MoE legs, same desk): compare it only between legs
  that load the GPU alike; user-active % does not depend on the model. Load balance is reported, not gated (ledger).
- One model on the machine at a time: another model resident or running is CONTENDED and voids the run. The harness
  unloads other ollama models and refuses to run ollama while mlx-serve serves; do not start other local inference
  during a run; `localbench park` keeps omp sessions' smol calls off the GPU (it parks the smol model, see below).
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
- Every measured `omp` child runs with `child_env()`: localbench's own agent dir (`runs/omp-agent`, base settings
  from `fixtures/omp/agent/config.yml`, pinned as `omp_agent_config`). None of the user's MCP servers, plugins,
  settings or memory banks reach a child. With the user's dir, children read other agents' databases under ~/.claude
  and called the user's Agent Mail MCP server (2026-09-23; `--no-tools` does not remove MCP tools). Reads of the
  user's own setup (`models`, routes) use `omp_env()`.
- Loopback endpoints only. A failed local call is a finding, never a cloud fallback.
- Goldens are written only by `localbench aa <spec> --write-golden` (the packet's "UPDATE_GOLDENS"), which refuses
  unsound A/A pairs; follow with `localbench show <golden> --diff HEAD` review in the same commit (pattern 3, golden
  regeneration reflex).
- The incumbent smol model is parked during test windows (`localbench park`); measure it by its parked name
  (`ollama:localbench-parked:5642e97495e1`, same digest). `localbench unpark` restores it.
- Numbers go into claims only from banked, same-generation receipts (packet §6, §8).

### Tick loop (standing; one product-moving unit per tick)

The product is the harness and its sound verdicts: what uses local models here and why, which models, how fast and
reliable each path is (main model, memory, side work), and when a new release is worth testing — not documents about
it. A tick that changes neither scores nothing.

1. Facts, stateless: `git status --short`, `localbench status`, open beads (`jq` on `.beads/issues.jsonl`), ledger
   rows whose retry predicate may now hold.
2. Pick one unit, in this order: a GENERATION-MISMATCH golden → the user's first tasks (memory fast and reliable; a
   local model fast and reliable; both together) → a ledger retry predicate that is now testable → an open `lb-*` bead.
3. Build, then verify against reality: a live run, a break-test with a known-bad AND a known-good leg, or a measured
   number. A check that only asks our own gates does not count.
4. Commit with its `[level]`; bank the receipt; a ledger row for every kept, refuted, or void hypothesis; close a
   bead only on cited evidence. A unit that cannot reach a verdict leaves an UNKNOWN row with a retry predicate,
   never a stub.
5. Halt and name the one decision when the next unit needs the user: changing their omp profiles, killing a process
   this session did not start, spend, publishing. Parking qwen3.8 for a test window is standing (unpark after).
   Three ticks in a row that move nothing: stop and report.
6. When two agents share this tree, the other agent grades each `[test]`/`[mutation]` commit: plant your own
   defects through scripts/mutate.py, run the smoke, append a row (PASS, or FIX naming the defect; the fix gets
   its own row) to docs/evidence/reviews/<date>-cross-grades.md. Name agents and panes by tmux id (`%20`), never
   by pane number. One owner per file; the hook runs the suite on the shared tree, so commit small and green.
   With no second agent, the commit still gets a row, UNGRADED, and stays unclaimed until graded. A grade by a
   fresh-context subagent on the author's own model is recorded as that, never as a cross-model grade.
   A FIX names a plausible defect: one a reasonable edit could introduce, with a consequence a user or the next
   agent would see. A planted side effect no test claims to watch is out of scope, or it shows the claim
   overreaches, and then the fix narrows the claim. Without this bar, grading never ends: some write always
   lands where a finite test does not look.
   `scripts/tick_driver.py` (a hub process) wakes an idle loop pane, never during a run; `touch runs/LOOP-STOP`
   stops it.

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
