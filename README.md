# localbench

Local-model measurements and two fixed OMP end-to-end task-case checks on macOS; banked goldens are host-specific.

localbench records machine state alongside measurements, compares runs with host goldens, and reports observed GPU
use. These are tool capabilities, not a certified speedup, reliability rate, or guarantee of an uncontended run.

## Requirements

- macOS on Apple Silicon only (GPU counters come from ioreg; machine state from sysctl, pmset, memory_pressure).
- Python >= 3.12 and [uv](https://docs.astral.sh/uv/); no third-party Python packages. A local server:
  [ollama](https://ollama.com), mlx-serve, oMLX or mlxfast (Layr-Labs' `mlx-server` from the MLX.FAST Bonsai 2 engine;
  localbench starts and stops the last three).
- omp on `PATH` or in `LOCALBENCH_OMP`: runs pin its version, and the e2e, rel, mem and sess tiers drive it.
- Optional: `sudo scripts/install-sudoers.sh` once, for powermetrics and `--purge` (localbench only calls `sudo -n`).

## Install

```sh
git clone https://github.com/JYeswak/localbench && cd localbench && uv tool install -e .
localbench --version
```

Install it editable: the clone is the data root, so goldens, receipts and `runs/` are written inside it. For a
non-editable install, set `LOCALBENCH_HOME=<clone>`.

## First run

```sh
localbench stats                              # the machine snapshot every receipt records
localbench aa ollama:<model> --write-golden   # A/A pair; writes a golden only when sound
localbench run ollama:<model>                 # later: measure again and judge against that golden
```

Goldens live in `goldens/<host_id>/` and are never compared across hosts. On a new host, `localbench aa <spec> --write-golden` writes the first golden only for a sound A/A pair; an unsound pair banks its receipt but leaves the golden unwritten.
The goldens and receipts in this repository were measured on the author's Mac
Studio (M3 Ultra, 80 GPU cores, 512 GB); read them as evidence and examples. A spec is `ollama:<model>`,
`mlx-serve:<model dir>`, `omlx:<model dir>` or `mlxfast:<model dir>`. If omp routes any role to a local model, `localbench park` before
measuring and `localbench unpark` after: a second model resident or running voids the run.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | No unsound verdict detected for the requested command; this does not certify unmeasured behavior. |
| 1 | Unsound, regressed or refused: a MUST FAIL, a CONTENDED run, a golden row REGRESSED / FAIL / MISSING / GENERATION-MISMATCH / TOL-UNPROVEN, or a refusal (a run already alive, preflight, a download that does not fit). Do not rest a claim on it. |
| 2 | Usage error: an unknown verb, a bad flag or argument value, or no data root. |
| 141 | stdout closed early (for example, piped into `head`). |

Failures print to stderr; stdout carries only results (with `--json`, one JSON document). `report` on an empty
window exits 0. The same table is in `localbench --help`.

## Environment

| Variable | Default | What it sets |
|---|---|---|
| `LOCALBENCH_HOME` | the checkout the install runs from | the clone holding `fixtures/`, `goldens/`, `runs/`; verbs other than stats, memory, keep, pull and create exit 2 when `fixtures/omp` is missing under it |
| `LOCALBENCH_OMP`, `LOCALBENCH_MLX_SERVE`, `LOCALBENCH_MLXFAST` | `PATH` | the omp, mlx-serve and mlx-server (mlxfast; needs `mlx.metallib` beside it) executables |
| `LOCALBENCH_HF_DIR` | `~/.cache/localbench/hf` | where `pull hf:<org>/<repo>` downloads |

## What the evidence says

The registered fixture below records historical request shapes, not the current omp prompt or model speed. Its pins
are in [docs/evidence/incumbents.md](docs/evidence/incumbents.md) and the fixture sidecars.

- In the omp 18.2.11 recorded default-profile requests, lean flags reduced prompt size from 73,779 to 11,433 tokens.

Lean flags: `--no-skills --no-rules --no-lsp --no-title --tools=read,bash,edit,write,grep,glob,todo`.
"Memory off" is `--config fixtures/omp/child-config.yml`; the reason is in the ledger (2026-09-23 rows).

Not claimed yet:

- Current-version prefill speedup versus the dense incumbent and cold first-turn wall: prior receipts used earlier
  Ollama/omp pins, so they are not current-generation speed proof (D6). Re-measure before making either claim.

- mlx-serve vs ollama on the same model: the latest A/B
  ([receipt](docs/evidence/receipts/ab-mlxserve-vs-ollama-moe.json)) is uncontended, but two of its legs have a MUST
  failure (the receipt's `problems`), so the release gate (packet §9) does not admit it.
- "Passes every MUST case": on both servers the MoE has sometimes answered "Reply with exactly: OK" wrongly on a
  cold first turn (raw tool-call text, "Hi, how can I help you today?"). Warm-turn results and the investigation are
  in docs/evidence/NEGATIVE_EVIDENCE.md.
- SWE-bench latency (lb-08) has not run.

## Commands

```sh
localbench stats                                         # machine snapshot
localbench status                                        # which goldens are live, per tier; fixture freshness; GPU
localbench gpu [--seconds N]                             # GPU % by process, loaded models, who can send work, why
localbench memory [--prune]                              # omp memory banks; --prune removes localbench's own
localbench models [--days 14]                            # installed models: freshness, who routes to them, releases
localbench watch  |  report --since 24h                  # record local-model use each minute | what used it, how long
localbench park | unpark                                 # move omp's local smol model out of reach while measuring
localbench keep ollama:<m> [5m|30m|2h|0]               # finite default 5m; 0 unloads only when safe
localbench gateway status|install|start|stop|remove     # OMP loopback gateway and reversible profile routing
localbench pull ollama:<m>                               # download a library tag or hf.co/<org>/<repo>:<quant>
localbench quiet [--resume]                              # pause omp's managed browser for a run; resume after
localbench run <spec>                                    # conf,micro,replay,e2e vs the golden
localbench eval run <spec> [--resume runs/eval-...] --wait-idle 1800 # versioned omp behavioral campaign
localbench eval varied <spec> --phase heldout --seed N --trials 2 --dry-run # planned read/edit cases only; live model trial evidence UNVERIFIED
localbench eval rescore runs/eval-...                      # offline re-score from saved traces
localbench aa  <spec> --write-golden                     # A/A pair -> banked receipt + golden (the only way)
localbench ab  <A spec> <B spec> --bank <name> [--pairs N]# interleaved A,B,...,A; arms by leg median
localbench run <spec> --tiers rel|relcold|relfresh       # answer reliability, warm / cold / cold without warm-up
localbench run <spec> --tiers mem                        # does omp memory recall across sessions, at what cost
localbench run <spec> --tiers sess                       # memory LLM vs main model in one interactive omp session
localbench run <spec> --tiers think                      # reasoning length, wall, accuracy on six checkable questions
localbench compare <run dir>                             # re-judge a run against today's golden
localbench bank <run dir> <name>                         # bank a single run as a receipt
localbench show <receipt|golden|run dir> [--path P]      # compact reading view; --path: one subtree, unrounded
localbench show <golden> --diff HEAD                     # what a re-bank changed: pins, rows (Δ%), tiers, verdicts
localbench record --label lean <spec> -- <omp flags>     # record omp's request body + generation sidecar
uv run python scripts/prune_models.py [--delete NAME]    # why each model is kept; deletes only named unused ones
localbench doctor [--fix] [--json]                       # PASS/WARN/FAIL per subsystem, with the command that fixes it
localbench validate <receipt|golden|run dir>             # parses; receipt legs need commit, worker and pins (exit 1 if invalid)
localbench audit [--since 24h] [--json]  |  why <id>     # the mutation ledger | one row in full
```

Campaign runs are barred from performance and golden evidence; they resume only when model, runtime, and test-input pins match, and can be re-scored offline from saved traces.

## Mutations, dry runs and the audit ledger
The following describes the CLI's intended mutation contract, not a banked live safety or reliability result.

Every verb that changes state (park, unpark, smol set|start|stop|autostart|revert, memory --prune, keep, gateway
install|start|stop|remove, pull, create, quiet [--resume], bank, aa --write-golden, record) computes one plan and runs it
(aa without --write-golden takes --dry-run and --explain too, and writes no audit row):

- `--dry-run` prints the planned actions, one per line, and changes nothing (no audit row either). With `--json` it
  prints `{verb, actions, would_refuse, noop}`. A plan the real command would refuse (a run is alive, the name
  exists) prints the refusal on stderr and exits 1, like the command.
- `--explain` prints what each action does and why, then proceeds.
- Asking for what is already so is a no-op that exits 0 and says so: park twice, unpark with nothing parked,
  smol stop when it is down, smol autostart on when it is on, gateway install when healthy, or keep <m> 0 when absent.
- Each real invocation appends one row to `~/.localbench/audit.jsonl` with its argv, cwd, host, localbench
  version, the actions and the outcome: `done` (a no-op too, with no actions), `refused` (with the reason) or
  `failed` (with the error). `localbench audit` lists rows; `localbench why <id>` prints one; an unknown id exits 2.
  `doctor --fix` records each repair it makes the same way.

## OMP Ollama residency
The following describes implemented controls and their known limits; a clean live OMP outage/restart and established
external-client unload-refusal receipt have not been banked. Do not treat configuration readback as live routing proof.

Ollama keeps a model for five minutes after a request. localbench keep defaults to that finite lease; forever and negative
durations are rejected. A zero/unload request is refused while a gateway request or external-client activity is active or
uncertain. The finite expiry is recorded in the local audit ledger and shown by localbench gateway status.

`localbench park` acquires a durable SQLite admission fence for every planned Ollama model before copying or unloading
any tag. Gateway request admission and fence acquisition serialize in one SQLite write transaction; the fence survives
gateway restart. `runs/PARKED.json` is atomically checkpointed through copy, unload and delete, so `localbench unpark`
can reconcile a partial operation, verify the restored digest, and only then release gateway admission. This fence does
not control direct Ollama clients: the existing socket/telemetry probe fails closed for activity it can observe, but a
new direct client can race that probe. Exclusive control of non-gateway clients is not claimed.

`localbench gateway install` points configured OMP profiles at a loopback-only LaunchAgent. Gateway configuration
is intended to fail closed rather than send OMP requests directly to port 11434; a clean live OMP outage/restart
proof is still pending. `stop` leaves profiles pointed at the stopped gateway; installation does not reload existing
OMP processes, so restart existing sessions before relying on the new route.
`remove` is designed to restore only the exact managed provider block and preserve manual edits. Direct non-OMP
clients can still call Ollama directly and are outside this policy.
The gateway's socket check is designed to defer unload while an established non-gateway Ollama socket remains open;
quiet nettop bytes, GPU silence, or elapsed time cannot prove request idle (long prefills can be silent). This does not
claim that an open socket is actively doing inference. Gateway-owned leases retain the five-minute idle expiry.
Inbound POST bodies have a 128 MiB limit and a 30-second total read deadline; incomplete bodies are not forwarded. This
does not bound upstream inference or response streaming. Existing residents without a gateway lease remain unowned and are
never unloaded by profile migration.

## How it measures
These measurement and isolation rules describe the harness contract, not independent current-generation proof of
every preflight refusal, contention decision, or child-process boundary.

The machine is measured while it works. Apps, the screen and user activity are recorded per leg (`show` prints app
GPU mean/p95 and user-active %) and do not void a run. Another model resident or running does: that run is
CONTENDED. Preflight refuses while another model can run (loadable smol models, stuck sessions, another runner
busy); `--wait-idle SECONDS` re-checks every 30 s, and `--allow-busy` measures anyway but marks the run non-proof.
`ab --pairs N` interleaves A,B,...,A and takes each arm as the median of its legs.

Goldens bind per tier: a moved pin that a tier exercised (backend, model, child overlay, agent config, fixtures)
stales only that tier's rows, and an omp version bump alone stales none. Measured omp children run in localbench's
own agent dir (`runs/omp-agent`), so the user's MCP servers, plugins and memory banks never reach them. Endpoints
are loopback only, and a failed local call is recorded as a finding with no cloud fallback. Agents working in this
repository follow [AGENTS.md](AGENTS.md).

## Layout

- `localbench/`: the CLI and its modules: client (streaming timing), sysstats (machine receipts, GPU by process),
  backends, workloads (tiers conf, micro, replay, e2e, rel, relcold, relfresh, mem, sess, think), golden (per-tier
  compare and merge), proxy (timing and purpose of omp's traffic), park, memory, render (reading views).
- `tests/`: stdlib unittest suite for pure logic; selected regression tests are exercised by `scripts/mutate.py`.
- `scripts/`: gates (`check-*.sh`), `mutate.py`, `prune_models.py`, `omp-resolve.ts` (omp's own model resolver,
  used by `park`), `tick_driver.py`, and one-off probes.
- `fixtures/omp/`: recorded omp request bodies and their sidecars (replay sends these; `localbench record` records
  your own), the child overlays, and `agent/config.yml`, the base settings of measured omp children.
- `goldens/<host_id>/`: reviewed goldens; never compared across hosts or generations.
- `docs/evidence/`: negative-evidence ledger, break-tests, discrepancies, incumbents, banked receipts.
- `docs/planning/packet.md`: the plan of record.

## Development

```sh
uv run python -m unittest discover -s tests -t .         # regression suite (pre-commit and CI run it)
uv run python scripts/mutate.py <cases.json>             # plant a defect, require a test to fail, restore by sha
sh scripts/check-claim-discipline.sh                     # enforced README claims need banked proof; zero enforced fails
uvx ruff check localbench
```

## License

MIT. See [LICENSE](LICENSE).
