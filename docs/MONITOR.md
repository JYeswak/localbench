# Watching a localbench run (lb-09 watcher brief)

<!--
  Anti-ceremony (A12):
  - Consumer: every watcher agent in ntm session `omp-test` that observes a localbench run; the main agent reading their findings.
  - Gate: lb-09 — a watcher runs scripts/check-monitor-pane.sh on its own pane before watching and stops on any non-zero exit.
  - Defect class: the observer contaminating the measurement (a watcher's local model contending for GPU/unified memory), and silent bad runs (contention, thermal throttle, swap, stalls, regressions) that nobody flagged.
  - Delete when: localbench itself refuses to start while any omp on the machine resolves to a local model AND it emits every flag below as a machine-checked status in summary.json; then watchers have nothing left to add.
-->

You observe. You do not grade, re-run, fix, or steer. Your output is a list of
timestamped findings the main agent can act on.

## 1. Before you watch: screen your own pane for local-model use

Run the check on **your own** pane (your pane id: `tmux display -p '#{pane_id}'`):

```sh
~/Developer/localbench/scripts/check-monitor-pane.sh %NN; echo $?
```

| exit | meaning | what you do |
|---|---|---|
| 0 | no local main model detected by this check | proceed only if there is no `WARNING:` line (below); this is not proof that every model route is remote |
| 1 | `LOCAL MODEL: <selector>` | stop. Do not watch. Report the line. |
| 2 | undetermined (no such pane, no omp under it, selector without provider) | stop. Unknown is not safe. |

A `WARNING: smol role is LOCAL ...` line does not change the exit code, but it
means omp sends session titles (and memory work when `mnemopi.llmMode: smol`)
to a local model even though your main model is cloud. During an active run
that is contamination: a title generation loads the model the run may be
measuring against. Treat the warning as disqualifying while a run is active;
relaunching with `omp --smol <cloud-provider>/<model>` clears it in the check
(`--smol` beats config). The 2026-09-29 `localbench models` inventory found a local
`smol` route to `ollama/qwen3.8:27b-mlx` in 12 of 13 profiles; `agy` had no
installed local `smol` route. Several profiles also set `mnemopi.llmMode: smol`.
Check the actual pane before watching: a profile inventory does not prove the
route cached by an already-running omp process.

**The check's limit.** It is a point-in-time screen, not a guarantee. It reads
the omp process argv (`--model`, `--provider`, `--smol`, `--profile`), the
session `.jsonl` the process holds open (the last `model_change` / assistant
message, if `jq` is installed), and the profile's `config.yml`. It recognizes
local provider names and loopback providers declared in `models.yml`; other
provider names default to non-local, so exit 0 is not proof against an
undeclared local endpoint. A `/model` switch made after the check, or not yet
written to the session file, is invisible to it; without `jq`, session-file
evidence is skipped. Never switch models while watching; re-run the check
whenever you are unsure. An earlier exit 0 is not a standing permission.

## 2. What to read

```sh
RUN=~/Developer/localbench/runs/LATEST        # symlink to the newest run dir
tail -n +1 -F "$RUN/progress.jsonl"           # live follow; stop with Ctrl-C before the next command
uv run python scripts/monitor-report.py "$RUN/progress.jsonl"  # one-shot progress summary
```

`monitor-report.py` prints one finding line per contention, sample with a nonempty error, failed e2e, **completed**
gap over 10 minutes, preflight_wait series (count, first and last time, last problems), or a stream
whose last event is not `done`. It cannot identify an ongoing stall from a single file read; use
the clock check in section 3. A `done` with no such progress-event findings prints `done no findings`,
not an all-clear for result rows or machine state. It does not grade. `LATEST` is
repointed when the next run starts; note the resolved directory name (`readlink "$RUN"`) at the start and put it in every finding.
When the `done` event arrives, read `$RUN/summary.json` (`system.before`, `system.during`,
`system.after`, and raw `results`/`conformance`). If `$RUN/report.md` exists,
read its golden-comparison statuses; A/A and A/B legs need not have one.

Events include `start`, `preflight_wait`, `isolated`, `tier`, `sample`, `e2e`,
`contention`, `done` (field list: see the progress contract in the lb-09 packet).
An aborted run still terminates with `done`; when the watchdog triggered,
`done.watchdog_abort` contains the reason also recorded in
`summary.json` at `verdicts.watchdog_abort`. `done` marks stream completion,
not a passing verdict.


## 3. What to flag (each with its timestamp)

| flag | where you see it |
|---|---|
| any `contention` event | `progress.jsonl`; quote `resident` and `gpu_device_pct`. If the resident foreign model is under `mlx-smol` (or, before 2026-09-25, `qwen3.8:27b-mlx`), suspect an omp smol role (possibly a watcher) and say so. |
| GPU busy at start | `summary.json` `system.before.live.gpu_device_pct` / `system.preflight.problems` (including `--allow-busy` runs). `start` and `isolated` events mark timing but do not contain GPU usage. |
| thermal pressure other than `Nominal`, or any `thermal_warning` | `summary.json` `system.power.thermal_pressure` and `system.before.power.thermal_warning` / `system.after.power.thermal_warning`. `system.power.available: false` means sampled thermal pressure was not measured; check the separate before/after warning fields too. |
| swap growth | `summary.json` `system.before.live.swap_used_mb` vs `system.during.swap_used_mb.max` vs `system.after.live.swap_used_mb`; any increase. |
| sample errors | a `sample` event with a nonempty `error`; an `e2e` event with `ok: false`. |
| bad result rows | if present, `report.md` golden-comparison rows with `REGRESSED`, `FAIL`, `MUST-VOID`, `MISSING`, `GENERATION-MISMATCH`, or `TOL-UNPROVEN`; `summary.json` holds raw metrics/conformance, not these statuses. |
| stall | no new event in `progress.jsonl` for more than 10 minutes before `done` (compare the last event's `t` with `date +%s`). |
| run ended without `done` | the process for this run is gone and the last event is not `done`; `pgrep -f 'localbench'` alone cannot distinguish another run or command. |

Report what you saw; do not interpret it into a verdict. `XFAIL`,
`IMPROVED`, and `PASS` rows are not findings unless something above also applies.

## 4. What you never do

- Use a local model while a run is active: no `ollama run/generate`, no
  `omp --model ollama/...|mlx-serve/...|localbench/...`, no `/model` switch to
  one, no `localbench run/aa/ab`. Check your own pane first (section 1); stop
  on any non-zero exit.
- Send keys, signals, or prompts to a benchmark process or to another pane.
  Read files; do not touch processes.
- Grade, edit, bless, or regenerate goldens, registries, or `summary.json`.
  You observe; you do not grade.
- Start heavy local work (builds, indexing, large file scans) during a run: it
  shows up as load and contention in the measurement.

## 5. Report format

One line per finding, nothing else:

```
<UTC ISO time of the event> <run dir name> <event or source> <evidence line>
```

Examples:

```
2026-09-22T21:14:03Z 20260922T211002Z__ollama__qwen3.6-35b contention {"event":"contention","t":1790111643.2,"resident":{"ollama":["qwen3.8:27b-mlx"],"mlx-serve":[]},"gpu_device_pct":71}
2026-09-22T21:40:10Z 20260922T211002Z__ollama__qwen3.6-35b stall last event t=1790112010.5 (tier), no event for 11m
2026-09-22T21:52:44Z 20260922T211002Z__ollama__qwen3.6-35b summary.json row decode_tps/long status=REGRESSED
```

Convert `t` with `date -u -r <int t> +%Y-%m-%dT%H:%M:%SZ`. For a source event,
quote its evidence fields from the file without changing values; the helper
re-serializes events and synthesizes stall/wait/end lines, so its lines are not
verbatim `progress.jsonl` records. If the run is clean on both progress and
summary inspection, report one line: `<time> <run dir> done no findings`.
