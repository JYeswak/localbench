# Break-tests: each gate fired on a known-bad input, with the verbatim output

<!--
  Anti-ceremony (A12):
  - Consumer: the reviewer asking "does this gate actually fire?"; a later agent re-running a check after editing it.
  - Gate: a check is not trusted until its known-bad case is recorded here with the exact command and output.
  - Defect class: gates that always pass (wrong field, wrong pane, silent fallback) and so certify nothing.
  - Delete when: each recorded break-test runs automatically in CI/selftest with the known-bad fixture; then this file is a human index only.
-->

## lb-09 — check-monitor-pane.sh

Run 2026-09-22 21:07 MDT (2026-09-23T03:06:47Z) on `mac-studio-apple-m3-ultra-512gb`, read-only
(tmux list-panes, ps, lsof, config/session file reads; no keys sent, no model loaded).

Pane ids confirmed first; they match the packet (%12 = omp on `ollama/qwen3.8:27b-mlx`, %10 = omp on a
cloud Muse model, default profile):

```
$ tmux list-panes -s -t omp-test -F '#{pane_id} #{pane_title}'
%9 omp-test__user_0
%10 omp-test__omp_1
%15 omp-test__omp_4
%12 omp-test__omp_3_ollama/qwen3.8
```

### Known-bad: %12, omp launched with `--model ollama/qwen3.8:27b-mlx` — must exit 1

```
$ sh scripts/check-monitor-pane.sh %12; echo $?
pane %12 shell_pid=4782 omp_pid=6547 title='omp-test__omp_3_ollama/qwen3.8'
  argv: bun ~/.bun/bin/omp --model ollama/qwen3.8:27b-mlx
  profile: ~/.omp/agent (lsof agent.db)
  session: ~/.omp/agent/sessions/-Developer-omp-test/2026-09-20T23-32-55-019Z_01a0c12a-606b-7088-a66c-64f9b9053c39.jsonl current=ollama/qwen3.8:27b-mlx
  config: modelRoles.default=anthropic/claude-opus-5-5:xhigh modelRoles.smol=ollama/qwen3.8:27b-mlx mnemopi.llmMode=smol
WARNING: smol role is LOCAL (ollama/qwen3.8:27b-mlx) and mnemopi.llmMode=smol: titles and memory work hit the local model even when the main model is cloud
LOCAL MODEL: ollama/qwen3.8:27b-mlx
1
```

Fired: exit 1, argv and session file agree.

### Cloud main model: %10, omp on Muse, default profile — exit recorded, smol WARNING expected

```
$ sh scripts/check-monitor-pane.sh %10; echo $?
pane %10 shell_pid=22573 omp_pid=23710 title='omp-test__omp_1'
  argv: bun ~/.bun/bin/omp --auto-approve
  profile: ~/.omp/agent (lsof agent.db)
  session: ~/.omp/agent/sessions/-Developer-omp-test/2026-09-20T21-48-16-492Z_01a0c0ca-92ec-747b-a476-969ce2113db2.jsonl current=muse-code/muse-spark-1.3-contributor
  config: modelRoles.default=anthropic/claude-opus-5-5:xhigh modelRoles.smol=ollama/qwen3.8:27b-mlx mnemopi.llmMode=smol
WARNING: smol role is LOCAL (ollama/qwen3.8:27b-mlx) and mnemopi.llmMode=smol: titles and memory work hit the local model even when the main model is cloud
NON-LOCAL MODEL: muse-code/muse-spark-1.3-contributor
0
```

Exit 0 with the smol WARNING. Note what argv + config alone would have said: `anthropic/claude-opus-5-5:xhigh`
(no `--model`, config default). The pane actually runs Muse: its session file records
`model_change` to `xai-oauth/grok-4.6` at launch, then a `/model` switch to
`muse-code/muse-spark-1.3-contributor`, and 445 assistant messages from Muse. Only the session-file source
sees runtime switches, which is why it outranks argv/config in the script.

### Known-bad: bogus pane ids — must exit 2

```
$ sh scripts/check-monitor-pane.sh %999; echo $?
UNDETERMINED: no tmux pane %999 (tmux list-panes -a)
2
$ sh scripts/check-monitor-pane.sh bogus; echo $?
usage: scripts/check-monitor-pane.sh <tmux-pane-id like %12> (got 'bogus')
UNDETERMINED: not a tmux pane id
2
```

Why the pane list and not `tmux display`: `tmux display -p -t %999 '#{pane_pid}'` printed nothing and
exited 0 in this run, and inside tmux a failed target can resolve to the caller's own pane, so a
`display`-based check could certify the watcher's own pane under a bogus id.

### Extra cases observed in the same run

```
$ sh scripts/check-monitor-pane.sh %15; echo $?
pane %15 shell_pid=43462 omp_pid=52772 title='omp-test__omp_4'
  argv: bun ~/.bun/bin/omp
  profile: ~/.omp/agent (lsof agent.db)
  session: none open current=?
  config: modelRoles.default=anthropic/claude-opus-5-5:xhigh modelRoles.smol=ollama/qwen3.8:27b-mlx mnemopi.llmMode=smol
WARNING: smol role is LOCAL (ollama/qwen3.8:27b-mlx) and mnemopi.llmMode=smol: titles and memory work hit the local model even when the main model is cloud
NON-LOCAL MODEL: anthropic/claude-opus-5-5:xhigh
0
$ sh scripts/check-monitor-pane.sh %9; echo $?
pane %9 shell_pid=22555 title='omp-test__user_0'
UNDETERMINED: no omp process under pane %9
2
```

%15 holds no session file open, so its verdict rests on argv + config only (the launch-time limit
applies in full). %9 runs no omp; its child is `/Applications/Ollama.app/Contents/Resources/ollama serve`
(pid 78482) — the ollama daemon lives in this pane.

## lb-10 — claim-discipline gate and its canary (2026-09-22)

### Finding: the kit's checker never parsed a row on macOS /bin/sh

The kit splits TSV rows by translating tabs to `\001` and reading with `IFS=$'\001'`. macOS `/bin/sh` is
GNU bash 3.2.57, which strips `\001` instead of splitting on it:

```
$ printf 'a\001b\001\001d\n' > /tmp/lb-claimgate/soh.txt
$ /bin/sh -c 'IFS=$(printf "\001") read -r x y z w < /tmp/lb-claimgate/soh.txt; echo "sh: x=[$x] y=[$y] z=[$z] w=[$w]"'
sh: x=[abd] y=[] z=[] w=[]
$ zsh -c '...same...'
zsh: x=[a] y=[b] z=[] w=[d]
```

Consequence: every registry row parsed as one field, `enforce` was always empty, no row was ever enforced,
and the pre-commit canary was rejected only by the "zero enforced rows while README exists" rule — a pass
for the wrong reason. Kit checker (commit 52c6a2d) on a canary replica:

```
$ sh /tmp/lb-claimgate/old-checker.sh /tmp/lb-claimgate/canary.tsv /tmp/lb-claimgate/README.md; echo "exit=$?"
FAIL: no enforced claims (enforce=yes) in /tmp/lb-claimgate/canary.tsv while /tmp/lb-claimgate/README.md exists and is non-empty.
...
Rows seen (label => enforce): labelreadme_patterncapability_keyexpected_substrproof_pathenforcenotes=> canaryCANARY_OVERCLAIM_XYZ/tmp/lb-claimgate/canary-proof-missing.mdyescanary fixture=>
check-claim-discipline: 0 passed, 1 failed, 2 skipped (0 enforced, 0 actually checked, 0 pattern-unmatched).
exit=1
```

Fix: separator `\037` (unit separator), which /bin/sh, dash, zsh, and bash all split on. Same canary after the fix:

```
$ sh scripts/check-claim-discipline.sh /tmp/lb-claimgate/canary.tsv /tmp/lb-claimgate/README.md; echo "exit=$?"
FAIL  canary: proof artifact missing or empty: /tmp/lb-claimgate/canary-proof-missing.md
check-claim-discipline: 0 passed, 1 failed, 0 skipped (1 enforced, 1 actually checked, 0 pattern-unmatched).
exit=1
```

### Gate strengthening with two-direction evidence (B7): spike receipts are non-proof

`/tmp/lb-claimgate/c.tsv` holds two enforced rows: `spike_row` cites docs/evidence/receipts/2026-09-22-spike.md
(newly rejected), `real_row` cites docs/evidence/incumbents.md (still admitted). Identical output under /bin/sh and /bin/dash:

```
$ sh scripts/check-claim-discipline.sh /tmp/lb-claimgate/c.tsv /tmp/lb-claimgate/README.md; echo "exit=$?"
FAIL  spike_row: proof docs/evidence/receipts/2026-09-22-spike.md is a spike receipt (non-proof per packet §8); cite a banked harness receipt
PASS  real_row -> docs/evidence/incumbents.md
check-claim-discipline: 1 passed, 1 failed, 0 skipped (2 enforced, 2 actually checked, 0 pattern-unmatched).
exit=1
```

### Hook canary now asserts the failure reason

.githooks/pre-commit now requires the canary to fail with `FAIL  canary: proof artifact missing`. Run from a
scratch git repo with each checker (python harness, `sh .githooks/pre-commit`):

```
OLD checker -> 1
pre-commit: CANARY FAILED — the gate rejected the canary for the wrong reason:
FAIL: no enforced claims (enforce=yes) in .../claims.tsv while .../README.md exists and is non-empty.
---
FIXED checker -> 0
pre-commit: canary self-test ok (gate rejects overclaims for a missing proof).
```

## lb-01 — preflight refuses a busy machine (2026-09-22)

Host `mac-studio-apple-m3-ultra-512gb`, ollama 0.32.15, model `qwen3.6:35b-mlx`, localbench working tree
(uncommitted). `scripts/break-preflight.sh` plants the load, runs `localbench run <spec> --tiers conf --repeats 1`,
and exits 0 only if localbench exits non-zero with the planted reason. Preflight refuses before any tier runs.

```
$ date -u +%FT%TZ; sh scripts/break-preflight.sh cpu; echo "script=$?"; sh scripts/break-preflight.sh gpu; echo "script=$?"
2026-09-23T03:47:08Z
preflight refused: CPU already 53.6% busy  (rerun with --allow-busy to measure anyway; such a run is non-proof)
localbench exit=1
BREAK-TEST OK: preflight refused for the planted reason (cpu)
script=0
preflight refused: GPU already 90.8% busy  (rerun with --allow-busy to measure anyway; such a run is non-proof)
localbench exit=1
BREAK-TEST OK: preflight refused for the planted reason (gpu)
script=0
```

cpu = one `yes >/dev/null` per logical CPU (32); gpu = a loop of 400-token `/api/generate` calls on the MoE, 15 s
before preflight. Admitted direction (same code, unplanted): runs 20260923T034053Z and 20260923T034457Z passed
preflight with cpu_busy_pct 24.6 and 14.8, GPU mean 3.2% and 5.4%.

### Gate change with two-direction evidence (B7): CPU busy replaces load average

The load1 rule refused an idle machine: at 2026-09-23T03:30Z preflight printed
`preflight refused: load1 24.2 > 24 P-cores` while `/usr/bin/top -l 2 -s 2` reported 91.21% and 88.29% idle
(load averages 18.61 18.78 17.09). The replacement judges measured idle time and still refuses a planted CPU burn
(above). load1 stays in the receipt (`system.preflight.idle_check.load1`); it is no longer a refusal criterion.
Not tested: a planted GPU consumer other than ollama (e.g. a Metal game); ioreg's Device Utilization covers it in
principle.

### Planted: GPU signal unavailable

`/tmp/lb-fakebin/ioreg` is a stub that prints nothing and exits 0 (no IOAccelerator keys):

```
$ date -u +%FT%TZ; out=$(PATH=/tmp/lb-fakebin:$PATH localbench run ollama:qwen3.6:35b-mlx --tiers conf --repeats 1 2>&1); rc=$?; printf '%s\n' "$out" | grep -E 'preflight'; echo "localbench exit=$rc"
2026-09-23T05:02:06Z
preflight refused: GPU signal unavailable (ioreg IOAccelerator utilization keys missing)  (rerun with --allow-busy to measure anyway; such a run is non-proof)
localbench exit=1
```

### Organic: a second model loaded mid-run marks the run CONTENDED

Not planted — it happened: run 20260923T033709Z (receipt docs/evidence/receipts/2026-09-22-omp-side-calls.md §2)
had `qwen3.8-uncensored:latest` loaded by an omp smol fallback during the e2e tier; the sampler emitted two
`contention` events and `localbench run` printed `UNSOUND  CONTENDED: another model was resident during the run`
and exited 1.

## lb-06 — golden compare fails loudly (2026-09-22)

Golden `goldens/mac-studio-apple-m3-ultra-512gb/ollama__qwen3.6_35b-mlx.json` (banked from
docs/evidence/receipts/aa__ollama__qwen3.6_35b-mlx__20260923T035230Z.json). `scripts/break-golden.sh <run> <case>`
doctors the golden in place, re-judges the aa2 leg with `localbench compare` (no measurement), and restores it on exit.

```
$ R=runs/20260923T035637Z__aa2__ollama__qwen3.6_35b-mlx; for c in clean regress genmismatch tolunproven; do sh scripts/break-golden.sh "$R" $c; echo "script=$?"; done
localbench compare exit=0 (case clean, key micro.decode.decode_tps)
BREAK-TEST OK: clean
UNSOUND  micro.decode.decode_tps: REGRESSED
localbench compare exit=1 (case regress, key micro.decode.decode_tps)
BREAK-TEST OK: regress
UNSOUND  e2e.ok.first_llm_s: GENERATION-MISMATCH
… (every row GENERATION-MISMATCH)
localbench compare exit=1 (case genmismatch, key micro.decode.decode_tps)
BREAK-TEST OK: genmismatch
UNSOUND  micro.decode.decode_tps: TOL-UNPROVEN
localbench compare exit=1 (case tolunproven, key micro.decode.decode_tps)
BREAK-TEST OK: tolunproven
```

(each followed by `script=0`; `git status goldens/` unchanged afterwards). Plants: regress = golden value ×1.30;
genmismatch = `pins.backend_version` → `0.0.0-planted`; tolunproven = `tol_source` deleted.

Sensitivity finding (not a gate defect, a property of the bands): the first version of the script planted the ×1.30
on `micro.cache_hit_8k.speedup_x` (tol 0.2333 from an A/A spread of 7.8%) and compare returned PASS — the aa2 value
75.59 sits 20% under a doctored 94.59, inside the band. A regression smaller than a metric's A/A-derived band is
invisible by construction; the per-metric `tol` in the golden says how large a change each row can see. The script
now plants on the tightest higher-is-better band (decode_tps, tol 0.05).
The clean case passes only because DISC-001 lists `conf.greedy_deterministic` XFAIL for this backend/model;
scoping is checked by `listed_discrepancies` returning it for ollama/qwen3.6:35b-mlx and not for
mlx-serve/Qwen3.6-35B-A3B-MLX-Serve-4bit or ollama/localbench-parked:5642e97495e1.

## lb-02 / lb-04 — conformance exit status and replay VOID rules (2026-09-23)

mlx-serve started with `--ctx-size 8192` (via `--server-arg=--ctx-size --server-arg=8192`) and
fixtures/omp/full.meta.json's `omp_sha` planted as `0000planted0000` for the duration of the run (restored after,
`cmp` identical). omp 18.2.11, child overlay sha `62eed267e219ddde`, localbench a7752e2+.

```
$ localbench run mlx-serve:$HOME/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit --tiers conf,replay --repeats 1 --server-arg=--ctx-size --server-arg=8192
  {"event": "sample", "label": "conf.ctx_64k", … "error": "HTTPError: HTTP Error 400: Bad Request"}
| replay.full.cold_ttft_s | VOID (GENERATION-MISMATCH: fixture (omp, sha, child config) ('18.2.11', '0000planted0000', '62eed267e219ddde') vs running ('18.2.11', 'ce797fb3ed92e768', '62eed267e219ddde')) | …
| replay.lean.cold_ttft_s | VOID (loaded context 8192 < fixture prompt 11433) | …
| conf.no_truncation_64k | MUST | FAIL | FAIL |
UNSOUND  MUST FAIL: conf.no_truncation_64k
end   07-planted-ctx8192: exit=1
```

Three gates in one run: a MUST FAIL is unsound on its own (the line precedes any golden row); a fixture from another
omp binary is GENERATION-MISMATCH, never averaged; a sample whose loaded context is below the fixture prompt is VOID.
mlx-serve refused the over-context requests with HTTP 400 (no silent truncation), and reported the planted 8192 as
its context in /v1/models, which is what the VOID rule read. The run printed `NO GOLDEN` in campaign 2
(2026-09-23T06:17Z, same MUST FAIL line); this campaign-3 run was compared against the then-banked mlx-serve golden.

**Superseded 2026-09-23 (per-tier generation binding, below):** a fixture recorded under another omp binary is no
longer VOID. Replay measures the backend on the recorded bodies whatever omp runs today; its golden rows bind to
`fixtures_sha`, and freshness against the running omp is a `fixture_fresh` detail plus a `localbench status` line.
The context-below-prompt VOID and the MUST FAIL line are unchanged.

## lb-07 — omp's own `mlx-serve` provider, answer-checked, then with the server stopped (2026-09-23)

The user's provider entry (~/.omp/agent/models.yml `mlx-serve`, discovery on :11234), not the localbench proxy;
default profile env stripped; `--config fixtures/omp/child-config.yml`; server pinned to the model directory.

```
$ omp -p "Reply with exactly: OK" --model mlx-serve/Qwen3.6-35B-A3B-MLX-Serve-4bit --smol mlx-serve/Qwen3.6-35B-A3B-MLX-Serve-4bit --config fixtures/omp/child-config.yml --mode json --no-session --no-title   (x3, server up)
["OK","Qwen3.6-35B-A3B-MLX-Serve-4bit","mlx-serve","stop"]      # final assistant text, model, provider, stopReason; exit=0 each
$ (server stopped; same command)
["","error","Unable to connect. Is the computer able to access the url?"]    # exit=1
```

Loud, no fallback: the final message names provider `mlx-serve`, empty content, stopReason `error`, exit 1. Slow to
fail: omp auto-retried 10 times with backoff, 07:33:20Z → 07:39:59Z (6 m 39 s) before giving up.

## B6 — first enforced claims, and what the claim gate does and does not catch (2026-09-23)

At that time, three rows were enforced (lean_prompt_tokens, moe_prefill_ratio, first_turn_under_10s):
`check-claim-discipline: 3 passed, 0 failed, 3 skipped (3 enforced, 3 actually checked, 0 pattern-unmatched).`

Proof drift is caught — the receipt copy with `"value": 6.0838` changed to `9.9999`:

```
$ sh scripts/check-claim-discipline.sh /tmp/lb-claims-drift.tsv README.md; echo "exit=$?"
FAIL  first_turn_under_10s: proof /tmp/lb-proof-drift.json lacks expected text: "value": 6.0838
exit=1
```

README drift is not — the README copy changed "6.1 s" to "4.1 s":

```
$ sh scripts/check-claim-discipline.sh registries/claims.tsv /tmp/lb-README-drift.md; echo "exit=$?"
WARNING  first_turn_under_10s: enforce=yes but the pattern was NOT found in the README — row NOT checked.
exit=0
```

By the kit's design an unmatched pattern is a warning (a claim removed from the README must not fail the gate),
so an edited number becomes an unregistered claim the checker cannot see. README edits that touch a registered
sentence are review-enforced; the WARNING line is the reviewer's signal. Not changed here (a gate policy change,
not a fix).

### B6/kit-b14 gate repair (2026-09-29)

The warning/exit-0 result above is historical, not the current gate policy. An enforced README pattern that
disappears now prints `FAIL` with the claim label and exits 1; demote or retire the claim explicitly with
`enforce=no`. The test originally changed the registered 6.1 s sentence to 4.1 s; after D6 demotion it changes
the remaining enforced historical lean-token sentence from 11,433 to 8,192 and checks that refusal.
Before the two D6 demotions, `sh scripts/check-claim-discipline.sh` returned
`3 passed, 0 failed, 4 skipped (3 enforced, 3 actually checked, 0 pattern-unmatched)`, exit 0; a nonempty public
README with zero enforced claims returned `0 passed, 1 failed, 1 skipped (0 enforced)`, exit 1.
`sh scripts/check-readiness.sh` returned READY, exit 0; `tests.test_gate_scripts` ran 4 tests OK.
`tests/gate-mutations.json` has six deliberately planted defects, all `caught=true`, `restored=true`, with good
exit 0 and bad exit 1; the new unmatched-pattern mutation is caught by the claim-drift test. Mutation output:
`runs/kit-b14-gate-mutations-20260929T0505Z.jsonl` (gitignored local scratch). This repairs the specific
enforced-claim drift; it does not discover arbitrary unregistered sentences elsewhere in the README.

The D6 audit later demoted two speed claims whose banked receipts predate the current backend/omp pins; see the
2026-09-29 UNKNOWN rows in `docs/evidence/NEGATIVE_EVIDENCE.md`. The README retained only the pinned historical
request-shape claim. At that intermediate point the claim gate printed `1 passed, 0 failed, 6 skipped (1 enforced,
1 actually checked, 0 pattern-unmatched)` and exited 0. After the test switched to that sentence, all six
`tests/gate-mutations.json` defects were planted again and caught/restored, including the unmatched-pattern case.

The remaining enforced row cited `fixtures/omp/lean.meta.json`, an omp 18.2.11 fixture sidecar rather than a banked
same-generation receipt. Demoting it under D4 leaves no valid enforced claim. `sh scripts/check-claim-discipline.sh`
now exits 1: `0 passed, 1 failed, 7 skipped (0 enforced, 0 actually checked, 0 pattern-unmatched)`. `kit-b6` is
reopened. Do not re-enforce a historical sidecar or weaken the zero-enforced gate to make the pre-commit hook green;
a banked current-generation claim must be earned first. The old speed receipts and fixture remain historical evidence.

The gate's test fixture now supplies an independently valid enforced row and a drifted README, so the gate's
intended fail-closed production state does not mask the detector test. `uv run python -m unittest tests.test_gate_scripts`
ran 4 tests, OK. `tests/gate-mutations.json` caught and restored all six planted defects with good_rc 0 and bad_rc 1,
including the unmatched enforced claim and caller-cwd path defects.

## lb-01 — per-process GPU contention (Sampler gpu_foreign_max_pct, preflight attribution)

Run 2026-09-23 09:1x MDT on `mac-studio-apple-m3-ultra-512gb` against a real contaminant: ollama runner
pid 71271 (`--model qwen3.8:27b-mlx`, `Stopping...`) generating for this agent's own five scout subagents (scout.md `model: "@smol"`; ledger
2026-09-23). First attempt: the contention event opened on the first sample, before any GPU window existed, so
`foreign_gpu` was `[]` — fixed by making the first sample wait one interval. Script: /tmp/breaktest-gpu.py
(3.5 s Sampler per target, then `_busy_check()`).

### Known-bad: target qwen3.6 while the qwen3.8 runner is busy — must name it

```
target qwen3.6:35b-mlx -> contention [{"foreign": {"ollama": ["qwen3.8:27b-mlx"]}, "foreign_gpu": [[71271, "qwen3.8:27b-mlx", 91.9]]}] | run table [('ollama', 'qwen3.8:27b-mlx', 89.5), ('Terminal', None, 5.3), ('WindowServer', None, 3.4)]
```

### Known-good: target qwen3.8 (the busy runner is the backend's own) — must not fire

```
target qwen3.8:27b-mlx -> contention [] | run table [('ollama', 'qwen3.8:27b-mlx', 88.1), ('Terminal', None, 5.9), ('WindowServer', None, 5.7)]
```

### Preflight names the process instead of a bare device %

```
preflight problems: ['GPU already 100.0% busy (by process: ollama pid 71271 (qwen3.8:27b-mlx) 86.6%, WindowServer pid 179 6.1%, Terminal pid 27181 5.9%)', "omp smol model(s) ['qwen3.8:27b-mlx'] are loadable: every omp session's titles/memory will contend mid-run; run `localbench park` first"]
```

Not yet exercised: a foreign GPU process with no foreign resident model (e.g. a browser tab); the per-process
path is the same `foreign_gpu` list, but no known-bad of that shape was available.

## lb-06 — per-tier generation binding and partial re-bank (2026-09-23)

Why: omp ships most days (user, 2026-09-23). With every golden row bound to the omp binary, each release turned
every row GENERATION-MISMATCH, including micro/conf rows that never touch omp. Now a row goes stale only when a pin
its tier depends on moves (golden.tier_keys): conf/micro = backend + model + macOS; replay = those + fixtures_sha;
e2e/rel family = those + omp version/sha/child config. `aa --tiers … --write-golden` merges into the existing golden
(golden.merge), keeping untouched tiers with the pins they were banked under unless the backend or model moved.

Both directions, on synthetic goldens (script /tmp/check-tier-binding.py, all asserts passed):

```
omp moved: {'micro.decode.tps': 'PASS', 'replay.lean.cold_ttft_s': 'PASS', 'e2e.ok.first_wall_s': 'GENERATION-MISMATCH', 'conf.tools': 'PASS', 'e2e.ok.correct': 'GENERATION-MISMATCH'}
fixtures moved: {'micro.decode.tps': 'PASS', 'replay.lean.cold_ttft_s': 'GENERATION-MISMATCH', 'e2e.ok.first_wall_s': 'PASS', 'conf.tools': 'PASS', 'e2e.ok.correct': 'PASS'}
backend moved: {'GENERATION-MISMATCH'}
after e2e re-bank under new omp: {'micro.decode.tps': 'PASS', 'replay.lean.cold_ttft_s': 'PASS', 'e2e.ok.first_wall_s': 'PASS', 'conf.tools': 'PASS', 'e2e.ok.correct': 'PASS'}
legacy golden without fixtures_sha: {'micro.decode.tps': 'PASS', 'replay.lean.cold_ttft_s': 'GENERATION-MISMATCH', 'e2e.ok.first_wall_s': 'PASS', 'conf.tools': 'PASS', 'e2e.ok.correct': 'PASS'}
model changed, dropped: ['conf', 'micro', 'replay']
OK
```

Plus: a fixture re-record after the merge still invalidates the kept replay rows. Live, on the real goldens
(`localbench status`, omp 18.2.11): the ollama MoE golden banked under 18.2.10 reports conf, micro CURRENT and e2e
GENERATION-MISMATCH; every golden banked before this change reports replay GENERATION-MISMATCH
(`fixtures_sha: [null, …]`) until its replay tier is re-banked — no pins were backfilled by hand.

Caught by the real-run leg, not the synthetic one: the first version compared pins with `!=`, so re-judging
`runs/20260923T083344Z__run__ollama__qwen3.6_35b-mlx` (omp 18.2.11 bodies) against the 18.2.10 golden
(18.2.10 bodies) printed `8 replay PASS` — both lacked fixtures_sha, None == None. golden.tier_diff now treats a pin
the golden never recorded as moved; the same run dir then printed `8 replay GENERATION-MISMATCH`, with conf
(4 PASS, 1 XFAIL) and micro (9 PASS, 1 IMPROVED, 1 REGRESSED) judged — previously every one of those rows was
GENERATION-MISMATCH.

## lb-01 — backend saturation vs a foreign runner; preflight per-process rule (2026-09-23)

Why: both A/A re-banks started 2026-09-23 ~15:36Z were refused CONTENDED on WindowServer at 25.3–30.9% (device
98–100%). An exemption for the compositor was drafted on the theory that saturation inflates its accounted time; the
known-good leg below refuted that, and the exemption was reverted before landing (ledger REJECT row). Script:
/tmp/breaktest-compositor.py (Sampler targeting ollama qwen3.6:35b-mlx, 25% per-process veto; it ran twice in
parallel by mistake, both copies shown).

```
{"leg": "backend-only", "samples": 24, "device_mean": 91.2, "windowserver_max": 9.3, "contention_foreign_gpu": [], ...}
{"leg": "backend-only", "samples": 46, "device_mean": 88.6, "windowserver_max": 9.2, "contention_foreign_gpu": [], ...}
{"leg": "foreign-runner", "samples": 31, "device_mean": 95.4, "windowserver_max": 7.8, "contention_foreign_gpu": [[["ollama", "localbench-parked:5642e97495e1", 47.3]]], ...}
{"leg": "foreign-runner", "samples": 53, "device_mean": 92.8, "windowserver_max": 9.3, "contention_foreign_gpu": [[["ollama", "localbench-parked:5642e97495e1", 43.5]]], ...}
OK
```

Known-good: the backend alone saturating the GPU fires nothing and WindowServer stays ≤9.3%. Known-bad: a second
runner generating at the same time is named by the per-process path. So WindowServer at 25–31% for tens of minutes
was desktop activity competing for the GPU, and the veto was correct. Change kept: `_busy_check` now refuses (or
`--wait-idle` waits) when any process is above 25% before the run starts, so a busy desktop no longer costs a 35-min
run that is voided at the end.

## lb-06 — server flags are part of the generation (backend_args), 2026-09-23

Why: `--server-arg` (e.g. mlx-serve `--mtp`, `--drafter`, `--ctx-size`) changed what was measured without changing any
pin, so a flagged run compared against the default golden as the same generation, and `aa --write-golden
--server-arg …` would have replaced the default golden. Now MlxServe/Ollama pins carry `backend_args` (a BACKEND_KEYS
member); goldens banked before it record none and are read as "" — they were default launches (aa never received
server flags; their mlx-serve logs show `[args] ... ctx-size=0, pld=on`). Script /tmp/check-backend-args.py:

```
legacy golden vs default launch        -> PASS
legacy golden vs --mtp launch          -> GENERATION-MISMATCH
golden banked with --mtp vs --mtp      -> PASS
golden banked with --mtp vs default    -> GENERATION-MISMATCH
OK
$ localbench aa mlx-serve:/nonexistent --server-arg=--mtp --write-golden; echo $?
refusing --write-golden with --server-arg: the golden for a spec is its default launch; measure a server flag with `localbench ab <spec> <spec> --b-server-arg=<flag> --bank <name>`
1
```

First version tested the key's presence instead of its value (golden_tier_pins fills every key with None), which
turned every golden GENERATION-MISMATCH on `{"backend_args": [null, ""]}` — caught by `localbench status` on the real
goldens before commit; fixed to `is None`.

## kit-receipt-conf-details-vej — non-PASS detail retention (2026-09-29)

Before the change, `tests.test_cli.ReceiptView` failed: `_receipt_view` dropped FAIL conf/replay diagnostics and
the verdict-less `replay.full` perf count. The render test failed because banking that perf count expanded the
all-PASS inline view. The CLI duplicate-tier test failed for run, AA, and AB: a second case with the same ID could
overwrite the first case's detail/verdict. After the change, `uv run python -m unittest tests.test_cli
tests.test_render -v` ran 43 tests, all OK. FAIL and VOID diagnostics retain recorded/replayed/reason fields;
verdict-less replay perf detail retains the full-prompt token counts through `--path`; PASS case detail is excluded
and the all-PASS rendered view is the same length as a view without per-case results. Duplicate tiers are refused
by argparse before any measurement handler runs.

`env TMPDIR=~/Developer/localbench/runs uv run --quiet python scripts/mutate.py
tests/receipt-mutations.json` planted seven defects: drop FAIL, drop VOID, drop perf-only replay counts, bank PASS,
show perf-only counts inline, render PASS detail inline from a banked receipt, and accept duplicate tier IDs. Each
reported `caught=true`, `restored=true`, good exit 0 and bad exit 1; the named tests failed on the planted defects.

A synthetic AB receipt (no model invoked) exercised the real CLI reader: `uv run python -m localbench show
runs/receipt-detail-ab-smoke.json` displayed each leg's FAIL recorded/replayed fields without displaying PASS
or perf-only detail inline. `--path /legs/1/details/replay.lean.prompt_tokens` returned
`{"recorded":11433,"replayed":10853}`; `--path /legs/1/details/replay.full/prompt_tokens` returned `73779`.
`uv run python -m localbench run ollama:no-such-model --tiers replay,replay` exited 2 at argument parsing with
`duplicate tier in --tiers`, before backend access. The scratch receipt was removed. Existing banked receipts are
not retroactively filled, and this smoke does not establish a live inference result.

## kit-b3 — a receipt without commit or worker identity is invalid (2026-09-29)

`localbench validate` previously accepted an A/B leg after removing `provenance.localbench_rev` or
`provenance.fingerprint`, or after removing `fingerprint.model`; each of the three new
`tests.test_mutations.Validate` cases failed before the validator change with `AssertionError: 0 != 1`.
The validator now rejects those legs, naming the missing field. A banked historical A/B still
passes: `localbench validate docs/evidence/receipts/ab-incumbent-vs-moe.json` ->
`valid ab: docs/evidence/receipts/ab-incumbent-vs-moe.json`.
`uv run python -m unittest tests.test_mutations.Validate` -> 7 tests OK.
Read-only receipt inventory: 59 `docs/evidence/receipts/*.json` documents passed
`localbench.__main__.validate_doc` with zero invalid receipts; this checks field presence,
not soundness or current-generation applicability.
`env TMPDIR=~/Developer/localbench/runs uv run --quiet python scripts/mutate.py
tests/receipt-mutations.json` -> 10/10 `caught=true`, `restored=true`, with good exit 0
and bad exit 1. The three new plants accept a leg missing its commit revision,
worker fingerprint, or worker model; each named validator test rejects that regression.
Validation of the receipt's fields does not establish that its old Ollama/omp generation proves
today's speed; the two D6 demotions are recorded in `NEGATIVE_EVIDENCE.md:887-905`.

## oMLX worker attribution — own load versus foreign contention (2026-09-29)

The SOUND descriptive session receipt
`docs/evidence/receipts/ab-mlxserve-vs-omlx-qwen36-sess-descriptive-20260929.json`
records the oMLX GPU worker as `name=python3.13`, `cmd=omlx-server`, 63.9% in its
first B leg. Before the change, `gpu_is_ours` matched `omlx` but not
`omlx-server`, and `is_inference` required `omlx serve`: the worker inflated
app-GPU and the same foreign process could evade CONTENDED on an Ollama run.
The two new `tests.test_sysstats.RunnerAttribution.test_omlx_python_worker_*`
and `test_foreign_omlx_python_worker_voids_an_ollama_run` cases failed before
the fix with `False is not true` and `[] != [worker]`, respectively.

The classifier now recognizes `omlx-server` in a worker command even when its
process name is `python3.13`; a measured oMLX worker is its own GPU work, and
one above the foreign GPU threshold is a model contender for an Ollama run.
Both focused tests passed; `tests/sysstats-mutations.json` planted each
missing classification separately, with `caught=true`, `restored=true`,
good exit 0 and bad exit 1 in both cases. A throwaway read of the **actual
receipt row** after reloading the changed module returned
`own_worker=True`, `app_gpu_from_worker=0.0`,
`foreign_worker_blocks_ollama=True`. The full unittest suite ran 418 tests
OK (with a ResourceWarning for an unclosed SQLite connection); Ruff and
`git diff --check` passed. Existing receipts retain their recorded load
summaries: their app-GPU figures are not retroactively corrected, and
CONTENDED cannot be re-judged from an incomplete historical process series.

## Unknown residency in E2E campaigns — non-proof, not contention (2026-09-29)

`cmd_eval_run` now records a case as VOID if its sampler observed an unknown
resident-model state; a confirmed second model remains CONTENDED, while known
isolation and application GPU load alone do not void it. The offline status
smoke printed `isolated PASS`, `unknown VOID`, `contended VOID`.
`tests/unknown-residency-mutations.json` planted six defects: accepting an
unknown E2E sample, hiding a known competitor behind another endpoint's
unknown state, ignoring unknown samples in the shared soundness gate, and
persisting unknown-residency bank, A/A, or A/B receipts. Each named regression
failed, and all six mutants were restored (`good_rc=0`, `bad_rc=1`). Before the
A/A and A/B fix, their two unknown-leg tests failed because receipts existed;
after it, the focused 9-test suite passed, including known-leg banking. A
filesystem smoke also observed `bank` returning `(1, False)` for unknown
samples and `(0, True)` for known isolation (exit code, receipt exists). The
shared-tree suite then ran 478 tests OK; readiness, Ruff and `git diff --check`
passed. These are offline handler and oracle checks, not a live model campaign.

## Memory oracle — exact recall and successful control response (2026-09-29)

The one-shot memory oracle rejects a recall answer that merely contains the
planted value and voids a fresh-control result without a completed successful
main response. An offline `mem()` smoke using the existing subprocess/proxy
fixture produced `valid hits 3 no_leak PASS`, `stale_recall hits 0 no_leak PASS`,
and `failed_control hits 0 no_leak VOID`: no-leak is separate from recall
correctness. `tests/memory-oracle-mutations.json` planted a substring-match
recall and a control check that ignored proxy completion; each named test
failed and each mutant was restored (`good_rc=0`, `bad_rc=1`). The shared-tree
unittest suite ran 478 tests OK after correcting a test fixture that lacked
the real summary's required `conformance.level`; readiness, Ruff, and
`git diff --check` passed. This is offline oracle evidence, not a live
memory-plus-model acceptance run.

## Summary view and seeded READ campaign controls (2026-09-30)

`localbench show runs/20260930T155847Z__run__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit`
now prints `# summary · UNSOUND` and `resident model state unknown in 2 sampler sample(s); run is non-proof`.
The saved report already labeled this run UNSOUND, and the run exit predicate rejects unknown residency;
only the `show` summary view had called it SOUND.
The renderer tests distinguish unknown residency (`contended=no`) from a known second model
(`CONTENDED`, `contended=yes`) and preserve SOUND for known isolation despite 80% application GPU load.

The seeded READ campaign test executes an offline OMP fixture through `run_trial`, saves real workspace and
trajectory files, records both cases in `EvaluationCampaign`, and re-scores them. A matching successful
`target.txt` read passes; reading `decoy.txt` while returning the correct target answer fails. The fixture
checks unchanged target and decoy bytes, recorded paths and responses, and campaign trace provenance.
`uv run --quiet python -m unittest tests.test_render tests.test_load tests.test_evaluation tests.test_varied`
ran 63 tests OK; `uv run python -m unittest discover -s tests -t .` ran 495 tests OK (two SQLite
ResourceWarnings). Gateway status reported four in-flight requests, so no isolated GPU run or live held-out
agent trial was performed. Neither offline control closes the corresponding live-acceptance Bead.

## Shared parked alias — restore every tag before deleting the alias (2026-09-30)

Two original Ollama tags with the same digest receive one `localbench-parked:<digest>` alias.
Before the fix, the new roundtrip test failed with `cannot restore qwen3.8-uncensored:latest:
neither original nor parked copy has digest ...`; an interrupted restore test lost that alias
before retry (`KeyError`). `unpark()` now keeps the alias until every original is restored,
then deletes each unique alias before releasing the park journal and admission fence.
Both tests pass. An isolated fake-Ollama smoke called `park.park()` and `park.unpark()`:
both originals disappeared during park, then reappeared with the same digest; the shared
alias and journal were gone afterward. The early-deletion mutation was caught and restored
(`good_rc=0`, `bad_rc=1`); 84 park/gateway/smol tests passed after the initial repair.
No live model, real Ollama tag, or gateway request was touched by this smoke.

A separate pane-2 review found that two different full digests can share the 12-character alias:
the old park path moved the first name before detecting the collision, then `unpark` refused
an intact second original and held the journal/fence on every retry. Both new regression tests
failed before correction. `park` now refuses the collision before moving a tag; `unpark` can
recover a previously journaled partial collision after validating every original and the alias.
An alias matching no journaled digest is retained with its journal and fence instead of deleted.
The isolated fake-service smoke printed `collision_refused`, `tags_unchanged: True`,
`journal_absent: True`; three causal mutations each reported `caught=true`, `restored=true`,
`good_rc=0`, `bad_rc=1`. This is offline recovery proof, not a live Ollama park.

The same peer then found a collision between a *new* park plan and an alias already in a
partial `PARKED.json`: the old preflight added a journal entry and a new admission fence
before refusing. The new regression failed first; `safety_refusal()` now checks recorded
aliases before the plan, without probing or moving a previously parked name. The separate
causal mutation omitting that journal check failed the test and restored the source. An
isolated fake-service smoke printed `continuation_refused_before_journal_write: True`,
`new_fence_absent: True` and then `prior_fence_released: True` after `unpark()` recovered
both originals. The full 501-test suite passed; the live park gate remains unverified.

With gateway requests in flight at zero, `localbench gpu --seconds 5` still found
an established direct Ollama client (`omp profile=p15-fake`, PID 76764, `proj-b` scratch
project). `localbench park --dry-run` printed smol and fallback plans, then exited 1:
`cannot park qwen3.8:27b-mlx: established non-gateway Ollama client connection remains`.
This is a live known-bad refusal, not a successful park/unpark or isolated model run.

## Cross-process park and agent outcome controls (2026-09-30)

`park()` and `unpark()` now hold the same persistent `PARKED.json.lock`
across the plan recheck, journal, admission fence, tag changes and cleanup.
A pre-existing same-digest alias without journal ownership is refused before
mutation. The subprocess race test holds park immediately after its journal
write: unpark returns 1 without changing tags or journal; park completes,
then unpark restores both original tags and removes the fence. The alias
test preserves the original tags, journal absence and fence absence on
refusal. Both tests failed before the fix. `tests/park-safety-mutations.json`
planted a missing lock and an omitted alias refusal; both were caught and
restored (`good_rc=0`, `bad_rc=1`). These tests use fake services, not Ollama.
Parked-alias prune fence orphan (2026-10-01, uncommitted): an independent review found
`scripts/prune_models.py --delete` could remove a journaled parked alias and drop its PARKED.json
entries without releasing the gateway park fence, whose ids live only in the journal — leaving the
original model fenced with no `unpark` able to release it. `prune_models.delete()` now refuses a
journaled parked alias (naming `localbench unpark`), and rechecks the journal under park's
`PARKED.json.lock` at the mutation boundary, so a concurrent park/unpark cannot change the journal
`tests/test_prune_models.py` covers CLI and direct-delete refusal without any
model deletion, the lock-held mutation boundary, and truthful `KEPT` inventory
output. Targeted suite: `uv run --quiet python -m unittest tests.test_prune_models`
— 30 tests OK.

Session RPC teardown now closes and reaps the child on an interrupted turn
and when interruption occurs inside `close()`; both real child-process tests
failed before the fix and passed afterward. Both
`tests/memory-rpc-mutations.json` plants were caught and restored. A timed-out
varied trial now kills its OMP process group before capturing final files;
offline tests caught a child that otherwise rewrote `target.txt` after the
snapshot. Offline re-scoring rejects a hashed snapshot that disagrees with
the current workspace, and the scorer rejects boolean wall durations.
`tests/evaluation-mutations.json` caught and restored all eight defects,
including those three. The `localbench eval rescore` offline fake-OMP smoke
returned 0 and PASS for a matching workspace, then 1 with
`final snapshot differs from workspace: target.txt` for a contradictory one.
No live held-out OMP agent trial or memory plant/control/recall is proved by
these fixtures.

Direct finite gateway leases now reject NaN, infinity and overflowing
expiry before storing a row. `tests/gateway-mutations.json` caught and
restored all 13 defects, including this guard. Shared-tree verification:
`uv run python -m unittest discover -s tests -t .` ran 512 tests OK
(two SQLite ResourceWarnings); `uvx ruff check localbench`,
`sh scripts/check-readiness.sh`, and `git diff --check` passed. The unchanged
claim-discipline gate still exits 1: 0 enforced of 13 registered claims.

After the owner authorized pausing the verified proj-a clients, their three OMP
agent processes exited while the tmux shells remained. A separate
user-owned `ollama run nimble` client (PID 85093, localbench pane `%19`)
then held a direct Ollama socket. `localbench gpu --seconds 5` identified
that client and `localbench park --dry-run` exited 1 with
`cannot park qwen3.8:27b-mlx: established non-gateway Ollama client connection remains`.
No live park, model run, unknown-residency sample or model comparison was
started. the owner subsequently reported an Ollama upgrade; the earlier
0.34.4 binary readback does not pin its new generation.

## Peer-reviewed alias, RPC and varied-trial safeguards (2026-09-30)

Peer review found that `unpark()` could delete a matching parked alias while
a request used it, or treat an alias as owned before `_copy` completed.
Three fake-service regressions failed before the repair: an active gateway
request, simulated external client activity and another client's same-digest
alias after a failed copy. The earlier two-tag recovery test now refuses
cleanup until an unowned alias is removed. `unpark()` checks journal phase
ownership, fences the alias before restoring originals, rechecks unload
safety before removal and preserves the journal/fence on refusal. Targeted
park tests: 38 passed. Four `tests/park-safety-mutations.json` plants were
caught and restored (`good_rc=0`, `bad_rc=1`). Direct clients can still
connect between socket checks; no host-wide exclusion is claimed.

The RPC constructor now kills and reaps its child and closes pipes if pump
startup is interrupted. The real-child regression failed before the fix;
`tests.test_workloads.SessTurns tests.test_workloads.RpcCancellation` ran
12 tests OK. Three `tests/memory-rpc-mutations.json` plants were caught
and restored. This is not a live memory plant/control/recall result.

A detached OMP tool can escape the process-group kill and write after the
timeout snapshot, or keep stdout/stderr pipes open after the OMP parent exits.
The first scenario previously made a timed-out FAIL receipt unscorable
after its workspace hash changed; the second raised another timeout before
writing a result. Timed-out trials now hash only the bounded saved evidence,
not a workspace a detached child can still mutate; offline re-scoring keeps
the timeout as FAIL without treating the mutable workspace as success proof.
If detached children keep the pipes open, the runner closes its own pipe
ends, reaps OMP, saves available output and records a timeout instead of
discarding the verdict. Non-timeout snapshots and re-scoring open workspace
files with `O_NOFOLLOW` and check the opened descriptor is regular; two
tests deliberately made path-level symlink metadata stale, and both failed
before the repair. Absent process exit metadata now scores ERROR, not a
model FAIL. `tests.test_varied.VariedTrialRunner` plus
`tests.test_evaluation.VariedTrialScoring` ran 12 tests OK; all 12
`tests/evaluation-mutations.json` plants were caught and restored. A real
`localbench eval rescore` offline fake-OMP smoke returned 0/PASS for an
unchanged saved trial and 1 with `trace is missing or changed:
attempt/workspace/target.txt` after a contradictory workspace write.
No held-out real OMP agent trial is proved by these fixtures.

The first 520-test shared-suite run after the changes failed solely because
new `errno` and `stat` stdlib imports were missing from
`docs/port/interfaces.tsv`. Both rows were added; the specific
`tests.test_port_map.CodeIsMapped.test_every_import_has_a_row` check passed.
`localbench status` then reported Ollama backend 0.35.0 SHA
`add45eb02df0252f` versus the old golden's 0.34.4 SHA
`bba8b79eac84ab09` (GENERATION-MISMATCH), while mlx-serve's
conf/e2e/micro/replay golden was CURRENT. A new foreign direct Ollama
connection from decision service PID 12925 appeared in `localbench gpu --seconds 5`;
`localbench park --dry-run --explain` exited 1 with
`cannot park qwen3.8:27b-mlx: established non-gateway Ollama client connection remains`.
No live park, model run or unknown-residency acceptance leg was started.

## Live park and residency controls (2026-10-01)

The later direct Ollama client released its socket without intervention.
`localbench gpu --seconds 5` then reported no foreign Ollama client;
`localbench park --dry-run --explain` admitted the operation, and
`localbench park --explain` parked both the smol and fallback tags as
`localbench-parked:5642e97495e1` and
`localbench-parked:34875c4701a6`. `localbench status` reported both parked
and no Ollama model loaded. The live refusal above remains a separate
negative control; the park is not yet a proven recovery until an eventual
safe `localbench unpark` restores both tags.

With smol parked and a fresh GPU/gateway preflight, the pinned
`mlx-serve:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit`
ran `--tiers e2e --repeats 1 --wait-idle 1800` twice. The first receipt is
`runs/20261001T004006Z__run__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit`;
the predeclared second control is
`runs/20261001T004614Z__run__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit`.
Both e2e MUST cases passed on both runs and neither reported contention,
but **both receipts are UNSOUND/non-proof**: each sampler recorded two
`resident.mlx-serve=null` intervals. One preceded server startup and one
occurred while the model process was active. The `/v1/models` probe did not
return a parsable success response in those intervals; the saved evidence
does not establish why. Neither leg is banked as a golden or a
current-generation performance/reliability claim. Starting a second model
would violate the one-model measurement rule; the live competitor control
remains unproved.

The MLX-serve log does not show the GET timing or error, and its server
supports up to three resident models: the launched process's `--model`
argument cannot substitute for an authoritative resident-set response.
Keep the unknown samples non-proof until an independently timestamped,
complete resident-set observation is available; do not infer a timeout
from `null` alone.

The second shared 520-test run failed at the documentation reader count:
`docs/port/INTERFACES.md` still said 38 after the import table grew to 40.
The count was corrected and
`tests.test_port_map.TableShape.test_the_readers_counts_match_the_table`
passed. The post-integration `uv run python -m unittest discover -s tests -t .`
run passed all 520 tests. Those pure-logic checks do not upgrade the two
live UNSOUND E2E receipts.

The claim audit reconciled `registries/claims.tsv` with the current README:
13 rows, 8 populated README patterns and 8 exact matches; 5 rows have no
README pattern, including the withdrawn speed and cold-first-turn claims.
None of the 8 matched claims has a banked same-generation proof under the
observed Ollama 0.35.0 and omp 18.4.5 pins, so all 13 remain `enforce=no`.
The claim-discipline gate still fails honestly: `sh scripts/check-claim-discipline.sh`
observed 0 enforced, 0 actually checked, 0 pattern-unmatched, with all 13
rows at `enforce=no`. It was not relaxed and no historical proof was promoted.

## Live memory/session diagnostic (2026-10-01)

`localbench run
mlx-serve:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit
--tiers mem,sess --repeats 1 --mem-rounds 1 --wait-idle 1800` wrote
`runs/20261001T005315Z__run__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit`.
`localbench show` reported **SOUND**, 0 unknown-residency samples,
no contention and conformance 3/3 PASS. Three planted facts had exact
fresh-process recall; the independent controls did not return those facts.
The `sess` path acknowledged 12/12 turns with zero timeouts, made three
successful memory-extract calls (`not_ok=0`) and overlapped extraction
with main work by 5.265 s
(`summary.json#/results/6/detail`). This is one sound diagnostic, not a
banked A/A reliability distribution; the two earlier UNSOUND E2E receipts
remain non-proof.

## Held-out varied read/edit campaign (2026-10-01)

After `localbench gpu --seconds 5` showed no direct inference clients and
gateway status showed no in-flight requests, the preregistered command ran
once: `localbench eval varied
mlx-serve:~/.mlx-serve/models/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit
--phase heldout --seed 700 --trials 2 --wait-idle 1800`. The score is
`runs/eval-varied-1790817008175348000-Qwen3.6-35B-A3B-MLX-Serve-4bit/scores/b6fedd57b670dc9ea74256f711f01e9f1a86be9563e168414c7a7de1bf1ab3ba.json`:
4 graded, 1 PASS, 3 FAIL, 0 ERROR, 0 void (Wilson 95% interval
`[0.0456, 0.6994]`). `read-700` answered `INITIAL`: the server stream
stopped mid-value although the model's own reasoning quoted the full planted
value (`trials/read-700/bodies/eval-varied-03.response.jsonl`). `edit-700`
wrote the exact expected bytes with omp 18.4.5's default hashline `edit`
tool, whose only argument is `input` (the path sits in its `[PATH#TAG]`
header). The grader requires `args.path` (`localbench/evaluation.py:365-368`),
so this FAIL is a grader defect, not a model miss. `read-701` passed.
`edit-701` wrote the expected value without the required trailing newline
(33 of 34 bytes). Corrected 2026-10-01 from the saved trajectories; the
saved score is unchanged, and a fixed grader is a new campaign identity.
These fixed inputs were not rerolled. The small result is functional
evidence only, not an accuracy distribution or performance evidence.

## Live park restoration (2026-10-01)

With gateway healthy and zero requests in flight, and `localbench gpu --seconds 5`
showing no inference clients or resident models, `localbench unpark --explain`
restored `qwen3.8:27b-mlx` (`5642e97495e1`) and
`thinkingcap-qwen3.8:27b-nvfp4` (`34875c4701a6`). The CLI refreshed and read
back all 13 OMP catalogs; each listed both restored tags. `localbench park
--status` then reported `parked: []`, and `localbench models --json` reported
the tags installed and `parked: false`. Audit row
`20261001T012433Z-c1c7de` (`localbench why` rc=0) records the successful
restore and catalog refresh.

A subsequent `localbench gpu --seconds 5` observed the restored smol model
resident, with a live OMP client from
`~/Developer/proj-a` using the gateway. The model was
left resident; no client or process was killed. This later use is not part of
the parked-state measurement.

## Parked-alias prune fence leak and repair (2026-10-01)

Static source trace found that the old `scripts/prune_models.py` path could
delete a journaled parked alias and remove its `PARKED.json` entry without
releasing the gateway park fence. `park()` stores fence IDs only on journal
rows; `unpark()` releases only IDs remaining in that journal
(`localbench/park.py:322–334,493–498`). Losing the last carrier left the
original model permanently fenced (`localbench/gateway.py:293–296`).

Fixed in the working tree: `plan()` receives a deterministic journaled-alias
set and lists those entries as kept; `check_delete()` refuses with
`localbench unpark` guidance; `delete()` rechecks under the shared
`PARKED.json.lock` before touching the tag or journal. Red-first regression
tests cover the CLI and direct refusal, the lock-held mutation boundary, and
the `KEPT` inventory. No live tag or park state was changed. Verification:
the targeted prune suite passed 30 tests, and the post-integration full suite
passed 525 tests.

Separate static issue, not exercised at runtime: the gateway maps any
non-draining `GatewayError` from request admission to “OMP profile is not
registered,” including a parked-model fence refusal
(`localbench/gateway.py:753–761,293–296`).
