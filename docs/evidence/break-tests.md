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

Three rows enforced (lean_prompt_tokens, moe_prefill_ratio, first_turn_under_10s). Current tree:
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
