<!--
  Anti-ceremony (CHECKLIST.md A12):
  - Consumer: the Rust port's PLAN_TO_PORT_LOCALBENCH_TO_RUST.md (scope, exclusions, crate choices) and
    EXISTING_LOCALBENCH_STRUCTURE.md (the spec; its external-interfaces section cites interfaces.tsv rows), per the
    porting-to-rust skill. Anyone adding an import, a subprocess, an endpoint or an env read to localbench/.
  - Gate: tests/test_port_map.py (AST extraction over localbench/*.py and scripts/*.py; a missing or stale row fails the
    regression suite, which the pre-commit hook runs).
  - Defect class: a port that discovers a dependency mid-implementation (an ioreg key, an omp event field, a root-only
    binary, a byte format a golden depends on) instead of planning for it.
  - Delete when: the Rust port has replaced localbench/ and its FEATURE_PARITY.md tracks these interfaces; then this
    file and interfaces.tsv go with the Python code.
-->

# localbench external interfaces: summary for the Rust port

The source of truth is [interfaces.tsv](interfaces.tsv): one row per interface, columns
`kind  name  used_in  purpose  rust  notes`. This page only summarizes it; where they differ, the TSV wins.

`rust` names the crate or std API for the port, or `keep: shell out` with the reason. Service rows carry `base=<url>`;
http rows are named `METHOD service /path`.

## Counts

| kind | rows | checked against the code by tests/test_port_map.py |
|---|---:|---|
| py-stdlib | 36 | yes: every import has a row, every row is still imported, every row is a real stdlib module |
| subprocess | 27 | yes: every launched executable (literals, `omp_bin()`, `_run`/`_first_line`/`_Rpc` helpers, the program behind `sudo -n`) |
| http | 19 | yes: every URL and API path literal is covered, every row's path is still used |
| env | 15 | yes: `os.environ`/`getenv` reads, the keys `omp_env()` filters, implicit HOME/PATH/TMPDIR; shell-only rows are checked against the scripts named in `used_in` |
| sqlite | 3 | yes: every `*.db` file the code opens |
| file | 40 | hand-curated (must exist and be mapped) |
| text-format | 28 | hand-curated: the exact fields and lines parsed from other programs' output |
| os | 16 | hand-curated |
| tool | 13 | hand-curated (scripts, CI, dev loop) |
| service | 10 | hand-curated, with the versions pinned in docs/evidence/incumbents.md |

The Python side is stdlib only, so every library dependency the port takes on is new. The crates it names: serde +
serde_json, clap, ureq, httparse, rusqlite (bundled), sha2 + hex, regex, jiff, plist, uuid, url, which, walkdir, glob,
tempfile, wait-timeout, crossbeam-channel, nix, libc, sysctl, libproc, sysinfo, io-kit-sys + core-foundation (or
objc2-io-kit), and optionally rand_mt, scopeguard, ctrlc.

## Hardest to port, and why

1. **GPU attribution through IOKit** (`ioreg -c AGXDeviceUserClient`, `-c IOAccelerator`). The keys
   (`accumulatedGPUTime`, `IOUserClientCreator`, `Device Utilization %`) are Apple-private and undocumented. The text
   parser splits ioreg's output on each entry's `+-o ` marker and reads every client block whole, so key order
   inside a block does not matter (pairing line by line did: ioreg lists `AppUsage` before `IOUserClientCreator`,
   which credited the previous block's process; sysstats.gpu_time_by_pid docstring). It does depend on ioreg's
   tree format and on one creator per block. A native read (io-kit-sys) removes the text dependency but has to
   walk CF dictionaries. CONTENDED verdicts and `localbench gpu`
   both depend on it.
2. **The timing proxy** (`http.server` on :11299). Per-call TTFT and the `purpose` labels of every omp-bound tier are
   measured inside it. It re-frames chunked responses by hand, flushes every SSE line, and logs a stream omp abandons
   as `aborted`. No std HTTP server exists, and a buffering one would corrupt the numbers without any error.
3. **Byte-compatible records.** Goldens are git-reviewed JSON (`indent=2, sort_keys=True`). JSONL rows use Python's
   `', '`/`': '` separators. Python's `ensure_ascii` escapes non-ASCII, it writes NaN, and its float repr differs from
   serde_json in edge cases. Every metric passes through Python `round(x, 4)`, which is correctly rounded with
   round-half-even. Matching that means a custom serde_json formatter and a decimal-correct round; otherwise every
   golden gets re-banked.
4. **The seeded corpus** (`random.Random(7)` `sample`/`choice`/`randint`). Prompt bytes decide `prompt_tokens` for the
   micro and conf tiers. The cheap, exact option is to freeze CPython's output as fixture files. Emulating CPython's
   MT19937 seeding and sampling algorithms is the fragile one.
5. **omp's surfaces**: `--mode json` event lines, the `--mode rpc` protocol, `config list --json` settings, the
   side-call prompt markers, and the managed block written into the user's `~/.omp/agent/models.yml`. omp ships most
   days, so all of these change under the port.
6. **Process and socket introspection** (`ps`, `lsof`, `top`). libproc and sysinfo give different numbers: `ps` pcpu
   is a decaying average, while sysinfo's is a refresh delta. Another process's env is visible only for the same uid.
   Keeping these as shell-outs is a legitimate choice.
7. **HTTP client semantics.** urllib opens a new connection for every request, while ureq pools them, which can shift
   TTFT ([inference], not measured). Server status is three-valued (answer / refused = down / timeout = unknown). SSE
   is read line by line.
   `DELETE /api/delete` carries a body.
8. **Child processes.** std has no run-with-timeout, and `terminate` means SIGTERM, then SIGKILL after a wait. Rust also
   starts with SIGPIPE ignored, so `println!` panics on a closed pipe. The Python code gets the same behavior as a
   BrokenPipeError and handles it only around stdout, returning exit 141. A process-wide SIG_DFL killed the sess tier
   on 2026-09-23.
9. **Root probes** (`sudo -n` powermetrics and purge). These stay shell-outs. The binary must never be setuid.

## Surprises found while mapping

Evidence: reading the code as of 2026-09-23, except where a line says *observed*: a command run on this host that day
(metadata only, with no model or GPU involved).

- **mlx-serve's log lines (`[args]`, `[mtp]`, `[spec-stats] mode=…`) are parsed by no code.** They appear only as
  evidence read by hand. The ledger's MTP retry predicate depends on `[spec-stats] mode=mtp`, and nothing checks it.
- **`omp_sha` hashes a JavaScript file.** *Observed* with `readlink -f ~/.bun/bin/omp`: it resolves to
  `…/@oh-my-pi/pi-coding-agent/dist/cli.js`, so the pin covers that entry script, not a native binary.
- **The `[mem]` filter in `backends._first_line` is for mlx-serve.** *Observed* with `mlx-serve --version`: it prints
  `[mem] MLX buffer-pool cap …` before its version.
- **`sysstats.inference_clients` records PI_CODING_AGENT_DIR.** `omp_client_identity` reads it from `ps eww`.
  An omp with only that variable is `agent_dir`, not profile `default`. An omp with neither still defaults.
- **The ollama version and hash can come from different binaries.** `ollama --version` runs PATH's ollama, while
  `backend_sha` hashes `/Applications/Ollama.app/…/ollama`. *Observed*: here `/usr/local/bin/ollama` is a symlink to
  that file. On another host it need not be.
- **Two network reads leave the machine**: registry.ollama.ai (manifests) and huggingface.co (model metadata).
  Inference stays on loopback.
- **Paths ignore TMPDIR.** Run cwds and the mlx-serve log are hard-coded under `/tmp`. memory.py's harness detection
  depends on that.
- **Smaller gaps.** Splash's binary pin does not measure the model, and it does not stale ollama/mlx tiers. The `-dirty` marker on `localbench_rev` only looks at
  `localbench/`. The proxy's fixed port means two concurrent runs collide.

## Open questions for the port plan

- **Golden binding.** Pins carry no harness identity: `localbench_rev` is provenance only. A Rust harness that measures
  slightly differently (pooled connections, another proxy) would be judged against Python-banked goldens without a
  GENERATION-MISMATCH. Add a harness pin, or re-bank everything under the port.
- **Byte format.** Should JSON outputs be byte-identical (custom formatter) or semantically identical (one re-bank
  plus a reviewed `git diff goldens/`)?
- **Native or shell-out.** For ioreg, ps, lsof and top: which native readers are worth their macOS-version risk?
  Linux CI needs `cfg(target_os = "macos")` stubs either way.
- **Concurrency model.** Threads (as today) or tokio? Only the proxy benefits from async.
- **Scripts.** The gates (`scripts/*.sh`, `.githooks/pre-commit`) and break-tests stay POSIX sh unless moved into a
  cargo xtask. The break-tests grep the CLI's stderr (`UNSOUND  <key>: <status>`, `preflight refused: …`), so those
  strings are an interface.
- **SWE-bench oracle (LB-08).** swebench 5.0.2 and OrbStack 2.2.3 are pinned in incumbents.md, but no code uses them
  yet. When lb-08 lands they need rows.

## Keeping the map true

`uv run --quiet python -m unittest discover -s tests -t .` runs tests/test_port_map.py. A new import, launch,
URL/path literal, env read or `.db` file with no row fails it, and the failure names the item and `file:line`. So
does a row the code no longer uses. A launch whose `argv[0]` cannot be resolved statically also fails: map its helper
in `EXEC_HELPERS` or pass a literal. Rows of the curated kinds are not checked for staleness. Review them when the
code they describe changes.

Not done by this map: none of the `rust` choices has been compiled or benchmarked. They are recommendations for the
PLAN, not verified parity. The curated rows (file, text-format, os, service, tool) were checked only by reading the
code, and none of the parsers was run against live tool output for this document.
