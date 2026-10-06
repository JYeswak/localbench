# localbench CLI/tooling audit — 2026-10-02

## Denominators and positive controls

- Top-level verb denominator: 37 names parsed from `localbench --help` (`localbench/__main__.py` subparser registry). Positive control: `localbench status --help` exited 0 and contained `--json`.
- Help probe: all 37 `<verb> --help` invocations exited 0. Positive control: `localbench show --help` printed positional `target` and options.
- JSON probe: 37 top-level help surfaces checked for literal `--json`; gaps are filed as beads `kit-e30.26` before this report. Positive control: `features --help` contains `--json`.
- Mutation flag probe: same 37 surfaces checked for both `--dry-run` and `--explain`; gaps are filed as `kit-e30.27`. Positive control: `keep --help` contains both.
- Robot/capability probe: `discover-cli.sh .` reported `robot_mode=false`, `capabilities=false`, `robot_docs=false`; gap bead `kit-e30.28` filed before reporting.
- Doctor registry denominator: `localbench/doctor.py:289-294` contains 13 `CHECKS`; requested foundational subsystems (heavyslot lock, gateway, digest/omp-update LaunchAgents, omp-frozen snapshots, tick drivers, public export, hooks) are not named in that registry. Gap bead `kit-e30.29` filed before reporting. Positive control: `doctor --json` emitted `platform` and `goldens` rows.
- Non-TTY probe: `verify-non-tty-discipline.sh localbench status --json` returned RC 1: stdin timeout violation. Gap bead `kit-e30.32` filed before reporting.

## Exit/help observations

- `localbench --version`: RC 0, `localbench 0.1.0`.
- Unknown verb: RC 2 with argparse usage on stderr.
- `localbench status --json`: RC 1 because current golden state is not clean; JSON still emitted. This is evidence for documented failure semantics, not classified as a bug here.
- `localbench doctor --json`: RC 1 with structured rows; current WARNs include mlxfast, sudoers, and generation-mismatched golden.

## Read-only CLI latency

Measurement command: hyperfine, 3 warmups, 20 runs each; `docs/audit/perf/fingerprint.json` records load average, machine, Python, and git SHA. Raw exports: `docs/audit/perf/{status,features,show,audit,doctor}.json`; ranked p50/p95 table: `docs/audit/perf/ranked-table.json`. Nonzero read-command exits were measured with `--ignore-failure` and remain recorded in the raw command behavior.

| Rank by p95 | Verb | p50 (s) | p95 (s) | n |
|---:|---|---:|---:|---:|
| 1 | show | 0.184 | 0.207 | 20 |
| 2 | audit | 0.206 | 0.354 | 20 |
| 3 | features | 5.284 | 7.593 | 20 |
| 4 | doctor | 8.630 | 9.625 | 20 |
| 5 | status | 10.543 | 13.147 | 20 |

Top latency bead: `kit-e30.30`.

## Tooling inventory

Positive controls: `python3 scripts/{land.py,tick_driver.py,mutate.py,export_public.py} --help` each exited RC 0. `.githooks/pre-commit`, `.github/workflows/kit-gates.yml`, and all four scripts exist. Land and mutation execution remain separately gated by the shared stale Git lock/heavy slot; this audit did not run GPU work.
