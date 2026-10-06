# Machine-level settings changed for load (kit-kcct.9)

Host: Mac Studio, M3 Ultra, 32 logical CPUs (24 P-cores + 8 E-cores), macOS 25.5. Each row names the change, who made
it, what was measured before and after, and the revert command. Measurements are read-only (`ps -AM`, `/usr/bin/top -d`,
`git status`); raw data is under `var/agent-tmp/perf-load.1/` and `var/agent-tmp/load-hunt/` (untracked scratch).

| Date (UTC) | Change | By | Measured effect | Revert |
|---|---|---|---|---|
| 2026-10-03 21:4x | CPU-only heavy-slot gates run at nice 10, not macOS background QoS (`taskpolicy -b`) | %pane, commit 72411f8 | under load 100, background QoS gave a suite 34 s of CPU in 53 min (PRI 4, E-cores only); no landing in 4 h | `git revert 72411f8` |
| 2026-10-03 23:2x | Spotlight results disabled in System Settings | the owner | `mdworker_shared` processes 0; `mds_stores` still indexing (6.7 runnable threads); runnable threads 93.2 -> 81.7 per sample | System Settings -> Spotlight |
| 2026-10-03 23:3x | Spotlight indexing disabled on every volume (`sudo mdutil -a -i off`) | the owner | `mdutil -s /`: "Indexing disabled."; background-band runnable threads 32.6 -> 9.5 per sample; Spotlight threads 0 | `sudo mdutil -a -i on` |
| 2026-10-04 00:1x | `var/agent-tmp/` added to localbench `.gitignore` | %pane, commit cde37b6 | `git status --porcelain -z --untracked-files=all`: 51,911 -> 458 entries, 0.05 s; ntm's internal-monitor stats each entry every 15 s (3,922 BSD syscalls/s before) | `git revert cde37b6` |
| 2026-10-04 00:2x | `var/agent-tmp/` added to the global git excludes file `~/.config/git/ignore` | omp-test %pane | uds untracked entries 83,950 -> 856 (its ntm monitor: 8,363 syscalls/s before) | delete the line from `~/.config/git/ignore` |

Load average over the same window (`sysctl vm.loadavg`): 88-110 at 23:20Z, `21.36 35.93 55.46` at 00:2xZ. Several
changes landed together (gates stopped, Spotlight off, the ignores, proj-b and proj-a notified), so the drop is not
attributed to any single row. Per-row effects above are the only attributed ones.
