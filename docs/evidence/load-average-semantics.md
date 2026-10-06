# Load H3: macOS load average counts runnable threads

Status: **H3 confirmed by current public XNU source; exact source revision for macOS 25.5 not independently matched.** Read-only 30-second sample collected 2026-10-04 00:02:08–00:02:36Z on the 32-core M3 Ultra (24 P-cores + 8 E-cores).

## Result

`sysctl vm.loadavg` is not CPU-busy percentage and is not capped at the number of P- or E-cores. In public XNU, `compute_averages()` sets `nthreads = sched_run_buckets[TH_BUCKET_RUN] - 1`, stores that in `sched_nrun`, and feeds `nthreads * LOAD_SCALE` to the legacy average. `TH_BUCKET_RUN` is explicitly “All runnable threads.” `processor_avail_count` is used for the separate Mach-factor calculation, not to normalize the `avenrun` load sample. The adjacent `compute_sched_load()` path is a distinct priority-bucket scheduler estimator and does normalize bucket load by CPU count; it is not the `vm.loadavg` path.

The runnable count includes executing threads as well as threads ready to run: XNU increments the count when a thread becomes runnable; the dispatch path requeues a preempted thread, while the waiting path clears `TH_RUN` and decrements the count. `compute_averunnable()` exponentially averages this sample at 5-second intervals into 1-, 5-, and 15-minute values.

Therefore a load near 88 with roughly 53% CPU idle is not a contradiction: the load value describes a smoothed count of runnable threads, not 88 busy cores or 88% utilization. It can exceed 32 because it is not core-normalized. The current sample also shows why an instantaneous `ps` count will not equal a 1-minute EWMA.

## Measurement

For 15 samples at nominal 2-second cadence, the read-only collector sampled `sysctl -n vm.loadavg` and `/bin/ps -AM -o pid,stat,pri,ni`. The `ps` rows were counted when the appended `STAT` began with `R`; the appended `PRI` numeric prefix was grouped into 0–15, 16–31, 32–47, and 48+ bands. Those bins are observational `ps PRI` bands, **not** XNU's internal `TH_BUCKET_SHARE_*` QoS buckets. The parser used fields, `substr`, and numeric comparisons; no regular expression, `lsof`, or `find` was used in the sampling loop. Collector runtime: 30.23 seconds.

| UTC | load 1m | load 5m | load 15m | R threads | PRI 0–15 | PRI 16–31 | PRI 32–47 | PRI 48+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00:02:08 | 22.20 | 36.59 | 55.92 | 31 | 10 | 20 | 1 | 0 |
| 00:02:10 | 21.31 | 36.16 | 55.65 | 17 | 4 | 12 | 1 | 0 |
| 00:02:12 | 21.31 | 36.16 | 55.65 | 17 | 5 | 11 | 1 | 0 |
| 00:02:15 | 21.31 | 36.16 | 55.65 | 33 | 5 | 27 | 1 | 0 |
| 00:02:16 | 21.36 | 35.93 | 55.46 | 53 | 41 | 12 | 0 | 0 |
| 00:02:18 | 21.36 | 35.93 | 55.46 | 45 | 14 | 29 | 2 | 0 |
| 00:02:21 | 23.57 | 36.15 | 55.42 | 47 | 11 | 34 | 2 | 0 |
| 00:02:23 | 23.57 | 36.15 | 55.42 | 20 | 3 | 17 | 0 | 0 |
| 00:02:24 | 23.57 | 36.15 | 55.42 | 20 | 3 | 14 | 2 | 1 |
| 00:02:26 | 23.21 | 35.86 | 55.20 | 11 | 1 | 7 | 3 | 0 |
| 00:02:28 | 23.21 | 35.86 | 55.20 | 26 | 6 | 20 | 0 | 0 |
| 00:02:30 | 22.31 | 35.46 | 54.95 | 44 | 2 | 40 | 1 | 1 |
| 00:02:32 | 22.31 | 35.46 | 54.95 | 17 | 2 | 12 | 3 | 0 |
| 00:02:35 | 22.31 | 35.46 | 54.95 | 42 | 12 | 28 | 1 | 1 |
| 00:02:36 | 25.25 | 35.85 | 54.97 | 28 | 7 | 20 | 1 | 0 |

Across this window: 1-minute load 21.31–25.25 (mean 22.54); 5-minute load 35.46–36.59 (mean 35.96); 15-minute load 54.95–55.92 (mean 55.35). The instantaneous R count ranged 11–53 (mean 30.07); mean `ps PRI` bands were 8.4 / 20.2 / 1.27 / 0.20. The 1-minute average was already far below the earlier high-load window.

The prior capture at `var/agent-tmp/perf-load.1/load.233816/` gives the relevant contrast. `top.txt` at 2026-10-03 17:38:56 local recorded load 78.81 / 83.01 / 86.45 with 27.62% user, 18.87% system, and 53.50% idle; the nearby `loadavg` sample was `{ 88.87 85.59 87.51 }`. The same capture set records 446 CPU-seconds (14.9 cores) over 30 seconds and at least 645 process starts observed by a 1-second `ps` poll. The exact 88.87 sample is not timestamped identically to the 53.50%-idle `top` sample, so that pairing is approximate, not byte-aligned.

## Interpretation and admission metric

H3 is supported: load average is a runnable-thread population averaged over time, not utilization and not an E-core-only or core-clamped count. The 15-point comparison shows the expected time-smoothed relationship, not point-for-point equality. The earlier process-start storm is a plausible source of scheduler/UI overhead, but these measurements do not prove it is the UI-latency cause; a synchronized UI-latency/context-switch measurement would be needed for causality.

Admission retains `load1` as context but no longer gates on it. The slot admits only when `cpu_busy < 80%` and
`mem_pressure == normal`; CPU busy at or above 80%, `warn`/`critical` memory pressure, and unreadable or unknown
sensor values refuse. The 80% policy leaves 20% CPU headroom: the `kit-qkmi.11` reproduction at load1 106 had
64–66% idle (34–36% busy), while the earlier high-load capture at load1 88.87 had 53.50% idle (~46.5% busy,
approximate because timestamps differ); both are below the bound. A synthetic 95%-busy fixture is refused. These
examples justify separation from known acceptable and bad conditions, not an empirically optimized cutoff.
Memory admission uses `sysstats.memory()`'s OS pressure level: only `normal` admits; no free-percentage cutoff is
claimed or inferred.

## Primary sources

- Apple XNU `sched_average.c`: 5-second callback registration and `compute_averages()` sampling/averaging path: [callback](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_average.c#L111-L114), [raw runnable count and `average_now`](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_average.c#L286-L318), [separate priority-bucket `sched_load` path](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_average.c#L123-L130) and [CPU normalization](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_average.c#L254-L267).
- Apple XNU `sched.h`: `TH_BUCKET_RUN` is “All runnable threads”: [bucket enum and counter](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched.h#L228-L234).
- Apple XNU `sched_prim.c`: runnable-count increment on wake/unblock and decrement after clearing `TH_RUN` on the waiting path: [increment](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_prim.c#L831-L838), [preempt/wait dispatch paths](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/sched_prim.c#L3541-L3590).
- Apple XNU `kern_synch.c`: 1/5/15-minute exponential constants and `compute_averunnable()` formula: [implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_synch.c#L458-L476).
- Apple `getloadavg(3)` documents the 1-, 5-, and 15-minute values: [Apple Developer manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/getloadavg.3.html).
- Apple `sysctl(3)` documents `VM_LOADAVG` as a `struct loadavg` query: [Apple Developer manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html).

Version caveat: citations use the public `apple-oss-distributions/xnu` `main` branch; an exact source tag matching this Mac's macOS 25.5 build was not established.