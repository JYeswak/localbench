# Receipt — 2026-09-23: cold first-turn answers on Qwen3.6-35B-A3B, memory off

<!--
  Anti-ceremony (A12):
  - Consumer: the 2026-09-23 UNKNOWN ledger row on cold first-turn wrong answers; README "not claimed" list.
  - Gate: B3 provenance for a tally whose run dirs are gitignored scratch.
  - Defect class: a failure rate quoted without its denominator or its time window.
  - Delete when: the UNKNOWN row is resolved (cause found or retry predicate met).
-->

All runs: omp 18.2.11 (sha `ce797fb3ed92e768`), child overlay `62eed267e219ddde` (memory off), ollama 0.32.15
`qwen3.6:35b-mlx`, mlx-serve 26.9.2 `Qwen3.6-35B-A3B-MLX-Serve-4bit`, prompt "Reply with exactly: OK", the backend
re-isolated immediately before each counted attempt. Tally from each run's summary.json (`e2e.ok.correct`, the
first attempt is the cold one) and the relcold/relfresh details.

| window (UTC) | source | cold "ok" attempts | wrong |
|---|---|---:|---:|
| 07:07–07:32 | e2e tier inside full-tier runs: ab a1/b/a2, aa1/aa2 ollama, aa1/aa2 mlx-serve | 7 | 3 |
| 08:16–08:27 | relcold + relfresh, both backends (receipts relcold-*, relfresh-*) | 80 | 0 |
| 08:28–08:44 | conf,micro,replay then relcold, both backends (runs 20260923T082816Z, 20260923T083344Z) | 40 | 0 |
| 08:45–08:51 | e2e tier alone, 6 runs per backend (runs 20260923T084511Z … 20260923T085036Z) | 12 | 0 |
| total | | 139 | 3 |

The three wrong answers, all between 07:11Z and 07:21Z:

- runs/20260923T071108Z__ab_b__mlx-serve__…: thinking "The user asked me to respond with \"Hi, how can I help you
  today?\"", answer `Hi, how can I help you today?`
- runs/20260923T071426Z__ab_a2__ollama__qwen3.6_35b-mlx: thinking "…reply with \"OK\" in English and \"خوب\" in
  Farsi", answer `OK  \nخوب`
- runs/20260923T071800Z__aa1__ollama__qwen3.6_35b-mlx: thinking "The user wants me to reply with exactly \"OK\"…",
  answer `<function=read>\n<parameter=path>\n/home/<user>/code/llm-serve/src/routes/llm/v1/chat/completions.rs …`

Same prompt size as passing attempts (10,736 tokens on mlx-serve, 11,451 on ollama), cached_tokens 0. Request bodies
were not captured for these three (body capture landed at commit f98aac4, after them). Warm attempts in the same
generation: 120/120 correct (receipts rel-*.json). Wilson 95% interval for 3/139: roughly 0.7%–6.2%.
