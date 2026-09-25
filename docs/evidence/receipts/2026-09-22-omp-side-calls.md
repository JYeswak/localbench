# Receipt — 2026-09-22: omp auxiliary (smol) calls reaching other local models

<!--
  Anti-ceremony (A12):
  - Consumer: the three 2026-09-22 ledger rows on parking, fuzzy resolution, and `--smol` routing.
  - Gate: B3 provenance for ledger evidence whose raw source (runs/, /tmp/ollama-serve.log) is scratch.
  - Defect class: a ledger row whose evidence lived only in a gitignored run dir or a /tmp log.
  - Delete when: never while those ledger rows stand; these are verbatim excerpts, not measurements.
-->

Host `mac-studio-apple-m3-ultra-512gb`; ollama 0.32.15 (sha `eee609f0a6da58b9`); omp 18.2.10 at
`~/.bun/bin/omp` (sha `acf06c76a4969558`), default profile; localbench working tree (uncommitted, 2026-09-22).
Timing numbers below are single runs and non-proof (packet §8); the excerpts document *which model was loaded*.

## 1. Prefix-renamed parked model was loaded by a smol call

`localbench park` (old naming) had copied `qwen3.8:27b-mlx` to `localbench-parked/qwen3.8:27b-mlx` at 21:18 MDT.
/tmp/ollama-serve.log (ollama serve stdout):

```
[GIN] 2026/09/22 - 21:26:46 | 404 |      4.3055ms |       127.0.0.1 | POST     "/v1/responses"
time=2026-09-22T21:27:09.516-06:00 level=INFO source=client.go:391 msg="starting mlx runner subprocess" model=localbench-parked/qwen3.8:27b-mlx port=64721
[GIN] 2026/09/22 - 21:27:32 | 200 | 30.264115041s |       127.0.0.1 | POST     "/v1/responses"
```

`bun run scripts/omp-fuzzy-probe.ts` (pi-tui fuzzyMatch, query `qwen3.8:27b-mlx`):

```
{"query":"qwen3.8:27b-mlx","id":"localbench-parked/qwen3.8:27b-mlx","matches":true,"score":-2998.6800000000003}
{"query":"qwen3.8:27b-mlx","id":"localbench-parked:5642e97495e1","matches":false,"score":0}
{"query":"qwen3.8:27b-mlx","id":"qwen3.6:35b-mlx","matches":false,"score":0}
{"query":"qwen3.8:27b-mlx","id":"qwen3.8-uncensored:latest","matches":false,"score":0}
```

## 2. With the smol model parked, the e2e `omp -p` children loaded `qwen3.8-uncensored:latest`

Run `runs/20260923T033709Z__run__ollama__qwen3.6_35b-mlx` (tiers micro,replay,e2e; children without `--smol`),
progress.jsonl:

```
{"t": 1790134729.299, "event": "contention", "foreign": {"ollama": ["qwen3.8-uncensored:latest"]}, "resident": {"ollama": ["qwen3.6:35b-mlx", "qwen3.8-uncensored:latest"], "mlx-serve": []}, "gpu_device_pct": 97}
{"t": 1790134751.064, "event": "contention", "foreign": {"ollama": ["qwen3.8-uncensored:latest"]}, "resident": {"ollama": ["qwen3.6:35b-mlx", "qwen3.8-uncensored:latest"], "mlx-serve": []}, "gpu_device_pct": 80}
```

verdicts: `{"contended":true,"must_fail":["e2e.ok.correct"],"preflight_problems":[],"allow_busy":false}`

/tmp/ollama-serve.log in the same window:

```
time=2026-09-22T21:38:48.689-06:00 level=INFO source=images.go:381 msg="template selection" model=registry.ollama.ai/library/qwen3.8-uncensored:latest selected=gguf_chat_template
[GIN] 2026/09/22 - 21:38:50 | 200 |  3.992390958s |       127.0.0.1 | POST     "/v1/responses"
time=2026-09-22T21:39:11.067-06:00 level=INFO source=images.go:381 msg="template selection" model=registry.ollama.ai/library/qwen3.8-uncensored:latest selected=gguf_chat_template
[GIN] 2026/09/22 - 21:39:11 | 500 |   3.99255325s |       127.0.0.1 | POST     "/v1/responses"
```

## 3. Same tier with `--smol localbench/<model>` on every child

Run `runs/20260923T034053Z__run__ollama__qwen3.6_35b-mlx` (tiers e2e): verdicts
`{"contended":false,"must_fail":[],"preflight_problems":[],"allow_busy":false}`. Auxiliary calls now reach the model
under test through the proxy (omp_calls.jsonl rows with `tools: 0`):

```
{"t":1790134948.948344,"tools":0,"prompt_tokens":278,"total_s":3.722}
{"t":1790134967.132438,"tools":0,"prompt_tokens":266,"total_s":1.673}
{"t":1790134981.669643,"tools":0,"prompt_tokens":266,"total_s":1.665}
```

Not shown here: which omp code path issues the tools=0 call (inferred mnemopi memory via `mnemopi.llmMode: smol`;
not traced in omp source).
