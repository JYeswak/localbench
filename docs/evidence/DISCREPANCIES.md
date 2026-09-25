# Known Conformance Divergences

<!--
  Anti-ceremony (CHECKLIST.md A12):
  - Consumer: golden.compare() and golden.from_aa() (a MUST/SHOULD FAIL is XFAIL only if listed here); the reviewer.
  - Gate: lb-02 — a known divergence is XFAIL, never SKIP; an unlisted FAIL fails the run and refuses to bank.
  - Defect class: intentional divergences that look like bugs to the next agent; failures quietly frozen into goldens.
  - Delete when: never while conformance cases exist; entries are tombstoned (RESOLVED), not deleted.
-->

One entry per divergence, scoped to one backend/model. The machine reads the `## DISC-NNN: <case id> on
<backend>/<model>` header and its `- Status: XFAIL <case id>` line (golden.listed_discrepancies); an XFAIL never
excuses the same case on another backend or model. Everything else is for humans.

Entry shape:

    ## DISC-NNN: <case id> on <backend>/<model>
    - Status: XFAIL <case id> | INVESTIGATING | RESOLVED <date>
    - Observed: <what fails, with the run dir or receipt>
    - Why accepted: <reason>, or why it is being investigated
    - Review date: YYYY-MM-DD

## DISC-001: conf.greedy_deterministic on ollama/qwen3.6:35b-mlx
- Status: XFAIL conf.greedy_deterministic
- Observed: both legs of docs/evidence/receipts/aa__ollama__qwen3.6_35b-mlx__20260923T035230Z.json. Two identical
  temperature-0 requests ("List three prime numbers greater than 100…", max_tokens 400) start with the same reasoning
  and diverge mid-reasoning inside 400 tokens (both finish_reason=length, all 400 tokens reasoning). The same case
  PASSes on mlx-serve/Qwen3.6-35B-A3B-MLX-Serve-4bit (smoke run, 13 tokens, no reasoning by default) and on the dense
  ollama/localbench-parked:5642e97495e1 in both legs of receipts/aa__ollama__localbench-parked_5642e97495e1__20260923T042117Z.json,
  so the nondeterminism is specific to this model on this ollama.
- Why accepted: SHOULD-level; omp does not rely on bit-identical replies. Cause not isolated — ollama's MLX runner
  logs speculative-decode stats on this model, and batched/speculative kernels are not guaranteed bit-exact
  [inference, not verified]. Accepting it keeps every other row of the golden gate meaningful.
- Review date: 2026-10-22, or on any ollama/model pin change.

## DISC-002: replay.lean.prompt_tokens on mlx-serve/Qwen3.6-35B-A3B-MLX-Serve-4bit
- Status: RESOLVED 2026-09-23
- Observed: both legs of docs/evidence/receipts/aa__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit__20260923T040100Z.json:
  replayed 10,968 vs recorded 11,645 (drift 0.0581 > 0.02). The full fixture on the same runs: 72,976 vs 74,284
  (0.0176, PASS). omp's own e2e turns report the same gap (input 11,104–11,127 on mlx-serve vs 11,679 on ollama).
  Same shape under omp 18.2.11 + child overlay, both legs of receipts/aa__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit__20260923T072527Z.json:
  lean 10,754 vs 11,433 (0.0594, FAIL), full 72,471 vs 73,779 (0.0177, PASS).
- Why accepted: the sidecar count comes from ollama's renderer (the fixtures were recorded against
  ollama/qwen3.6:35b-mlx); mlx-serve's chat template renders the same request body, mostly the 7 tool schemas, into
  fewer tokens. Not truncation: loaded context 262,144 and the output answers the prompt. A per-backend recorded
  count would remove this entry (retry: record the lean fixture's sidecar count on mlx-serve too).
- Resolved: the case is VOID when the run backend is not the one that recorded the sidecar (14d1452). A VOID run against this golden's FAIL is not a regression. A later FAIL is no longer excused. A per-backend sidecar count is still unrecorded.
- Review date: 2026-10-22, or on any mlx-serve/model/omp pin change.
