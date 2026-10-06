# Claim coverage audit — 2026-09-29

**Verdict: uncovered, not certified.** This is a complete classification of the **bounded stratum below**, not a repo-wide claim-coverage percentage. In this stratum, **0/129 (0%)** material user-observable assertions have a matching registry row backed by a banked same-generation proof. Six are registered but unproven; 123 are unregistered. The claim-discipline gate correctly exits 1 with zero `enforce=yes` rows. Do not infer that the product lacks these behaviors; this measures claim evidence, not implementation.

The 129/6/123 figures below describe the dated bounded census, **not a post-remediation percentage**.
README and registry wording changed afterward; no updated atom-by-atom census has been performed.

## Scope and counting rule

Read README.md, docs/MONITOR.md, docs/evidence/incumbents.md, docs/planning/omp-residency-policy.md and Python/TypeScript comments in `localbench/` and `scripts/`. Count one bounded, independently falsifiable assertion per user-visible performance, reliability, correctness, safety, or live-configuration promise, including historical observations identified as such. Exclude CLI syntax, version/hash/date inventory, license, implementation-colocated explanations, instructions, fenced examples, anti-ceremony notes, and aspirational acceptance tests. A repeated assertion in another document counts there because that surface can drift independently. Code-comment promises already covered by the counted README/docs assertions are excluded from the separate comment stratum. The earlier 310 lexical candidates and 440 atomized factual details are **not** claim denominators.

| Surface | Current | Historical | Material claims | Registered without valid proof | Unregistered | Proven |
|---|---:|---:|---:|---:|---:|---:|
| README.md | 42 | 5 | 47 | 6 | 41 | 0 |
| docs/MONITOR.md | 9 | 3 | 12 | 0 | 12 | 0 |
| docs/evidence/incumbents.md | 16 | 3 | 19 | 0 | 19 | 0 |
| docs/planning/omp-residency-policy.md | 35 | 3 | 38 | 0 | 38 | 0 |
| Code comments, `localbench/` and `scripts/` | 13 | 0 | 13 | 0 | 13 | 0 |
| **Bounded total** | **115** | **14** | **129** | **6** | **123** | **0** |

The six registered-but-unproven README assertions are at README.md:66,118,144,148,151,163 (registry labels `lean_prompt_tokens`, `behavioral_eval_lane`, `gateway_fail_closed`, `gateway_unload_safety`, `gateway_body_bounds`, `golden_tier_generation`). All have `enforce=no`; the historical fixture sidecar at :66 is not a banked receipt. Existing banked performance receipts are not automatically proof of these exact assertions, nor of a clean source revision when `localbench_rev` is dirty.

## Reproduction anchors and visible gaps

- README.md:3-6,35-48,73-82,122-135,139-167 carry material assertions beyond the six registered sentences. Notable unproven live promises: mutation plan/refusal/no-op semantics (:122-135), keep and external-client safety (:139-153), and measured load/child isolation/fail-closed behavior (:157-167). The command-list caption at :188 was corrected from “every README claim” to the gate's actual enforced-claim contract.
- docs/MONITOR.md:24-47,57-60,69-78,93-94 include the pane check's exit/disqualification, point-in-time limitations, and monitor findings. Its 12/13 smol-route census at :34-38 is explicitly dated; it does not prove already-running OMP session routes.
- docs/evidence/incumbents.md:18-23,29-34,46-51 includes current configuration and generation rules. The old instruction to re-bank live tiers on any omp version change conflicted with `localbench/golden.py:tier_keys`; :46-51 now says omp version/SHA alone do not stale a golden.
- docs/planning/omp-residency-policy.md:3,7,13-22,33-49,64-67 includes safety and routing promises. :7 now identifies `keep forever` as **former** behavior; :3 and :37 distinguish verified gateway/profile configuration readback from **unverified live existing-session/built-in Ollama provider routing**. Its Scope (:18-22) and Lifecycle (:41-49) are counted as present-tense guarantees. If those 13 assertions are treated strictly as plan-only, the bounded denominator is **116**, unregistered **110**, proven **0**. This sensitivity does not turn an absent proof into a pass.
- Remaining distinct comment-only promises: `localbench/__main__.py:425-426,485-486,1717,1902-1903`; `localbench/gateway.py:644`; `localbench/proxy.py:134-135`; `localbench/workloads.py:225-226,615-616,814-815,888-889,1095-1096,1151`; `scripts/export_public.py:27-30`. Unjustified comments about universal probe cwds, cache topology inferred from timing, measured tolerance resolution, and historical fixture metadata as current proof were narrowed during this audit. Code comments are not banked receipts.

## Residual technical-doc survey (separate denominator)

Pane `%pane` audited additional assertion types not counted in the 129 material public/comment units above. Current
technical documentation: **0/13** matching registry rows and proof — docs/port/INTERFACES.md:37-40 (one
Python-dependency statement), :44-78 (nine independently falsifiable porting constraints), :120-124 (one test-map
contract), and active docs/evidence/DISCREPANCIES.md DISC-001/DISC-003 at :23-34,:50-62 (two status statements).
Dated historical observations: **0/11** at INTERFACES.md:84-100 (ten statements) and resolved DISC-002 at
DISCREPANCIES.md:36-48 (one). These are **not additive to the public-claim coverage rate**: technical porting
constraints and dated discrepancy records are not necessarily product promises. The README-specific gate requires
an exact prose pattern to enforce a row; the registry also permits blank-pattern, unenforced inventory rows
(claims.tsv:14-16). Technical-doc-only statements cannot gain **enforced README coverage** from those rows alone,
and none of the 24 has a matching proof. No unsupported registry rows were added.

The remaining docs inventory excluded version/file inventories in docs/REFERENCES.md and INTERFACES.md:22-35,
prospective plans and INTERFACES.md:102-116, process requirements in CHECKLIST.md / definition-of-done.md /
demotion-rules.md, and evidence artifacts (receipts, ledger, break-tests and cross-grades). The survey does not
certify every document in the repository and cannot support a repo-wide percentage.

## Remediation gate

`kit-b12` remains open. Do not register 123 assertions with empty or misleading proof merely to raise a coverage metric, delete truthful implementation comments for optics, or count source text/fixture metadata as banked same-generation evidence. For each material **public** promise, either supply a matching current-generation independent receipt and registry entry, or demote/remove the unsupported sentence; prioritize gateway live-routing and unload safety before speed copy. For comment-only contracts, either move the genuine user-facing promise to a verified public surface or keep the comment explicitly scoped as implementation intent. Re-run the same bounded census after those changes and separately inventory the other docs before making any repo-wide percentage claim.

## 2026-09-30 bounded remediation checkpoint

README claim classes were reviewed against the exact-match `readme_pattern`, `enforce`, and proof-path rules
in `scripts/check-claim-discipline.sh:37-103`. The current registry has eight README-matching inventory
rows, all `enforce=no`: historical request shape (`lean_prompt_tokens`), campaign restriction
(`behavioral_eval_lane`), gateway fail-closed intent / external socket / body bounds
(`gateway_fail_closed`, `gateway_unload_safety`, `gateway_body_bounds`), per-tier freshness
(`golden_tier_generation`), local-model scope (`local_model_scope`), and first golden
(`aa_first_golden`). The other three performance/conformance entries (`moe_prefill_ratio`,
`first_turn_under_10s`, `must_conformance`) and the two comparison/workload entries
(`mlx_serve_vs_ollama`, `swe_slice_latency`) are also `enforce=no`; their current README text
does not promote the withdrawn speed, first-turn, all-MUST, or SWE-bench claims.

Classification of the flagged README surfaces after these edits (line ranges refer to this checkpoint):

| Surface | Class and evidence disposition |
|---|---|
| :3-6, :10-15, :24-39 | Mission/scope, requirements, install and first-golden behavior. Scope and first-golden sentences have unenforced inventory rows; source paths and dated examples are not same-generation behavioral receipts. The broad speed/reliability promise at :5-6 was removed, while functional instructions remain. |
| :41-59 | Exit/error and environment contracts, not banked current-generation proofs of every branch. Exit 0 is now explicitly bounded to the requested command instead of certifying all behavior. No matching enforced rows. |
| :61-82 | Dated omp 18.2.11 prompt observation (`lean_prompt_tokens`) is explicitly historical; sidecars are not a current banked same-generation receipt. Prior A/B failures and cold MUST wrong answers are negative evidence, not permission to assert a speedup or all-MUST pass. |
| :84-119 | The command block is mainly syntax/examples (excluded by the stated counting rule); its descriptive captions and campaign restriction can still assert behavior. `behavioral_eval_lane` remains unenforced; planned varied dry-run is explicitly marked live model trial evidence UNVERIFIED. |
| :121-137 | Mutation dry-run, refusal, no-op and audit-ledger guarantees are an unregistered implementation contract, not independently banked live evidence. The section now says so; do not represent code paths or tests as live certification. |
| :139-165 | Residency, park, gateway routing, external-client and request-body safety contracts: three matching gateway rows remain unenforced. Gateway readback is not live OMP outage/restart routing proof; established external-client unload refusal lacks a banked live receipt. The socket sentence was demoted from an absolute safety result to a designed guard with an explicit direct-client race. |
| :167-181 | Contention/preflight, interleaving, per-tier generation and child isolation: `golden_tier_generation` is registered but unenforced; the rest lacks matching enforced claims. This is now labeled a harness contract rather than independent proof of every live boundary. |
| :183-207 | Layout/development inventories, commands and license are excluded as syntax/inventory by the counting rule; such text cannot serve as a receipt for claims above. |

The dated **123 unregistered** units include 41 README units plus 82 claims in the other four
counted surfaces; these are the *prior census*, not a recalculated post-edit total. Do not
blanket-delete useful technical docs or blanket-register their claims with empty proof. The
additional 24 technical-doc statements are a separate, non-additive survey (:28-44).
No `enforce=yes` row was earned: the 2026-09-29 micro A/A inspected at
`docs/evidence/receipts/aa__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit__20260929T085025Z.json:5-48`
does pin omp 18.4.3 and reports no contention/MUST failure, but its `localbench_rev` is
`9230b9d-dirty` and its tiers are only `micro`; it cannot certify live gateway unload/routing,
mission reliability, or all README assertions.

The gate still fails by construction when README is nonempty and zero rows are enforced
(`scripts/check-claim-discipline.sh:89-98`); this checkpoint does **not** claim a new gate run.
Neither README nor registry specifies a defensible "at stated cadence" monitoring or recheck
period; do not invent one. Executable scope decision: keep `kit-b12` open; bank an independent,
clean-source, same-generation live OMP outage/restart and established-external-socket unload-refusal
receipt before promoting those exact gateway claims. Then rerun the bounded census and the claim
gate with those artifacts, without changing the gate's acceptance rules.
