# FrankenSuite Starter Checklist

Two phases. Phase A asks: **is the plan execution-ready?** Phase B asks:
**is execution staying honest?** Every item carries its origin — the suite
repository and file it was observed in. Nothing here is invented; items with
thin evidence are marked PROVISIONAL instead of being hidden.

Classes:
- **DAY-1 MECHANICAL** — enforced by a script in this kit, at the enforcement point each item names (per-commit, phase gate, or init).
- **DAY-1 PROCEDURAL** — a convention the team adopts on day one; enforced by review.
- **MATURITY** — observed in the suite but expensive; adopt when the project earns it.
- **PROVISIONAL** — thin suite evidence (few projects did it); adopt with eyes open.

`scripts/init.sh` seeds every item as a bead in `.beads/issues.jsonl`, with the
Done criteria as the bead's acceptance criteria.

## Phase A — Planning: is the plan execution-ready?

### A1 — Non-goals written ("what this is not")
- **What:** docs/planning/packet.md contains an explicit non-goals section listing what the project will not do.
- **Why:** Every non-goal written now is a scope argument the team will not have to have later; undefined scope is where overclaims breed.
- **Done criteria:** CHECK: NON-GOALS section is non-empty with named exclusions.
- **Verified by:** scripts/check-readiness.sh (machine).
- **Origin:** frankengit "What FrankenGit is not" (exclusion list); franken_markdown docs/planning/COMPREHENSIVE_PLAN.md §2 non-goals.
- **Class:** DAY-1 PROCEDURAL

### A2 — Incumbent oracle pinned before implementation
- **What:** The incumbent (or spec/census, if there is no incumbent) is pinned at version + commit SHA before any implementation work, recorded in the packet's SOTA section.
- **Why:** A self-speedup without the incumbent live in the same invocation is maintenance, not a win; an unpinned oracle drifts and silently invalidates comparisons.
- **Done criteria:** SOTA section names the pinned revision + SHA; oracle binary recorded in docs/evidence/. Skipped only with a recorded reason if no incumbent exists.
- **Verified by:** scripts/check-readiness.sh CHECK: SOTA (presence); mechanical contract test at maturity.
- **Origin:** franken_whisper docs/INCUMBENT_CONTRACT.json ("A competitive ratio is only bankable against a PINNED incumbent"); franken_lean AGENTS.md Rule D8; Gate 7 observed in 21/44.
- **Class:** DAY-1 PROCEDURAL

### A3 — Work packetized with a mandatory schema; missing field means NOT READY
- **What:** Work is decomposed into packets, each carrying the full mandatory field schema (goal, legacy anchors, target files, oracle tests, fixture manifest happy/edge/adversarial, risk note, acceptance gate).
- **Why:** A plan with optional fields is a plan with holes the agents will find for you; machine-checked completeness is the cheapest gate in the kit.
- **Done criteria:** CHECK: PACKETS section lists packet IDs with their required fields; scripts/check-readiness.sh exits 0.
- **Verified by:** scripts/check-readiness.sh (machine: section presence, substance signals, and section vocabulary — see "Machine-check limits" below); semantic truth by review (A13).
- **Machine-check limits:** the checker verifies structure, not truth. It requires every section to be present, non-trivially filled, and using the vocabulary its contract demands — but a determined agent can still write plausible-sounding empty prose that passes. The independent review (A13) is the semantic backstop. Faking a pass now costs a structured, section-fluent draft instead of one junk line; at that point most of the real planning work is done anyway.
- **Origin:** franken_numpy docs/planning/PHASE2C_EXTRACTION_PACKET.md (14 mandatory fields; "Missing any field => NOT READY"); frankentorch docs/SCHEMA_LOCK_V1.md; Gate 2 (6/44, 5 mechanical).
- **Class:** DAY-1 MECHANICAL

### A4 — Claim inventory registered before claims are made
- **What:** Every public claim the project will make is registered in registries/claims.tsv with status planned and a proof slot, before the README makes it.
- **Why:** Claims registered after the fact are rationalizations; the registry is the difference between a claim and a boast.
- **Done criteria:** registries/claims.tsv exists with one row per planned claim; CHECK: CLAIM-INVENTORY non-empty.
- **Verified by:** scripts/check-claim-discipline.sh (machine, runs in pre-commit); scripts/check-readiness.sh.
- **Origin:** franken_markdown scripts/claims.tsv; frankensympy registries/claims.toml (status_semantics: planned/documented/implemented_uncertified/validated/certified/blocked/retired); Gate 9 (22/44).
- **Class:** DAY-1 PROCEDURAL

### A5 — Negative-evidence ledger initialized with its schema
- **What:** docs/evidence/NEGATIVE_EVIDENCE.md exists with the row schema (hypothesis, A/B, A/A null, verdict, retry predicate, lesson) before execution starts.
- **Why:** Falsified hypotheses recorded nowhere are relitigated forever; the ledger is the project's memory of what did not work.
- **Done criteria:** ledger file exists with schema header (created by init.sh); rows appended as work proceeds.
- **Verified by:** init.sh seeds the ledger (machine); the pre-commit hook validates every added/modified row's full content and rejects missing or weasel ("later", "TBD", "n/a") retry predicates (machine, per commit).
- **Origin:** frankenscipy docs/NEGATIVE_EVIDENCE.md; frankensearch docs/evidence/e8h-hypothesis-ledger.md; Gate 10 (31/44).
- **Class:** DAY-1 MECHANICAL

### A6 — Demotion rules written down
- **What:** docs/evidence/demotion-rules.md exists, stating what moves claims down, who may demote, and what cannot be overridden.
- **Why:** Promotion needs a gate; demotion needs only counter-evidence — but only if the rules say so in advance, or demotion becomes a political act.
- **Done criteria:** demotion rules file exists covering the claim-effect lattice, P0 blocking, gate changes, and tombstoning.
- **Verified by:** review (procedural); CHECK: HONESTY-MACHINERY references the rules.
- **Origin:** frankenfs docs/planning/MODULARITY_RUNBOOK.md + docs/runbooks/readiness-action-autopilot.md (multi-layer auto-demotion); Gate 19 (30/44, only 5 mechanical — hence procedural-first).
- **Class:** DAY-1 PROCEDURAL

### A7 — Phase exit criteria defined before agents start
- **What:** The packet names each phase, its entry conditions, and its exit proof; no phase gate may claim a result whose transitive dependency closure contains an unresolved [OPEN].
- **Why:** Phases without exit criteria end when enthusiasm does; exit criteria are what make "done" a fact instead of a feeling.
- **Done criteria:** CHECK: EXIT-CRITERIA non-empty; every phase lists entry conditions and exit proof.
- **Verified by:** scripts/check-readiness.sh (presence); review (substance).
- **Origin:** franken_nlp COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md §11; frankentorch docs/planning/PLAN_TO_PORT_PYTORCH_TO_RUST.md §6; Gate 3 (25/44).
- **Class:** DAY-1 PROCEDURAL

### A8 — Definition of Done written with command evidence
- **What:** A written Definition of Done requiring command evidence for done and artifact evidence for blocked.
- **Why:** "Done" without a command that proves it is a status update, not a state; "blocked" without the error text is a mood.
- **Done criteria:** DoD document exists stating: mark done only with command evidence and updated documentation; mark blocked only with concrete artifact evidence (error text + command + path). Installed by `init.sh` as `docs/definition-of-done.md` (from `templates/definition-of-done.md`) — tailor it, keep the evidence rules.
- **Verified by:** review; bead close_reasons cite commands.
- **Origin:** franken_whisper docs/definition_of_done.md; asupersync ATP_DOD_CHECKLIST.md (enforced by scripts/validate_dod.sh).
- **Class:** DAY-1 PROCEDURAL

### A9 — Anti-reward-hacking law adopted
- **What:** The 12 forbidden reward-hacking patterns are named in the project's agent instructions, with the three load-bearing rules (never weaken a gate to land a change; no self-grading without independent verification; demotions always allowed).
- **Why:** 29 of 44 suite projects needed this doctrine written down because agents under pressure will otherwise optimize the metric instead of the work.
- **Done criteria:** the 12 patterns (appendix below) appear verbatim in the project's agent instructions; the three rules are stated as law. Installed by `init.sh` as `AGENTS.md` (from `templates/agents.md`) — tailor the project sections, keep the patterns verbatim.
- **Verified by:** review (grep for the patterns in AGENTS.md or equivalent).
- **Origin:** suite-wide AGENTS.md Rule 0.5, via franken_markdown AGENTS.md and frankenfs AGENTS.md; Gate 18 (29/44). The suite-wide file lives outside the repos, so the list is vendored verbatim into this kit's appendix.
- **Class:** DAY-1 PROCEDURAL

### A10 — Acceptance shape: positive observable + planted negative + no-claim line [PROVISIONAL]
- **What:** Every work item states its acceptance as: the positive observable, a planted negative (what must fail), and the no-claim line (what this does not prove).
- **Why:** A test suite with no planted negative cannot distinguish a passing gate from a gate that cannot fail; the no-claim line bounds the blast radius of the claim.
- **Done criteria:** bead acceptance_criteria contain all three lines.
- **Verified by:** review.
- **Origin:** frankenterm docs/proof-taxonomy.json; docs/reality-check-bridge-plan-2026-09-01.md acceptance shape; Gates 13 (4/44) and 14 (5/44).
- **Class:** PROVISIONAL

### A11 — Proof taxonomy declared [PROVISIONAL]
- **What:** The packet declares the admissible proof categories for the project and an explicit non-proof list (things that look like evidence but are not).
- **Why:** Without a taxonomy, every dispute about "is this proof?" is argued from scratch; with one, it is a lookup.
- **Done criteria:** CHECK: PROOF-TAXONOMY lists categories and non-proof classifications.
- **Verified by:** scripts/check-readiness.sh (presence).
- **Origin:** frankenterm docs/proof-taxonomy.json (11 proof categories, 4 non-proof classifications); Gate 16 (1/44 — the only suite project).
- **Class:** PROVISIONAL

### A12 — Anti-ceremony: every artifact names its consumer, gate, defect class, deletion condition [PROVISIONAL]
- **What:** Each process artifact created states: who consumes it, which gate it enforces, which observed defect class justifies it, and when it gets deleted.
- **Why:** Process artifacts accumulate until the process is the product; an artifact without a deletion condition is permanent by default.
- **Done criteria:** each template/registry doc carries the four lines.
- **Verified by:** review.
- **Origin:** franken_tts Doctrine #0 ("A process artifact may exist only as a hard gate for a named capability"); Gate 21 (4/44).
- **Class:** PROVISIONAL

### A13 — Plan independently reviewed before execution
- **What:** A second agent (fresh eyes, different model, or adversarial duel) reviews the packet and records what changed.
- **Why:** The author of a plan cannot see its holes; independent review is the cheapest way to find the ones that matter.
- **Done criteria:** CHECK: REVIEW names the reviewer, the method, and what changed. Solo projects with no second reviewer record the sanctioned attestation instead: "Independent review: not performed (solo) — nothing was independently changed. Residual risk accepted by <name>, <date>." The machine verifies the attestation exists; it can never verify the review's quality — READY is not the same as reviewed.
- **Verified by:** scripts/check-readiness.sh (presence); the review artifact itself.
- **Origin:** frankensearch docs/divergence register (fresh-eyes second-agent review); frankenterm docs/DUELING_WIZARDS_REPORT.md; Gate 20 (16/44).
- **Class:** DAY-1 PROCEDURAL

### A14 — Release gate defined: what blocks publication
- **What:** The packet states what blocks publication, which clauses cannot be waived, and which can be waived only with a public, expiring, recorded waiver (owner, rationale, expiry, compensating controls).
- **Why:** A release gate defined at release time is negotiated under pressure; defined now, it is a contract.
- **Done criteria:** CHECK: RELEASE-GATE non-empty; non-waivable clauses named.
- **Verified by:** scripts/check-readiness.sh (presence); review (substance).
- **Origin:** frankenterm docs/release/attestation-checklist.md ("incomplete producer evidence MUST block publication"); frankentui docs/pane-release-gate-policy.md; asupersync docs/wasm_ga_go_no_go_evidence_packet.md; Gate 4 (26/44).
- **Class:** DAY-1 PROCEDURAL

## Phase B — Beads/execution: is execution staying honest?

### B1 — The beads graph is the executable form; prose is the rationale of record
- **What:** The beads graph is the plan of record; documents explain why, beads say what is being done and whether it closed.
- **Why:** Prose plans drift from reality silently; a graph with open/closed states cannot drift without showing it.
- **Done criteria:** .beads/issues.jsonl is non-empty and current; no parallel prose tracker exists.
- **Verified by:** inspection; init.sh seeds the graph from this checklist.
- **Origin:** frankensearch docs/plans/quill-distillation ("The beads graph ... is the executable form; this document is the rationale of record"); Gate 1 (38/44).
- **Class:** DAY-1 PROCEDURAL

### B2 — Close only on cited evidence; close_reason required
- **What:** A bead closes only with a close_reason citing the evidence (commit, receipt, ledger row); prose assertions never close beads.
- **Why:** "It works" closes nothing; a pointer to the artifact that proves it closes the bead.
- **Done criteria:** every closed bead has a non-empty close_reason naming evidence.
- **Verified by:** review of closed beads.
- **Origin:** asupersync docs/atp_rq_beat_rsync_ledger.md ("closure on cited evidence"); frankentui .beads/policy.yaml (close requires evidence token); franken_whisper done/blocked decision rules.
- **Class:** DAY-1 PROCEDURAL

### B3 — Receipts carry provenance; generation binding
- **What:** Every evidence receipt records commit, tool versions, host, and worker; a receipt from a different generation is not evidence for this release.
- **Why:** Stale receipts are the quietest form of overclaim — the measurement was real, once, somewhere else.
- **Done criteria:** receipt format includes commit/version/host/worker fields; no receipt is cited across generations.
- **Verified by:** review; receipt schema in CHECK: EVIDENCE-DESIGN.
- **Origin:** frankenterm docs/reality-check-bridge-plan-2026-09-01.md §4b (generation binding); Gate 8 (31/44).
- **Class:** DAY-1 PROCEDURAL

### B4 — Every falsified hypothesis banked with a retry predicate
- **What:** Falsified hypotheses go into the negative-evidence ledger with hypothesis, method, observable result, why rejected, retry predicate, and provenance.
- **Why:** An unbaked rejection gets relitigated; a banked one with a retry predicate becomes a scheduled experiment instead of a rumor.
- **Done criteria:** ledger rows exist for falsified hypotheses; every REJECT row has a retry predicate.
- **Verified by:** pre-commit hook (machine): each added/modified row is read in full from the staged file; rows with missing or weasel ("later", "TBD", "n/a") retry predicates block the commit. Testability of the predicate itself is judged by review.
- **Origin:** frankensearch docs/evidence/e8h-hypothesis-ledger.md ("Rejects require a retry-condition predicate — never 'later'"); Gate 10 (31/44).
- **Class:** DAY-1 PROCEDURAL

### B5 — Pre-commit honesty gate runs and self-tests with a canary
- **What:** The pre-commit hook runs the claim-discipline check and first proves its own teeth by failing a deliberately overclaimed canary fixture.
- **Why:** A gate that cannot fail is decoration; the canary proves the gate can still bite, on every single commit.
- **Done criteria:** `.git/hooks/pre-commit` installed and live (the copy git actually executes); `.githooks/pre-commit` tracked in the repo as the source of truth; canary self-test passes (i.e. the canary is rejected). Edit `.githooks/pre-commit`, then re-run `scripts/init.sh` (idempotent) to reinstall the live copy — editing `.git/hooks/pre-commit` directly is overwritten on the next init.
- **Verified by:** the hook itself, every commit (machine); install verified by init.sh output.
- **Escape hatch (named, not hidden):** `git commit --no-verify` bypasses this hook silently, by git's design — no hook can prevent that. The backstop is CI: `.github/workflows/kit-gates.yml` (installed by init.sh) re-runs every gate where `--no-verify` cannot reach. A locally bypassed gate is a process violation, not a silent pass; treat it as one. Local hooks are advisory without CI.
- **Origin:** franken_whisper .githooks/pre-commit + examples/ledger_preflight.rs (exit 2 blocks); franken_markdown check-claim-discipline.sh self-test; frankensympy tools/validate_planning.py --self-test; Gates 6 (10/44) and 17 (7/44).
- **Class:** DAY-1 MECHANICAL

### B6 — Claim discipline enforced: README claims cross-checked against proofs
- **What:** scripts/check-claim-discipline.sh runs (via the hook): every enforced README claim must resolve to an existing proof artifact containing the expected evidence.
- **Why:** README drift is a governance failure, not documentation debt — the README is the project's public claims surface.
- **Done criteria:** the script exits 0; every enforce=yes row passes; at least one row is enforced (enforce=yes) once README.md exists and is non-empty — a public README with zero enforced claims fails the check, because an unenforced registry is undecorated discipline.
- **Verified by:** pre-commit hook (machine).
- **Origin:** franken_markdown scripts/check-claim-discipline.sh + scripts/claims.tsv; Gate 9 (22/44).
- **Class:** DAY-1 MECHANICAL

### B7 — Never weaken a gate to land a change; gate changes need two-direction evidence
- **What:** Changing a gate's thresholds, counters, or exception semantics requires evidence for newly admitted valid cases AND cases that remain rejected.
- **Why:** The most common way a gate dies is a well-meaning edit under deadline pressure; this rule makes that edit cost evidence.
- **Done criteria:** no gate change lands without the two-direction evidence attached; rule D3 in demotion rules.
- **Verified by:** review of gate-changing commits.
- **Origin:** frankenfs docs/planning/MODULARITY_RUNBOOK.md; frankenredis docs/GATE_VALIDITY.md; Gate 18 doctrine.
- **Class:** DAY-1 PROCEDURAL

### B8 — Demotion rules executed on schedule
- **What:** The demotion rules (D1–D7) are run: expired manual proofs demote, open P0s block release claims, retired ids are tombstoned.
- **Why:** Demotion rules nobody runs are wishes; scheduled execution is what makes them machinery.
- **Done criteria:** demotion pass performed at the cadence the rules state; demotions recorded in the ledger.
- **Verified by:** review; ledger entries.
- **Origin:** frankenfs multi-layer demotion; frankentui docs/claims-ledger.md (90-day manual expiry); Gate 19.
- **Class:** DAY-1 PROCEDURAL

### B9 — Waivers are public, time-bounded, and recorded
- **What:** Any waiver of a release-blocking gate states rationale, owner, expiry, and compensating controls; waivers bypassing a release-blocking gate force NO_GO unless all four are recorded and signed off.
- **Why:** Secret waivers are gate deletions with better PR; public expiring waivers keep the gate honest while allowing judgment.
- **Done criteria:** every waiver recorded with the four fields; no waiver outlives its expiry.
- **Verified by:** review.
- **Origin:** asupersync docs/wasm_ga_go_no_go_evidence_packet.md (waiver policy); franken_lean docs/AGENT_FRONTIER_PROTOCOL.md (expiring waivers); frankentui docs/pane-release-gate-policy.md; Gate 23 (8/44).
- **Class:** DAY-1 PROCEDURAL

### B10 — Session completion: land the plane
- **What:** Every work session ends with the landing ritual: file beads for remaining work, run the quality gates, update bead states, sync the tracker, hand off context.
- **Why:** Sessions that end mid-thought leave the next agent to reconstruct intent from debris; the ritual makes handoff a habit, not a favor.
- **Done criteria:** the five landing steps are performed at session end.
- **Verified by:** review; handoff notes reference closed/filed beads.
- **Origin:** asupersync AGENTS.md ("Landing the Plane"); frankenterm AGENTS.md; Gate 22 (4/44).
- **Class:** DAY-1 PROCEDURAL

### B11 — Ledger resurrection cadence
- **What:** Rejected ledger rows are re-audited on a schedule; rows whose rejection is void (measurement could not detect the effect) are resurrected, not relitigated from zero.
- **Why:** The frankenscipy fleet audit found 56.2% of audited REJECT rows were void — dead levers were not dead, the measurements were. Without resurrection, the ledger becomes a graveyard of false negatives.
- **Done criteria:** resurrection audit performed at the stated cadence; void rows marked VOID-<class>; resurrected rows link the old row.
- **Verified by:** review; ledger history.
- **Origin:** frankenscipy docs/LEDGER_RESURRECTION.md; frankenfs docs/LEDGER_RESURRECTION.md; Gate 11 (11/44).
- **Class:** MATURITY

### B12 — Claim-coverage audit
- **What:** A periodic sweep measures what fraction of claims in the wild (README, docs, comments) are registered in claims.tsv with proof.
- **Why:** Claims breed in prose faster than registries grow; the audit measures the gap. The suite's best audits found 2.0% and 6.8% coverage — assume yours is low until measured.
- **Done criteria:** audit run at the stated cadence; coverage number recorded; unregistered claims registered or removed.
- **Verified by:** the audit artifact.
- **Origin:** franken_networkx docs/CLAIM_COVERAGE_AUDIT.md (2.0% of 591 claims); frankensearch (6.8%); Gate 12 (8/44).
- **Class:** MATURITY

### B13 — False-closure repair sweep
- **What:** A periodic audit re-opens thin closes — beads closed on prose, stale receipts, or missing evidence — as debt beads until re-proven.
- **Why:** Agents close beads; pressure closes beads early. The suite's largest repair found 739 debt beads hiding inside a "complete" graph.
- **Done criteria:** close-audit run at the stated cadence; thin closes re-opened with reasons.
- **Verified by:** the audit artifact.
- **Origin:** franken_node (739 debt beads from the bead-completion illusion); frankentui close-audit.
- **Class:** MATURITY

### B14 — Gate self-test: deliberately break a gate
- **What:** New gates are deliberately broken in a scratch copy to prove they fail loudly; the breakage and the failure are recorded.
- **Why:** An untested gate is a hypothesis about your tooling; the mutation proves it is a fact.
- **Done criteria:** each new gate has a recorded break-test showing it fails loudly.
- **Verified by:** the break-test artifact.
- **Origin:** frankenredis docs/GATE_VALIDITY.md ("new gates must be deliberately broken in a scratch copy"); frankensympy tools/gate_review_battery.py (18 mutation cases, all must be KILLED); Gate 17 (7/44).
- **Class:** MATURITY

## Class legend

- **DAY-1 MECHANICAL** — enforced by a script in this kit, at a stated enforcement point:
  - B5, B6 run on every commit via the pre-commit hook;
  - A3 runs at the Phase A→B gate via `scripts/check-readiness.sh` — blocking in CI (`.github/workflows/kit-gates.yml`), advisory warning in the hook when the packet is staged;
  - A5 is seeded by `init.sh`, and every staged ledger row is linted per commit.
- **DAY-1 PROCEDURAL** — adopt on day one; enforced by review and habit.
- **MATURITY** — observed in the suite but expensive; adopt when the project earns it (B11–B14).
- **PROVISIONAL** — thin suite evidence; adopt with eyes open (A10, A11, A12).

## Appendix: the 12 forbidden reward-hacking patterns

Quoted verbatim from the suite-wide agent law (AGENTS.md Rule 0.5), as
observed in franken_markdown and frankenfs. Name all twelve in the project's
agent instructions:

1. gate self-weakening ("temporarily" loosening a check to land a change)
2. proof-class inflation (relabeling weak evidence as strong)
3. golden regeneration reflex (regenerating goldens until they pass)
4. commit-stream pumping (many trivial commits to look productive)
5. tautological tests (tests that assert what the code does, not what it should do)
6. easy-lever cherry-picking (only attempting levers known to pass)
7. close-pump abuse (closing beads to move a metric)
8. scope-splitting (splitting work to dodge gate thresholds)
9. spec-editing as progress (editing the spec instead of the code)
10. conformance metastasis (growing the conformance suite instead of fixing failures)
11. dependency smuggling (undeclared dependencies)
12. bench-path hardcoding (special-casing the benchmark input)

Three load-bearing rules sit above the list:
- Never weaken a gate to land a change (see B7).
- No self-grading without independent verification.
- Demotions are always allowed (see D4).

Variant note: the execution-readiness catalog (Gate 18) quotes frankengit's
`AGENTS.md` §5/§16 with 4 of the 12 worded differently — easy-*bead*
cherry-picking, *follow-up laundering*, *demo-path* hardcoding,
*spec-editing* — versus this list's easy-*lever* cherry-picking,
*conformance metastasis*, *bench-path* hardcoding, *spec-editing as
progress*. "Verbatim" above means the franken_markdown/frankenfs lineage;
treat the twelve as the doctrine's core, not a single canonical text. The
suite-wide file itself was never read directly (it lives outside the repos).
