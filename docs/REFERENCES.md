# REFERENCES — where every mechanism came from

Each row: the kit mechanism, the originating suite repository and exact file,
what was borrowed, and what the kit changed. Repository names are the
`dicklesworthstone` GitHub set assessed in the research program (43 "franken"
repos + asupersync); file paths are relative to each repo's root at the
assessed commit.

| Kit mechanism | Origin repo + file | Borrowed | Changed |
|---|---|---|---|
| Planning packet schema; missing field ⇒ NOT READY | franken_numpy `docs/planning/PHASE2C_EXTRACTION_PACKET.md` | the 14-field mandatory packet and the NOT READY rule | trimmed to 12 sections; sections map to checklist gates, not NumPy extraction |
| Packet schema lock | frankentorch `docs/SCHEMA_LOCK_V1.md` | "missing any file ⇒ NOT_READY" as a machine verdict | same verdict, generic sections |
| `check-readiness.sh` | frankensympy `tools/validate_planning.py` + frankentorch packet validator | lint-the-plan-before-execution; fail loudly listing every gap | POSIX sh + grep/awk instead of Python; marker-based instead of schema-parsed |
| Claim registry + discipline check | franken_markdown `scripts/claims.tsv` + `scripts/check-claim-discipline.sh` | the enforce-only-when-README-claims-it rule; proof_path + expected_substr cross-check | dropped the capabilities-`--json` flag check (suite-specific); paths resolve to repo root |
| Self-proving gate (canary false claim) | franken_markdown `scripts/check-claim-discipline.sh` self-test; frankensympy `tools/validate_planning.py --self-test` | the pattern: the gate must fail a deliberately false fixture, or it has no teeth | moved into the pre-commit hook so it runs on every commit, not just in CI |
| Pre-commit honesty hook | franken_whisper `.githooks/pre-commit` + `examples/ledger_preflight.rs` | exit-2 blocking on invalid rows; hook lives in the repo | pure shell (no cargo build); added the canary self-test and ledger-row lint |
| Hook must stay cheap | frankensim `scripts/hooks/pre-commit` | the advisory wisdom: a hook that builds the workspace gets `--no-verify`'d | kept as a code comment in the hook |
| Negative-evidence ledger + resurrection | frankenscipy `docs/NEGATIVE_EVIDENCE.md` + `docs/LEDGER_RESURRECTION.md`; frankenfs `docs/LEDGER_RESURRECTION.md` | row schema; verdict taxonomy; the resurrection audit (56.2% of REJECT rows were void) | starter schema only; resurrection is a MATURITY checklist item, not day-one |
| Retry predicates on rejects | frankensearch `docs/evidence/e8h-hypothesis-ledger.md` | "Rejects require a retry-condition predicate — never 'later'" | enforced by the hook's staged-row lint |
| Demotion rules (multi-layer) | frankenfs `docs/planning/MODULARITY_RUNBOOK.md` + `docs/runbooks/readiness-action-autopilot.md` | claim-effect lattice; P0 auto-block; gate-change two-direction evidence | marked procedural-first (only 5/44 achieved mechanical auto-demotion); mechanical forms deferred to maturity |
| Demotions always allowed | frankensim `docs/MATURITY_LEVELS.md` | the rule verbatim | none |
| Tombstoning retired claims | frankensympy `registries/claims.toml` | "retired claim IDs must remain" | none |
| Proof expiry (90 days, manual) | frankentui `docs/claims-ledger.md` | manual proofs expire after 90 days | procedural in the starter set |
| SHA-mismatch auto-demotion | frankengit `registries/claims.tsv` + `docs/VERIFY_SPEC.md` | automatic, non-overridable demotion on hash mismatch | maturity-only (needs a hash-carrying registry first) |
| Beads JSONL schema | franken_numpy `.beads/issues.jsonl` (30-field schema) | field names and JSONL-first convention | trimmed to 14 fields; added `acceptance_criteria` as required |
| "Beads graph is the executable form" | frankensearch `docs/plans/quill-distillation-2026-08-27.md` | the doctrine verbatim | none |
| JSONL is truth, sqlite disposable | frankensearch; frankenscipy (only `issues.jsonl` ships) | the convention | written into `.beads/config.yaml` + `.beads/README.md` by init.sh |
| Plan→beads conversion requirements | frankensympy `docs/AGENT_NATIVE_PROTOCOL.md` §14 | bounded deliverable, objective acceptance commands, forbidden shortcuts, rollback semantics, named verifier | compressed into the bead schema's required fields |
| Acceptance shape (positive + planted negative + no-claim) | frankenterm `docs/reality-check-bridge-plan-2026-09-01.md` §4b | the three-line acceptance shape | PROVISIONAL checklist item (4–5/44 did this) |
| Proof taxonomy | frankenterm `docs/proof-taxonomy.json` | 11 proof categories + 4 non-proof classifications as the idea | PROVISIONAL; the kit asks for a taxonomy, it does not ship frankenterm's |
| Generation binding / receipt provenance | frankenterm `docs/reality-check-bridge-plan-2026-09-01.md` §4b | commit+version+host+worker on receipts; cross-generation receipts are not evidence | none |
| Attestation / release gate | frankenterm `docs/release/attestation-checklist.md`; frankentui `docs/pane-release-gate-policy.md` | "incomplete producer evidence MUST block publication"; fail loudly on missing artifacts | generic release-gate section in the packet |
| Incumbent pinning | franken_whisper `docs/INCUMBENT_CONTRACT.json` | "a competitive ratio is only bankable against a PINNED incumbent"; harness fails closed on drift | pinning is day-one procedural; fail-closed contract test is maturity |
| Definition of Done | franken_whisper `docs/definition_of_done.md`; asupersync `ATP_DOD_CHECKLIST.md` | done only with command evidence; blocked only with artifact evidence | checklist item, not a separate doc |
| Anti-reward-hacking law (12 patterns) | suite-wide `AGENTS.md` Rule 0.5, via franken_markdown `AGENTS.md` and frankenfs `AGENTS.md` | the 12 patterns quoted verbatim | vendored into CHECKLIST.md appendix — the suite-wide file lives outside the repos and was never read directly |
| Gate self-test (deliberate breakage) | frankenredis `docs/GATE_VALIDITY.md`; frankensympy `tools/gate_review_battery.py` | break new gates in a scratch copy; 18 mutation cases must be KILLED | maturity checklist item |
| Waiver policy | asupersync `docs/wasm_ga_go_no_go_evidence_packet.md`; franken_lean `docs/AGENT_FRONTIER_PROTOCOL.md` | rationale + owner + expiry + compensating controls; bypassing a release gate forces NO_GO | checklist item, procedural |
| Session completion ("Landing the Plane") | asupersync `AGENTS.md`; frankenterm `AGENTS.md` | file issues → run gates → update states → sync → hand off | checklist item, procedural |
| Phase exit criteria | franken_nlp `COMPREHENSIVE_PLAN_FOR_FRANKEN_NLP.md` §11; frankentorch `docs/planning/PLAN_TO_PORT_PYTORCH_TO_RUST.md` §6 | "no phase gate may claim a result whose dependency closure contains an unresolved [OPEN]" | packet section + checklist item |
| Non-goals ("what X is not") | frankengit "What FrankenGit is not"; franken_markdown `docs/planning/COMPREHENSIVE_PLAN.md` §2 | explicit exclusion lists as scope control | packet section |
| SOTA survey (adopt/adapt/reject) | frankensympy `docs/SOURCE_PROJECT_AUDIT.md`; frankensearch plan §4 | per-source verdicts with retry conditions for rejects | packet section |
| Independent review (fresh-eyes / duel) | frankensearch divergence register; frankenterm `docs/DUELING_WIZARDS_REPORT.md` | second-agent review before execution | packet section + checklist item |
| "Only a terminal receipt proves execution" | asupersync `docs/claim_evidence_graph_contract.md` | closure on cited evidence | bead close rules |
| Anti-ceremony doctrine | franken_tts Doctrine #0 (AGENTS.md) | artifact must name consumer, gate, defect class, deletion condition | PROVISIONAL checklist item (4/44) |
| Claim-coverage audit | franken_networkx `docs/CLAIM_COVERAGE_AUDIT.md` | audit the prose-vs-registry gap | maturity checklist item |
| False-closure repair | franken_node (739 debt beads); frankentui close-audit | re-open thin closes as debt | maturity checklist item |
| Kill gate | franken_threed master plan §5.8 | a named gate that can kill the project | not included — 1/44, too thin even for PROVISIONAL; noted here so the omission is deliberate |

## Where the evidence was too thin

- **PROVISIONAL items** (A10, A11, A12): each observed in ≤5 of 44 projects. Included because the mechanisms are load-bearing where they exist (frankenterm's taxonomy, frankenterm's acceptance shape, franken_tts's anti-ceremony), but marked so no one mistakes them for settled practice.
- **Kill gate** (1/44): excluded from the checklist; listed above so the omission is visible.
- **Per-change toolchain gate** (fmt/lint/test/build — Gate 5, 21/44): not a checklist item, by design rather than oversight. The kit's pre-commit hook deliberately avoids builds (frankensim: a hook that builds the workspace gets `--no-verify`'d on its second use), so the toolchain gate lives in the CI template (`.github/workflows/kit-gates.yml`, installed by init.sh) as commented-out fmt/lint/test steps the project fills per language. Specified as an extension point, not half-specified as a hook step.
- **Mechanical auto-demotion** (5/44): the kit keeps demotion procedural on day one with mechanical forms specified for maturity — matching the evidence rather than the aspiration.
- **check-readiness.sh substance heuristics**: the checker enforces structure (section presence, ≥3 real content lines, no repeated filler, section-required vocabulary, dated sign-off), not semantic truth. A determined agent can still write plausible empty prose that passes; the independent review (A13) remains the truth backstop. Documented in CHECKLIST.md A3 so the limit is part of the contract, not a footnote.
- **The 12 patterns**: quoted verbatim from the profiles' quotation of the suite-wide file; the suite-wide file itself was never read directly (it lives outside the repos). Additionally, the execution-readiness catalog documents a frankengit variant in which 4 of the 12 are worded differently — disclosed in the CHECKLIST.md appendix. If the wording ever needs to be authoritative, re-verify against the source.

## Cold-test additions (2026-09-22)

Five independent agents used the kit cold (fresh sandboxes, no grading
material), each building a different toy project. Full stumble log:
`.coldtest-lessons.md`. 15 lessons accepted, 4 rejected. Every patch below
is documentation, template, or diagnostic hardening — **no new external
mechanisms were introduced**, so nothing here needs a new row in the
mechanism table:

- `templates/agents.md` + `templates/definition-of-done.md`: installed by
  `init.sh` as `AGENTS.md` / `docs/definition-of-done.md`. Content is the
  CHECKLIST.md appendix (12 patterns, verbatim) and A8's evidence rules —
  both already traced above; the change is mechanical (a seed file instead
  of hand-copying), not doctrinal.
- Ledger lint hardened: the hook now lints every row of the staged ledger
  file (body-only edits to existing rows previously evaded the check).
  Same B4 doctrine, stronger enforcement.
- Claim checker: `enforce=yes` rows whose `readme_pattern` misses the README
  print a loud WARNING (previously a quiet SKIP); the summary distinguishes
  enforced / actually-checked / pattern-unmatched; the B6 zero-enforced
  error message now prescribes a working fix.
- All shipped templates carry REFERENCE COPY headers and filled A12
  four-line blocks; the packet template documents each section's
  machine-checkable vocabulary and offers sanctioned solo-review language
  for §11/§12 (the machine verifies the attestation exists, never its
  quality).
- Quickstart reordered: proofs → enforced claims → README LAST (B6 blocks
  any non-empty README until a claim is enforced with proof); first commit
  and JSONL bead-closing shown explicitly; `init.sh` warns on missing git
  identity.
