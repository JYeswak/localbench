<!-- Working copy of templates/definition-of-done.md (installed by scripts/init.sh). -->
<!--
  Anti-ceremony (CHECKLIST.md A12):
  - Consumer: every agent closing a bead or landing a change; the reviewer auditing closes (B2).
  - Gate: A8 Definition of Done — "done" and "blocked" are states with evidence, not status updates.
  - Defect class: prose-closed beads and mood-based "blocked" states with no artifact.
  - Delete when: the project adopts a machine-enforced DoD check (asupersync's validate_dod.sh pattern); until then, keep this file.
-->

# Definition of Done

Installed by the FrankenSuite Starter Kit (CHECKLIST.md A8); examples tailored to localbench.

## Mark DONE only with

1. **Command evidence** — the exact command(s) that prove the outcome, e.g.
   `sh scripts/check-readiness.sh` → `READY`; `localbench run --backend ollama --model qwen3.6:35b-mlx`
   → exit 0 with `runs/<stamp>/summary.json`, banked to `docs/evidence/receipts/` when a claim cites it.
   Paste the command and its result into the bead's `close_reason`.
2. **Updated documentation** — the docs that describe the changed behavior
   are updated in the same change. Code without doc updates is not done.

## Mark BLOCKED only with

Concrete artifact evidence: the error text, the command that produced it,
and the path. "Blocked" without these three is a mood, not a state.

## Never

- Close a bead because code was committed; close it when the named gate
  artifacts pass. (frankensympy AGENTS.md §12)
- Close a bead on a prose assertion. Closure is on cited evidence.
  (asupersync: "only a terminal receipt proves execution")
- Weaken a gate to land a change. (AGENTS.md, pattern 1)

Origin: franken_whisper docs/definition_of_done.md; asupersync
ATP_DOD_CHECKLIST.md (enforced there by scripts/validate_dod.sh).
