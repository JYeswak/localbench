# Demotion rules (starter set)

<!-- Working copy of templates/demotion-rules.md. localbench change: D6 also triggers on any pin change (packet §7). -->
<!--
  Anti-ceremony (CHECKLIST.md A12):
  - Consumer: the reviewer running the demotion pass (B8); any agent whose evidence weakens a claim.
  - Gate: B8 demotion rules executed on schedule.
  - Defect class: stale claims that are never demoted because demotion was never written down (30/44 repos have demotion rules; only 5 enforce them mechanically).
  - Delete when: demotion is encoded mechanically in CI (frankenfs auto-demotion pattern).
-->
Demotion is the machinery that moves claims *down* when evidence weakens.
Promotion needs a gate; demotion needs only counter-evidence. Adapted from
frankenfs's multi-layer pattern — the most developed in the suite
(docs/planning/MODULARITY_RUNBOOK.md, docs/runbooks/readiness-action-autopilot.md,
docs/LEDGER_RESURRECTION.md). Each rule is marked [MECHANICAL] (enforced by a
script) or [PROCEDURAL] (enforced by convention and review). This starter set
is deliberately procedural-first: only 5 of 44 suite projects achieved
mechanical auto-demotion, so the kit does not pretend it is day-one cheap.

## D1 — Claim-effect lattice [PROCEDURAL]
- **Trigger:** advisory (weak) evidence arrives about a claim.
- **Action:** advisory inputs may only `no_change`, `block_upgrade`, or `downgrade_required`. The `upgrade_eligible` effect is reserved for authoritative proof.
- **Override:** none — an upgrade on advisory evidence is a gate violation, not a judgment call.
- **Origin:** frankenfs docs/runbooks/readiness-action-autopilot.md §4.

## D2 — Open P0 blocks release claims [PROCEDURAL → MECHANICAL at maturity]
- **Trigger:** a priority-0 issue is open.
- **Action:** no release-readiness claim may be made until the P0 is closed or reprioritized with recorded evidence.
- **Override:** reprioritization requires the evidence for the new priority, written down.
- **Mechanical form (maturity):** a test asserting release-readiness is blocked by an open P0 (frankenfs: `release_readiness_blocked_by_open_p0`).
- **Origin:** frankenfs docs/tracker-hygiene.md.

## D3 — Gate-change rule [PROCEDURAL]
- **Trigger:** someone proposes changing a gate's thresholds, counters, or exception semantics to land a change.
- **Action:** the change requires evidence in both directions: cases newly admitted as valid AND cases that remain rejected. A gate may never be weakened to land a change.
- **Override:** none.
- **Origin:** frankenfs docs/planning/MODULARITY_RUNBOOK.md ("Changing thresholds, counters, or exception semantics is a gate change"); frankenredis docs/GATE_VALIDITY.md.

## D4 — Demotions are always allowed [PROCEDURAL]
- **Trigger:** counter-evidence against any claim, from any agent.
- **Action:** any agent may demote the claim immediately, no permission needed. Record the demotion and the evidence in the negative-evidence ledger.
- **Override:** n/a — this rule exists to make demotion cheaper than silence.
- **Origin:** frankensim docs/MATURITY_LEVELS.md ("Demotions are always allowed and are never blocked").

## D5 — Tombstoning [PROCEDURAL]
- **Trigger:** a claim is retired.
- **Action:** the claim id stays in `registries/claims.tsv` marked `retired` in notes; it is never reused and never deleted. Future agents must be able to see what was claimed and why it died.
- **Override:** none.
- **Origin:** frankensympy registries/claims.toml ("retired claim IDs must remain").

## D6 — Proof expiry [PROCEDURAL → MECHANICAL at maturity]
- **Trigger:** a claim's proof is older than 90 days and was attested manually (not by a machine-checkable artifact),
  OR (localbench) any pinned component in docs/evidence/incumbents.md changed version, binary hash, or model digest
  after the proof was produced — speed proofs die with their generation.
- **Action:** the claim demotes to `implemented_uncertified` until re-verified.
- **Override:** re-verification on the new generation resets the clock.
- **Mechanical form (maturity):** the claim registry carries proof dates; the claim-discipline check fails expired manual proofs.
- **Origin:** frankentui docs/claims-ledger.md ("manual: proofs expire after 90 days").

## D7 — SHA-mismatch auto-demotion [MECHANICAL at maturity]
- **Trigger:** an artifact's hash does not match the registry.
- **Action:** automatic demotion — not reviewer-overridable, not waivable.
- **Override:** none. Fix the artifact or fix the registry; do not patch around the mismatch.
- **Origin:** frankengit registries/claims.tsv + docs/VERIFY_SPEC.md.
