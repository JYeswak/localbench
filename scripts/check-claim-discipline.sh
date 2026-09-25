#!/bin/sh
# check-claim-discipline.sh [claims.tsv] [README.md]
# Adapted from franken_markdown scripts/check-claim-discipline.sh.
# For every row with enforce=yes:
#   - if readme_pattern is set but NOT found in the README, the row is NOT
#     silently skipped: it prints a loud WARNING. A pattern that misses the
#     README means the claim is registered but UNVERIFIED — the usual cause
#     is a pattern that doesn't match the README prose EXACTLY
#     (case-sensitive, same line breaks). The registry never forces a claim
#     the README does not actually make (franken_markdown).
#   - otherwise proof_path must exist and be non-empty, and expected_substr
#     (if set) must appear inside the proof artifact.
# Exit 0: every enforced row passes. Exit 1: any enforced row fails.
# Also exit 1 when zero rows are enforced while README.md exists and is
# non-empty: a public README with an unenforced registry is undecorated
# discipline (CHECKLIST.md B6). With no README yet, that state is a warning.
# Paths in claims.tsv are relative to the repository root; absolute paths are
# honored as-is.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLAIMS="${1:-$ROOT/registries/claims.tsv}"
README_F="${2:-$ROOT/README.md}"
case "$CLAIMS" in /*) ;; *) CLAIMS="$ROOT/$CLAIMS" ;; esac
case "$README_F" in /*) ;; *) README_F="$ROOT/$README_F" ;; esac

if [ ! -f "$CLAIMS" ]; then
  echo "check-claim-discipline: $CLAIMS not found."
  exit 1
fi

# NOTE: tab is IFS *whitespace*, so IFS=<tab> read collapses empty fields.
# Translate tabs to \037 (unit separator, not IFS whitespace) to preserve empty columns.
# localbench fix (2026-09-22): the kit used \001, which macOS /bin/sh (bash 3.2) strips instead of splitting
# on — every row then parsed as one field, no row was ever enforced, and the canary "passed" only through the
# zero-enforced-rows rule. See docs/evidence/break-tests.md (lb-10).
SEP=$(printf '\037')
pass=0; fail=0; skipped=0; enforced=0; checked=0; unmatched=0
rowlist=""

while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  IFS="$SEP" read -r label pattern capkey substr proof enforce notes <<EOF
$(printf '%s' "$line" | tr '\t' '\037')
EOF
  [ "$label" = "label" ] && continue   # header row
  [ -z "$label" ] && continue          # blank row
  rowlist="${rowlist}${label}=>${enforce} "
  if [ "$enforce" != "yes" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  enforced=$((enforced + 1))
  if [ -n "$pattern" ]; then
    if [ ! -f "$README_F" ] || ! grep -qF -- "$pattern" "$README_F"; then
      echo "WARNING  $label: enforce=yes but the pattern was NOT found in the README — row NOT checked."
      echo "         pattern must match README prose EXACTLY (case-sensitive, same line breaks)."
      echo "         This claim is registered but UNVERIFIED until the pattern matches."
      unmatched=$((unmatched + 1))
      continue
    fi
  fi
  checked=$((checked + 1))
  # Proof paths are relative to the repo root; absolute paths are honored
  # as-is (an absolute path that silently resolved under $ROOT used to
  # false-FAIL — now it resolves to itself).
  case "$proof" in
    /*) p="$proof" ;;
    *)  p="$ROOT/$proof" ;;
  esac
  # localbench (packet §8, lb-10): spike receipts are declared non-proof; an enforced claim may not cite one.
  case "$proof" in
    *docs/evidence/receipts/*spike*)
      echo "FAIL  $label: proof $proof is a spike receipt (non-proof per packet §8); cite a banked harness receipt"
      fail=$((fail + 1))
      continue
      ;;
  esac
  if [ -z "$proof" ] || [ ! -s "$p" ]; then
    echo "FAIL  $label: proof artifact missing or empty: ${proof:-<none>}"
    fail=$((fail + 1))
    continue
  fi
  if [ -n "$substr" ] && ! grep -qF -- "$substr" "$p"; then
    echo "FAIL  $label: proof $proof lacks expected text: $substr"
    fail=$((fail + 1))
    continue
  fi
  echo "PASS  $label -> $proof"
  pass=$((pass + 1))
done < "$CLAIMS"

if [ "$enforced" -eq 0 ]; then
  if [ -f "$README_F" ] && [ -s "$README_F" ]; then
    echo "FAIL: no enforced claims (enforce=yes) in $CLAIMS while $README_F exists and is non-empty."
    echo "A public README with zero enforced claims is undecorated discipline (CHECKLIST.md B6)."
    echo "Rows seen (label => enforce): ${rowlist:-<none>}"
    echo "Fix ONE of:"
    echo "  (a) set enforce=yes on at least one real claim whose proof artifact exists;"
    echo "  (b) keep the README empty/absent until your first claim has a proof."
    echo "Note: trimming prose cannot help — ANY non-empty README requires an enforced claim."
    fail=$((fail + 1))
  else
    echo "WARNING: no enforced claims (enforce=yes) in $CLAIMS; the registry is decorative until rows are enforced."
  fi
fi
echo "check-claim-discipline: $pass passed, $fail failed, $skipped skipped ($enforced enforced, $checked actually checked, $unmatched pattern-unmatched)."
[ "$fail" -eq 0 ]
