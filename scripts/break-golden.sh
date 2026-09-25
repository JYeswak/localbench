#!/bin/sh
# Break-test golden compare (lb-06 planted negatives; docs/evidence/break-tests.md).
# Consumer: the reviewer asking "does compare fail loudly?"; re-run after editing golden.py or cmd_compare.
# Takes a run dir whose golden exists (e.g. the aa2 leg that banked it), doctors a COPY of the golden in place,
# re-judges the run with `localbench compare` (no measurement), and restores the golden on exit.
#   clean       unmodified golden                      -> exit 0 (the run is inside its own A/A band)
#   regress     one higher-is-better value x1.30       -> exit 1, that row REGRESSED
#   genmismatch pins.backend_version changed           -> exit 1, every row GENERATION-MISMATCH
#   tolunproven one metric's tol_source removed        -> exit 1, that row TOL-UNPROVEN
set -u
RUN=${1:?usage: scripts/break-golden.sh <run dir> <case>}
CASE=${2:?case: clean|regress|genmismatch|tolunproven}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
G=$(jq -r '.provenance.pins | "goldens/\(.host_id)/\(.backend)__\(.model)"' "$RUN/summary.json" | sed 's/[^A-Za-z0-9._\/-]/_/g').json
G="$ROOT/$G"
[ -f "$G" ] || { echo "no golden at $G" >&2; exit 2; }
BACKUP=$(mktemp)
cp "$G" "$BACKUP"
trap 'cp "$BACKUP" "$G"; rm -f "$BACKUP"' EXIT INT TERM

# Plant on the higher-is-better metric with the TIGHTEST band: a 30% shift is inside a wide band by design
# (2026-09-22: micro.cache_hit_8k.speedup_x, tol 0.2333, absorbed a 30% doctored golden -> PASS).
KEY=$(jq -r '[.metrics | to_entries[] | select(.value.better == "higher")] | sort_by(.value.tol) | .[0].key' "$G")
case "$CASE" in
clean) want_rc=0; want="" ;;
regress)
  jq --arg k "$KEY" '.metrics[$k].value *= 1.30' "$BACKUP" > "$G"; want_rc=1; want="$KEY: REGRESSED" ;;
genmismatch)
  jq '.pins.backend_version = "0.0.0-planted"' "$BACKUP" > "$G"; want_rc=1; want=": GENERATION-MISMATCH" ;;
tolunproven)
  jq --arg k "$KEY" 'del(.metrics[$k].tol_source)' "$BACKUP" > "$G"; want_rc=1; want="$KEY: TOL-UNPROVEN" ;;
*) echo "unknown case $CASE" >&2; exit 2 ;;
esac

out=$(localbench compare "$RUN" 2>&1)
rc=$?
printf '%s\n' "$out" | grep '^UNSOUND' | head -5
echo "localbench compare exit=$rc (case $CASE, key $KEY)"
if [ "$rc" -eq "$want_rc" ] && { [ -z "$want" ] || printf '%s\n' "$out" | grep -q "UNSOUND  .*$want"; }; then
  echo "BREAK-TEST OK: $CASE"
  exit 0
fi
echo "BREAK-TEST FAILED: expected exit $want_rc${want:+ with '$want'}" >&2
exit 1
