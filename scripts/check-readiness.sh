#!/bin/sh
# check-readiness.sh — machine-check docs/planning/packet.md for required fields.
# Pattern: franken_numpy PHASE2C_EXTRACTION_PACKET.md ("Missing any field =>
# packet state NOT READY"); frankentorch SCHEMA_LOCK_V1.md ("Missing any file
# => packet status NOT_READY").
# Usage: ./scripts/check-readiness.sh [path/to/packet.md]
#        KIT_MIN_LINES=3 ./scripts/check-readiness.sh   # override substance floor
# Exit 0: READY. Exit 1: NOT READY (lists every missing/incomplete section).
# A pristine (unfilled) template is NOT READY by design: blockquote guidance
# lines never count as content, so only real prose satisfies a section.
#
# What the machine checks (and what it cannot):
#   - section presence (the <!-- CHECK: --> markers);
#   - substance signals: >= MIN_LINES real content lines per section, and no
#     single line repeated more than twice inside a section (repeated filler);
#   - section vocabulary: each section must use the words its contract
#     requires (e.g. PACKETS must name the mandatory packet fields; SIGN-OFF
#     must carry a date). The vocabulary lists live in section_spec() below.
# What it cannot check: whether the prose is TRUE. A determined agent can
# still write plausible-sounding empty prose that passes these heuristics.
# The independent review (CHECKLIST.md A13) is the semantic backstop. What the
# heuristics buy: faking a pass now costs a structured, section-fluent draft
# instead of one junk line — at which point most of the real planning work is
# done anyway. (CHECKLIST.md A3 documents this limit.)
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACKET="${1:-$ROOT/docs/planning/packet.md}"
case "$PACKET" in /*) ;; *) PACKET="$ROOT/$PACKET" ;; esac
MIN_LINES="${KIT_MIN_LINES:-3}"

if [ ! -f "$PACKET" ]; then
  echo "NOT READY: $PACKET not found."
  echo "Copy templates/planning-packet.md to $PACKET and fill every section."
  exit 1
fi

TMPD="${TMPDIR:-/tmp}/kit-readiness-$$"
rm -rf "$TMPD"
mkdir -p "$TMPD" || { echo "NOT READY: cannot create temp dir."; exit 1; }
trap 'rm -rf "$TMPD"' EXIT

# section_spec <marker>: prints two lines —
#   line 1: minimum real content lines for the section
#   line 2: need|kw1,kw2,... — at least `need` of the keywords must appear
#           (case-insensitive). A keyword prefixed with "w:" must match as a
#           whole word, not a substring.
section_spec() {
  case "$1" in
    *PROBLEM*)          printf '3\n0|\n' ;;
    *NON-GOALS*)        printf '3\n1|w:not,w:never,w:no,out of scope,will not,won'"'"'t,does not\n' ;;
    *SOTA*)             printf '3\n1|sha,commit,version,pinned\n' ;;
    *PACKETS*)          printf '3\n5|goal,anchor,target,oracle,fixture,risk,acceptance\n' ;;
    *CLAIM-INVENTORY*)  printf '3\n1|planned,claims.tsv\n' ;;
    *EVIDENCE-DESIGN*)  printf '3\n2|commit,version,host,worker\n' ;;
    *HONESTY-MACHINERY*) printf '3\n2|ledger,demot,resurrect,predicate\n' ;;
    *PROOF-TAXONOMY*)   printf '3\n1|non-proof,non_proof\n' ;;
    *RELEASE-GATE*)     printf '3\n1|waiv\n' ;;
    *EXIT-CRITERIA*)    printf '3\n2|exit,entry\n' ;;
    *REVIEW*)           printf '3\n1|chang\n' ;;
    *SIGN-OFF*)         printf '2\n1|w:signed,w:sign-off,w:signoff\n' ;;
    *)                  printf '%s\n0|\n' "$MIN_LINES" ;;
  esac
}

# marker | human-readable label
SECTIONS='<!-- CHECK: PROBLEM -->|problem statement
<!-- CHECK: NON-GOALS -->|non-goals ("what this is not")
<!-- CHECK: SOTA -->|state-of-the-art survey
<!-- CHECK: PACKETS -->|work packets
<!-- CHECK: CLAIM-INVENTORY -->|claim inventory
<!-- CHECK: EVIDENCE-DESIGN -->|evidence design
<!-- CHECK: HONESTY-MACHINERY -->|honesty machinery
<!-- CHECK: PROOF-TAXONOMY -->|proof taxonomy
<!-- CHECK: RELEASE-GATE -->|release gate
<!-- CHECK: EXIT-CRITERIA -->|phase exit criteria
<!-- CHECK: REVIEW -->|independent review
<!-- CHECK: SIGN-OFF -->|execution sign-off'

fail=0
missing=""

OLDIFS="$IFS"
NL='
'
IFS="$NL"
for entry in $SECTIONS; do
  IFS="$OLDIFS"
  if [ -z "$entry" ]; then
    IFS="$NL"
    continue
  fi
  marker="${entry%%|*}"
  label="${entry#*|}"
  SECF="$TMPD/section.txt"
  : > "$SECF"

  if ! grep -qF -- "$marker" "$PACKET"; then
    missing="${missing}  - ${label}: marker ${marker} not found
"
    fail=1
    IFS="$NL"
    continue
  fi

  # Extract the section body: lines after the marker up to the next
  # "## " heading or the next CHECK marker. Drop blank lines, blockquote
  # guidance ("> ..."), and HTML comments — none of those count as content.
  awk -v m="$marker" '
    /^## / { if (found) exit; next }
    /^<!-- CHECK:/ { if (found) exit; if (index($0, m)) found = 1; next }
    found { print }
  ' "$PACKET" | grep -v '^[[:space:]]*$' | grep -v '^[[:space:]]*>' | grep -v '^[[:space:]]*<!--' > "$SECF"

  spec=$(section_spec "$marker")
  min_lines=$(printf '%s' "$spec" | sed -n '1p')
  vocspec=$(printf '%s' "$spec" | sed -n '2p')
  need="${vocspec%%|*}"
  kws="${vocspec#*|}"

  n=$(wc -l < "$SECF" | tr -d ' ')
  if [ "$n" -lt "$min_lines" ]; then
    missing="${missing}  - ${label}: only ${n} content line(s); need >= ${min_lines} real (non-guidance) lines
"
    fail=1
    IFS="$NL"
    continue
  fi

  # Repeated-filler check: no normalized line may appear more than twice.
  # (Squeeze spaces/tabs only — never newlines, or every line merges into one.)
  if tr 'A-Z' 'a-z' < "$SECF" | tr -s ' \t' ' ' | grep -v '^ *$' \
      | sort | uniq -c | sort -rn | head -1 | grep -qE '^ *([3-9]|[1-9][0-9])'; then
    missing="${missing}  - ${label}: a line is repeated 3+ times — filler, not content
"
    fail=1
    IFS="$NL"
    continue
  fi

  # Section vocabulary check.
  if [ "$need" -gt 0 ] && [ -n "$kws" ]; then
    hits=0
    IFS=','
    for kw in $kws; do
      IFS="$OLDIFS"
      case "$kw" in
        w:*)
          w="${kw#w:}"
          if grep -qiE "(^|[^[:alnum:]_])${w}([^[:alnum:]_]|$)" "$SECF"; then
            hits=$((hits + 1))
          fi
          ;;
        *)
          if grep -qiF -- "$kw" "$SECF"; then
            hits=$((hits + 1))
          fi
          ;;
      esac
      IFS=','
    done
    IFS="$OLDIFS"
    if [ "$hits" -lt "$need" ]; then
      missing="${missing}  - ${label}: section vocabulary too thin (${hits}/${need} required terms: ${kws})
"
      fail=1
      IFS="$NL"
      continue
    fi
  fi

  # SIGN-OFF must be dated (frankenscipy PHASE2C_SIGNOFF: dated, attributed).
  case "$marker" in
    *SIGN-OFF*)
      if ! grep -qE '[0-9]{4}-[0-9]{2}-[0-9]{2}' "$SECF"; then
        missing="${missing}  - ${label}: sign-off must carry a date (YYYY-MM-DD)
"
        fail=1
      fi
      ;;
  esac
  IFS="$NL"
done
IFS="$OLDIFS"

if [ "$fail" -eq 0 ]; then
  echo "READY: all 12 planning-packet sections present with substance."
  exit 0
fi
printf 'NOT READY: %s is missing/incomplete in:\n' "$PACKET"
printf '%s' "$missing"
exit 1
