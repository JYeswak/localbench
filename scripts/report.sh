#!/bin/sh
# Durable worker -> orchestrator report: the text lands in a file first, then a one-line pointer is typed into the
# orchestrator's pane and its arrival is checked there.
#
# Anti-ceremony (A12):
# - Consumer: the orchestrator pane (%pane by default), which reads runs/inbox.jsonl each tick.
# - Gate: none; it is a transport whose delivery is verified at the receiving end.
# - Defect class: a long report typed into a busy pane is lost and the sender closes its bead on its own send
#   (2026-10-02: %pane's 9-leap LEAPS message never reached %pane; kit-z6iw.23 was closed as "delivered").
# - Delete when: Agent Mail inbox events are read by the orchestrator each tick.
#
#   scripts/report.sh <bead> <READY|DONE|BLOCKED|LEAPS|...> <file-or-text> [to-pane]
#
# Exit 0 = report filed AND the pointer was seen in the target pane; 1 = filed but the pointer was not seen
# (the report is still in runs/inbox.jsonl, so the orchestrator finds it on its next tick).
set -u
[ $# -ge 3 ] || { echo "usage: scripts/report.sh <bead> <status> <file-or-text> [to-pane]" >&2; exit 2; }
bead=$1 status=$2 body=$3 to=${4:-%pane}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
ts=$(date -u +%Y%m%dT%H%M%SZ)
# Agent tool shells often lack TMUX_PANE (2026-10-02: every worker report arrived as "unknown"); REPORT_FROM wins.
from=${REPORT_FROM:-$(tmux display-message -p -t "${TMUX_PANE:-}" '#{pane_id}' 2>/dev/null || echo "${TMUX_PANE:-unknown}")}
dir="$ROOT/runs/inbox"; mkdir -p "$dir" || exit 1
file="$dir/$ts-${from#%}-$bead.md"
if [ -f "$body" ]; then cp "$body" "$file"; else printf '%s\n' "$body" > "$file"; fi
op="rpt-${from#%}-$ts"
printf '{"t":"%s","op":"%s","from":"%s","to":"%s","bead":"%s","status":"%s","file":"%s"}\n' \
  "$ts" "$op" "$from" "$to" "$bead" "$status" "$file" >> "$ROOT/runs/inbox.jsonl"
first=$(head -c 160 "$file" | tr '\n' ' ')
TMUX_TMPDIR=${TMUX_TMPDIR:-~/.tmux-sockets} timeout 25 ntm send localbench --panes="$to" --no-cass-check \
  --force-non-interactive "$op [$from -> $to] $status $bead: $first... FULL: $file" >/dev/null 2>&1
sleep 5
if TMUX_TMPDIR=${TMUX_TMPDIR:-~/.tmux-sockets} tmux capture-pane -p -J -t "$to" -S -400 | grep -qF "$op"; then
  echo "report: delivered $op ($file)"; exit 0
fi
echo "report: filed $file but $op not seen in $to; it stays in runs/inbox.jsonl for the next tick" >&2
exit 1
