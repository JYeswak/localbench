#!/bin/sh
# Regression-suite verdict for one git tree: the commit's content, never the shared working tree.
#
# Anti-ceremony (A12):
# - Consumer: .githooks/pre-commit (step 5) and the landing scripts, which call it with the tree they are about to
#   commit so the commit's own hook is a cache hit.
# - Gate: the regression suite, unchanged; only WHAT it runs on and how often changed.
# - Defect class: a commit judged by other panes' uncommitted work (2026-10-02: %pane's WIP turned every shared-tree
#   commit red), and a 4-minute suite run while git holds index.lock (two stale locks the same day).
# - Delete when: commits are only ever made from per-commit snapshots by one landing tool.
#
#   scripts/suite_verdict.sh [TREE]     TREE defaults to `git write-tree` (the index, or the temporary index of
#                                       `git commit --only`). Exit 0 = PASS (fresh or cached), 1 = FAIL,
#                                       75 = the heavy slot is busy and SUITE_VERDICT_WAIT ran out (default 7200 s).
#
# The commit hook sets SUITE_VERDICT_WAIT=0: a hook that queues for the slot holds index.lock for the whole wait,
# blocking every other commit, and an agent that times the commit out leaves a stale lock (2026-10-03, twice). Run
# this script first (it queues), then commit: the hook is a cache hit.
#
# PASS is cached in ~/.localbench/suite-verdicts/<tree>-<python version>: a tree's content fully determines the
# suite's inputs (tests are hermetic: CI runs them without omp/ollama). FAIL is never cached.
# SUITE_VERDICT_FRESH=1 bypasses only this read; the normal export/slot/run/log path and PASS cache write still run.
set -u
child=
cleanup() {
  if [ -n "$child" ]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    child=
  fi
}
on_signal() {
  code=$1
  trap - EXIT TERM INT HUP
  cleanup
  exit "$code"
}
trap cleanup EXIT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
ROOT=$(git rev-parse --show-toplevel) || exit 1
tree=${1:-$(git write-tree)} || exit 1
pyver=$(uv run --quiet --project "$ROOT" python -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null) || {
  echo "suite_verdict: uv/python unavailable; the suite cannot run" >&2; exit 1; }
cache="$HOME/.localbench/suite-verdicts"
key="$tree-$pyver"
if [ "${SUITE_VERDICT_FRESH:-0}" != 1 ] && [ -f "$cache/$key" ]; then
  echo "suite_verdict: cached PASS for tree $tree: $(cat "$cache/$key")"
  exit 0
fi
snap="$ROOT/var/agent-tmp/suite-snap.$$"     # owned scratch; the reaper removes it once this pid is gone
mkdir -p "$snap" || exit 1
printf 'pid=%s label=suite-snap repo=%s created=%s\n' "$$" "$ROOT" "$(date -u +%FT%TZ)" > "$snap/.owner"
git archive "$tree" | tar -x -C "$snap" || { echo "suite_verdict: cannot export tree $tree" >&2; exit 1; }
start=$(date +%s)
# Under the heavy slot as a CPU-only holder (background priority; one heavy job at a time): 2026-10-03 two suites
# ran side by side outside the slot while load climbed to 47.
out_file="$snap/suite.out"
(
  cd "$snap"
  exec uv run --quiet --project "$ROOT" python "$ROOT/scripts/heavy_run.py" "${SUITE_VERDICT_WAIT:-7200}" >"$out_file" 2>&1
) &
child=$!
wait "$child"; rc=$?
out=$(cat "$out_file" 2>/dev/null || true)
child=
if [ "$rc" -eq 0 ]; then
  ran=$(printf '%s\n' "$out" | grep -E '^Ran [0-9]+ tests' | head -1)
  mkdir -p "$cache" && printf '%s; wall %ss; %s\n' "$ran" "$(($(date +%s) - start))" "$(date -u +%FT%TZ)" > "$cache/$key"
  echo "suite_verdict: PASS for tree $tree: $ran"
  exit 0
fi
if [ "$rc" -eq 75 ]; then
  printf '%s\n' "$out" | grep -v '^INFO' | tail -3 >&2
  echo "suite_verdict: heavy slot busy; run 'sh scripts/suite_verdict.sh' (it queues), then commit again" >&2
  exit 75
fi
printf '%s\n' "$out" | grep -v '^INFO' | tail -40 >&2
echo "suite_verdict: FAIL for tree $tree" >&2
exit 1
