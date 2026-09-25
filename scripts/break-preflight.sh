#!/bin/sh
# Break-test the preflight gate (lb-01 planted negatives; docs/evidence/break-tests.md).
# Consumer: the reviewer asking "does preflight refuse a busy machine?"; re-run after editing preflight().
# Case cpu: saturate every core with `yes`, then ask localbench to measure -> must exit 1 "CPU already ...% busy".
# Case gpu: keep ollama generating on the MoE in a loop, then ask again -> must exit 1 "GPU already ...% busy".
# Neither case reaches a tier: preflight refuses before any model is loaded for measurement.
set -u
CASE=${1:?usage: scripts/break-preflight.sh cpu|gpu}
SPEC=${2:-ollama:qwen3.6:35b-mlx}
PIDS=""
cleanup() { [ -n "$PIDS" ] && kill $PIDS 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

case "$CASE" in
cpu)
  n=$(sysctl -n hw.ncpu)
  i=0
  while [ "$i" -lt "$n" ]; do yes >/dev/null & PIDS="$PIDS $!"; i=$((i + 1)); done
  sleep 3
  want="preflight refused: CPU already"
  ;;
gpu)
  model=${SPEC#ollama:}
  (while :; do
     curl -s http://127.0.0.1:11434/api/generate \
       -d "{\"model\":\"$model\",\"prompt\":\"Write a long essay about hash maps.\",\"stream\":false,\"options\":{\"num_predict\":400}}" \
       >/dev/null
   done) & PIDS="$PIDS $!"
  sleep 15
  want="preflight refused: GPU already"
  ;;
*) echo "unknown case $CASE" >&2; exit 2 ;;
esac

out=$(localbench run "$SPEC" --tiers conf --repeats 1 2>&1)
rc=$?
printf '%s\n' "$out" | grep 'preflight'
echo "localbench exit=$rc"
if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -q "^$want"; then
  echo "BREAK-TEST OK: preflight refused for the planted reason ($CASE)"
  exit 0
fi
echo "BREAK-TEST FAILED: expected exit 1 with '$want'" >&2
exit 1
