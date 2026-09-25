#!/bin/sh
# check-monitor-pane.sh — is the omp in a tmux pane driven by a LOCAL model?
# A watcher of a localbench run must not load a local model while the run is
# active (it would contend for the GPU/unified memory the run is measuring).
# Consumer: every lb-09 watcher, on its own pane, before it starts watching
# (docs/MONITOR.md). Break-tests: docs/evidence/break-tests.md (lb-09).
#
# Usage: scripts/check-monitor-pane.sh <tmux-pane-id>      e.g. %12
# Exit 0: the omp in that pane uses a non-local main model.
# Exit 1: 'LOCAL MODEL: <selector>' — a local provider (ollama, mlx-serve,
#         localbench, lm-studio, llama.cpp, vllm, or any provider whose
#         models.yml baseUrl is loopback) appears in the evidence.
# Exit 2: cannot determine (no such pane, no omp under it, selector without a
#         provider, unreadable config). Fail closed: unknown is not safe.
# A 'WARNING:' line (exit code unaffected) means the profile's smol role is
# local: omp sends titles, and memory work when mnemopi.llmMode is smol, to
# smol even when the main model is cloud.
#
# Evidence, strongest last-writer first:
#   (d) the session .jsonl the omp process holds open (lsof): the last
#       model_change (role default) or assistant message provider/model.
#       This is the only source that sees a runtime /model switch.
#   (a) the omp process argv: --model / --provider / --smol / --profile.
#   (b) the profile's config.yml modelRoles.default (used only when neither
#       (d) nor (a) names a model). Profile dir: the agent.db the process has
#       open (lsof); else --profile / OMP_PROFILE / PI_PROFILE /
#       PI_CODING_AGENT_DIR from `ps eww` (macOS usually hides other
#       processes' env, so this is often empty); else ~/.omp/agent.
#   (c) the pane title — printed as corroboration only, never decides.
# Any local evidence from (a) or (d) => exit 1, even if the user later
# switched to a cloud model: they can switch back.
#
# LIMIT (honest): this is a point-in-time check, not a guarantee. A user can
# run /model at any moment after the check; argv and config never see that,
# and the session file only records it once written (and file order can
# disagree with the active branch after /tree navigation). Smol calls are not
# recorded per call. Re-run the check whenever you are unsure; never treat an
# earlier exit 0 as a standing permission.
set -u

PANE="${1:-}"
case "$PANE" in
  %[0-9]*) ;;
  *) echo "usage: $0 <tmux-pane-id like %12> (got '${PANE}')" >&2
     echo "UNDETERMINED: not a tmux pane id" ; exit 2 ;;
esac

command -v tmux >/dev/null 2>&1 || { echo "UNDETERMINED: tmux not found"; exit 2; }

# `tmux display -t %999` prints nothing and exits 0 (and inside tmux may fall
# back to the caller's own pane), so resolve the id from the full pane list.
LINE=$(tmux list-panes -a -F '#{pane_id} #{pane_pid} #{pane_title}' 2>/dev/null \
  | awk -v p="$PANE" '$1==p {print; exit}')
if [ -z "$LINE" ]; then
  echo "UNDETERMINED: no tmux pane $PANE (tmux list-panes -a)"
  exit 2
fi
PANE_PID=$(printf '%s\n' "$LINE" | awk '{print $2}')
PANE_TITLE=$(printf '%s\n' "$LINE" | cut -d' ' -f3-)

# First omp process among the pane shell's descendants (breadth-first).
OMP=$(ps -A -o pid=,ppid=,command= | awk -v root="$PANE_PID" '
  { pid=$1; ppid=$2; $1=""; $2=""; sub(/^  /, ""); cmd[pid]=$0; kids[ppid]=kids[ppid] " " pid }
  END {
    q = root; 
    while (q != "") {
      n = split(q, cur, " "); q = ""
      for (i = 1; i <= n; i++) {
        p = cur[i]; if (p == "") continue
        if (p != root) {
          m = split(cmd[p], w, " ")
          for (j = 1; j <= 2 && j <= m; j++)
            if (w[j] ~ /(^|\/)omp$/) { print p " " cmd[p]; exit }
        }
        q = q " " kids[p]
      }
    }
  }')
if [ -z "$OMP" ]; then
  echo "pane $PANE shell_pid=$PANE_PID title='$PANE_TITLE'"
  echo "UNDETERMINED: no omp process under pane $PANE"
  exit 2
fi
OMP_PID=${OMP%% *}
OMP_CMD=${OMP#* }

# argv flags (ps flattens quoting; model selectors contain no spaces).
argv_flag() {
  printf '%s\n' "$OMP_CMD" | awk -v f="$1" '{
    for (i = 1; i <= NF; i++) {
      if ($i == f && i < NF) { print $(i+1); exit }
      if (index($i, f "=") == 1) { print substr($i, length(f) + 2); exit }
    }
  }'
}
ARG_MODEL=$(argv_flag --model)
ARG_PROVIDER=$(argv_flag --provider)
ARG_SMOL=$(argv_flag --smol)
ARG_PROFILE=$(argv_flag --profile)

ENVV=$(ps eww -o command= -p "$OMP_PID" 2>/dev/null | tr ' ' '\n')
env_var() { printf '%s\n' "$ENVV" | sed -n "s/^$1=//p" | head -n 1; }

OPEN=$(lsof -a -p "$OMP_PID" -Fn 2>/dev/null | sed -n 's/^n//p')
DB=$(printf '%s\n' "$OPEN" | grep '/agent\.db$' | head -n 1)
SESSION=$(printf '%s\n' "$OPEN" | grep '/sessions/.*\.jsonl$' | tail -n 1)

if [ -n "$DB" ]; then
  AGENT_DIR=${DB%/agent.db}; PROFILE_SRC="lsof agent.db"
elif [ -n "$ARG_PROFILE" ]; then
  AGENT_DIR="$HOME/.omp/profiles/$ARG_PROFILE/agent"; PROFILE_SRC="argv --profile"
elif [ -n "$(env_var PI_CODING_AGENT_DIR)" ]; then
  AGENT_DIR=$(env_var PI_CODING_AGENT_DIR); PROFILE_SRC="env PI_CODING_AGENT_DIR"
elif [ -n "$(env_var OMP_PROFILE)$(env_var PI_PROFILE)" ]; then
  NAME=$(env_var OMP_PROFILE); [ -n "$NAME" ] || NAME=$(env_var PI_PROFILE)
  AGENT_DIR="$HOME/.omp/profiles/$NAME/agent"; PROFILE_SRC="env OMP_PROFILE/PI_PROFILE"
else
  AGENT_DIR="$HOME/.omp/agent"; PROFILE_SRC="default"
fi
CONFIG="$AGENT_DIR/config.yml"
MODELS="$AGENT_DIR/models.yml"

# `section.key` from a 2-space-indented YAML block (no full YAML parser).
yaml_get() {
  [ -f "$CONFIG" ] || return 0
  awk -v sec="$1" -v key="$2" '
    /^[^ #]/ { insec = ($0 ~ "^" sec ":") ; next }
    insec && $0 ~ "^  " key ":" {
      v = $0; sub("^  " key ":[ ]*", "", v); sub(/[ ]+#.*$/, "", v)
      gsub(/^["\047]|["\047]$/, "", v); print v; exit
    }' "$CONFIG"
}
CFG_DEFAULT=$(yaml_get modelRoles default)
CFG_SMOL=$(yaml_get modelRoles smol)
MNEMOPI_MODE=$(yaml_get mnemopi llmMode)

# Providers declared in models.yml with a loopback / non-loopback baseUrl.
yml_providers() {
  [ -f "$MODELS" ] || return 0
  awk -v want="$1" '
    /^providers:/ { inp = 1; next }
    /^[^ #]/ { inp = 0 }
    inp && /^  [A-Za-z0-9._-]+:[ ]*$/ { prov = $1; sub(/:$/, "", prov); next }
    inp && /^    baseUrl:/ {
      loop = ($0 ~ /(127\.0\.0\.1|localhost|\[::1\]|0\.0\.0\.0)/)
      if ((want == "loop" && loop) || (want == "remote" && !loop)) print prov
    }' "$MODELS"
}
LOOP_PROVIDERS=$(yml_providers loop | tr '\n' ' ')
REMOTE_PROVIDERS=$(yml_providers remote | tr '\n' ' ')

# classify <selector>: prints local | cloud | unknown
classify() {
  sel=$1
  case "$sel" in
    */*) prov=${sel%%/*} ;;
    *) printf 'unknown\n'; return ;;
  esac
  prov=$(printf '%s' "$prov" | tr 'A-Z' 'a-z')
  case " $LOOP_PROVIDERS " in *" $prov "*) printf 'local\n'; return ;; esac
  case "$prov" in
    ollama|mlx-serve|mlx|localbench|lm-studio|lmstudio|llama.cpp|llamacpp|llama-cpp|llamafile)
      printf 'local\n'; return ;;
    vllm)
      case " $REMOTE_PROVIDERS " in *" vllm "*) printf 'cloud\n' ;; *) printf 'local\n' ;; esac
      return ;;
  esac
  printf 'cloud\n'
}

SESS_MODEL=""
if [ -n "$SESSION" ] && [ -r "$SESSION" ]; then
  if command -v jq >/dev/null 2>&1; then
    SESS_MODEL=$(jq -Rrn 'reduce (inputs | fromjson?) as $l (null;
        if $l.type == "model_change" and (($l.role // "default") == "default") then $l.model
        elif $l.type == "message" and $l.message.role == "assistant"
             and (($l.message.provider // "") != "") then ($l.message.provider + "/" + ($l.message.model // ""))
        else . end) // empty' "$SESSION" 2>/dev/null)
  else
    echo "NOTE: jq not found; session-file evidence skipped"
  fi
fi

echo "pane $PANE shell_pid=$PANE_PID omp_pid=$OMP_PID title='$PANE_TITLE'"
echo "  argv: $OMP_CMD"
echo "  profile: $AGENT_DIR ($PROFILE_SRC)"
echo "  session: ${SESSION:-none open} current=${SESS_MODEL:-?}"
echo "  config: modelRoles.default=${CFG_DEFAULT:-?} modelRoles.smol=${CFG_SMOL:-?} mnemopi.llmMode=${MNEMOPI_MODE:-?}"

# Evidence set for the main model.
EVIDENCE=""
[ -n "$ARG_MODEL" ] && EVIDENCE="$EVIDENCE $ARG_MODEL"
[ -z "$ARG_MODEL" ] && [ -n "$ARG_PROVIDER" ] && EVIDENCE="$EVIDENCE $ARG_PROVIDER/"
[ -n "$SESS_MODEL" ] && EVIDENCE="$EVIDENCE $SESS_MODEL"
[ -z "$EVIDENCE" ] && [ -n "$CFG_DEFAULT" ] && EVIDENCE=" $CFG_DEFAULT"

# Smol role: argv --smol > PI_SMOL_MODEL > config.
SMOL=$ARG_SMOL
[ -n "$SMOL" ] || SMOL=$(env_var PI_SMOL_MODEL)
[ -n "$SMOL" ] || SMOL=$CFG_SMOL
if [ -n "$SMOL" ] && [ "$(classify "$SMOL")" = local ]; then
  if [ "$MNEMOPI_MODE" = smol ]; then
    echo "WARNING: smol role is LOCAL ($SMOL) and mnemopi.llmMode=smol: titles and memory work hit the local model even when the main model is cloud"
  else
    echo "WARNING: smol role is LOCAL ($SMOL): titles/lightweight tasks hit the local model even when the main model is cloud"
  fi
fi

TITLE_LOCAL=$(printf '%s' "$PANE_TITLE" | grep -Eio 'ollama|mlx-serve|localbench|lm-?studio|llama\.?cpp|vllm' | head -n 1)

if [ -z "$EVIDENCE" ]; then
  echo "UNDETERMINED: no --model in argv, no session model, no modelRoles.default in $CONFIG"
  exit 2
fi
UNKNOWN=""
for sel in $EVIDENCE; do
  case $(classify "$sel") in
    local) echo "LOCAL MODEL: $sel"; exit 1 ;;
    unknown) UNKNOWN="$UNKNOWN $sel" ;;
  esac
done
if [ -n "$UNKNOWN" ]; then
  echo "UNDETERMINED: selector without provider:$UNKNOWN"
  exit 2
fi
[ -n "$TITLE_LOCAL" ] && echo "NOTE: pane title mentions '$TITLE_LOCAL' but argv/session/config say cloud; re-check with the pane owner"
echo "NON-LOCAL MODEL:$EVIDENCE"
exit 0
