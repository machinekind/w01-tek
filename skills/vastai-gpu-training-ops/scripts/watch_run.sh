#!/usr/bin/env bash
# Monitor a remote training log and process until completion or failure.
#
# Usage: watch_run.sh -s "ssh -p PORT root@HOST" -l REMOTE_LOG \
#   -p "package[.]train" [-m LOCAL_MIRROR] [-i SECONDS]
#
# Exit 0: done marker. Exit 1: crash signature. Exit 2: silent process exit.
set -euo pipefail

SSH=""
LOG=""
PATTERN=""
MIRROR="./run_mirror.log"
INTERVAL=60

while getopts "s:l:p:m:i:" option; do
  case "$option" in
    s) SSH=$OPTARG ;;
    l) LOG=$OPTARG ;;
    p) PATTERN=$OPTARG ;;
    m) MIRROR=$OPTARG ;;
    i) INTERVAL=$OPTARG ;;
    *) exit 64 ;;
  esac
done

if [[ -z "$SSH" || -z "$LOG" || -z "$PATTERN" ]]; then
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
  exit 64
fi

SIGNATURES='^steps |Traceback|RuntimeError|XlaRuntimeError|RESOURCE_EXHAUSTED|CUDA_ERROR|OOM|Killed|done ->|PROCESS_EXITED'

: > "$MIRROR"

# SSH is intentionally a command prefix such as "ssh -p PORT root@HOST".
# shellcheck disable=SC2086
$SSH "tail -n +1 -F '$LOG'" >> "$MIRROR" 2>/dev/null &
STREAM_PID=$!

(
  gone=0
  while true; do
    sleep "$INTERVAL"
    # A transient SSH failure is unknown state, not process death: it neither
    # counts toward GONE nor resets the streak. Only three consecutive
    # successful polls that find no process declare death.
    # shellcheck disable=SC2086
    alive=$($SSH "pgrep -f '$PATTERN' >/dev/null && echo RUN || echo GONE" 2>/dev/null || echo SSH_FAIL)
    case "$alive" in
      GONE) gone=$((gone + 1)) ;;
      RUN)  gone=0 ;;
    esac
    if (( gone >= 3 )); then
      sleep 10  # let the log stream deliver any terminal line first
      echo "PROCESS_EXITED" >> "$MIRROR"
      break
    fi
  done
) &
WATCHDOG_PID=$!

trap 'kill "$STREAM_PID" "$WATCHDOG_PID" 2>/dev/null || true' EXIT

while IFS= read -r line; do
  printf '%s\n' "$line"
  case "$line" in
    *"done ->"*) exit 0 ;;
    *PROCESS_EXITED*) exit 2 ;;
    *Traceback*|*RuntimeError*|*RESOURCE_EXHAUSTED*|*CUDA_ERROR*|*OOM*|*Killed*) exit 1 ;;
  esac
done < <(tail -n +1 -F "$MIRROR" | grep --line-buffered -aE "$SIGNATURES")
