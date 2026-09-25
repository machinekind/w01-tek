#!/usr/bin/env bash
# Make sure a rented machine dies, whatever happens to the session that
# rented it.
#
#   scripts/rental_watchdog.sh arm 90 --kill 'vastai destroy instance 123'
#   scripts/rental_watchdog.sh arm 23:55 --kill 'curl -sf -X DELETE ...' --name enverge
#   scripts/rental_watchdog.sh arm 60 --name enverge        # remind only
#   scripts/rental_watchdog.sh fire                         # kill now
#   scripts/rental_watchdog.sh status
#   scripts/rental_watchdog.sh disarm
#
# Provider-agnostic sibling of vast_destroy_at.sh: that one knows the vast
# CLI, this one runs whatever kill command you hand it, for providers whose
# meter runs until the instance is DELETED (Enverge bills create-to-delete;
# "restart" does not stop the meter, and there is no documented spend cap).
#
# Why launchd rather than `sleep && kill` in a terminal: a backgrounded sleep
# dies with its shell -- including when an agent session ends, which is the
# exact scenario this exists for. Why not cron: cron silently skips a job
# whose time passed while the Mac slept, which is the case that costs money
# overnight; launchd re-fires a missed StartCalendarInterval on wake.
#
# WHAT THIS IS NOT: enforcement. It runs on your laptop, so a Mac that is off
# at the deadline protects nothing, and a provider without a delete API can
# only be shouted about, not stopped. The hard limit is a card with a cap or
# prepaid credits; this is the second line.
set -euo pipefail

NAME="rental"
KILL_CMD=""
DEADLINE=""

usage() {
  cat >&2 <<USAGE
usage:
  $0 arm <MINUTES|HH:MM> [--kill '<command>'] [--name <label>]
  $0 fire | status | disarm [--name <label>]
USAGE
  exit 1
}

cmd="${1:-status}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill) KILL_CMD="${2:?--kill needs a command}"; shift 2 ;;
    --name) NAME="${2:?--name needs a label}"; shift 2 ;;
    *) [[ -z "$DEADLINE" ]] && { DEADLINE="$1"; shift; } || usage ;;
  esac
done

LABEL="com.wojtek.rental-watchdog.${NAME}"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="$HOME/Library/Logs/wojtek-rental-watchdog-${NAME}.log"
STATE="$HOME/.wojtek-rental-watchdog-${NAME}"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

notify() {  # loud, because a silent watchdog is indistinguishable from none
  osascript -e "display notification \"$1\" with title \"Wojtek rental watchdog\"" 2>/dev/null || true
  echo "$(date '+%F %T') $1" >> "$LOG"
}

case "$cmd" in

fire)
  mkdir -p "$(dirname "$LOG")"
  kill_cmd="$(cat "$STATE" 2>/dev/null || true)"
  {
    echo "=== $(date '+%F %T %Z') watchdog firing for '${NAME}'"
    if [[ -n "$kill_cmd" ]]; then
      echo "running: $kill_cmd"
      if bash -lc "$kill_cmd" 2>&1; then
        echo "kill command OK"
      else
        echo "KILL COMMAND FAILED -- the instance may still be billing"
      fi
    else
      echo "no kill command was armed: DELETE THE INSTANCE YOURSELF"
    fi
  } >> "$LOG" 2>&1
  if [[ -n "$kill_cmd" ]]; then
    notify "fired for ${NAME} -- check the log that it actually died"
  else
    notify "DEADLINE for ${NAME}: delete the instance in the provider portal NOW"
  fi
  # One-shot: a stale job must never kill a future instance.
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST" "$STATE"
  ;;

arm)
  [[ -n "$DEADLINE" ]] || usage
  if [[ "$DEADLINE" == *:* ]]; then
    hh="${DEADLINE%%:*}"; mm="${DEADLINE##*:}"
  else                                  # minutes from now
    hh="$(date -v "+${DEADLINE}M" '+%H')"; mm="$(date -v "+${DEADLINE}M" '+%M')"
  fi
  mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
  # The kill command lives outside the repo: it can carry an instance id or a
  # token, and neither belongs in git.
  printf '%s' "$KILL_CMD" > "$STATE"
  chmod 600 "$STATE"
  cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${SELF}</string>
    <string>fire</string>
    <string>--name</string>
    <string>${NAME}</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>$((10#$hh))</integer>
    <key>Minute</key><integer>$((10#$mm))</integer>
  </dict>
  <key>StandardOutPath</key><string>${LOG}</string>
  <key>StandardErrorPath</key><string>${LOG}</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>${HOME}/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string></dict>
</dict>
</plist>
PLIST_EOF
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  printf 'armed "%s" for %02d:%02d\n' "$NAME" "$((10#$hh))" "$((10#$mm))"
  if [[ -n "$KILL_CMD" ]]; then
    # Redact anything that looks like a key before echoing: kill commands get
    # pasted into terminals, transcripts and screen shares, and a 32+ char
    # hex blob in one is a leaked credential (learned the hard way).
    echo "  will run: $(printf '%s' "$KILL_CMD" | sed -E 's/[0-9a-fA-F]{32,}/<redacted>/g')"
  else
    echo "  NO kill command: it will only shout at you. Provider portals that"
    echo "  bill until delete need a real command here."
  fi
  echo "  log:    $LOG"
  echo "  disarm: $0 disarm --name $NAME"
  ;;

status)
  # Two independent facts, because "launchctl has not registered it yet" and
  # "nothing is scheduled" look identical if you only ask launchd, and the
  # difference is whether a rented machine is unprotected.
  loaded=no; scheduled=no
  launchctl list 2>/dev/null | grep -q "$LABEL" && loaded=yes
  [[ -f "$PLIST" ]] && scheduled=yes
  if [[ "$scheduled" == yes ]]; then
    echo "ARMED: $NAME (plist present, launchd registered: $loaded)"
    grep -A3 StartCalendarInterval "$PLIST" 2>/dev/null | grep -oE '<integer>[0-9]+</integer>' \
      | sed -E 's/[^0-9]//g' | paste -sd: - | sed 's/^/  at /'
    echo "  kill: $(cat "$STATE" 2>/dev/null || echo '(none -- reminder only)')"
    [[ "$loaded" == no ]] && echo "  WARNING: scheduled but not loaded -- re-arm it"
  else
    echo "NOT ARMED: $NAME -- nothing will stop a running rental"
  fi
  [[ -f "$LOG" ]] && { echo "--- last log:"; tail -3 "$LOG"; }
  ;;

disarm)
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST" "$STATE"
  echo "disarmed: $NAME"
  ;;

*) usage ;;
esac
