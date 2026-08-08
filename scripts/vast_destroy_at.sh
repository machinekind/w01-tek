#!/usr/bin/env bash
# Guarantee a vast instance dies at a wall-clock time, without depending on
# any shell staying open.
#
#   scripts/vast_destroy_at.sh install <instance-id> [HH:MM]   # default 23:55
#   scripts/vast_destroy_at.sh run     <instance-id>           # destroy now
#   scripts/vast_destroy_at.sh status
#   scripts/vast_destroy_at.sh cancel
#
# Why launchd and not `sleep && destroy` in a terminal: a backgrounded sleep
# dies with its session, and vast keeps billing. Why not cron: cron silently
# skips a job whose time passed while the Mac was asleep, which is exactly
# the case that costs money overnight. launchd re-fires a missed
# StartCalendarInterval job on wake.
#
# vast has no server-side scheduled destroy (`vastai` exposes only show/delete
# scheduled-job, no create), so this is client-side by necessity. If the Mac
# is off at the appointed time the instance keeps running until the machine
# wakes -- for a hard guarantee, top-up limits are the only server-side lever.
set -euo pipefail

LABEL="com.wojtek.vast-destroy"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="$HOME/Library/Logs/wojtek-vast-destroy.log"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

cmd="${1:-status}"
case "$cmd" in

run)
  id="${2:?usage: $0 run <instance-id>}"
  {
    echo "=== $(date '+%F %T %Z') destroying vast instance $id"
    yes | vastai destroy instance "$id" 2>&1 || echo "destroy call failed"
    sleep 5
    left="$(vastai show instances --raw 2>/dev/null | tr -d '[:space:]')"
    echo "instances after: ${left:-?}"
    vastai show user --raw 2>/dev/null | python3 -c \
      'import json,sys; print("credit left:", json.load(sys.stdin).get("credit"))' 2>/dev/null || true
  } >> "$LOG" 2>&1
  # One-shot: unload so a stale job cannot destroy a FUTURE instance that
  # happens to reuse this id.
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  ;;

install)
  id="${2:?usage: $0 install <instance-id> [HH:MM]}"
  when="${3:-23:55}"
  hh="${when%%:*}"; mm="${when##*:}"
  mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
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
    <string>run</string>
    <string>${id}</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>${hh#0}</integer>
    <key>Minute</key><integer>${mm#0}</integer>
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
  echo "scheduled: destroy instance ${id} at ${when} daily-until-fired"
  echo "  log:    $LOG"
  echo "  cancel: $0 cancel"
  ;;

status)
  if launchctl list | grep -q "$LABEL"; then
    echo "ARMED:"
    grep -A2 -E "StartCalendarInterval|<string>[0-9]{6,}</string>" "$PLIST" 2>/dev/null | tr -d '\n' | sed 's/  */ /g'
    echo
    grep -oE "<string>[0-9]{6,}</string>" "$PLIST" 2>/dev/null | head -1
  else
    echo "not armed"
  fi
  echo "--- live instances:"
  vastai show instances --raw 2>/dev/null | head -c 120
  echo
  [ -f "$LOG" ] && { echo "--- last log:"; tail -3 "$LOG"; }
  ;;

cancel)
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "cancelled"
  ;;

*)
  echo "usage: $0 {install <id> [HH:MM]|run <id>|status|cancel}" >&2
  exit 1
  ;;
esac
