#!/usr/bin/env bash
# Spend cap for a rented Vast.ai instance: destroy it once the deadline passes.
#
# Polls wall-clock every 30 s instead of sleeping once for the whole window, so a
# laptop suspend cannot stretch the deadline. Reads the API key at fire time, so
# the key is never copied into this script or its log.
#
#   INSTANCE_ID=NNNNNNNN MINUTES=90 ./vast_autodestroy.sh &
#
#   # extend (any later epoch, any time)
#   echo $(( $(date +%s) + 3600 )) > "$STATE_DIR/vast_deadline_epoch"
#   # cancel
#   rm "$STATE_DIR/vast_deadline_epoch"
#
# It fires exactly on time with no warning. Size the window for the whole
# session: re-renting costs a fresh ~9 min weight download.
set -uo pipefail

INSTANCE_ID=${INSTANCE_ID:?"set INSTANCE_ID"}
MINUTES=${MINUTES:-60}
ENV_FILE=${ENV_FILE:-"$PWD/.env"}          # must define VAST_API_KEY=
STATE_DIR=${STATE_DIR:-"$(cd "$(dirname "$0")" && pwd)"}
POLL_S=${POLL_S:-30}

DEADLINE_FILE="$STATE_DIR/vast_deadline_epoch"
LOG="$STATE_DIR/vast_autodestroy.log"

log() { echo "[$(date -u +%FT%TZ)] $*" >>"$LOG"; }

[ -f "$DEADLINE_FILE" ] || echo $(( $(date +%s) + MINUTES * 60 )) >"$DEADLINE_FILE"
log "armed for instance $INSTANCE_ID, deadline $(cat "$DEADLINE_FILE")"

while :; do
    if [ ! -f "$DEADLINE_FILE" ]; then
        log "deadline file removed -- watchdog cancelled, instance left running"
        exit 0
    fi
    deadline=$(cat "$DEADLINE_FILE")
    if [ "$(date +%s)" -ge "$deadline" ]; then
        key=$(grep '^VAST_API_KEY=' "$ENV_FILE" | cut -d= -f2)
        resp=$(curl -s -X DELETE -H "Authorization: Bearer $key" \
            -H 'Content-Type: application/json' \
            "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" -d '{}')
        log "deadline reached -- destroy response: $resp"
        exit 0
    fi
    sleep "$POLL_S"
done
