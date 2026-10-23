#!/usr/bin/env bash
# perf.sh -- profile the robot computer from the PC, one scenario at a time.
#
# Brings up the robot's own launch (the arguments of its wojtek-robot
# service, read from the RPi) with the motors mocked (mock_hw:=true) and
# telemetry on, adds the scenario's extras, runs probe.py on the RPi and
# saves its JSON under results/. The boot service is stopped while a
# perf stack runs and started again by `down`; the perf unit conflicts
# with it, so the two can never run side by side.
#
#   ./ros/hw_tests/perf/perf.sh sync                 # copy probe to the RPi
#   ./ros/hw_tests/perf/perf.sh run <scenario> [secs] [extra launch args...]
#   ./ros/hw_tests/perf/perf.sh up <scenario> [extra launch args...]
#   ./ros/hw_tests/perf/perf.sh probe <label> [secs] [--viewer] [--camera]
#                                                  [--topic NAME=TYPE ...]
#   ./ros/hw_tests/perf/perf.sh down                 # restore the boot service
#   ./ros/hw_tests/perf/perf.sh logs                 # tail the perf stack's log
#   ./ros/hw_tests/perf/perf.sh scenarios            # list them
#
# Nothing here arms the robot: real_io_node comes up DISARMED as always,
# and with mock_hw:=true there are no drives to arm. MOCK=0 runs the real
# drivers instead (motors powered, human present -- see ros/README.md).
# Run `down` before ros/deploy.sh: deploy restarts the boot service.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
# Local overrides (RPI_HOST, REMOTE_WS, DECK_PORT): ros/.env is the file
# deploy.sh reads (see ros/.env.example); the repository root's is honoured
# too.
# shellcheck disable=SC1091
[ -f "$REPO/.env" ] && set -a && . "$REPO/.env" && set +a
# shellcheck disable=SC1091
[ -f "$REPO/ros/.env" ] && set -a && . "$REPO/ros/.env" && set +a
RPI_HOST="${RPI_HOST:-rpi@10.42.0.2}"
RPI_ADDR="${RPI_HOST#*@}"
REMOTE_WS="${REMOTE_WS:-wojtek_ws}"       # relative to the RPi user's home
REMOTE_DIR="${REMOTE_DIR:-wojtek_perf}"     # relative to the RPi user's home
DECK_PORT="${DECK_PORT:-8090}"              # the gateway's port (deck_port:=)
UNIT="wojtek-perf-stack"
MOCK="${MOCK:-1}"
RESULTS="$HERE/results"

ssh_rpi() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$RPI_HOST" "$@"; }

# Scenario -> launch arguments on top of the service's own. Every rung
# names the extras it turns off as well as on, so a service that already
# runs the panel, the camera or the bridge cannot blur the ladder.
scenario_args() {
  case "$1" in
    stack)  echo "deck:=false deck_camera:=false deck_camera_depth:=false foxglove:=false" ;;
    deck)   echo "deck:=true deck_camera:=false deck_camera_depth:=false foxglove:=false" ;;
    camera) echo "deck:=true deck_camera:=true deck_camera_depth:=false foxglove:=false" ;;
    vlm)    echo "deck:=true deck_camera:=true deck_camera_depth:=true foxglove:=false" ;;
    full)   echo "deck:=true deck_camera:=true deck_camera_depth:=true foxglove:=true" ;;
    *) echo "unknown scenario '$1' (see: perf.sh scenarios)" >&2; return 1 ;;
  esac
}

# key:=value arguments, the last value per key wins (bash 3 compatible, no
# associative arrays: macOS ships bash 3.2).
merge_args() {
  local -a args=("$@")
  local out="" keys=" " kv k i
  for ((i = ${#args[@]} - 1; i >= 0; i--)); do
    kv="${args[$i]}"
    [ -n "$kv" ] || continue
    k="${kv%%:=*}"
    case "$keys" in *" $k "*) continue ;; esac
    keys="$keys$k "
    out="$kv${out:+ $out}"
  done
  printf '%s' "$out"
}

usage() { sed -n '2,/^set -/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

cmd_scenarios() {
  cat <<'EOF'
stack   control stack only (ros2_control, real_io, policy, pad teleop)
deck    + deck_gateway (panel, websocket), no camera
camera  + RealSense colour stream (compressed transport) feeding the gateway
vlm     camera + depth from the same node (the probe times the raw depth,
        the PC-side viewer times the colour JPEG)
full    vlm + foxglove_bridge (connect Foxglove to ws://<robot>:8765)
Every scenario runs with bag:=false; pass bag:=true to measure the recorder.
Add a viewer to any of them with `probe ... --viewer` (the PC pulls the
MJPEG stream while the probe measures); `run` does it for camera/vlm/full.
`probe ... --topic NAME=TYPE` times any extra topic raw (repeatable).
EOF
}

cmd_sync() {
  ssh_rpi "mkdir -p ~/$REMOTE_DIR"
  rsync -a "$HERE/probe.py" "$RPI_HOST:$REMOTE_DIR/"
  echo "probe synced to $RPI_HOST:~/$REMOTE_DIR"
}

cmd_up() {
  local scenario="${1:?scenario}"; shift
  local extra; extra="$(scenario_args "$scenario")"
  local mock="mock_hw:=true"
  [ "$MOCK" = "0" ] && mock=""
  cmd_sync >/dev/null
  # The service's own launch arguments, so the perf stack is the robot's
  # stack and not a copy that drifts from it. bag:=false: the service's
  # recorder would write ~200 MB/min to the card during every window.
  # ssh joins its arguments into one remote command line, hence the %q.
  local overrides
  # shellcheck disable=SC2086
  overrides="$(merge_args $mock telemetry:=true bag:=false $extra "$@")"
  ssh_rpi "bash -s -- $UNIT $(printf %q "$REMOTE_WS") $(printf %q "$overrides")" <<'REMOTE'
set -euo pipefail
unit="$1"; ws="$2"; overrides="$3"
exec_line="$(systemctl cat wojtek-robot.service | grep '^ExecStart=.\+' | tail -1)"
base_args="$(sed -n "s/.*robot\.launch\.py\([^']*\)'.*/\1/p" <<<"$exec_line")"
if [ -z "$base_args" ]; then
  echo "could not read the service's launch arguments" >&2; exit 1
fi
# An override replaces the service's value for the same key.
for kv in $overrides; do
  base_args="$(sed -E "s/(^| )${kv%%:=*}:=[^ ]*//g" <<<"$base_args")"
done
env_py="$(systemctl show wojtek-robot.service -p Environment --value)"
# Remember whether the boot service was running, for `down`.
if systemctl is-active -q wojtek-robot.service; then
  touch "/tmp/$unit.restore"
  sudo systemctl stop wojtek-robot.service
fi
sudo systemctl stop "$unit" 2>/dev/null || true
sudo systemctl reset-failed "$unit" 2>/dev/null || true
env_args=()
for kv in $env_py; do env_args+=(-E "$kv"); done
# Conflicts=: starting either unit stops the other, so the boot service's
# real drivers can never come up next to a running perf stack (the panel's
# restart button, deploy.sh and a stray `systemctl start` all go through
# systemd). The unit runs as the ssh login, in its workspace.
sudo systemd-run --unit="$unit" --uid="$(id -un)" --gid="$(id -gn)" \
  -p Conflicts=wojtek-robot.service -p Description="Wojtek perf stack" \
  -p LimitRTPRIO=99 -p LimitMEMLOCK=infinity "${env_args[@]}" \
  /usr/bin/taskset -c 2,3 /bin/bash -lc "source /opt/ros/jazzy/setup.bash && source $HOME/$ws/install/setup.bash && exec ros2 launch wojtek_bringup robot.launch.py $base_args $overrides" >/dev/null
# systemd-run only says the unit was created; a launch that dies at once
# (a mistyped argument, a package missing after a partial build) would
# leave the boot service stopped and "started" on the screen.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  if ! systemctl is-active -q "$unit"; then
    echo "$unit died right after start:" >&2
    journalctl -u "$unit" -n 30 --no-pager -o cat >&2 || true
    if [ -f "/tmp/$unit.restore" ]; then
      sudo systemctl start wojtek-robot.service && rm -f "/tmp/$unit.restore"
      echo "wojtek-robot.service started again" >&2
    fi
    exit 1
  fi
done
echo "started $unit:$base_args $overrides"
REMOTE
}

cmd_down() {
  ssh_rpi bash -s -- "$UNIT" <<'REMOTE'
set -euo pipefail
unit="$1"
sudo systemctl stop "$unit" 2>/dev/null || true
sudo systemctl reset-failed "$unit" 2>/dev/null || true
if [ -f "/tmp/$unit.restore" ]; then
  # The marker goes only once the service is back, so a failed start can
  # be retried with another `down`.
  sudo systemctl start wojtek-robot.service
  rm -f "/tmp/$unit.restore"
  echo "perf stack stopped; wojtek-robot.service started again"
else
  echo "perf stack stopped (wojtek-robot.service was not running before)"
fi
REMOTE
}

cmd_logs() { ssh_rpi "journalctl -u $UNIT -n ${1:-80} --no-pager -o cat"; }

# What a probe leaves behind if it dies half way: the background viewer and
# the temporary files. Set by cmd_probe, cleared on its way out.
PROBE_VPID=""
PROBE_TMP=()
probe_cleanup() {
  [ -n "$PROBE_VPID" ] && kill "$PROBE_VPID" 2>/dev/null || true
  [ ${#PROBE_TMP[@]} -gt 0 ] && rm -f "${PROBE_TMP[@]}" || true
}

cmd_probe() {
  local label="${1:?label}"; shift
  if ! [[ "$label" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "label must be [A-Za-z0-9._-]+ (it names the result file): '$label'" >&2
    return 1
  fi
  local secs=30
  if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then secs="$1"; shift; fi
  local viewer=0 want_topic=0
  local -a probe_flags=()
  for a in "$@"; do
    if [ "$want_topic" = 1 ]; then
      probe_flags+=(--topic "$a"); want_topic=0; continue
    fi
    case "$a" in
      --viewer) viewer=1 ;;
      --camera) probe_flags+=(--camera) ;;
      --topic) want_topic=1 ;;
      *) echo "unknown probe flag $a" >&2; return 1 ;;
    esac
  done
  if [ "$want_topic" = 1 ]; then echo "--topic needs NAME=TYPE" >&2; return 1; fi
  mkdir -p "$RESULTS"
  local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
  local out="$RESULTS/$stamp-$label.json"
  local viewer_out; viewer_out="$(mktemp)"
  PROBE_TMP=("$out.tmp" "$viewer_out")
  trap probe_cleanup EXIT
  if [ "$viewer" = 1 ]; then
    # The gateway comes up with the stack: wait for its port, then start the
    # viewer just before the window (the probe settles 5 s first). The
    # viewer's own clock starts at its first frame, so a camera still
    # warming up does not deflate its numbers.
    local deadline=$((SECONDS + 120))
    until curl -s -o /dev/null -m 2 "http://$RPI_ADDR:$DECK_PORT/"; do
      if [ $SECONDS -ge $deadline ]; then echo "gateway never came up on :$DECK_PORT" >&2; break; fi
      sleep 2
    done
    python3 "$HERE/viewer.py" "http://$RPI_ADDR:$DECK_PORT/stream.mjpg" \
      --duration "$secs" --connect-timeout "$((secs + 30))" >"$viewer_out" &
    PROBE_VPID=$!
  fi
  local since; since="$(ssh_rpi date +%s)"
  # The probe joins the graph the way the service does: same domain, RMW
  # and DDS config, read from the unit rather than copied here.
  local flags_q=""
  [ ${#probe_flags[@]} -gt 0 ] && flags_q="$(printf ' %q' "${probe_flags[@]}")"
  ssh_rpi "bash -s -- $(printf %q "$REMOTE_WS") $(printf %q "$REMOTE_DIR") $secs $(printf %q "$label")$flags_q" <<'REMOTE' >"$out.tmp"
set -euo pipefail
ws="$1"; dir="$2"; shift 2
source /opt/ros/jazzy/setup.bash
source "$HOME/$ws/install/setup.bash"
env_py="$(systemctl show wojtek-robot.service -p Environment --value)"
if [ -n "$env_py" ]; then export $env_py; fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///etc/cyclonedds-rpi.xml}"
secs="$1"; label="$2"; shift 2
exec taskset -c 0 nice -n 5 python3 "$HOME/$dir/probe.py" --duration "$secs" --label "$label" "$@"
REMOTE
  # Controller overruns and warnings the stack logged during the window.
  local overruns warns
  overruns="$(ssh_rpi "journalctl -u $UNIT --since @$since -o cat --no-pager | grep -ci \"overrun detected\"" || true)"
  warns="$(ssh_rpi "journalctl -u $UNIT --since @$since -o cat --no-pager | grep -E 'WARN|ERROR' | sed 's/\[[0-9.]*\]//' | sort | uniq -c | sort -rn | head -8" || true)"
  if [ -n "$PROBE_VPID" ]; then wait "$PROBE_VPID" || true; PROBE_VPID=""; fi
  python3 - "$out.tmp" "$out" "$viewer_out" "$overruns" "$warns" <<'PY'
import json, sys
src, dst, viewer_path, overruns, warns = sys.argv[1:6]
report = json.load(open(src))
try:
    report["viewer"] = json.load(open(viewer_path))
except (OSError, ValueError):
    pass
report["log"] = {"overrun_lines": int(overruns or 0),
                 "warnings": [w.strip() for w in warns.splitlines() if w.strip()]}
json.dump(report, open(dst, "w"), indent=1)
PY
  probe_cleanup; PROBE_TMP=(); trap - EXIT
  python3 "$HERE/compare.py" "$out"
  echo "saved $out"
  if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("stack_ready", True) else 1)' "$out"; then
    echo "!! the stack was not publishing when the window started; see: perf.sh logs" >&2
    return 1
  fi
}

cmd_run() {
  local scenario="${1:?scenario}"; shift
  local secs=30
  if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then secs="$1"; shift; fi
  # A deck_port override moves the viewer too.
  local a
  for a in "$@"; do case "$a" in deck_port:=*) DECK_PORT="${a#deck_port:=}" ;; esac; done
  cmd_up "$scenario" "$@"
  case "$scenario" in
    camera) cmd_probe "$scenario" "$secs" --viewer ;;
    vlm|full) cmd_probe "$scenario" "$secs" --viewer \
           --topic "/camera/camera/depth/image_rect_raw=sensor_msgs/msg/Image" ;;
    *) cmd_probe "$scenario" "$secs" ;;
  esac
}

case "${1:-}" in
  sync) shift; cmd_sync "$@" ;;
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  probe) shift; cmd_probe "$@" ;;
  run) shift; cmd_run "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  scenarios) cmd_scenarios ;;
  *) usage; exit 1 ;;
esac
