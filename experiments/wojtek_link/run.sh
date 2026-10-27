#!/usr/bin/env bash
# The DGX side of the link in one command: the relay (the robot, and the
# camera Pi), the pixel resolver, the brain and the operator page, all on
# the DGX's own ROS domain, localhost only. Run it on the brain's host.
#
#   experiments/wojtek_link/run.sh up [--send] [--in-camera]
#                    start; RECEIVE ONLY unless --send (then the page and
#                    the brain drive the robot). --in-camera: the camera
#                    through the robot's bridge, the sim test only
#   experiments/wojtek_link/run.sh down      stop all of it
#   experiments/wojtek_link/run.sh status    what runs, both links, the page
#   experiments/wojtek_link/run.sh test      the model-free tests, in the image
#
# Addresses come from the environment or a gitignored .env (this
# directory's, then the repo root's), never from this file:
#   WOJTEK_BRIDGE_URL   the robot's foxglove_bridge, ws://<robot>:8765 (required)
#   WOJTEK_CAMERA_URL   the camera Pi's bridge, ws://<camera Pi>:8765 (optional)
# With defaults:
#   WOJTEK_REPO         a checkout with experiments/wojtek_vlm_gui (default: this one)
#   WOJTEK_LINK_DOMAIN  the DGX's ROS domain (44)
#   WOJTEK_PAGE_PORT    the page, on 127.0.0.1 only (8501): ssh -L 8501:127.0.0.1:8501 <host>
#   WOJTEK_IMAGE        the robot image (docker-wojtek_robot:latest)
#   VLM_URL, VLM_MODEL  the brain's model server, as for ros/sim.sh (default: local Ollama)
#
# It changes nothing on the robot or the camera Pi, and runs the relay
# exactly as tested (python3 -m wojtek_link.relay_node, this directory).
# The resolver and the brain run in the wojtek_robot container, whose
# workspace is built (create it once with ./ros/dev.sh or ./ros/sim.sh);
# only the processes this script started are stopped by `down`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINK_CT=wojtek_link          # the relay and the page
ROBOT_CT=wojtek_robot        # the resolver and the brain (the built workspace)

# KEY from the environment, else the last KEY= line of the .env files, else a default.
setting() {
  local key="$1" default="${2-}" f line
  if [[ -n "${!key-}" ]]; then printf '%s' "${!key}"; return; fi
  for f in "$HERE/.env" "${REPO_FOR_ENV:-}/.env"; do
    [[ -f "$f" ]] || continue
    line="$(grep -E "^${key}=" "$f" | tail -1 || true)"
    if [[ -n "$line" ]]; then printf '%s' "${line#*=}"; return; fi
  done
  printf '%s' "$default"
}

default_repo="$(cd "$HERE/../.." && pwd)"
REPO_FOR_ENV="$default_repo"
REPO="$(setting WOJTEK_REPO "$default_repo")"
REPO_FOR_ENV="$REPO"
IMAGE="$(setting WOJTEK_IMAGE docker-wojtek_robot:latest)"
DOMAIN="$(setting WOJTEK_LINK_DOMAIN 44)"
PORT="$(setting WOJTEK_PAGE_PORT 8501)"
BRIDGE_URL="$(setting WOJTEK_BRIDGE_URL)"
CAMERA_URL="$(setting WOJTEK_CAMERA_URL)"
VLM_URL_V="$(setting VLM_URL)"
VLM_MODEL_V="$(setting VLM_MODEL)"
ROS_ENV="source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=$DOMAIN ROS_LOCALHOST_ONLY=1;"
QUIET='ROS_LOCALHOST_ONLY is deprecated|localhost_only. is enabled'

die() { echo "run.sh: $*" >&2; exit 1; }
running() { [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == true ]]; }

link_status() {   # the relay's own /wojtek/link/status, summarised
  docker exec "$LINK_CT" bash -c "$ROS_ENV timeout 5 ros2 topic echo --once --full-length \
    /wojtek/link/status std_msgs/msg/String --field data 2>/dev/null" \
  | python3 -c '
import json, sys
line = next((l for l in sys.stdin if l.lstrip().startswith("{")), None)
if line is None:
    print("  link status: none (the relay is not running?)"); sys.exit()
s = json.loads(line)
print("  robot : connected %s | send %s | odom msgs %s | %s kB/s | clock %s ms | %s" % (
    s["connected"], s["send"], s["in"].get("/wojtek/odom", 0), s["kB_s_in"], s["clock_offset_ms"],
    s["error"] or "ok"))
c = s.get("camera")
if c:
    print("  camera: connected %s | %s kB/s | clock %s ms | %s" % (
        c["connected"], c["kB_s_in"], c["clock_offset_ms"], c["error"] or "ok"))
print("  camera mount from the robot: %s | out %s | dropped %s" % (s["camera_mount"], s["out"], s["dropped_out"]))'
}

cmd_status() {
  local ct st
  for ct in "$LINK_CT" "$ROBOT_CT"; do
    st="$(docker inspect -f '{{.State.Status}}' "$ct" 2>/dev/null | tr -d '\n')" || true
    echo "container $ct: ${st:-absent}"
  done
  if running "$LINK_CT"; then
    docker exec "$LINK_CT" bash -c 'ps -eo args | grep -E "^python3 -m [w]ojtek_(link.relay_node|vlm_gui.server)" | sed "s/^/  /" | cut -c1-160' || true
    link_status
  fi
  if running "$ROBOT_CT"; then
    docker exec "$ROBOT_CT" bash -c 'for f in /tmp/wojtek_link_resolver.pgid /tmp/wojtek_link_brain.pgid; do
      [ -f "$f" ] && { g=$(cat "$f"); kill -0 -"$g" 2>/dev/null && echo "  $(basename "$f" .pgid): running (group $g)" || echo "  $(basename "$f" .pgid): stopped"; }; done; true'
  fi
  echo "  page: $(curl -s -m 3 "http://127.0.0.1:$PORT/health" | cut -c1-120 || true)"
}

cmd_down() {
  if running "$LINK_CT"; then
    docker exec "$LINK_CT" bash -c 'pkill -TERM -f "^python3 -m [w]ojtek_link.relay_node"; pkill -TERM -f "^python3 -m [w]ojtek_vlm_gui.server"; sleep 2; true'
  fi
  docker rm -f "$LINK_CT" >/dev/null 2>&1 || true
  if running "$ROBOT_CT"; then
    docker exec "$ROBOT_CT" bash -c 'for f in /tmp/wojtek_link_resolver.pgid /tmp/wojtek_link_brain.pgid; do
      [ -f "$f" ] && { kill -INT -"$(cat "$f")" 2>/dev/null; rm -f "$f"; }; done; true'
  fi
  echo "down: the relay, the page, the resolver and the brain are stopped (nothing else touched)"
}

cmd_up() {
  local send=() in_camera=0 arg
  for arg in "$@"; do
    case "$arg" in
      --send) send=(--send) ;;
      --in-camera) in_camera=1 ;;
      *) die "unknown option $arg (up takes --send, --in-camera)" ;;
    esac
  done
  [[ "$BRIDGE_URL" == ws://* ]] || die "WOJTEK_BRIDGE_URL must be ws://<robot>:8765 (environment or .env)"
  [[ -z "$CAMERA_URL" || "$CAMERA_URL" == ws://* ]] || die "WOJTEK_CAMERA_URL must be ws://<camera Pi>:8765"
  [[ -d "$REPO/experiments/wojtek_vlm_gui" ]] || die "no experiments/wojtek_vlm_gui under WOJTEK_REPO=$REPO"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "no image $IMAGE (build it with ./ros/dev.sh)"
  docker inspect "$ROBOT_CT" >/dev/null 2>&1 || die "no $ROBOT_CT container (create and build it once with ./ros/dev.sh)"
  running "$LINK_CT" && die "already up ($LINK_CT runs): ./run.sh down first"

  local relay=(--url "$BRIDGE_URL" "${send[@]}")
  if (( in_camera )); then
    relay+=(--in-camera)
  elif [[ -n "$CAMERA_URL" ]]; then
    relay+=(--camera-url "$CAMERA_URL")
  fi

  echo "== $LINK_CT: the relay and the page (domain $DOMAIN, localhost only)"
  docker run -d --name "$LINK_CT" --network host --ipc host \
    -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID="$DOMAIN" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -v "$HERE:/link:ro" -v "$REPO:/repo:ro" "$IMAGE" sleep infinity >/dev/null
  # The page's two web dependencies are not in the robot image.
  docker exec "$LINK_CT" bash -c 'pip3 install -q --break-system-packages "starlette>=0.40" "uvicorn>=0.30" 2>&1 | grep -v WARNING | tail -1; python3 -c "import starlette, uvicorn"' \
    || die "the page's dependencies did not install (pip needs the internet)"
  docker exec -d "$LINK_CT" bash -c "source /opt/ros/jazzy/setup.bash; cd /link; \
    PYTHONPATH=/link:\$PYTHONPATH setsid python3 -m wojtek_link.relay_node $(printf '%q ' "${relay[@]}") \
    > /tmp/relay.log 2>&1 < /dev/null &"

  echo "== $ROBOT_CT: the resolver and the brain (domain $DOMAIN)"
  running "$ROBOT_CT" || docker start "$ROBOT_CT" >/dev/null
  cmd_robot_stop_ours
  local vlm=()
  [[ -n "$VLM_URL_V" ]] && vlm+=(-e "VLM_URL=$VLM_URL_V")
  [[ -n "$VLM_MODEL_V" ]] && vlm+=(-e "VLM_MODEL=$VLM_MODEL_V")
  local ws="$ROS_ENV source /ros2_ws/install/setup.bash;"
  docker exec -d ${vlm[@]+"${vlm[@]}"} "$ROBOT_CT" bash -c "$ws setsid bash -c 'echo \$\$ > /tmp/wojtek_link_resolver.pgid; exec ros2 run wojtek_nav pixel_goal_node' > /tmp/wojtek_link_resolver.log 2>&1 < /dev/null &"
  docker exec -d ${vlm[@]+"${vlm[@]}"} "$ROBOT_CT" bash -c "$ws setsid bash -c 'echo \$\$ > /tmp/wojtek_link_brain.pgid; exec ros2 launch wojtek_nav brain.launch.py' > /tmp/wojtek_link_brain.log 2>&1 < /dev/null &"

  echo "== the page on 127.0.0.1:$PORT"
  docker exec -d "$LINK_CT" bash -c "source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=$DOMAIN; \
    cd /repo/experiments/wojtek_vlm_gui; PYTHONPATH=\$PWD:\$PYTHONPATH setsid python3 -m wojtek_vlm_gui.server \
    --host 127.0.0.1 --port $PORT > /tmp/page.log 2>&1 < /dev/null &"

  sleep 12
  echo "== up ($([[ ${#send[@]} -gt 0 ]] && echo 'SENDING: the page and the brain drive the robot' || echo 'receive only'))"
  docker exec "$LINK_CT" bash -c 'grep -aE "link up|link down|lacks|refuses" /tmp/relay.log | tail -4 | cut -c1-170' | grep -vE "$QUIET" || true
  docker exec "$ROBOT_CT" bash -c 'grep -ahE "pixel goal up|vlm brain up|Error|error" /tmp/wojtek_link_resolver.log /tmp/wojtek_link_brain.log | cut -c1-170 | head -4' | grep -vE "$QUIET" || true
  cmd_status
  echo "page: ssh -L $PORT:127.0.0.1:$PORT <this host>, then http://localhost:$PORT"
}

cmd_robot_stop_ours() {   # an earlier run's resolver and brain, not anything else in the container
  docker exec "$ROBOT_CT" bash -c 'for f in /tmp/wojtek_link_resolver.pgid /tmp/wojtek_link_brain.pgid; do
    [ -f "$f" ] && { kill -INT -"$(cat "$f")" 2>/dev/null; rm -f "$f"; }; done; true'
}

cmd_test() {
  docker run --rm -v "$HERE:/exp:ro" -w /exp "$IMAGE" bash -c \
    'source /opt/ros/jazzy/setup.bash; PYTHONPATH=/exp:$PYTHONPATH python3 -m pytest -q -p no:cacheprovider'
}

case "${1-}" in
  up) shift; cmd_up "$@" ;;
  down) cmd_down ;;
  status) cmd_status ;;
  test) cmd_test ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
