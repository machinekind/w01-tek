#!/usr/bin/env bash
# On the camera Pi: the D435 and a read-only foxglove_bridge that the DGX's
# relay reads (wojtek_link.relay_node --camera-url ws://<camera Pi>:8765).
#
#   bash run.sh          # foreground; Ctrl-C (or the SSH session ending) stops both
#
# Installs nothing and changes no configuration: it runs two nodes from the
# Pi's /opt/ros/jazzy and stops them on exit. Nothing here touches the
# robot's Pi.
#
# The camera Pi's ROS graph is localhost only on its own domain: no DDS
# traffic joins the robot's graph or the DGX's over the LAN; the DGX reads
# the pictures only through the bridge's websocket.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WOJTEK_CAMERA_DOMAIN="${WOJTEK_CAMERA_DOMAIN:-43}"   # the camera Pi's own graph (the DGX's is 44)
WOJTEK_CAMERA_PIN="${WOJTEK_CAMERA_PIN:-auto}"       # auto: the isolated cores, if any; 0: no pinning

set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="$WOJTEK_CAMERA_DOMAIN"

# The Wojtek image isolates cores (isolcpus) for the control loop, which a
# camera Pi does not run, and nproc sees only the rest. Give the driver and
# the bridge an isolated core each rather than crowd them onto the wifi's.
iso="$(cat /sys/devices/system/cpu/isolated 2>/dev/null || true)"
cam_pin=()
bridge_pin=()
if [[ "$WOJTEK_CAMERA_PIN" != 0 ]]; then
  cpus=()
  for part in ${iso//,/ }; do
    if [[ "$part" == *-* ]]; then
      for ((c = ${part%-*}; c <= ${part#*-}; c++)); do cpus+=("$c"); done
    else
      cpus+=("$part")
    fi
  done
  if (( ${#cpus[@]} >= 2 )); then
    cam_pin=(taskset -c "${cpus[0]}")
    bridge_pin=(taskset -c "${cpus[1]}")
  fi
fi
echo "camera Pi: domain $ROS_DOMAIN_ID, localhost only; isolated cores: '${iso}';" \
     "driver ${cam_pin[*]:-unpinned}, bridge ${bridge_pin[*]:-unpinned}"

# The executables themselves, not `ros2 run`: a signal then reaches the node.
rs="$(ros2 pkg prefix realsense2_camera)/lib/realsense2_camera/realsense2_camera_node"
fb="$(ros2 pkg prefix foxglove_bridge)/lib/foxglove_bridge/foxglove_bridge"

pids=()
stop() {
  trap - EXIT INT TERM HUP
  echo "stopping the camera and the bridge"
  kill -INT "${pids[@]}" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "${pids[@]}" 2>/dev/null || break
    sleep 0.5
  done
  kill -9 "${pids[@]}" 2>/dev/null || true
}
trap stop EXIT INT TERM HUP

"${cam_pin[@]}" "$rs" --ros-args -r __ns:=/camera -r __node:=camera \
  --params-file "$HERE/realsense.yaml" &
pids+=("$!")
"${bridge_pin[@]}" "$fb" --ros-args --params-file "$HERE/bridge.yaml" &
pids+=("$!")
echo "running: driver pid ${pids[0]}, bridge pid ${pids[1]} on :8765 (read only)"
wait -n "${pids[@]}" || true
echo "one of the two exited"
