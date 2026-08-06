#!/usr/bin/env bash
# One command for the full local simulation: MuJoCo sim + policy + viz +
# operator console (drive pad), all in the dev container. Works on Linux and
# macOS; first run builds the Docker image (slow once), after that it starts
# in seconds.
#
# The console is the WEB one by default -- http://localhost:8080 -- so no X
# server is ever required for driving. Visualization:
#
#   ./sim.sh                Linux: RViz (X11) + web console
#                           macOS: foxglove_bridge -> open the Foxglove app
#                                  and connect to ws://localhost:8765
#                                  + web console on http://localhost:8080
#   ./sim.sh --foxglove     force Foxglove on Linux too
#   ./sim.sh --rviz         force X11 RViz on macOS (slow; needs XQuartz)
#   ./sim.sh --qt-console   the Qt operator console instead of the web one
#                           (needs X11; on macOS: XQuartz, see below)
#   ./sim.sh --gamepad      drive with a bluetooth Xbox pad instead of the web
#                           console (left stick vx/yaw, right stick strafe,
#                           A arms, D-pad height). Pair the pad on the HOST
#                           first. Linux: joy driver in the container via the
#                           /dev mount. macOS: Docker can't pass input devices,
#                           so this falls back to the web console, which reads
#                           the pad in the browser (same mapping) -- press any
#                           pad button on the page to activate.
#
# macOS one-time setup, only for the X11 paths (--rviz / --qt-console):
#   brew install --cask xquartz
#   defaults write org.xquartz.X11 nolisten_tcp -bool false   # allow TCP :6000
# After that this script starts XQuartz itself when needed.
#
# Anything else is passed through to `ros2 run wojtek_bringup robot --sim`
# (e.g. --plotjuggler, --no-viz, --no-console) -- including ros2-launch
# name:=value arguments, which robot.py forwards to sim.launch.py:
#
#   ./sim.sh camera:=false      # no virtual D435 (default on; the camera
#                               # publishes /camera/camera/depth/* + color
#                               # like the real perception stack)
#   ./sim.sh boot_pose:=folded  # start in the robot's boot/zeroing pose
#   ./sim.sh hw:=mujoco         # physics plant (hw:=mock is the default:
#                               # the full node graph, no dynamics)
#
# Since 2026-08-06 the simulation IS the robot bringup with the hardware
# plugin swapped, so the startup procedure is the robot's own: zero ->
# stand_up -> arm (see docs/sim-test-contract.md). The old sim-only
# `initial_pose:=` is now `boot_pose:=`, the same argument the robot takes.
#
# Ctrl-C tears the session down; the container stays up for reuse (same one
# ./dev.sh attaches to).
set -eo pipefail
cd "$(dirname "$0")/docker"

VIZ=auto
CONSOLE=web
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --rviz) VIZ=rviz ;;
    --foxglove) VIZ=foxglove ;;
    --qt-console) CONSOLE=qt ;;
    --gamepad) CONSOLE=gamepad ;;
    --no-console) CONSOLE=none ;;
    *) EXTRA+=("$arg") ;;
  esac
done

MAC=false
if [ "$(uname -s)" = "Darwin" ]; then
  MAC=true
fi

if [ "$CONSOLE" = gamepad ] && $MAC; then
  # Docker on macOS can't pass input devices into the container, but the web
  # console reads the pad in the BROWSER (Gamepad API) with the same mapping
  # (left stick vx/yaw, right stick strafe, A arms, D-pad height).
  echo ">> macOS: the container can't see the pad -- using the web console,"
  echo ">> which reads it in the browser. Pair the pad with this Mac, open"
  echo ">> http://localhost:8080 and press any pad button to activate."
  CONSOLE=web
fi

COMPOSE=(docker compose)
if $MAC; then
  # Bridge networking + published Foxglove/console ports; see compose.mac.yaml.
  COMPOSE+=(-f compose.yaml -f compose.mac.yaml)
elif docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
  # Hardware GL/EGL in the container (RViz + the sim camera) instead of the
  # llvmpipe fallback that renders everything on the CPU; see compose.gpu.yaml.
  COMPOSE+=(-f compose.yaml -f compose.gpu.yaml)
fi

# ---- X server (only RViz and the Qt console need one) -----------------------
HAVE_X=false
DOCKER_ENV=()
if $MAC; then
  # Only the X11 paths (--rviz / --qt-console) need XQuartz -- the default
  # Foxglove + web console session must not start it. XQuartz serves X over
  # TCP :6000 (needs the one-time nolisten_tcp setup above); start it if
  # it's installed but not yet listening.
  if { [ "$VIZ" = rviz ] || [ "$CONSOLE" = qt ]; } && [ -x /opt/X11/bin/xhost ]; then
    if ! nc -z 127.0.0.1 6000 >/dev/null 2>&1; then
      open -a XQuartz || true
      for _ in $(seq 1 10); do
        nc -z 127.0.0.1 6000 >/dev/null 2>&1 && break
        sleep 1
      done
    fi
    if nc -z 127.0.0.1 6000 >/dev/null 2>&1; then
      # Container connections arrive via the Docker VM as 127.0.0.1. xhost
      # itself talks to XQuartz over the local unix socket (:0) -- the shell's
      # DISPLAY is typically unset until the next login after installing
      # XQuartz, and xhost silently does nothing without one.
      DISPLAY="${DISPLAY:-:0}" /opt/X11/bin/xhost +127.0.0.1 >/dev/null 2>&1 || true
      # MIT-SHM can't cross the TCP link; Qt/X11 must not try it.
      DOCKER_ENV=(-e DISPLAY=host.docker.internal:0 -e QT_X11_NO_MITSHM=1)
      HAVE_X=true
    elif [ "$VIZ" = rviz ] || [ "$CONSOLE" = qt ]; then
      echo ">> XQuartz is installed but not listening on :6000 -- run once:"
      echo ">>   defaults write org.xquartz.X11 nolisten_tcp -bool false"
      echo ">> then quit and reopen XQuartz."
    fi
  fi
elif [ -n "${DISPLAY:-}" ]; then
  command -v xhost >/dev/null 2>&1 && xhost +local:docker >/dev/null 2>&1 || true
  HAVE_X=true
fi

if [ "$VIZ" = auto ]; then
  # macOS: X11 RViz (GL over XQuartz) is slow at best -- default to the
  # native Foxglove app even when XQuartz is around. Headless Linux gets
  # Foxglove too; X11 Linux keeps the existing RViz setup.
  if $MAC || ! $HAVE_X; then
    VIZ=foxglove
  else
    VIZ=rviz
  fi
fi

if { [ "$VIZ" = rviz ] || [ "$CONSOLE" = qt ]; } && ! $HAVE_X; then
  echo "!! --rviz/--qt-console need an X server (macOS: XQuartz, see header of this script)" >&2
  exit 1
fi

CMD=(ros2 run wojtek_bringup robot --sim)
if [ "$VIZ" = foxglove ]; then CMD+=(--foxglove); fi
if [ "$CONSOLE" = web ]; then CMD+=(--web-console); fi
if [ "$CONSOLE" = gamepad ]; then CMD+=(--no-console --gamepad); fi
if [ "$CONSOLE" = none ]; then CMD+=(--no-console); fi

"${COMPOSE[@]}" up -d

if [ "$VIZ" = foxglove ]; then
  echo ">> Foxglove: open the app and connect to ws://localhost:8765"
fi
if [ "$CONSOLE" = web ]; then
  echo ">> console:  open http://localhost:8080 (drive pad, arm/pose buttons)"
fi
if [ "$CONSOLE" = gamepad ]; then
  echo ">> gamepad:  left stick vx/yaw, right stick strafe, A arms, D-pad height"
fi

# docker exec bypasses the entrypoint and non-interactive bash skips .bashrc,
# so source ROS + the workspace overlay explicitly (single-quoted: expanded
# inside the container, where ROS_DISTRO comes from the base image).
exec docker exec -it ${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"} wojtek_robot bash -c '
  source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
  source /ros2_ws/install/setup.bash
  exec "$@"' _ "${CMD[@]}" "${EXTRA[@]}"
