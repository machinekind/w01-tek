#!/usr/bin/env bash
# One command for the full local simulation, from the HOST: starts the dev
# container, sorts out X11 and the platform's viz default, then runs the
# session inside. Works on Linux and macOS; first run builds the Docker image
# (slow once), after that it starts in seconds.
#
# Everything session-side lives in sim.launch.py now -- this wrapper adds only
# what a launch cannot do from inside the container: the compose file stack,
# XQuartz/X11 access, and picking RViz vs Foxglove for the platform. All
# name:=value arguments pass straight through to sim.launch.py, so its
# vocabulary is the only one to learn (`ros2 launch wojtek_pc sim.launch.py
# --show-args` inside ./dev.sh lists everything):
#
#   ./sim.sh                    Linux: RViz (X11) + web console
#                               macOS: foxglove_bridge -> open the Foxglove app
#                                      and connect to ws://localhost:8765
#                               web console on http://localhost:8080 either way
#   ./sim.sh --foxglove         force Foxglove on Linux too
#   ./sim.sh --rviz             force X11 RViz on macOS (slow; needs XQuartz)
#   ./sim.sh --no-viz           no 3D view at all
#   ./sim.sh --plotjuggler      also open PlotJuggler
#   ./sim.sh console:=qt        the Qt operator console instead of the web one
#                               (needs X11; on macOS: XQuartz, see below)
#   ./sim.sh console:=none      no operator console
#   ./sim.sh gamepad:=true      bluetooth Xbox pad via the in-container joy
#                               driver (left stick vx/yaw, right stick strafe,
#                               A arms, D-pad height). Pair the pad on the HOST
#                               first. Linux only: Docker on macOS can't pass
#                               input devices -- there the web console reads
#                               the pad in the browser (same mapping), press
#                               any pad button on the page to activate.
#   ./sim.sh hw:=mock           no dynamics, just the node graph (hw:=mujoco,
#                               the physics plant, is the default)
#   ./sim.sh camera:=false      no virtual D435 (default on; it publishes
#                               /camera/camera/depth/* + color like the real
#                               perception stack)
#   ./sim.sh boot_pose:=folded  start in the robot's boot/zeroing pose
#   ./sim.sh policy:=<ref>      org/name[@rev] or a local artifact directory
#
# macOS one-time setup, only for the X11 paths (--rviz / console:=qt):
#   brew install --cask xquartz
#   defaults write org.xquartz.X11 nolisten_tcp -bool false   # allow TCP :6000
# After that this script starts XQuartz itself when needed.
#
# The simulation IS the robot bringup with the hardware plugin swapped, so the
# startup procedure is the robot's own: zero -> stand_up -> arm (see
# docs/sim-test-contract.md). Ctrl-C tears the session down; the container
# stays up for reuse (same one ./dev.sh attaches to).
set -eo pipefail
cd "$(dirname "$0")/docker"

VIZ=auto
PLOTJUGGLER=false
WANT_QT=false
WANT_GAMEPAD=false
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --rviz) VIZ=rviz ;;
    --foxglove) VIZ=foxglove ;;
    --no-viz) VIZ=none ;;
    --plotjuggler) PLOTJUGGLER=true ;;
    # Old-vocabulary aliases, kept so muscle memory still works; the launch
    # arguments on the right are the real interface.
    --qt-console) EXTRA+=("console:=qt") ;;
    --no-console) EXTRA+=("console:=none") ;;
    --gamepad) EXTRA+=("gamepad:=true") ;;
    *) EXTRA+=("$arg") ;;
  esac
done
for arg in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$arg" in
    console:=qt) WANT_QT=true ;;
    gamepad:=true) WANT_GAMEPAD=true ;;
  esac
done

MAC=false
if [ "$(uname -s)" = "Darwin" ]; then
  MAC=true
fi

if $WANT_GAMEPAD && $MAC; then
  # Docker on macOS can't pass input devices into the container, but the web
  # console reads the pad in the BROWSER (Gamepad API) with the same mapping.
  echo ">> macOS: the container can't see the pad -- gamepad:=true dropped."
  echo ">> The web console reads it in the browser instead: pair the pad with"
  echo ">> this Mac, open http://localhost:8080, press any pad button."
  FILTERED=()
  for arg in "${EXTRA[@]}"; do
    [ "$arg" = "gamepad:=true" ] || FILTERED+=("$arg")
  done
  EXTRA=(${FILTERED[@]+"${FILTERED[@]}"})
fi

# Compose file stack, same choice ./dev.sh makes: macOS needs bridge
# networking + published Foxglove/console ports, an NVIDIA runtime gets
# hardware GL/EGL (RViz + the sim camera) instead of the llvmpipe fallback.
COMPOSE=(docker compose)
if $MAC; then
  COMPOSE+=(-f compose.yaml -f compose.mac.yaml)
elif docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
  COMPOSE+=(-f compose.yaml -f compose.gpu.yaml)
fi

# ---- X server (only RViz and the Qt console need one) -----------------------
HAVE_X=false
DOCKER_ENV=()
if $MAC; then
  # Only the X11 paths (--rviz / console:=qt) need XQuartz -- the default
  # Foxglove + web console session must not start it. XQuartz serves X over
  # TCP :6000 (needs the one-time nolisten_tcp setup above); start it if
  # it's installed but not yet listening.
  if { [ "$VIZ" = rviz ] || $WANT_QT; } && [ -x /opt/X11/bin/xhost ]; then
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
    else
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

if { [ "$VIZ" = rviz ] || $WANT_QT; } && ! $HAVE_X; then
  echo "!! --rviz/console:=qt need an X server (macOS: XQuartz, see header of this script)" >&2
  exit 1
fi

# robot --sim orchestrates the session and its teardown: sim.launch.py (which
# owns console:= and gamepad:=) plus viz.launch.py with the sim RViz layout.
# --no-console is unconditional here -- the console is the launch's job, this
# flag only stops robot.py from starting a second one of its own.
CMD=(ros2 run wojtek_bringup robot --sim --no-console)
case "$VIZ" in
  foxglove) CMD+=(--foxglove) ;;
  none) CMD+=(--no-viz) ;;
esac
if $PLOTJUGGLER; then CMD+=(--plotjuggler); fi

"${COMPOSE[@]}" up -d

if [ "$VIZ" = foxglove ]; then
  echo ">> Foxglove: open the app and connect to ws://localhost:8765"
fi
if ! $WANT_QT && ! printf '%s\n' ${EXTRA[@]+"${EXTRA[@]}"} | grep -q '^console:=none$'; then
  echo ">> console:  open http://localhost:8080 (drive pad, arm/pose buttons)"
fi
if $WANT_GAMEPAD && ! $MAC; then
  echo ">> gamepad:  left stick vx/yaw, right stick strafe, A arms, D-pad height"
fi

# docker exec bypasses the entrypoint and non-interactive bash skips .bashrc,
# so source ROS + the workspace overlay explicitly (single-quoted: expanded
# inside the container, where ROS_DISTRO comes from the base image).
exec docker exec -it ${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"} wojtek_robot bash -c '
  source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
  source /ros2_ws/install/setup.bash
  exec "$@"' _ "${CMD[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
