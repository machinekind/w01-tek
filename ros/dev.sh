#!/usr/bin/env bash
# Get a shell inside the piesek_robot dev container: builds the image and/or
# starts the container if needed (idempotent - safe to run repeatedly, a
# no-op if it's already running), then attaches.
#
# Once inside, ROS is already sourced (~/.bashrc in the image, since `docker
# exec` bypasses the image's ENTRYPOINT) - just run plain ROS commands, e.g.:
#   ros2 launch wojtek_bringup real.launch.py dry_run:=true
#   ros2 launch wojtek_bringup sim.launch.py rviz:=false
#   colcon build   # after editing anything under src/ on the host
#
# Handles X11 access for RViz GUI (sim.launch.py's rviz:=true default)
# itself - skipped safely if there's no display (headless/SSH without X
# forwarding) or `xhost` isn't installed.
set -eo pipefail
cd "$(dirname "$0")/docker"

if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

docker compose up -d
exec docker exec -it piesek_robot bash
