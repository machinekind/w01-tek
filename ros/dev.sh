#!/usr/bin/env bash
# Get a shell inside the wojtek_robot dev container: builds the image and/or
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

# Make sure the RPi SSH key is unlocked in the host agent before the container
# starts. The key (id_ed25519) has a passphrase; the container forwards this
# agent (compose.yaml) so `ros2 run wojtek_bringup robot` / deploy.sh can ssh to
# the RPi without a password. Without a loaded key ssh silently falls back to
# password auth. One passphrase prompt here, then none inside the container.
if [ -n "${SSH_AUTH_SOCK:-}" ] && command -v ssh-add >/dev/null 2>&1; then
  if ! ssh-add -l >/dev/null 2>&1; then
    echo ">> no keys in the SSH agent -- loading ~/.ssh/id_ed25519 for RPi access"
    ssh-add ~/.ssh/id_ed25519 || echo "!! ssh-add failed; SSH to the RPi may prompt for a password"
  fi
else
  echo "!! no SSH agent ($SSH_AUTH_SOCK) -- SSH to the RPi will prompt for a password"
fi

# Same file stack as ../sim.sh so the two never recreate the container back
# and forth over a config diff (macOS needs bridge networking + the published
# Foxglove port -- see compose.mac.yaml).
COMPOSE=(docker compose)
if [ "$(uname -s)" = "Darwin" ]; then
  COMPOSE+=(-f compose.yaml -f compose.mac.yaml)
fi

"${COMPOSE[@]}" up -d
exec docker exec -it wojtek_robot bash
