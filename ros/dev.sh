#!/usr/bin/env bash
# Get a shell inside the wojtek_robot dev container: builds the image and/or
# starts the container if needed (idempotent - safe to run repeatedly, a
# no-op if it's already running), then attaches.
#
# This is THE way into the workspace: enter once, work in the shell. Inside,
# ROS is already sourced (~/.bashrc in the image, since `docker exec` bypasses
# the image's ENTRYPOINT) - just run plain ROS commands, e.g.:
#   ros2 launch wojtek_pc sim.launch.py            # full simulation session
#   ros2 launch wojtek_bringup real.launch.py dry_run:=true
#   colcon build --packages-skip md80_hardware_interface
# (md80 is skipped because this bind mount hides the candle checkout the image
# built it from; its install/ from image build time stays valid.)
#
# Open a second shell the same way for service calls, topic echoes and teleop
# while a launch runs in the first.
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

# A running container means someone's session may be inside: attach straight
# away. `compose up` on a running container is NOT a no-op when the compose
# stack differs from the one that started it (e.g. GPU overlay here vs a
# plain start elsewhere) -- it RECREATES the container, killing the session
# and discarding the build overlay.
if [ "$(docker inspect -f '{{.State.Running}}' wojtek_robot 2>/dev/null)" = "true" ]; then
  exec docker exec -it wojtek_robot bash
fi

# Compose file stack: macOS needs bridge networking + the published Foxglove
# port (compose.mac.yaml), an NVIDIA runtime gets hardware GL (compose.gpu.yaml).
# The runtime being registered is not enough -- with the kernel driver not
# loaded the GPU stack fails at container create ("Driver Not Loaded"), so
# require a working nvidia-smi too.
COMPOSE=(docker compose)
if [ "$(uname -s)" = "Darwin" ]; then
  COMPOSE+=(-f compose.yaml -f compose.mac.yaml)
elif docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' \
  && nvidia-smi -L >/dev/null 2>&1; then
  COMPOSE+=(-f compose.yaml -f compose.gpu.yaml)
fi

"${COMPOSE[@]}" up -d
exec docker exec -it wojtek_robot bash
