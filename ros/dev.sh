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

# Compose file stack: macOS needs bridge networking + the published Foxglove
# port (compose.mac.yaml), an NVIDIA runtime gets hardware GL (compose.gpu.yaml).
COMPOSE=(docker compose)
if [ "$(uname -s)" = "Darwin" ]; then
  COMPOSE+=(-f compose.yaml -f compose.mac.yaml)
elif docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
  COMPOSE+=(-f compose.yaml -f compose.gpu.yaml)
fi

"${COMPOSE[@]}" up -d

# Personal values the launches read, carried in from the host environment or
# the gitignored repo-root .env (see .env.example): the keeper org of the
# pinned default policy and the token that downloads it, the VLM brain's
# model server. Same list and same rule as sim.sh; `docker exec` passes no
# host environment on its own.
DOCKER_ENV=()
for var in HF_ORGANIZATION HF_TOKEN WOJTEK_POLICY VLM_URL VLM_MODEL VLLM_API_KEY; do
  if [ -z "${!var:-}" ]; then
    for envfile in ../../.env ../.env; do
      [ -f "$envfile" ] || continue
      # `|| true`: a key absent from the file is the normal case, not an
      # error for set -e/pipefail to kill the script on (it did, silently).
      val=$({ grep -E "^${var}=" "$envfile" || true; } | tail -1 | cut -d= -f2- | tr -d '"'"'")
      if [ -n "$val" ]; then export "$var=$val"; break; fi
    done
  fi
  # The training tools take a host path in WOJTEK_POLICY too (an export
  # dir, a policy.npz); the container cannot see host paths, so only a
  # Hugging Face reference (org/name[@rev]) goes in.
  if [ "$var" = WOJTEK_POLICY ]; then case "${WOJTEK_POLICY:-}" in
    /*|.*|~*) echo ">> WOJTEK_POLICY is a host path -- not forwarded; the launch runs the pin (or pass policy:=)"
              unset WOJTEK_POLICY ;;
  esac; fi
  [ -n "${!var:-}" ] && DOCKER_ENV+=(-e "$var=${!var}")
done
if [ -z "${HF_ORGANIZATION:-}" ] && [ -z "${WOJTEK_POLICY:-}" ] && [ ! -s ../policy_override ]; then
  echo "!! neither HF_ORGANIZATION nor WOJTEK_POLICY set (repo-root .env), no ros/policy_override: launches need an explicit policy:=" >&2
fi

exec docker exec -it ${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"} wojtek_robot bash
