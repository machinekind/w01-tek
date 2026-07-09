#!/usr/bin/env bash
# Build the ros/ workspace Docker image. Only needs Docker on the host - see
# docker/Dockerfile for what's inside (ROS 2 Jazzy + rosdep-resolved deps,
# built once at image-build time).
set -eo pipefail
cd "$(dirname "$0")/docker"
docker compose build "$@"

cat <<'EOF'

Build done. Next step:
  ./dev.sh   # starts the container if needed, then drops you into a shell
             # (ROS already sourced - e.g. ros2 launch piesek_bringup real.launch.py dry_run:=true)
EOF
