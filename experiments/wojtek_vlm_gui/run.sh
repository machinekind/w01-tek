#!/usr/bin/env bash
# Entry point for the Wojtek VLM GUI experiment. Self-contained on purpose:
# nothing outside this directory is configured to know about it. See README.md.
#
# Everything runs in the wojtek_vlm_gui container (docker/compose.yaml), a
# sibling of wojtek_robot on the same host network and DDS domain. Start the
# simulation first, from the repository root:
#   ./ros/sim.sh model_xml:=scene_nav.xml leg_odom:=true nav:=true vlm:=true
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
CONTAINER=wojtek_vlm_gui

usage() {
  cat >&2 <<'USAGE'
usage: run.sh {build|up|down|shell|gui|test} [args]
  build    build the wojtek_vlm_gui image (once per machine)
  up       start the container (idempotent)
  down     stop and remove it
  shell    interactive shell inside it (ROS 2 sourced, venv on PATH)
  gui      the operator page (JS, served by wojtek_vlm_gui.server) on http://localhost:8501
           (this computer only; `run.sh gui --host 0.0.0.0` on Linux, or edit the ports
           line in docker/compose.mac.yaml on a Mac, to reach it from another device);
           ROS_DOMAIN_ID from this shell is passed in (the cable bench uses 43)
  walker   the bench's legs: /cmd_vel -> odom->base_link + the page's Walk guide
           (a person carries the camera). On the real robot without leg odometry:
           run.sh walker --ros-args -p role:=odometry (never with leg odometry on)
  test     model-free unit tests (EXP_PY=<host python> to run outside docker)
USAGE
}

# Compose file stack, the same choice ros/sim.sh makes: on macOS Docker
# runs in a VM and the sim is on the bridge network, so the page joins that
# bridge with its port published (compose.mac.yaml); Linux keeps host
# networking next to the sim.
COMPOSE=(-f compose.yaml)
if [ "$(uname -s)" = "Darwin" ]; then
  COMPOSE+=(-f compose.mac.yaml)
fi

# ROS_LOCALHOST_ONLY: compose takes it only from this shell (the .env it
# reads by itself is docker/.env here), but ros/sim.sh also falls back to the
# repo-root .env and then ros/.env. Look in the same places, so this
# container and wojtek_robot always get the same value.
if [ -z "${ROS_LOCALHOST_ONLY:-}" ]; then
  for envfile in "$HERE/../../.env" "$HERE/../../ros/.env"; do
    [ -f "$envfile" ] || continue
    # `|| true`: a key missing from the file is normal, not a reason for
    # set -e and pipefail to stop the script. Quotes, a Windows line end and
    # spaces around the value are cut off, the same as forward_env.sh does.
    val=$({ grep -E '^ROS_LOCALHOST_ONLY=' "$envfile" || true; } | tail -1 | cut -d= -f2- \
      | tr -d '\r"'"'" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    if [ -n "$val" ]; then
      export ROS_LOCALHOST_ONLY="$val"
      break
    fi
  done
fi

compose() {
  (cd "$HERE/docker" && docker compose "${COMPOSE[@]}" "$@")
}

up() {
  compose up -d wojtek_vlm_gui
}

# Runs a command inside the container through the image entrypoint, which
# sources ROS 2 and cds into /exp.
cexec() {
  docker exec -i "$CONTAINER" /entrypoint.sh "$@"
}

case "${1:-}" in
  build)
    shift
    compose build "$@"
    ;;
  up)
    up
    ;;
  down)
    compose down
    ;;
  shell)
    up
    exec docker exec -it "$CONTAINER" /entrypoint.sh bash
    ;;
  gui)
    shift
    up
    # On a Mac the container sits on Docker's bridge network, so the server
    # listens on the container's own address and the ports line in
    # compose.mac.yaml keeps it to the Mac itself. On Linux the container
    # shares this computer's network and the server's default (this
    # computer only) applies. The operator's own --host comes later on the
    # line and wins.
    GUI_ARGS=(python3 -m wojtek_vlm_gui.server --port 8501)
    if [ "$(uname -s)" = "Darwin" ]; then
      GUI_ARGS+=(--host 0.0.0.0)
    fi
    # -i only (no -t): works from a plain terminal and from a background job.
    exec docker exec -i -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" "$CONTAINER" /entrypoint.sh \
      "${GUI_ARGS[@]}" "$@"
    ;;
  walker)
    shift
    up
    exec docker exec -i -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" "$CONTAINER" /entrypoint.sh \
      python3 -m wojtek_vlm_gui.walker_node "$@"
    ;;
  test)
    shift
    if [ -n "${EXP_PY:-}" ]; then
      exec "$EXP_PY" -m pytest tests -q "$@"
    fi
    up
    # The venv sees the system site-packages, where ros-dev-tools' apt
    # pytest plugins (launch_testing) may be built against an older pytest
    # and break entry-point autoload; load only the plugin this suite needs.
    cexec env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p pytest_timeout tests -q "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
