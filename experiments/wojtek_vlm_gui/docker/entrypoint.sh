#!/usr/bin/env bash
# Sources ROS 2, puts the GUI venv first on PATH, then runs the given
# command. Used both as the image ENTRYPOINT and by run.sh's `docker exec`
# calls, so there is exactly one place that knows the sourcing order.
set -eo pipefail
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
export PATH="${GUI_VENV:-/opt/gui-venv}/bin:${PATH}"
export PYTHONPATH="/exp${PYTHONPATH:+:$PYTHONPATH}"
cd /exp
exec "$@"
