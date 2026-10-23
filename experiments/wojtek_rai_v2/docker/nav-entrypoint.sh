#!/usr/bin/env bash
# Sources ROS 2, puts the experiment on PYTHONPATH, runs the given command.
set -eo pipefail
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
export PYTHONPATH="/exp${PYTHONPATH:+:$PYTHONPATH}"
cd /exp
exec "$@"
