#!/bin/bash
set -e
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash
exec "$@"
