#!/bin/bash
set -e
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash

# Host ~/.ssh is bind-mounted read-only at /host-ssh, still owned by the host
# user's uid. We run as root in here, and OpenSSH refuses to read config/key
# files it doesn't own ("Bad owner or permissions ..."). Copy into a
# root-owned /root/.ssh once per container start so `ssh`/`rsync` work the
# same as they do on the host (auth itself still goes through the forwarded
# agent, see SSH_AUTH_SOCK -- this copy is just to satisfy ssh's file checks).
if [ -d /host-ssh ] && [ ! -e /root/.ssh/.synced-from-host ]; then
    mkdir -p /root/.ssh
    cp -a /host-ssh/. /root/.ssh/
    chown -R root:root /root/.ssh
    chmod 700 /root/.ssh
    find /root/.ssh -type f -exec chmod 600 {} \;
    touch /root/.ssh/.synced-from-host
fi

exec "$@"
