# Navigation stack for the Wojtek experiment: Nav2 + slam_toolbox +
# depth-to-laserscan + the experiment's own glue nodes (odom relay, cmd_vel
# watchdog). Runs as the `wojtek_nav` compose service next to wojtek_robot
# and wojtek_rai: host network, same domain and DDS profile.
#
#   ./experiments/wojtek_rai_v2/run.sh nav build
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-navigation2 \
        ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-slam-toolbox \
        ros-${ROS_DISTRO}-depthimage-to-laserscan \
        ros-${ROS_DISTRO}-depth-image-proc \
        ros-${ROS_DISTRO}-pointcloud-to-laserscan \
        ros-${ROS_DISTRO}-tf2-tools \
        ros-${ROS_DISTRO}-tf-transformations \
        ros-${ROS_DISTRO}-cv-bridge \
        python3-numpy python3-yaml && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
COPY docker/nav-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /exp
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
