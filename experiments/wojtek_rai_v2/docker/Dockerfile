# RAI (RobotecAI) runtime for the Wojtek simulation experiment. A second
# container next to wojtek_robot: same host network, same ROS_DOMAIN_ID and
# CycloneDDS profile (see compose.yaml), nothing installed into wojtek_robot.
#
# Build once per machine (x86_64 and aarch64 both work: ros:jazzy-ros-base is
# multi-arch and the RAI stack is pure Python + opencv-python-headless wheels):
#   ./experiments/wojtek_rai_v2/run.sh build
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ARG RAI_CORE_VERSION=2.12.1
ARG RAI_WHOAMI_VERSION=0.0.5
ENV DEBIAN_FRONTEND=noninteractive

# cv_bridge: rai.tools.ros2.generic.topics imports it at module scope.
# nav2-msgs + nav2-simple-commander: rai.tools.ros2 imports its Nav2 tools
# unconditionally (none are used here, but the package must import).
# tf-transformations: exec dependency of rai_interfaces.
# rmw-cyclonedds-cpp: Wojtek's DDS (ros/docker/compose.yaml).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv git ros-dev-tools \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-nav2-msgs \
        ros-${ROS_DISTRO}-nav2-simple-commander \
        ros-${ROS_DISTRO}-tf-transformations && \
    rm -rf /var/lib/apt/lists/*

# rai_interfaces (HRIMessage etc.), pinned by SHA in ros/rai_interfaces.repos.
# Not on packages.ros.org, so it is built here. Its exec-only deps (Nav2,
# portaudio) are skipped: this experiment uses neither.
WORKDIR /rai_ws
COPY ros/rai_interfaces.repos /rai_ws/rai_interfaces.repos
RUN mkdir -p src && vcs import src < rai_interfaces.repos && \
    apt-get update && \
    rosdep update --rosdistro "${ROS_DISTRO}" && \
    rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys "portaudio19-dev" && \
    rm -rf /var/lib/apt/lists/* && \
    /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
        colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF"

# The RAI stack in a venv that sees the system site-packages (rclpy,
# cv_bridge, numpy 1.26). A venv rather than the system interpreter because
# pip cannot replace Debian-installed packages that lack a RECORD file
# (pluggy, for one); the venv shadows them instead. It lives in the image,
# never under the bind-mounted checkout, so no root-owned files appear there.
ENV RAI_VENV=/opt/rai-venv
RUN python3 -m venv --system-site-packages "${RAI_VENV}" && \
    "${RAI_VENV}/bin/pip" install --no-cache-dir \
        "rai-core==${RAI_CORE_VERSION}" \
        "rai-whoami==${RAI_WHOAMI_VERSION}" \
        "pytest>=8.0" "pytest-timeout>=2.3"
ENV PATH="${RAI_VENV}/bin:${PATH}"

# Interactive shells and every `docker exec` through the entrypoint see ROS,
# rai_interfaces and the experiment on PYTHONPATH.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /rai_ws/install/setup.bash" >> /root/.bashrc
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The experiment directory is bind-mounted here (compose.yaml). RAI resolves
# config.toml relative to the cwd, so every process starts in /exp.
WORKDIR /exp
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
