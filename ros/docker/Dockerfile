ARG ROS_DISTRO=jazzy
# Pinned to match src/md80_hardware_interface/3rd_party/candle's submodule
# commit. The container fetches it itself rather than relying on whoever
# built the image having run `git submodule update --init` on the host -
# a plain `git clone` (no --recursive) of this repo must be enough.
ARG CANDLE_COMMIT=7c201d744baa107ee408e3e01694efd1a764a174

FROM ros:${ROS_DISTRO}-ros-core
ARG CANDLE_COMMIT

STOPSIGNAL SIGINT

WORKDIR /ros2_ws

# Manifests only, not the full tree yet: apt-get/rosdep/candle below are the
# slow part (network installs) but don't actually depend on source content,
# only on declared dependencies. Copying just package.xml means ordinary code
# edits under src/ (frequent) don't bust this layer's cache - only a changed
# dependency (rare) does. Can't glob this (every package.xml has the same
# basename and would collide in the destination), so list packages explicitly.
COPY ./src/wojtek_description/package.xml ./src/wojtek_description/package.xml
COPY ./src/md80_hardware_interface/package.xml ./src/md80_hardware_interface/package.xml
COPY ./src/bmi160_serial_hardware_interface/package.xml ./src/bmi160_serial_hardware_interface/package.xml
COPY ./src/bmx160_serial_hardware_interface/package.xml ./src/bmx160_serial_hardware_interface/package.xml
COPY ./src/imu_i2c_hardware_interface/package.xml ./src/imu_i2c_hardware_interface/package.xml
COPY ./src/wojtek_policy/package.xml ./src/wojtek_policy/package.xml
COPY ./src/wojtek_bringup/package.xml ./src/wojtek_bringup/package.xml
COPY ./src/wojtek_pc/package.xml ./src/wojtek_pc/package.xml
COPY ./src/wojtek_teleop/package.xml ./src/wojtek_teleop/package.xml
COPY ./src/magnetometer_broadcaster/package.xml ./src/magnetometer_broadcaster/package.xml

# This is a dev/build environment, not a slimmed-down production image:
# ros-dev-tools (colcon, rosdep, compilers, git, ...) stay installed so
# anyone can edit code (src/ is bind-mounted at runtime, see compose.yaml)
# and rebuild inside the container without reinstalling anything.
# The mesa EGL/OSMesa libraries are for MuJoCo's offscreen renderer (the
# sim's virtual D435 camera): surfaceless EGL works headless with no GPU,
# OSMesa is the software fallback. Without them `MUJOCO_GL=egl` raises at
# import (the sim node then degrades to camera-off).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-dev-tools udev git python3-pip setserial openssh-client \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-plotjuggler-ros \
        libegl1 libegl-mesa0 libgl1 libglx-mesa0 libosmesa6 && \
    mkdir -p src/md80_hardware_interface/3rd_party && \
    rm -rf src/md80_hardware_interface/3rd_party/candle && \
    git clone https://github.com/mabrobotics/candle src/md80_hardware_interface/3rd_party/candle && \
    git -C src/md80_hardware_interface/3rd_party/candle checkout "$CANDLE_COMMIT" && \
    (rosdep init || true) && \
    rosdep update --rosdistro "$ROS_DISTRO" && \
    rosdep install --from-paths src -y -i && \
    pip3 install --break-system-packages mujoco huggingface_hub && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The real source, now that the slow network-installs above are cached.
# Everything from here on re-runs on every src/ change, but that's just the
# compile step below, not the ROS/apt/candle download.
COPY ./src /ros2_ws/src/

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --symlink-install \
        --packages-select wojtek_description md80_hardware_interface \
            bmx160_serial_hardware_interface bmi160_serial_hardware_interface \
            imu_i2c_hardware_interface \
            wojtek_teleop wojtek_policy wojtek_bringup wojtek_pc magnetometer_broadcaster \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF"

# So an interactive `docker exec -it <container> bash` also has ROS +
# the workspace overlay sourced (docker exec bypasses ENTRYPOINT).
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
# Stays up as a standing dev environment; exec into it and run
# `ros2 run`/`ros2 launch`/`colcon build` manually. Not meant to
# auto-launch the robot - this is a develop/test image, not a
# production one yet.
CMD ["sleep", "infinity"]
