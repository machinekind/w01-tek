# The operator GUI's runtime: ROS 2 Jazzy + a small ASGI server for the JS page. A
# container next to wojtek_robot on the same host network and DDS domain
# (compose.yaml); nothing is installed into wojtek_robot. The page is a
# thin ROS 2 client of wojtek_nav's vlm_brain_node: no model, no motion
# logic here.
#
# Build once per machine (x86_64 and aarch64 both work: ros:jazzy-ros-base
# is multi-arch and the venv is pure-Python wheels):
#   ./experiments/wojtek_vlm_gui/run.sh build
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# rmw-cyclonedds-cpp: Wojtek's DDS (ros/docker/compose.yaml). std_msgs,
# sensor_msgs and std_srvs come with ros-base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

# The web server in a venv that sees the system site-packages (rclpy, the
# message packages, numpy 1.26). numpy is pinned below 2 so the venv never
# shadows the numpy rclpy's C extensions were built against. It lives in
# the image, never under the bind-mounted checkout.
ENV GUI_VENV=/opt/gui-venv
RUN python3 -m venv --system-site-packages "${GUI_VENV}" && \
    "${GUI_VENV}/bin/pip" install --no-cache-dir \
        "starlette>=0.40" "uvicorn>=0.30" "websockets>=12" "pillow>=10" \
        "numpy<2" "pytest>=8.0" "pytest-timeout>=2.3"
ENV PATH="${GUI_VENV}/bin:${PATH}"

# Interactive shells and every `docker exec` through the entrypoint see
# ROS and the experiment on PYTHONPATH.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The experiment directory is bind-mounted here (compose.yaml).
WORKDIR /exp
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
