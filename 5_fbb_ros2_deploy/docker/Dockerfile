# Sim + RViz + (optionally) real-robot deps for the fbb RL policy stack.
# Build from the REPO ROOT so both source trees are in context:
#   docker build -f 5_fbb_ros2_deploy/docker/Dockerfile -t fbb-deploy .
# or just: 5_fbb_ros2_deploy/run.sh build

FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]
STOPSIGNAL SIGINT

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-humble-xacro \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers \
        ros-humble-teleop-twist-keyboard \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo for the sim node (physics only, no GL viewer needed).
RUN pip3 install --no-cache-dir mujoco==3.2.7

WORKDIR /ros2_ws
COPY quadruped_ros2_original/four_bar_bot_description src/four_bar_bot_description
COPY 5_fbb_ros2_deploy/ros2_ws/src/fbb_policy src/fbb_policy

# The description package does not install its mujoco/ dir; fbb_policy ships
# its own copies of the MJX XMLs, so only launch/meshes/urdf/rviz are needed.
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select \
        four_bar_bot_description fbb_policy

COPY 5_fbb_ros2_deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
