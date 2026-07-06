#!/usr/bin/env bash
# Entry points for the fbb ROS 2 deploy stack (native ROS 2 Humble).
# Docker variants (docker-*) exist only for hosts without ROS installed.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd .. && pwd)"
RL_PY="$REPO/4_four_bar_bot_rl/.venv/bin/python"

ros_env() {
  if [ -d "/opt/ros/${ROS_DISTRO:-humble}" ]; then
    # shellcheck disable=SC1091
    source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
  else
    # No system ROS (e.g. Fedora): use the RoboStack conda env instead.
    # Its activation scripts reference unset vars, so relax -u around them.
    set +u
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ros-humble
    set -u
  fi
  set +u
  [ -f ros2_ws/install/setup.bash ] && source ros2_ws/install/setup.bash
  set -u
  return 0
}

case "${1:-}" in
  deps)       # one-time dependencies
    if command -v apt-get >/dev/null; then   # Ubuntu/Debian with system ROS
      sudo apt-get install -y ros-humble-xacro ros-humble-robot-state-publisher \
        ros-humble-teleop-twist-keyboard ros-humble-rviz2 \
        ros-humble-ros2-control ros-humble-ros2-controllers
      pip3 install --user mujoco
    else                                     # Fedora etc.: RoboStack conda env
      # shellcheck disable=SC1091
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
      conda create -n ros-humble --override-channels \
        -c conda-forge -c robostack-staging -y \
        ros-humble-desktop ros-humble-xacro ros-humble-teleop-twist-keyboard \
        colcon-common-extensions compilers cmake pkg-config make ninja python=3.11
      # setuptools>=80 drops `setup.py develop --editable` which colcon
      # --symlink-install still uses
      conda run -n ros-humble pip install mujoco "setuptools<80"
    fi ;;

  build)      # colcon build (description package + fbb_policy)
    ros_env
    ln -sfn ../../../quadruped_ros2_original/four_bar_bot_description \
      ros2_ws/src/four_bar_bot_description
    cd ros2_ws && colcon build --symlink-install \
      --packages-select four_bar_bot_description fbb_policy ;;

  sim)        # MuJoCo sim + policy + RViz
    ros_env
    ros2 launch fbb_policy sim.launch.py "${@:2}" ;;

  teleop)     # keyboard teleop (run next to sim)
    ros_env
    ros2 run teleop_twist_keyboard teleop_twist_keyboard ;;

  wasd)       # hold-to-move WASD teleop (release key = stop)
    ros_env
    python3 tools/wasd_teleop.py ;;

  real-build) # additionally build the hardware interfaces (on the robot)
    ros_env
    git -C "$REPO" submodule update --init \
      quadruped_ros2_original/md80_hardware_interface/3rd_party/candle
    ln -sfn ../../../quadruped_ros2_original/md80_hardware_interface \
      ros2_ws/src/md80_hardware_interface
    ln -sfn ../../../quadruped_ros2_original/bmx160_serial_hardware_interface \
      ros2_ws/src/bmx160_serial_hardware_interface
    cd ros2_ws && colcon build --symlink-install --packages-select \
      md80_hardware_interface bmx160_serial_hardware_interface ;;

  real)       # on the robot; starts DISARMED, see README
    ros_env
    ros2 launch fbb_policy real.launch.py "${@:2}" ;;

  export-policy)  # re-export the checkpoint into the ROS package
    (cd "$REPO/4_four_bar_bot_rl" && ./run.sh export --run "${2:-policies/fbb_v3}")
    cp "$REPO/4_four_bar_bot_rl/${2:-policies/fbb_v3}/deploy/policy.npz" \
       "$REPO/4_four_bar_bot_rl/${2:-policies/fbb_v3}/deploy/policy_meta.json" \
       ros2_ws/src/fbb_policy/config/
    echo "policy refreshed -- rebuild: ./run.sh build" ;;

  fit-map)    # regenerate joint_map.yaml (URDF<->MuJoCo joint conventions)
    TMP=$(mktemp -d)
    cp "$REPO/quadruped_ros2_original/four_bar_bot_description/urdf/"*.xacro "$TMP/"
    sed -i "s|\$(find four_bar_bot_description)/urdf|$TMP|g" "$TMP/"*.xacro
    sed -i "s|\$(find four_bar_bot_description)|$REPO/quadruped_ros2_original/four_bar_bot_description|g" "$TMP/"*.xacro
    sed -i "s|\$(find quadruped_controller)|$REPO/quadruped_ros2_original/quadruped_controller|g" "$TMP/"*.xacro
    FBB_URDF_XACRO="$TMP/four_bar_bot.urdf.xacro" "$RL_PY" tools/fit_joint_map.py
    rm -rf "$TMP" ;;

  test)       # pure-python unit tests (no ROS needed)
    "$RL_PY" -m pytest ros2_ws/src/fbb_policy/test -q "${@:2}" ;;

  docker-build)   # fallback for hosts without ROS
    docker build -f docker/Dockerfile -t fbb-deploy "$REPO" ;;
  docker-sim)
    xhost +local:docker >/dev/null 2>&1 || true
    docker compose -f docker/compose.yaml up sim ;;
  docker-teleop)
    docker compose -f docker/compose.yaml run --rm teleop ;;
  docker-check)   # headless end-to-end smoke test in the container
    docker run --rm -v "$PWD/docker/ci_check.sh:/ci_check.sh:ro" fbb-deploy bash /ci_check.sh ;;

  *)
    echo "usage: run.sh {deps|build|sim|teleop|real-build|real|export-policy|fit-map|test|docker-*}"
    exit 1 ;;
esac
