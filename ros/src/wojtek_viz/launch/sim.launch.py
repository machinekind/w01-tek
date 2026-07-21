"""MuJoCo sim + policy + RViz.

    ros2 launch wojtek_viz sim.launch.py [rviz:=false] [initial_pose:=folded]

initial_pose:=folded spawns the robot in the real robot's boot/zeroing pose
(lying flat, knees folded -- see real_io_node) instead of standing at home.
Note the policy starts immediately, so from folded it will try to stand on
its own; use this to inspect/verify the boot pose, not for clean walking.

Drive with any Twist teleop, e.g.:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
    ros2 launch wojtek_viz gamepad.launch.py   # bluetooth Xbox pad
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("wojtek_viz")
    policy_share = get_package_share_directory("wojtek_policy")
    xacro_file = os.path.join(share, "urdf", "wojtek_sim.urdf.xacro")
    robot_description = ParameterValue(Command(f"xacro {xacro_file}"), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("initial_pose", default_value="home"),
            # HF repo id (org/name[@revision]) or a local directory with
            # policy.npz + policy_meta.json -- see wojtek_policy/policy_source.py.
            DeclareLaunchArgument(
                "policy",
                default_value="<HF_ORGANIZATION>/wojtek-springy-locomotion",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="wojtek_viz",
                executable="mujoco_sim_node",
                output="screen",
                parameters=[{"initial_pose": LaunchConfiguration("initial_pose")}],
            ),
            Node(
                package="wojtek_policy",
                executable="policy_node",
                output="screen",
                parameters=[
                    {
                        "policy": LaunchConfiguration("policy"),
                        "imu_mount_rpy": [0.0, 0.0, 0.0],  # sim IMU is in base_link
                        "auto_enable": True,
                        "soft_start_s": 0.5,
                        "clamp_knee": False,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(policy_share, "rviz", "wojtek.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
