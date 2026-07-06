"""MuJoCo sim + policy + RViz.

    ros2 launch fbb_policy sim.launch.py [rviz:=false]

Drive with any Twist teleop, e.g.:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
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
    share = get_package_share_directory("fbb_policy")
    xacro_file = os.path.join(share, "urdf", "fbb_sim.urdf.xacro")
    robot_description = ParameterValue(Command(f"xacro {xacro_file}"), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="fbb_policy",
                executable="mujoco_sim_node",
                output="screen",
            ),
            Node(
                package="fbb_policy",
                executable="policy_node",
                output="screen",
                parameters=[
                    {
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
                arguments=["-d", os.path.join(share, "rviz", "fbb.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
